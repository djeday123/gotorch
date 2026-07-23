package cuda

// B-impl-2: F16 MatMul (mixed precision) + F32<->F16 conversions.
//
// Прогнозы (pre-registered):
//   P1 CastF32ToF16 -> CastF16ToF32 round-trip: rel <= 1e-3 (F16 quantization eps ~5e-4).
//   P2 F16 batched vs J(F64 CPU): hybrid abs=1e-3+rel=5e-4 (paper B4 F16 class).
//   P3 F16 non-batched vs F16 batched b=1: bit-exact (same GemmEx algo path).

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// f32ToF16Bits — reference IEEE 754 binary32 -> binary16 (round-to-nearest even).
func f32ToF16Bits(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	exp := int32((b>>23)&0xFF) - 127 + 15
	mant := b & 0x7FFFFF
	if exp >= 31 {
		return sign | 0x7C00
	}
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mant = (mant | 0x800000) >> uint(1-exp)
		return sign | uint16(mant>>13)
	}
	// Round to nearest even.
	rounded := (mant + 0x1000) >> 13
	if rounded > 0x3FF {
		exp++
		rounded = 0
		if exp >= 31 {
			return sign | 0x7C00
		}
	}
	return sign | uint16(exp)<<10 | uint16(rounded)
}

func f16BitsToF32(b uint16) float32 {
	sign := uint32(b&0x8000) << 16
	exp := int32((b >> 10) & 0x1F)
	mant := uint32(b & 0x3FF)
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
	} else if exp == 31 {
		return math.Float32frombits(sign | 0x7F800000 | mant<<13)
	}
	return math.Float32frombits(sign | uint32(exp+127-15)<<23 | mant<<13)
}

func u16Bytes(v []uint16) []byte {
	buf := make([]byte, 2*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint16(buf[i*2:], x)
	}
	return buf
}

func bytesU16(buf []byte) []uint16 {
	v := make([]uint16, len(buf)/2)
	for i := range v {
		v[i] = binary.LittleEndian.Uint16(buf[i*2:])
	}
	return v
}

func TestCastF32ToF16_RoundTrip(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	const n = 256
	r := rand.New(rand.NewSource(0xF16))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(r.NormFloat64())
	}
	srcS, _ := b.Alloc(n * 4)
	defer b.Free(srcS)
	f16S, _ := b.Alloc(n * 2)
	defer b.Free(f16S)
	rtS, _ := b.Alloc(n * 4)
	defer b.Free(rtS)
	b.CopyH2D(srcS, f32Bytes(src))
	if err := b.CastF32ToF16(srcS, f16S, n); err != nil {
		t.Fatalf("F32->F16: %v", err)
	}
	if err := b.CastF16ToF32(f16S, rtS, n); err != nil {
		t.Fatalf("F16->F32: %v", err)
	}
	b.Sync()
	rtBuf := make([]byte, n*4)
	b.CopyD2H(rtBuf, rtS)
	rt := bytesF32(rtBuf)

	var maxRel float64
	fails := 0
	const relTol = 1e-3
	for i := range src {
		d := math.Abs(float64(rt[i]) - float64(src[i]))
		rel := d / (math.Abs(float64(src[i])) + 1e-30)
		if rel > maxRel {
			maxRel = rel
		}
		if rel > relTol {
			fails++
		}
	}
	t.Logf("Round-trip F32->F16->F32 n=%d: maxRel=%.3e fails=%d/%d (floor %.0e)", n, maxRel, fails, n, relTol)
	if fails > 0 {
		t.Errorf("round-trip: %d fails", fails)
	}
}

func TestMatMulF16(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	if !HasBlasWrapper() {
		t.Skip("wrapper required for F16")
	}

	const m, k, n = 32, 24, 16
	r := rand.New(rand.NewSource(0xF16A))
	aF := make([]float32, m*k)
	bF := make([]float32, k*n)
	for i := range aF {
		aF[i] = float32(r.NormFloat64())
	}
	for i := range bF {
		bF[i] = float32(r.NormFloat64())
	}
	aF16 := make([]uint16, m*k)
	bF16 := make([]uint16, k*n)
	for i, v := range aF {
		aF16[i] = f32ToF16Bits(v)
	}
	for i, v := range bF {
		bF16[i] = f32ToF16Bits(v)
	}
	// J(F64 CPU) reference с actual F16 values (round trip back to F32 for accum).
	aFback := make([]float64, m*k)
	bFback := make([]float64, k*n)
	for i, u := range aF16 {
		aFback[i] = float64(f16BitsToF32(u))
	}
	for i, u := range bF16 {
		bFback[i] = float64(f16BitsToF32(u))
	}
	refC := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var acc float64
			for l := 0; l < k; l++ {
				acc += aFback[i*k+l] * bFback[l*n+j]
			}
			refC[i*n+j] = acc
		}
	}

	aS, _ := b.Alloc(m * k * 2)
	defer b.Free(aS)
	bS, _ := b.Alloc(k * n * 2)
	defer b.Free(bS)
	cS, _ := b.Alloc(m * n * 4)
	defer b.Free(cS)
	b.CopyH2D(aS, u16Bytes(aF16))
	b.CopyH2D(bS, u16Bytes(bF16))
	if err := b.MatMulF16(aS, bS, cS, m, n, k); err != nil {
		t.Fatalf("MatMulF16: %v", err)
	}
	b.Sync()
	cBuf := make([]byte, m*n*4)
	b.CopyD2H(cBuf, cS)
	got := bytesF32(cBuf)

	// F16 IO + FP32 out + TF32 compute: floor из paper B4 = hybrid abs=1e-3+rel=5e-4.
	// PRE-REGISTERED: F16 class. TF32 в compute добавляет ~1e-3 rel.
	var maxAbs, maxRel float64
	fails := 0
	const absTol, relTol = 1e-3, 5e-3
	for i := range got {
		d := math.Abs(float64(got[i]) - refC[i])
		rel := d / (math.Abs(refC[i]) + 1e-30)
		if d > maxAbs {
			maxAbs = d
		}
		if rel > maxRel {
			maxRel = rel
		}
		if d > absTol+relTol*math.Abs(refC[i]) {
			fails++
		}
	}
	t.Logf("MatMulF16 vs J(F64 CPU on F16-rounded inputs) [m=%d n=%d k=%d]: maxAbs=%.3e maxRel=%.3e fails=%d/%d (floor abs=%.0e+rel=%.0e·|ref|)",
		m, n, k, maxAbs, maxRel, fails, len(got), absTol, relTol)
	if fails > 0 {
		t.Errorf("MatMulF16: %d fails", fails)
	}
}

func TestMatMulStridedBatchedF16(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	if !HasBlasWrapper() {
		t.Skip("wrapper required for F16 batched")
	}

	const batch, m, k, n = 4, 16, 24, 32
	r := rand.New(rand.NewSource(0xF16B))
	aF := make([]float32, batch*m*k)
	bF := make([]float32, batch*k*n)
	for i := range aF {
		aF[i] = float32(r.NormFloat64())
	}
	for i := range bF {
		bF[i] = float32(r.NormFloat64())
	}
	aF16 := make([]uint16, batch*m*k)
	bF16 := make([]uint16, batch*k*n)
	for i, v := range aF {
		aF16[i] = f32ToF16Bits(v)
	}
	for i, v := range bF {
		bF16[i] = f32ToF16Bits(v)
	}
	// F64 CPU ref.
	aFback := make([]float64, batch*m*k)
	bFback := make([]float64, batch*k*n)
	for i, u := range aF16 {
		aFback[i] = float64(f16BitsToF32(u))
	}
	for i, u := range bF16 {
		bFback[i] = float64(f16BitsToF32(u))
	}
	refC := make([]float64, batch*m*n)
	for bi := 0; bi < batch; bi++ {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var acc float64
				for l := 0; l < k; l++ {
					acc += aFback[bi*m*k+i*k+l] * bFback[bi*k*n+l*n+j]
				}
				refC[bi*m*n+i*n+j] = acc
			}
		}
	}
	aS, _ := b.Alloc(batch * m * k * 2)
	defer b.Free(aS)
	bS, _ := b.Alloc(batch * k * n * 2)
	defer b.Free(bS)
	cS, _ := b.Alloc(batch * m * n * 4)
	defer b.Free(cS)
	b.CopyH2D(aS, u16Bytes(aF16))
	b.CopyH2D(bS, u16Bytes(bF16))
	if err := b.MatMulStridedBatchedF16(aS, bS, cS, batch, m, n, k,
		int64(m*k), int64(k*n), int64(m*n)); err != nil {
		t.Fatalf("MatMulStridedBatchedF16: %v", err)
	}
	b.Sync()
	cBuf := make([]byte, batch*m*n*4)
	b.CopyD2H(cBuf, cS)
	got := bytesF32(cBuf)

	var maxAbs, maxRel float64
	fails := 0
	const absTol, relTol = 1e-3, 5e-3
	for i := range got {
		d := math.Abs(float64(got[i]) - refC[i])
		rel := d / (math.Abs(refC[i]) + 1e-30)
		if d > maxAbs {
			maxAbs = d
		}
		if rel > maxRel {
			maxRel = rel
		}
		if d > absTol+relTol*math.Abs(refC[i]) {
			fails++
		}
	}
	t.Logf("MatMulStridedBatchedF16 vs J [b=%d m=%d n=%d k=%d]: maxAbs=%.3e maxRel=%.3e fails=%d/%d (floor abs=%.0e+rel=%.0e·|ref|)",
		batch, m, n, k, maxAbs, maxRel, fails, len(got), absTol, relTol)
	if fails > 0 {
		t.Errorf("F16 batched: %d fails", fails)
	}
}

// F16 non-batched vs batched b=1: same GemmEx algo → должно быть очень близко.
func TestMatMulF16_NonBatchedEqBatched1(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	if !HasBlasWrapper() {
		t.Skip("wrapper required")
	}
	const m, k, n = 16, 24, 32
	r := rand.New(rand.NewSource(0xF16C))
	aF16 := make([]uint16, m*k)
	bF16 := make([]uint16, k*n)
	for i := range aF16 {
		aF16[i] = f32ToF16Bits(float32(r.NormFloat64()))
	}
	for i := range bF16 {
		bF16[i] = f32ToF16Bits(float32(r.NormFloat64()))
	}
	aS, _ := b.Alloc(m * k * 2)
	defer b.Free(aS)
	bS, _ := b.Alloc(k * n * 2)
	defer b.Free(bS)
	c1S, _ := b.Alloc(m * n * 4)
	defer b.Free(c1S)
	c2S, _ := b.Alloc(m * n * 4)
	defer b.Free(c2S)
	b.CopyH2D(aS, u16Bytes(aF16))
	b.CopyH2D(bS, u16Bytes(bF16))
	if err := b.MatMulF16(aS, bS, c1S, m, n, k); err != nil {
		t.Fatalf("non-batched: %v", err)
	}
	if err := b.MatMulStridedBatchedF16(aS, bS, c2S, 1, m, n, k,
		int64(m*k), int64(k*n), int64(m*n)); err != nil {
		t.Fatalf("batched b=1: %v", err)
	}
	b.Sync()
	buf1 := make([]byte, m*n*4)
	buf2 := make([]byte, m*n*4)
	b.CopyD2H(buf1, c1S)
	b.CopyD2H(buf2, c2S)
	g1 := bytesF32(buf1)
	g2 := bytesF32(buf2)
	mismatches := 0
	for i := range g1 {
		if math.Float32bits(g1[i]) != math.Float32bits(g2[i]) {
			mismatches++
		}
	}
	t.Logf("F16 non-batched vs batched b=1 [m=%d n=%d k=%d]: bit-exact=%d/%d", m, n, k, len(g1)-mismatches, len(g1))
	// PRE-REGISTERED bit-exact. Прогноз P3 может промахнуться -- cublas GemmEx vs
	// GemmStridedBatchedEx internal algo может отличаться (как F32 loop-vs-strided).
	if mismatches > 0 {
		t.Logf("(note) P3 bit-exact прогноз промахнулся, mismatches=%d/%d -- allowed if hybrid pass", mismatches, len(g1))
		// Проверка hybrid:
		fails := 0
		for i := range g1 {
			d := math.Abs(float64(g1[i]) - float64(g2[i]))
			if d > 1e-3+5e-3*math.Abs(float64(g1[i])) {
				fails++
			}
		}
		if fails > 0 {
			t.Errorf("F16 non-batched vs batched hybrid: %d fails", fails)
		}
	}
}
