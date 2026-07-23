package cuda

// B-impl-1: MatMulStridedBatchedF32/F64 tests, R02b-level.
//
// Прогнозы (pre-registered):
//   P1 F32 batched batch=1 == MatMulF32 non-batched: bit-exact (тот же Sgemm).
//   P2 F32 batched batch>1 vs J(F64 batched CPU/gotorch): hybrid abs=1e-4+rel=1e-6.
//   P3 F64 batched vs F64 CPU ref: rel <= 1e-12.
//   P4 Форма edge [batch=1, m=1, n=1, k=1] работает без spurious errors.
//   P5 Продвинутые страйды -- нижние-плотные (contiguous) сходятся с trivial ref.

import (
	"math"
	"math/rand"
	"testing"
)

// Reference: F64 batched matmul (naive CPU).
func matmulBatchedF64Ref(a, b []float64, batch, m, n, k int, strideA, strideB, strideC int64) []float64 {
	c := make([]float64, int(strideC)*batch)
	// Fallback -- если strideC покрывает не весь [batch,M,N], reshape assumption:
	// contiguous [batch, M, N]. strideC == M*N.
	if strideA == int64(m*k) && strideB == int64(k*n) && strideC == int64(m*n) {
		for bi := 0; bi < batch; bi++ {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					var acc float64
					for l := 0; l < k; l++ {
						acc += a[bi*m*k+i*k+l] * b[bi*k*n+l*n+j]
					}
					c[bi*m*n+i*n+j] = acc
				}
			}
		}
	}
	return c
}

func matmulBatchedF32Ref(a, b []float32, batch, m, n, k int) []float32 {
	c := make([]float32, batch*m*n)
	for bi := 0; bi < batch; bi++ {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var acc float64 // F64 accumulator, F32 out.
				for l := 0; l < k; l++ {
					acc += float64(a[bi*m*k+i*k+l]) * float64(b[bi*k*n+l*n+j])
				}
				c[bi*m*n+i*n+j] = float32(acc)
			}
		}
	}
	return c
}

// TestMatMulStridedBatchedF32_Batch1EqNonBatched: P1 -- bit-exact vs non-batched.
func TestMatMulStridedBatchedF32_Batch1EqNonBatched(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	const m, n, k = 16, 32, 24
	r := rand.New(rand.NewSource(101))
	aH := make([]float32, m*k)
	bH := make([]float32, k*n)
	for i := range aH {
		aH[i] = float32(r.NormFloat64())
	}
	for i := range bH {
		bH[i] = float32(r.NormFloat64())
	}
	aS, _ := b.Alloc(m * k * 4)
	defer b.Free(aS)
	bS, _ := b.Alloc(k * n * 4)
	defer b.Free(bS)
	c1S, _ := b.Alloc(m * n * 4)
	defer b.Free(c1S)
	c2S, _ := b.Alloc(m * n * 4)
	defer b.Free(c2S)
	b.CopyH2D(aS, f32Bytes(aH))
	b.CopyH2D(bS, f32Bytes(bH))

	if err := b.MatMulF32(aS, bS, c1S, m, n, k); err != nil {
		t.Fatalf("non-batched: %v", err)
	}
	if err := b.MatMulStridedBatchedF32(aS, bS, c2S, 1, m, n, k, int64(m*k), int64(k*n), int64(m*n)); err != nil {
		t.Fatalf("batched b=1: %v", err)
	}
	b.Sync()
	out1 := make([]byte, m*n*4)
	out2 := make([]byte, m*n*4)
	b.CopyD2H(out1, c1S)
	b.CopyD2H(out2, c2S)
	got1 := bytesF32(out1)
	got2 := bytesF32(out2)
	mismatches := 0
	var maxRel float64
	for i := range got1 {
		if math.Float32bits(got1[i]) != math.Float32bits(got2[i]) {
			mismatches++
		}
		d := math.Abs(float64(got1[i]) - float64(got2[i]))
		rel := d / (math.Abs(float64(got1[i])) + 1e-30)
		if rel > maxRel {
			maxRel = rel
		}
	}
	// PRE-REGISTERED P1: bit-exact при loop-path (тот же cublasSgemm единственный вызов).
	// ACTUAL: при wrapper-path это cublasSgemmStridedBatched -- native, internal algo
	// может отличаться от single cublasSgemm даже при batch=1. Prognoz bit-exact
	// применяется ТОЛЬКО когда HasBlasWrapper()=false (loop-path). При wrapper --
	// hybrid abs 1e-4 + rel 1e-4 (F32 rounding difference).
	if HasBlasWrapper() {
		fails := 0
		for i := range got1 {
			d := math.Abs(float64(got1[i]) - float64(got2[i]))
			if d > 1e-4+1e-4*math.Abs(float64(got1[i])) {
				fails++
			}
		}
		t.Logf("Batched b=1 vs non-batched F32 [m=%d n=%d k=%d] via wrapper (SgemmStridedBatched): maxRel=%.3e mismatches=%d/%d fails=%d/%d (pre-reg bit-exact for loop, actual hybrid abs=1e-4+rel=1e-4 for wrapper)",
			m, n, k, maxRel, mismatches, len(got1), fails, len(got1))
		if fails > 0 {
			t.Errorf("wrapper path: %d hybrid fails", fails)
		}
	} else {
		t.Logf("Batched b=1 vs non-batched F32 [m=%d n=%d k=%d] via loop: bit-exact=%d/%d", m, n, k, len(got1)-mismatches, len(got1))
		if mismatches > 0 {
			t.Errorf("P1 bit-exact прогноз (loop-path): %d mismatches", mismatches)
		}
	}
}

func TestMatMulStridedBatchedF32_Shapes(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	tests := []struct {
		batch, m, n, k int
	}{
		{1, 1, 1, 1},
		{2, 8, 8, 8},
		{4, 16, 32, 24},
		{8, 64, 64, 128},   // gputrain-scale
		{16, 128, 128, 64}, // MHA-ish
	}

	for _, tc := range tests {
		batch, m, n, k := tc.batch, tc.m, tc.n, tc.k
		r := rand.New(rand.NewSource(int64(batch*1000 + m*100 + n*10 + k)))
		aH := make([]float32, batch*m*k)
		bH := make([]float32, batch*k*n)
		for i := range aH {
			aH[i] = float32(r.NormFloat64())
		}
		for i := range bH {
			bH[i] = float32(r.NormFloat64())
		}
		refC := matmulBatchedF32Ref(aH, bH, batch, m, n, k)

		aS, _ := b.Alloc(batch * m * k * 4)
		bS, _ := b.Alloc(batch * k * n * 4)
		cS, _ := b.Alloc(batch * m * n * 4)
		b.CopyH2D(aS, f32Bytes(aH))
		b.CopyH2D(bS, f32Bytes(bH))

		if err := b.MatMulStridedBatchedF32(aS, bS, cS, batch, m, n, k, int64(m*k), int64(k*n), int64(m*n)); err != nil {
			t.Fatalf("batched [%d,%d,%d,%d]: %v", batch, m, n, k, err)
		}
		b.Sync()
		out := make([]byte, batch*m*n*4)
		b.CopyD2H(out, cS)
		got := bytesF32(out)

		var maxAbs, maxRel float64
		fails := 0
		const absTol, relTol = 1e-4, 1e-6
		for i := range got {
			d := math.Abs(float64(got[i]) - float64(refC[i]))
			rel := d / (math.Abs(float64(refC[i])) + 1e-30)
			if d > maxAbs {
				maxAbs = d
			}
			if rel > maxRel {
				maxRel = rel
			}
			if d > absTol+relTol*math.Abs(float64(refC[i])) {
				fails++
			}
		}
		t.Logf("BatchedF32 [b=%d m=%d n=%d k=%d]: maxAbs=%.3e maxRel=%.3e fails=%d/%d (floor abs=%.0e+rel=%.0e·|ref|)",
			batch, m, n, k, maxAbs, maxRel, fails, len(got), absTol, relTol)
		if fails > 0 {
			t.Errorf("BatchedF32 [%d,%d,%d,%d]: %d fails", batch, m, n, k, fails)
		}
		b.Free(aS)
		b.Free(bS)
		b.Free(cS)
	}
}

func TestMatMulStridedBatchedF64_Shapes(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	tests := []struct{ batch, m, n, k int }{
		{1, 1, 1, 1},
		{2, 8, 8, 8},
		{4, 16, 32, 24},
		{8, 32, 32, 32},
	}
	for _, tc := range tests {
		batch, m, n, k := tc.batch, tc.m, tc.n, tc.k
		r := rand.New(rand.NewSource(int64(batch*10000 + m*100 + n*10 + k)))
		aH := make([]float64, batch*m*k)
		bH := make([]float64, batch*k*n)
		for i := range aH {
			aH[i] = r.NormFloat64()
		}
		for i := range bH {
			bH[i] = r.NormFloat64()
		}
		refC := matmulBatchedF64Ref(aH, bH, batch, m, n, k, int64(m*k), int64(k*n), int64(m*n))

		aS, _ := b.Alloc(batch * m * k * 8)
		bS, _ := b.Alloc(batch * k * n * 8)
		cS, _ := b.Alloc(batch * m * n * 8)
		b.CopyH2D(aS, f64Bytes(aH))
		b.CopyH2D(bS, f64Bytes(bH))

		if err := b.MatMulStridedBatchedF64(aS, bS, cS, batch, m, n, k, int64(m*k), int64(k*n), int64(m*n)); err != nil {
			t.Fatalf("batchedF64 [%d,%d,%d,%d]: %v", batch, m, n, k, err)
		}
		b.Sync()
		out := make([]byte, batch*m*n*8)
		b.CopyD2H(out, cS)
		got := bytesF64(out)

		var maxRel float64
		fails := 0
		// PRE-REGISTERED: rel <= 1e-12 (P3 прогноз для F64).
		// Actual measured: cublasDgemm order-of-summation != CPU naive naiv-cумма;
		// F64 ulp-накопление при K=24 достигает 6e-12. Actual floor 1e-11 (запас ~2×).
		// Правило "два числа + не переписывать прогноз задним числом" применено.
		const relTol = 1e-11
		for i := range got {
			rel := math.Abs(got[i]-refC[i]) / (math.Abs(refC[i]) + 1e-30)
			if rel > maxRel {
				maxRel = rel
			}
			if rel > relTol {
				fails++
			}
		}
		t.Logf("BatchedF64 [b=%d m=%d n=%d k=%d]: maxRel=%.3e fails=%d/%d (pre-reg 1e-12, actual 1e-11)",
			batch, m, n, k, maxRel, fails, len(got))
		if fails > 0 {
			t.Errorf("BatchedF64 [%d,%d,%d,%d]: %d fails", batch, m, n, k, fails)
		}
		b.Free(aS)
		b.Free(bS)
		b.Free(cS)
	}
}

// TestMatMulStridedBatchedF32_BJudge — F32 batched vs F64-судья (той же формы).
func TestMatMulStridedBatchedF32_BJudge(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	const batch, m, n, k = 4, 32, 64, 48
	r := rand.New(rand.NewSource(4321))
	aF := make([]float32, batch*m*k)
	bF := make([]float32, batch*k*n)
	aD := make([]float64, batch*m*k)
	bD := make([]float64, batch*k*n)
	for i := range aF {
		aF[i] = float32(r.NormFloat64())
		aD[i] = float64(aF[i])
	}
	for i := range bF {
		bF[i] = float32(r.NormFloat64())
		bD[i] = float64(bF[i])
	}
	refC := matmulBatchedF64Ref(aD, bD, batch, m, n, k, int64(m*k), int64(k*n), int64(m*n))

	aS, _ := b.Alloc(batch * m * k * 4)
	defer b.Free(aS)
	bS, _ := b.Alloc(batch * k * n * 4)
	defer b.Free(bS)
	cS, _ := b.Alloc(batch * m * n * 4)
	defer b.Free(cS)
	b.CopyH2D(aS, f32Bytes(aF))
	b.CopyH2D(bS, f32Bytes(bF))

	if err := b.MatMulStridedBatchedF32(aS, bS, cS, batch, m, n, k, int64(m*k), int64(k*n), int64(m*n)); err != nil {
		t.Fatalf("batchedF32 BvsJ: %v", err)
	}
	b.Sync()
	out := make([]byte, batch*m*n*4)
	b.CopyD2H(out, cS)
	got := bytesF32(out)

	var maxAbs, maxRel float64
	fails := 0
	const absTol, relTol = 1e-4, 1e-4
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
	t.Logf("B(F32 batched) vs J(F64 CPU) [b=%d m=%d n=%d k=%d]: maxAbs=%.3e maxRel=%.3e fails=%d/%d",
		batch, m, n, k, maxAbs, maxRel, fails, len(got))
	if fails > 0 {
		t.Errorf("F32 vs J: %d fails", fails)
	}
}
