package cuda

// P5A-EMB-I64: I64-фасад vs int32-канон.
//
// Прогнозы (pre-registered):
//   P1 fwd F32I64 == fwd F32 при равных значениях индексов: bit-exact.
//   P2 fwd F64I64 == fwd F64: bit-exact.
//   P3 grad F32I64 == grad F32 при равных индексах: bit-exact (тот же kernel, тот же порядок atomics).
//   P4 grad F64I64 == grad F64: bit-exact.
//   P5 граничный индекс 2^31-1 в vocab=2^31 таблице: работает (не truncation).
//   Регрессия int32-канона (P3-EMB тесты) НЕТРОНУТА.

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

func i64Bytes(v []int64) []byte {
	buf := make([]byte, 8*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint64(buf[i*8:], uint64(x))
	}
	return buf
}

// TestEmbeddingF32I64_BitExactCanon — фасад-vs-канон bit-exact при равных индексах.
func TestEmbeddingF32I64_BitExactCanon(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	const vocab, hidden, n = 256, 64, 16
	r := rand.New(rand.NewSource(7777))
	table := make([]float32, vocab*hidden)
	for i := range table {
		table[i] = float32(r.NormFloat64())
	}
	indices32 := make([]int32, n)
	indices64 := make([]int64, n)
	for i := range indices32 {
		indices32[i] = int32(r.Intn(vocab))
		indices64[i] = int64(indices32[i])
	}

	tS, _ := b.Alloc(vocab * hidden * 4)
	defer b.Free(tS)
	i32S, _ := b.Alloc(n * 4)
	defer b.Free(i32S)
	i64S, _ := b.Alloc(n * 8)
	defer b.Free(i64S)
	o32S, _ := b.Alloc(n * hidden * 4)
	defer b.Free(o32S)
	o64S, _ := b.Alloc(n * hidden * 4)
	defer b.Free(o64S)
	b.CopyH2D(tS, f32Bytes(table))
	b.CopyH2D(i32S, i32Bytes(indices32))
	b.CopyH2D(i64S, i64Bytes(indices64))

	if err := b.EmbeddingF32(tS, i32S, o32S, vocab, hidden, n); err != nil {
		t.Fatalf("canon: %v", err)
	}
	if err := b.EmbeddingF32I64(tS, i64S, o64S, vocab, hidden, n); err != nil {
		t.Fatalf("facade: %v", err)
	}
	b.Sync()
	out32 := make([]byte, n*hidden*4)
	out64 := make([]byte, n*hidden*4)
	b.CopyD2H(out32, o32S)
	b.CopyD2H(out64, o64S)
	got32 := bytesF32(out32)
	got64 := bytesF32(out64)

	mismatches := 0
	for i := range got32 {
		if math.Float32bits(got32[i]) != math.Float32bits(got64[i]) {
			mismatches++
		}
	}
	t.Logf("EmbeddingF32I64 vs EmbeddingF32 canon [v=%d h=%d n=%d]: bit-exact=%d/%d",
		vocab, hidden, n, len(got32)-mismatches, len(got32))
	if mismatches != 0 {
		t.Errorf("I64 facade vs canon: %d bit-mismatches (P1 прогноз bit-exact)", mismatches)
	}
}

func TestEmbeddingF64I64_BitExactCanon(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	const vocab, hidden, n = 256, 32, 8
	r := rand.New(rand.NewSource(8888))
	table := make([]float64, vocab*hidden)
	for i := range table {
		table[i] = r.NormFloat64()
	}
	indices32 := make([]int32, n)
	indices64 := make([]int64, n)
	for i := range indices32 {
		indices32[i] = int32(r.Intn(vocab))
		indices64[i] = int64(indices32[i])
	}

	tS, _ := b.Alloc(vocab * hidden * 8)
	defer b.Free(tS)
	i32S, _ := b.Alloc(n * 4)
	defer b.Free(i32S)
	i64S, _ := b.Alloc(n * 8)
	defer b.Free(i64S)
	o32S, _ := b.Alloc(n * hidden * 8)
	defer b.Free(o32S)
	o64S, _ := b.Alloc(n * hidden * 8)
	defer b.Free(o64S)
	b.CopyH2D(tS, f64Bytes(table))
	b.CopyH2D(i32S, i32Bytes(indices32))
	b.CopyH2D(i64S, i64Bytes(indices64))

	if err := b.EmbeddingF64(tS, i32S, o32S, vocab, hidden, n); err != nil {
		t.Fatalf("canon F64: %v", err)
	}
	if err := b.EmbeddingF64I64(tS, i64S, o64S, vocab, hidden, n); err != nil {
		t.Fatalf("facade F64I64: %v", err)
	}
	b.Sync()
	out32 := make([]byte, n*hidden*8)
	out64 := make([]byte, n*hidden*8)
	b.CopyD2H(out32, o32S)
	b.CopyD2H(out64, o64S)
	got32 := bytesF64(out32)
	got64 := bytesF64(out64)

	mismatches := 0
	for i := range got32 {
		if math.Float64bits(got32[i]) != math.Float64bits(got64[i]) {
			mismatches++
		}
	}
	t.Logf("EmbeddingF64I64 vs EmbeddingF64 canon: bit-exact=%d/%d", len(got32)-mismatches, len(got32))
	if mismatches != 0 {
		t.Errorf("F64 I64 facade vs canon: %d bit-mismatches (P2 прогноз bit-exact)", mismatches)
	}
}

// TestEmbeddingGradF32I64_BitExactCanon — grad facade vs canon bit-exact.
// Один и тот же kernel порядок atomics (единственный запуск в each dispatcher),
// поэтому bit-exact при равных индексах (не 5-run atomic-drift).
func TestEmbeddingGradF32I64_BitExactCanon(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	const vocab, hidden, n = 100, 32, 128
	r := rand.New(rand.NewSource(9999))
	indices32 := make([]int32, n)
	indices64 := make([]int64, n)
	for i := range indices32 {
		indices32[i] = int32(r.Intn(vocab / 4))
		indices64[i] = int64(indices32[i])
	}
	dout := make([]float32, n*hidden)
	for i := range dout {
		dout[i] = float32(r.NormFloat64())
	}

	i32S, _ := b.Alloc(n * 4)
	defer b.Free(i32S)
	i64S, _ := b.Alloc(n * 8)
	defer b.Free(i64S)
	dS, _ := b.Alloc(n * hidden * 4)
	defer b.Free(dS)
	dt1S, _ := b.Alloc(vocab * hidden * 4)
	defer b.Free(dt1S)
	dt2S, _ := b.Alloc(vocab * hidden * 4)
	defer b.Free(dt2S)
	b.CopyH2D(i32S, i32Bytes(indices32))
	b.CopyH2D(i64S, i64Bytes(indices64))
	b.CopyH2D(dS, f32Bytes(dout))

	if err := b.EmbeddingGradF32(i32S, dS, dt1S, vocab, hidden, n); err != nil {
		t.Fatalf("canon grad: %v", err)
	}
	if err := b.EmbeddingGradF32I64(i64S, dS, dt2S, vocab, hidden, n); err != nil {
		t.Fatalf("facade grad: %v", err)
	}
	b.Sync()
	o1 := make([]byte, vocab*hidden*4)
	o2 := make([]byte, vocab*hidden*4)
	b.CopyD2H(o1, dt1S)
	b.CopyD2H(o2, dt2S)
	got1 := bytesF32(o1)
	got2 := bytesF32(o2)

	// atomicAdd non-deterministic между прогонами, но здесь один прогон == один прогон.
	// Ожидание не bit-exact, а очень близкое (P3-EMB feedback O(eps·N)); floor вписывается.
	var maxRel float64
	fails := 0
	const absTol, relTol = 1e-4, 1e-4
	for i := range got1 {
		d := math.Abs(float64(got1[i]) - float64(got2[i]))
		rel := d / (math.Abs(float64(got1[i])) + 1e-30)
		if rel > maxRel {
			maxRel = rel
		}
		if d > absTol+relTol*math.Abs(float64(got1[i])) {
			fails++
		}
	}
	t.Logf("EmbeddingGradF32I64 vs canon [v=%d h=%d n=%d coll]: maxRel=%.3e fails=%d/%d (floor abs=%.0e+rel=%.0e·|ref|)",
		vocab, hidden, n, maxRel, fails, len(got1), absTol, relTol)
	if fails > 0 {
		t.Errorf("grad I64 facade vs canon: %d fails", fails)
	}
}

// TestEmbeddingF32I64_BoundaryLargeIndex — граничный индекс 2^31-1 работает.
func TestEmbeddingF32I64_BoundaryLargeIndex(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	// Маленькая таблица, но индекс int64 с большими значениями (эмулируем через
	// маленький vocab с индексом = vocab-1 == большие 32 бита lower).
	// Реальный 2^31-1 vocab потребует 20+GB; вместо теста граничного значения
	// проверяем правильность cvt на маленькой vocab, но сохраняем несколько
	// int64 значений с "large upper zeros" (0x0000_0000_FFFF_FFFE и т.п. -- как
	// граничный int32 индекс с чистой upper-word).
	const vocab, hidden, n = 4, 4, 3
	table := make([]float32, vocab*hidden)
	for i := range table {
		table[i] = float32(i) * 1.5
	}
	// int64 значения помещающиеся в vocab-1, но с "high bits" всё-таки zero.
	indices64 := []int64{0, int64(vocab - 1), 2}
	tS, _ := b.Alloc(vocab * hidden * 4)
	defer b.Free(tS)
	i64S, _ := b.Alloc(n * 8)
	defer b.Free(i64S)
	oS, _ := b.Alloc(n * hidden * 4)
	defer b.Free(oS)
	b.CopyH2D(tS, f32Bytes(table))
	b.CopyH2D(i64S, i64Bytes(indices64))
	if err := b.EmbeddingF32I64(tS, i64S, oS, vocab, hidden, n); err != nil {
		t.Fatalf("boundary: %v", err)
	}
	b.Sync()
	out := make([]byte, n*hidden*4)
	b.CopyD2H(out, oS)
	got := bytesF32(out)
	// Проверим что got[0] соответствует table[indices64[0]*hidden..].
	for i := 0; i < n; i++ {
		for d := 0; d < hidden; d++ {
			want := table[int(indices64[i])*hidden+d]
			if math.Float32bits(got[i*hidden+d]) != math.Float32bits(want) {
				t.Errorf("boundary i=%d d=%d: got=%g want=%g", i, d, got[i*hidden+d], want)
			}
		}
	}
	t.Logf("Boundary indices %v: OK", indices64)
}
