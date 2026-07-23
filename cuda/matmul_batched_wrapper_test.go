package cuda

// B-impl-1 доработка Step C: A/B wrapper vs loop.
//
// Прогноз P1: NOT bit-exact -- wrapper использует native cublasSgemmStridedBatched,
// loop использует single cublasSgemm по batch. Internal algo может отличаться.
// Actual floor: hybrid abs=1e-4 + rel=1e-4 (F32 rounding class).
//
// Прогноз P2: F64 wrapper vs loop -- rel <= 1e-11 (не 1e-12 per B-impl-1 findings).

import (
	"math"
	"math/rand"
	"testing"
)

func TestWrapperVsLoop_F32(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	if !HasBlasWrapper() {
		t.Skip("wrapper .so not found -- test requires wrapper path; set GOTORCH_LIBS_DIR or build libs/blas_wrapper")
	}

	const batch, m, k, n = 4, 16, 24, 32
	r := rand.New(rand.NewSource(0xB1DAF))
	aH := make([]float32, batch*m*k)
	bH := make([]float32, batch*k*n)
	for i := range aH {
		aH[i] = float32(r.NormFloat64())
	}
	for i := range bH {
		bH[i] = float32(r.NormFloat64())
	}

	aS, _ := b.Alloc(batch * m * k * 4)
	defer b.Free(aS)
	bS, _ := b.Alloc(batch * k * n * 4)
	defer b.Free(bS)
	c1S, _ := b.Alloc(batch * m * n * 4) // wrapper
	defer b.Free(c1S)
	c2S, _ := b.Alloc(batch * m * n * 4) // loop
	defer b.Free(c2S)
	b.CopyH2D(aS, f32Bytes(aH))
	b.CopyH2D(bS, f32Bytes(bH))

	// Wrapper path (public method).
	if err := b.MatMulStridedBatchedF32(aS, bS, c1S, batch, m, n, k, int64(m*k), int64(k*n), int64(m*n)); err != nil {
		t.Fatalf("wrapper: %v", err)
	}
	// Loop path (internal method).
	if err := b.matMulBatchedF32Loop(aS, bS, c2S, batch, m, n, k, int64(m*k), int64(k*n), int64(m*n)); err != nil {
		t.Fatalf("loop: %v", err)
	}
	b.Sync()
	out1 := make([]byte, batch*m*n*4)
	out2 := make([]byte, batch*m*n*4)
	b.CopyD2H(out1, c1S)
	b.CopyD2H(out2, c2S)
	got1 := bytesF32(out1)
	got2 := bytesF32(out2)

	var maxAbs, maxRel float64
	bitExactCount := 0
	fails := 0
	const absTol, relTol = 1e-4, 1e-4
	for i := range got1 {
		if math.Float32bits(got1[i]) == math.Float32bits(got2[i]) {
			bitExactCount++
		}
		d := math.Abs(float64(got1[i]) - float64(got2[i]))
		rel := d / (math.Abs(float64(got1[i])) + 1e-30)
		if d > maxAbs {
			maxAbs = d
		}
		if rel > maxRel {
			maxRel = rel
		}
		if d > absTol+relTol*math.Abs(float64(got1[i])) {
			fails++
		}
	}
	t.Logf("Wrapper vs Loop F32 [b=%d m=%d n=%d k=%d]: bit-exact=%d/%d maxAbs=%.3e maxRel=%.3e fails=%d/%d (floor abs=%.0e+rel=%.0e·|ref|)",
		batch, m, n, k, bitExactCount, len(got1), maxAbs, maxRel, fails, len(got1), absTol, relTol)
	if fails > 0 {
		t.Errorf("wrapper vs loop F32: %d fails", fails)
	}
}

func TestWrapperVsLoop_F64(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	if !HasBlasWrapper() {
		t.Skip("wrapper .so not found -- test requires wrapper path")
	}

	const batch, m, k, n = 4, 16, 24, 32
	r := rand.New(rand.NewSource(0xB1DAF7))
	aH := make([]float64, batch*m*k)
	bH := make([]float64, batch*k*n)
	for i := range aH {
		aH[i] = r.NormFloat64()
	}
	for i := range bH {
		bH[i] = r.NormFloat64()
	}

	aS, _ := b.Alloc(batch * m * k * 8)
	defer b.Free(aS)
	bS, _ := b.Alloc(batch * k * n * 8)
	defer b.Free(bS)
	c1S, _ := b.Alloc(batch * m * n * 8)
	defer b.Free(c1S)
	c2S, _ := b.Alloc(batch * m * n * 8)
	defer b.Free(c2S)
	b.CopyH2D(aS, f64Bytes(aH))
	b.CopyH2D(bS, f64Bytes(bH))

	if err := b.MatMulStridedBatchedF64(aS, bS, c1S, batch, m, n, k, int64(m*k), int64(k*n), int64(m*n)); err != nil {
		t.Fatalf("wrapper F64: %v", err)
	}
	if err := b.matMulBatchedF64Loop(aS, bS, c2S, batch, m, n, k, int64(m*k), int64(k*n), int64(m*n)); err != nil {
		t.Fatalf("loop F64: %v", err)
	}
	b.Sync()
	out1 := make([]byte, batch*m*n*8)
	out2 := make([]byte, batch*m*n*8)
	b.CopyD2H(out1, c1S)
	b.CopyD2H(out2, c2S)
	got1 := bytesF64(out1)
	got2 := bytesF64(out2)

	var maxRel float64
	fails := 0
	const relTol = 1e-11
	for i := range got1 {
		rel := math.Abs(got1[i]-got2[i]) / (math.Abs(got1[i]) + 1e-30)
		if rel > maxRel {
			maxRel = rel
		}
		if rel > relTol {
			fails++
		}
	}
	t.Logf("Wrapper vs Loop F64 [b=%d m=%d n=%d k=%d]: maxRel=%.3e fails=%d/%d (floor rel=%.0e)",
		batch, m, n, k, maxRel, fails, len(got1), relTol)
	if fails > 0 {
		t.Errorf("wrapper vs loop F64: %d fails", fails)
	}
}
