package cuda

// Stage 5 composite operations tests: Sum/Mean.

import (
	"math"
	"math/rand"
	"testing"
)

func TestSumF64(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	sizes := []int{1, 256, 100000}
	for _, n := range sizes {
		r := rand.New(rand.NewSource(int64(n)))
		xs := make([]float64, n)
		var ref float64
		for i := range xs {
			xs[i] = r.NormFloat64()
			ref += xs[i]
		}
		aS, _ := b.Alloc(n * 8)
		defer b.Free(aS)
		b.CopyH2D(aS, f64Bytes(xs))
		got, err := b.SumF64(aS, n)
		if err != nil {
			t.Fatalf("n=%d SumF64: %v", n, err)
		}
		relErr := math.Abs(got-ref) / (math.Abs(ref) + 1e-300)
		t.Logf("SumF64 n=%d: got=%.10g ref=%.10g relErr=%.3e", n, got, ref, relErr)
		if relErr > 1e-10 {
			t.Errorf("SumF64 n=%d: relErr=%.3e > 1e-10", n, relErr)
		}
	}
}

func TestSumF32(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	sizes := []int{1, 256, 100000}
	for _, n := range sizes {
		r := rand.New(rand.NewSource(int64(n) + 1))
		xs := make([]float32, n)
		var ref float64
		for i := range xs {
			xs[i] = float32(r.NormFloat64())
			ref += float64(xs[i])
		}
		aS, _ := b.Alloc(n * 4)
		defer b.Free(aS)
		b.CopyH2D(aS, f32Bytes(xs))
		got, err := b.SumF32(aS, n)
		if err != nil {
			t.Fatalf("n=%d SumF32: %v", n, err)
		}
		relErr := math.Abs(float64(got)-ref) / (math.Abs(ref) + 1e-300)
		t.Logf("SumF32 n=%d: got=%.7g ref=%.7g relErr=%.3e", n, got, ref, relErr)
		// F32 tol on n=100k: 1e-4 (ТЗ)
		tol := 1e-6
		if n >= 100000 {
			tol = 1e-4
		}
		if relErr > tol {
			t.Errorf("SumF32 n=%d: relErr=%.3e > %.0e", n, relErr, tol)
		}
	}
}

func TestMeanF64(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	n := 4096
	r := rand.New(rand.NewSource(42))
	xs := make([]float64, n)
	var sum float64
	for i := range xs {
		xs[i] = r.NormFloat64()
		sum += xs[i]
	}
	ref := sum / float64(n)
	aS, _ := b.Alloc(n * 8)
	defer b.Free(aS)
	b.CopyH2D(aS, f64Bytes(xs))
	got, err := b.MeanF64(aS, n)
	if err != nil {
		t.Fatalf("MeanF64: %v", err)
	}
	relErr := math.Abs(got-ref) / math.Abs(ref)
	t.Logf("MeanF64: got=%.10g ref=%.10g relErr=%.3e", got, ref, relErr)
	if relErr > 1e-10 {
		t.Errorf("MeanF64: relErr=%.3e > 1e-10", relErr)
	}
}

func TestMeanF32(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	n := 4096
	r := rand.New(rand.NewSource(43))
	xs := make([]float32, n)
	var sum float64
	for i := range xs {
		xs[i] = float32(r.NormFloat64())
		sum += float64(xs[i])
	}
	ref := float32(sum / float64(n))
	aS, _ := b.Alloc(n * 4)
	defer b.Free(aS)
	b.CopyH2D(aS, f32Bytes(xs))
	got, err := b.MeanF32(aS, n)
	if err != nil {
		t.Fatalf("MeanF32: %v", err)
	}
	relErr := math.Abs(float64(got-ref)) / math.Abs(float64(ref))
	t.Logf("MeanF32: got=%.7g ref=%.7g relErr=%.3e", got, ref, relErr)
	if relErr > 1e-5 {
		t.Errorf("MeanF32: relErr=%.3e > 1e-5", relErr)
	}
}

// --- Softmax ---

func cpuSoftmaxF64(a []float64, rows, cols int) []float64 {
	out := make([]float64, rows*cols)
	for r := 0; r < rows; r++ {
		mx := math.Inf(-1)
		for j := 0; j < cols; j++ {
			if a[r*cols+j] > mx {
				mx = a[r*cols+j]
			}
		}
		var s float64
		for j := 0; j < cols; j++ {
			out[r*cols+j] = math.Exp(a[r*cols+j] - mx)
			s += out[r*cols+j]
		}
		for j := 0; j < cols; j++ {
			out[r*cols+j] /= s
		}
	}
	return out
}

func cpuSoftmaxF32(a []float32, rows, cols int) []float32 {
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		mx := float32(math.Inf(-1))
		for j := 0; j < cols; j++ {
			if a[r*cols+j] > mx {
				mx = a[r*cols+j]
			}
		}
		var s float64
		for j := 0; j < cols; j++ {
			e := math.Exp(float64(a[r*cols+j] - mx))
			out[r*cols+j] = float32(e)
			s += e
		}
		for j := 0; j < cols; j++ {
			out[r*cols+j] = float32(float64(out[r*cols+j]) / s)
		}
	}
	return out
}

func TestSoftmaxF64(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	shapes := []struct{ rows, cols int }{{1, 1}, {3, 7}, {128, 512}}
	for _, sh := range shapes {
		n := sh.rows * sh.cols
		r := rand.New(rand.NewSource(int64(n)))
		xs := make([]float64, n)
		for i := range xs {
			xs[i] = r.NormFloat64() * 3
		}
		ref := cpuSoftmaxF64(xs, sh.rows, sh.cols)
		aS, _ := b.Alloc(n * 8)
		defer b.Free(aS)
		cS, _ := b.Alloc(n * 8)
		defer b.Free(cS)
		b.CopyH2D(aS, f64Bytes(xs))
		if err := b.SoftmaxF64(aS, cS, sh.rows, sh.cols); err != nil {
			t.Fatalf("SoftmaxF64 %dx%d: %v", sh.rows, sh.cols, err)
		}
		b.Sync()
		buf := make([]byte, n*8)
		b.CopyD2H(buf, cS)
		got := bytesF64(buf)
		var maxRel float64
		for i := range got {
			r := math.Abs(got[i]-ref[i]) / (math.Abs(ref[i]) + 1e-300)
			if r > maxRel {
				maxRel = r
			}
		}
		t.Logf("SoftmaxF64 %dx%d: maxRel=%.3e", sh.rows, sh.cols, maxRel)
		if maxRel > 1e-13 {
			t.Errorf("SoftmaxF64 %dx%d: maxRel=%.3e > 1e-13", sh.rows, sh.cols, maxRel)
		}
	}
}

func TestSoftmaxF32(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	shapes := []struct{ rows, cols int }{{1, 1}, {3, 7}, {128, 512}}
	for _, sh := range shapes {
		n := sh.rows * sh.cols
		r := rand.New(rand.NewSource(int64(n)))
		xs := make([]float32, n)
		for i := range xs {
			xs[i] = float32(r.NormFloat64() * 3)
		}
		ref := cpuSoftmaxF32(xs, sh.rows, sh.cols)
		aS, _ := b.Alloc(n * 4)
		defer b.Free(aS)
		cS, _ := b.Alloc(n * 4)
		defer b.Free(cS)
		b.CopyH2D(aS, f32Bytes(xs))
		if err := b.SoftmaxF32(aS, cS, sh.rows, sh.cols); err != nil {
			t.Fatalf("SoftmaxF32 %dx%d: %v", sh.rows, sh.cols, err)
		}
		b.Sync()
		buf := make([]byte, n*4)
		b.CopyD2H(buf, cS)
		got := bytesF32(buf)
		stats := f32Compare(got, ref)
		t.Logf("SoftmaxF32 %dx%d: maxAbs=%.3e maxRel=%.3e",
			sh.rows, sh.cols, stats.maxAbsErr, stats.maxRelErr)
		if stats.maxAbsErr > 1e-5 {
			t.Errorf("SoftmaxF32 %dx%d: maxAbs=%.3e > 1e-5",
				sh.rows, sh.cols, stats.maxAbsErr)
		}
	}
}
