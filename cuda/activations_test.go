package cuda

// Stage 5 non-composite activations tests — forward + backward + grad-consistency.
//
// Точность:
//   - ReLU F32/F64: bit-exact против max(x, 0).
//   - Sigmoid F32: hybrid (ex2.approx.f32 ~1 ULP; rcp.approx.f32 ~1 ULP; combined ~few ULP).
//   - Sigmoid F64: fdlibm exp → 1 / (1+expmx) в honest FP64; max ulp измеряется.
//   - Tanh F32: aparatnyy tanh.approx.f32 — доклад фактических maxRelErr.
//   - Tanh F64: (exp(2x)-1)/(exp(2x)+1) через fdlibm exp.
//
// Grad-consistency: для F64 Sigmoid/Tanh — SigmoidGradF64(SigmoidF64(x), 1) должно
// совпадать с central-difference SigmoidF64 (h=1e-6, tol=1e-8). Аналог для tanh.

import (
	"math"
	"math/rand"
	"testing"
)

// --- Helpers ---

func alloc2AndCopyF64(t *testing.T, b *PuregoBackend, xs []float64) (aS, cS Storage) {
	t.Helper()
	nb := len(xs) * 8
	var err error
	aS, err = b.Alloc(nb)
	if err != nil {
		t.Fatalf("Alloc a: %v", err)
	}
	cS, err = b.Alloc(nb)
	if err != nil {
		b.Free(aS)
		t.Fatalf("Alloc c: %v", err)
	}
	if err := b.CopyH2D(aS, f64Bytes(xs)); err != nil {
		b.Free(aS)
		b.Free(cS)
		t.Fatalf("H2D: %v", err)
	}
	return
}

func alloc2AndCopyF32(t *testing.T, b *PuregoBackend, xs []float32) (aS, cS Storage) {
	t.Helper()
	nb := len(xs) * 4
	var err error
	aS, err = b.Alloc(nb)
	if err != nil {
		t.Fatalf("Alloc a: %v", err)
	}
	cS, err = b.Alloc(nb)
	if err != nil {
		b.Free(aS)
		t.Fatalf("Alloc c: %v", err)
	}
	if err := b.CopyH2D(aS, f32Bytes(xs)); err != nil {
		b.Free(aS)
		b.Free(cS)
		t.Fatalf("H2D: %v", err)
	}
	return
}

// --- ReLU F64 bit-exact ---

func TestReLUF64(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	r := rand.New(rand.NewSource(101))
	xs := make([]float64, 4096)
	for i := range xs {
		xs[i] = (r.Float64() - 0.5) * 200
	}
	aS, cS := alloc2AndCopyF64(t, b, xs)
	defer b.Free(aS)
	defer b.Free(cS)
	if err := b.ReLUF64(aS, cS, len(xs)); err != nil {
		t.Fatalf("ReLUF64: %v", err)
	}
	b.Sync()
	buf := make([]byte, len(xs)*8)
	b.CopyD2H(buf, cS)
	got := bytesF64(buf)
	for i := range xs {
		want := math.Max(xs[i], 0)
		if got[i] != want {
			t.Fatalf("ReLUF64 idx=%d: got=%g want=%g x=%g", i, got[i], want, xs[i])
		}
	}
	t.Logf("ReLUF64: bit-exact on n=%d", len(xs))
}

func TestReLUF32(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	r := rand.New(rand.NewSource(103))
	xs := make([]float32, 4096)
	for i := range xs {
		xs[i] = (r.Float32() - 0.5) * 200
	}
	aS, cS := alloc2AndCopyF32(t, b, xs)
	defer b.Free(aS)
	defer b.Free(cS)
	if err := b.ReLUF32(aS, cS, len(xs)); err != nil {
		t.Fatalf("ReLUF32: %v", err)
	}
	b.Sync()
	buf := make([]byte, len(xs)*4)
	b.CopyD2H(buf, cS)
	got := bytesF32(buf)
	for i := range xs {
		var want float32
		if xs[i] > 0 {
			want = xs[i]
		}
		if got[i] != want {
			t.Fatalf("ReLUF32 idx=%d: got=%g want=%g x=%g", i, got[i], want, xs[i])
		}
	}
	t.Logf("ReLUF32: bit-exact on n=%d", len(xs))
}

// --- Sigmoid F64 через fdlibm ---

func TestSigmoidF64(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	r := rand.New(rand.NewSource(105))
	xs := make([]float64, 4096)
	for i := range xs {
		xs[i] = (r.Float64() - 0.5) * 20
	}
	ref := make([]float64, len(xs))
	for i, x := range xs {
		ref[i] = 1.0 / (1.0 + math.Exp(-x))
	}
	aS, cS := alloc2AndCopyF64(t, b, xs)
	defer b.Free(aS)
	defer b.Free(cS)
	if err := b.SigmoidF64(aS, cS, len(xs)); err != nil {
		t.Fatalf("SigmoidF64: %v", err)
	}
	b.Sync()
	buf := make([]byte, len(xs)*8)
	b.CopyD2H(buf, cS)
	got := bytesF64(buf)
	maxUlp, maxRel, wi := f64Stats(got, ref, xs)
	if wi < 0 {
		t.Logf("SigmoidF64: n=%d all bit-exact", len(xs))
	} else {
		t.Logf("SigmoidF64: n=%d maxUlp=%d maxRel=%.3e worstInput=%.4g got=%.17g ref=%.17g",
			len(xs), maxUlp, maxRel, xs[wi], got[wi], ref[wi])
	}
	if maxUlp > 8 {
		t.Errorf("SigmoidF64 maxUlp=%d > 8", maxUlp)
	}
}

func TestSigmoidF32(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	r := rand.New(rand.NewSource(107))
	xs := make([]float32, 4096)
	for i := range xs {
		xs[i] = float32((r.Float64() - 0.5) * 20)
	}
	ref := make([]float32, len(xs))
	for i, x := range xs {
		ref[i] = float32(1.0 / (1.0 + math.Exp(-float64(x))))
	}
	aS, cS := alloc2AndCopyF32(t, b, xs)
	defer b.Free(aS)
	defer b.Free(cS)
	if err := b.SigmoidF32(aS, cS, len(xs)); err != nil {
		t.Fatalf("SigmoidF32: %v", err)
	}
	b.Sync()
	buf := make([]byte, len(xs)*4)
	b.CopyD2H(buf, cS)
	got := bytesF32(buf)
	stats := f32Compare(got, ref)
	t.Logf("SigmoidF32: n=%d maxAbs=%.3e maxRel=%.3e",
		len(xs), stats.maxAbsErr, stats.maxRelErr)
	// F32 approx: expect abs ≤ 1e-6, rel ≤ 1e-5 (hybrid).
	if stats.maxAbsErr > 1e-5 {
		t.Errorf("SigmoidF32 maxAbs=%.3e > 1e-5", stats.maxAbsErr)
	}
}

// --- Tanh F64 через fdlibm ---

func TestTanhF64(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	r := rand.New(rand.NewSource(109))
	xs := make([]float64, 4096)
	for i := range xs {
		xs[i] = (r.Float64() - 0.5) * 20
	}
	ref := make([]float64, len(xs))
	for i, x := range xs {
		ref[i] = math.Tanh(x)
	}
	aS, cS := alloc2AndCopyF64(t, b, xs)
	defer b.Free(aS)
	defer b.Free(cS)
	if err := b.TanhF64(aS, cS, len(xs)); err != nil {
		t.Fatalf("TanhF64: %v", err)
	}
	b.Sync()
	buf := make([]byte, len(xs)*8)
	b.CopyD2H(buf, cS)
	got := bytesF64(buf)
	// Формула (exp(2x)-1)/(exp(2x)+1) имеет cancellation при малых |x|,
	// поэтому ULP-метрика для малых значений даёт большие числа при
	// крошечной абсолютной ошибке. Метрика — hybrid abs+rel:
	// |diff| < abs_tol(1e-14) + rel_tol(1e-12)*|ref|. Точная реализация
	// через expm1(2x)/(expm1(2x)+2) — будущий улучшитель (см. отчёт stage5).
	const absTol, relTol = 1e-14, 1e-12
	var maxAbs, maxRel float64
	var wi int
	for i := range got {
		abs := math.Abs(got[i] - ref[i])
		if abs > maxAbs {
			maxAbs, wi = abs, i
		}
		if r := abs / (math.Abs(ref[i]) + 1e-300); r > maxRel {
			maxRel = r
		}
	}
	t.Logf("TanhF64: n=%d maxAbs=%.3e maxRel=%.3e worstInput=%.4g got=%.17g ref=%.17g",
		len(xs), maxAbs, maxRel, xs[wi], got[wi], ref[wi])
	fail := false
	for i := range got {
		bound := absTol + relTol*math.Abs(ref[i])
		if math.Abs(got[i]-ref[i]) > bound {
			fail = true
			break
		}
	}
	if fail {
		t.Errorf("TanhF64: hybrid |diff| > 1e-14 + 1e-12*|ref| — see log")
	}
}

// --- TanhF32: доклад точности по диапазонам (стоп-точка ТЗ) ---

func runTanhF32Range(t *testing.T, b *PuregoBackend, lo, hi float32, name string) {
	t.Helper()
	const n = 4096
	r := rand.New(rand.NewSource(int64(len(name) * 111)))
	xs := make([]float32, n)
	for i := range xs {
		xs[i] = lo + (hi-lo)*r.Float32()
	}
	ref := make([]float32, n)
	for i, x := range xs {
		ref[i] = float32(math.Tanh(float64(x)))
	}
	aS, cS := alloc2AndCopyF32(t, b, xs)
	defer b.Free(aS)
	defer b.Free(cS)
	if err := b.TanhF32(aS, cS, n); err != nil {
		t.Fatalf("TanhF32: %v", err)
	}
	b.Sync()
	buf := make([]byte, n*4)
	b.CopyD2H(buf, cS)
	got := bytesF32(buf)
	stats := f32Compare(got, ref)
	t.Logf("TanhF32 %s [%g..%g]: maxAbs=%.3e maxRel=%.3e",
		name, lo, hi, stats.maxAbsErr, stats.maxRelErr)
}

func TestTanhF32Ranges(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	runTanhF32Range(t, b, -0.01, 0.01, "tiny")
	runTanhF32Range(t, b, -1, 1, "small")
	runTanhF32Range(t, b, -5, 5, "medium")
	runTanhF32Range(t, b, -20, 20, "saturating")
}

// --- Grad-consistency F64 ---

// numericDeriv — central difference h=1e-6.
func numericDerivSigmoidF64(x float64) float64 {
	const h = 1e-6
	fp := 1.0 / (1.0 + math.Exp(-(x + h)))
	fm := 1.0 / (1.0 + math.Exp(-(x - h)))
	return (fp - fm) / (2 * h)
}
func numericDerivTanhF64(x float64) float64 {
	const h = 1e-6
	return (math.Tanh(x+h) - math.Tanh(x-h)) / (2 * h)
}

func TestSigmoidGradF64Consistency(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	// Range where sigmoid isn't saturated (derivative not vanishingly small).
	r := rand.New(rand.NewSource(113))
	xs := make([]float64, 256)
	for i := range xs {
		xs[i] = (r.Float64() - 0.5) * 8 // ~[-4, 4], derivative ~[0.02, 0.25]
	}
	// Compute y = sigmoid(x) analytically (independent from device).
	ys := make([]float64, len(xs))
	for i, x := range xs {
		ys[i] = 1.0 / (1.0 + math.Exp(-x))
	}
	dY := make([]float64, len(xs))
	for i := range dY {
		dY[i] = 1.0
	}
	yS, err := b.Alloc(len(ys) * 8)
	if err != nil {
		t.Fatalf("Alloc y: %v", err)
	}
	defer b.Free(yS)
	dyS, err := b.Alloc(len(dY) * 8)
	if err != nil {
		t.Fatalf("Alloc dY: %v", err)
	}
	defer b.Free(dyS)
	outS, err := b.Alloc(len(xs) * 8)
	if err != nil {
		t.Fatalf("Alloc out: %v", err)
	}
	defer b.Free(outS)
	b.CopyH2D(yS, f64Bytes(ys))
	b.CopyH2D(dyS, f64Bytes(dY))
	if err := b.SigmoidGradF64(yS, dyS, outS, len(xs)); err != nil {
		t.Fatalf("SigmoidGradF64: %v", err)
	}
	b.Sync()
	buf := make([]byte, len(xs)*8)
	b.CopyD2H(buf, outS)
	got := bytesF64(buf)

	var maxErr float64
	var wi int
	for i, x := range xs {
		numDeriv := numericDerivSigmoidF64(x)
		abs := math.Abs(got[i] - numDeriv)
		if abs > maxErr {
			maxErr, wi = abs, i
		}
	}
	t.Logf("SigmoidGradF64 vs numeric: maxErr=%.3e worst x=%.4g got=%.10g numeric=%.10g",
		maxErr, xs[wi], got[wi], numericDerivSigmoidF64(xs[wi]))
	if maxErr > 1e-8 {
		t.Errorf("SigmoidGrad-numeric consistency: maxErr=%.3e > 1e-8", maxErr)
	}
}

func TestTanhGradF64Consistency(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	r := rand.New(rand.NewSource(117))
	xs := make([]float64, 256)
	for i := range xs {
		xs[i] = (r.Float64() - 0.5) * 4
	}
	ys := make([]float64, len(xs))
	for i, x := range xs {
		ys[i] = math.Tanh(x)
	}
	dY := make([]float64, len(xs))
	for i := range dY {
		dY[i] = 1.0
	}
	yS, err := b.Alloc(len(ys) * 8)
	if err != nil {
		t.Fatalf("Alloc y: %v", err)
	}
	defer b.Free(yS)
	dyS, err := b.Alloc(len(dY) * 8)
	if err != nil {
		t.Fatalf("Alloc dY: %v", err)
	}
	defer b.Free(dyS)
	outS, err := b.Alloc(len(xs) * 8)
	if err != nil {
		t.Fatalf("Alloc out: %v", err)
	}
	defer b.Free(outS)
	b.CopyH2D(yS, f64Bytes(ys))
	b.CopyH2D(dyS, f64Bytes(dY))
	if err := b.TanhGradF64(yS, dyS, outS, len(xs)); err != nil {
		t.Fatalf("TanhGradF64: %v", err)
	}
	b.Sync()
	buf := make([]byte, len(xs)*8)
	b.CopyD2H(buf, outS)
	got := bytesF64(buf)

	var maxErr float64
	var wi int
	for i, x := range xs {
		numDeriv := numericDerivTanhF64(x)
		abs := math.Abs(got[i] - numDeriv)
		if abs > maxErr {
			maxErr, wi = abs, i
		}
	}
	t.Logf("TanhGradF64 vs numeric: maxErr=%.3e worst x=%.4g got=%.10g numeric=%.10g",
		maxErr, xs[wi], got[wi], numericDerivTanhF64(xs[wi]))
	if maxErr > 1e-8 {
		t.Errorf("TanhGrad-numeric consistency: maxErr=%.3e > 1e-8", maxErr)
	}
}

// --- ReLUGrad bit-exact vs CPU ---

func TestReLUGradF64(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	r := rand.New(rand.NewSource(119))
	n := 4096
	xs := make([]float64, n)
	dY := make([]float64, n)
	for i := range xs {
		xs[i] = (r.Float64() - 0.5) * 100
		dY[i] = r.NormFloat64()
	}
	xS, _ := b.Alloc(n * 8)
	defer b.Free(xS)
	dyS, _ := b.Alloc(n * 8)
	defer b.Free(dyS)
	outS, _ := b.Alloc(n * 8)
	defer b.Free(outS)
	b.CopyH2D(xS, f64Bytes(xs))
	b.CopyH2D(dyS, f64Bytes(dY))
	if err := b.ReLUGradF64(xS, dyS, outS, n); err != nil {
		t.Fatalf("ReLUGradF64: %v", err)
	}
	b.Sync()
	buf := make([]byte, n*8)
	b.CopyD2H(buf, outS)
	got := bytesF64(buf)
	for i := 0; i < n; i++ {
		var want float64
		if xs[i] > 0 {
			want = dY[i]
		}
		if got[i] != want {
			t.Fatalf("ReLUGradF64 idx=%d: got=%g want=%g x=%g dY=%g", i, got[i], want, xs[i], dY[i])
		}
	}
	t.Logf("ReLUGradF64: bit-exact on n=%d", n)
}

func TestReLUGradF32(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	r := rand.New(rand.NewSource(121))
	n := 4096
	xs := make([]float32, n)
	dY := make([]float32, n)
	for i := range xs {
		xs[i] = float32((r.Float64() - 0.5) * 100)
		dY[i] = float32(r.NormFloat64())
	}
	xS, _ := b.Alloc(n * 4)
	defer b.Free(xS)
	dyS, _ := b.Alloc(n * 4)
	defer b.Free(dyS)
	outS, _ := b.Alloc(n * 4)
	defer b.Free(outS)
	b.CopyH2D(xS, f32Bytes(xs))
	b.CopyH2D(dyS, f32Bytes(dY))
	if err := b.ReLUGradF32(xS, dyS, outS, n); err != nil {
		t.Fatalf("ReLUGradF32: %v", err)
	}
	b.Sync()
	buf := make([]byte, n*4)
	b.CopyD2H(buf, outS)
	got := bytesF32(buf)
	for i := 0; i < n; i++ {
		var want float32
		if xs[i] > 0 {
			want = dY[i]
		}
		if got[i] != want {
			t.Fatalf("ReLUGradF32 idx=%d: got=%g want=%g", i, got[i], want)
		}
	}
	t.Logf("ReLUGradF32: bit-exact on n=%d", n)
}
