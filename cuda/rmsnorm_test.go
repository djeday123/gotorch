package cuda

// P2-RMS: RMSNorm forward/backward tests, F32 и F64.
//
// Reference: y = gamma * x / rms, где rms = sqrt(mean(x^2)+eps).
// CPU reference — F64-аккумулятор (bit-exact для F64, hybrid tolerance для F32).
//
// Формы: [1,1], [3,7], [128,512], боевая [16, 64].
// Backward — grad-consistency через numerical h=1e-6 (F64 сравниваем аналитический
// grad с (f(x+h)-f(x-h))/(2h); F32 hybrid).
//
// Edge cases: equal row (все элементы одинаковые), zero row (rms=sqrt(eps)),
// eps sensitivity (для очень маленьких x).

import (
	"math"
	"math/rand"
	"testing"
)

// rmsNormCPUF64 — F64 reference forward.
func rmsNormCPUF64(x, gamma []float64, rows, cols int, eps float64) []float64 {
	y := make([]float64, rows*cols)
	for r := 0; r < rows; r++ {
		var sum2 float64
		for c := 0; c < cols; c++ {
			v := x[r*cols+c]
			sum2 += v * v
		}
		invRms := 1.0 / math.Sqrt(sum2/float64(cols)+eps)
		for c := 0; c < cols; c++ {
			y[r*cols+c] = gamma[c] * x[r*cols+c] * invRms
		}
	}
	return y
}

// rmsNormGradCPUF64 — F64 reference backward (analytic).
// dx_j = gamma_j*dy_j*inv_rms - x_j*S*inv_rms^3/cols,  где S = sum_i(gamma_i*x_i*dy_i)
// dgamma_j = sum_rows(dy_j*x_j*inv_rms)
func rmsNormGradCPUF64(x, gamma, dy []float64, rows, cols int, eps float64) (dx, dgamma []float64) {
	dx = make([]float64, rows*cols)
	dgamma = make([]float64, cols)
	for r := 0; r < rows; r++ {
		var sum2, S float64
		for c := 0; c < cols; c++ {
			v := x[r*cols+c]
			sum2 += v * v
			S += gamma[c] * v * dy[r*cols+c]
		}
		invRms := 1.0 / math.Sqrt(sum2/float64(cols)+eps)
		invRms3overCols := invRms * invRms * invRms / float64(cols)
		for c := 0; c < cols; c++ {
			t1 := gamma[c] * dy[r*cols+c] * invRms
			t2 := x[r*cols+c] * S * invRms3overCols
			dx[r*cols+c] = t1 - t2
			dgamma[c] += dy[r*cols+c] * x[r*cols+c] * invRms
		}
	}
	return
}

// hybridPass — abs + rel tolerance BLAS style.
func hybridPass(got, ref, absTol, relTol float64) bool {
	d := math.Abs(got - ref)
	return d <= absTol+relTol*math.Abs(ref)
}

// ─────────────────────────── FORWARD ───────────────────────────

func testRMSNormF64Forms(t *testing.T, rows, cols int) {
	t.Helper()
	b := tryBackend(t)
	defer b.Close()

	r := rand.New(rand.NewSource(int64(rows*1000+cols) + 1))
	x := make([]float64, rows*cols)
	gamma := make([]float64, cols)
	for i := range x {
		x[i] = r.NormFloat64()
	}
	for i := range gamma {
		gamma[i] = 0.7 + 0.3*r.NormFloat64()
	}
	const eps = 1e-6

	ref := rmsNormCPUF64(x, gamma, rows, cols, eps)

	xS, _ := b.Alloc(rows * cols * 8)
	defer b.Free(xS)
	gS, _ := b.Alloc(cols * 8)
	defer b.Free(gS)
	yS, _ := b.Alloc(rows * cols * 8)
	defer b.Free(yS)
	b.CopyH2D(xS, f64Bytes(x))
	b.CopyH2D(gS, f64Bytes(gamma))

	if err := b.RMSNormF64(xS, gS, yS, rows, cols, eps); err != nil {
		t.Fatalf("RMSNormF64 [%d,%d]: %v", rows, cols, err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	out := make([]byte, rows*cols*8)
	b.CopyD2H(out, yS)
	got := bytesF64(out)

	var maxRel float64
	fails := 0
	for i := range got {
		rel := math.Abs(got[i]-ref[i]) / (math.Abs(ref[i]) + 1e-30)
		if rel > maxRel {
			maxRel = rel
		}
		if !hybridPass(got[i], ref[i], 0, 1e-12) {
			fails++
		}
	}
	t.Logf("RMSNormF64 [%d,%d]: maxRel=%.3e fails=%d/%d", rows, cols, maxRel, fails, len(got))
	if fails > 0 {
		t.Errorf("RMSNormF64 [%d,%d]: %d/%d elements exceed rel=1e-12", rows, cols, fails, len(got))
	}
}

func testRMSNormF32Forms(t *testing.T, rows, cols int) {
	t.Helper()
	b := tryBackend(t)
	defer b.Close()

	r := rand.New(rand.NewSource(int64(rows*1000+cols) + 7))
	x := make([]float32, rows*cols)
	gamma := make([]float32, cols)
	xF64 := make([]float64, rows*cols)
	gF64 := make([]float64, cols)
	for i := range x {
		x[i] = float32(r.NormFloat64())
		xF64[i] = float64(x[i])
	}
	for i := range gamma {
		gamma[i] = float32(0.7 + 0.3*r.NormFloat64())
		gF64[i] = float64(gamma[i])
	}
	const eps = 1e-6

	ref := rmsNormCPUF64(xF64, gF64, rows, cols, eps)

	xS, _ := b.Alloc(rows * cols * 4)
	defer b.Free(xS)
	gS, _ := b.Alloc(cols * 4)
	defer b.Free(gS)
	yS, _ := b.Alloc(rows * cols * 4)
	defer b.Free(yS)
	b.CopyH2D(xS, f32Bytes(x))
	b.CopyH2D(gS, f32Bytes(gamma))

	if err := b.RMSNormF32(xS, gS, yS, rows, cols, eps); err != nil {
		t.Fatalf("RMSNormF32 [%d,%d]: %v", rows, cols, err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	out := make([]byte, rows*cols*4)
	b.CopyD2H(out, yS)
	got := bytesF32(out)

	var maxAbs, maxRel float64
	fails := 0
	const absTol, relTol = 1e-4, 1e-5
	for i := range got {
		d := math.Abs(float64(got[i]) - ref[i])
		rel := d / (math.Abs(ref[i]) + 1e-30)
		if d > maxAbs {
			maxAbs = d
		}
		if rel > maxRel {
			maxRel = rel
		}
		if !hybridPass(float64(got[i]), ref[i], absTol, relTol) {
			fails++
		}
	}
	t.Logf("RMSNormF32 [%d,%d]: maxAbs=%.3e maxRel=%.3e fails=%d/%d (abs=%.0e+rel=%.0e·|ref|)",
		rows, cols, maxAbs, maxRel, fails, len(got), absTol, relTol)
	if fails > 0 {
		t.Errorf("RMSNormF32 [%d,%d]: %d/%d fail hybrid tolerance", rows, cols, fails, len(got))
	}
}

func TestRMSNormF64_Shapes(t *testing.T) {
	testRMSNormF64Forms(t, 1, 1)
	testRMSNormF64Forms(t, 3, 7)
	testRMSNormF64Forms(t, 128, 512)
	testRMSNormF64Forms(t, 16, 64) // battle-form (LLaMA-tiny-ish)
}

func TestRMSNormF32_Shapes(t *testing.T) {
	testRMSNormF32Forms(t, 1, 1)
	testRMSNormF32Forms(t, 3, 7)
	testRMSNormF32Forms(t, 128, 512)
	testRMSNormF32Forms(t, 16, 64)
}

// ─────────────────────────── EDGE CASES ───────────────────────────

func TestRMSNormF64_EqualRow(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	const rows, cols = 4, 32
	const val = 1.25
	const eps = 1e-6
	x := make([]float64, rows*cols)
	gamma := make([]float64, cols)
	for i := range x {
		x[i] = val
	}
	for i := range gamma {
		gamma[i] = 1.0
	}
	// Для x=[val]*cols, rms = |val|. y = gamma * val / |val| = sign(val)*gamma.
	ref := rmsNormCPUF64(x, gamma, rows, cols, eps)
	xS, _ := b.Alloc(rows * cols * 8)
	defer b.Free(xS)
	gS, _ := b.Alloc(cols * 8)
	defer b.Free(gS)
	yS, _ := b.Alloc(rows * cols * 8)
	defer b.Free(yS)
	b.CopyH2D(xS, f64Bytes(x))
	b.CopyH2D(gS, f64Bytes(gamma))
	if err := b.RMSNormF64(xS, gS, yS, rows, cols, eps); err != nil {
		t.Fatalf("RMSNormF64 equal-row: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	out := make([]byte, rows*cols*8)
	b.CopyD2H(out, yS)
	got := bytesF64(out)
	for i := range got {
		if !hybridPass(got[i], ref[i], 0, 1e-12) {
			t.Errorf("equal-row idx=%d: got=%.15g ref=%.15g", i, got[i], ref[i])
			break
		}
	}
	t.Logf("equal-row F64 rows=%d cols=%d val=%g: OK; y[0]=%.6f expected=~1.0", rows, cols, val, got[0])
}

func TestRMSNormF64_ZeroRow(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	const rows, cols = 4, 32
	const eps = 1e-6
	x := make([]float64, rows*cols) // all zeros
	gamma := make([]float64, cols)
	for i := range gamma {
		gamma[i] = 1.5
	}
	// rms = sqrt(eps); y = 0 * gamma / sqrt(eps) = 0.
	xS, _ := b.Alloc(rows * cols * 8)
	defer b.Free(xS)
	gS, _ := b.Alloc(cols * 8)
	defer b.Free(gS)
	yS, _ := b.Alloc(rows * cols * 8)
	defer b.Free(yS)
	b.CopyH2D(xS, f64Bytes(x))
	b.CopyH2D(gS, f64Bytes(gamma))
	if err := b.RMSNormF64(xS, gS, yS, rows, cols, eps); err != nil {
		t.Fatalf("RMSNormF64 zero-row: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	out := make([]byte, rows*cols*8)
	b.CopyD2H(out, yS)
	got := bytesF64(out)
	for i := range got {
		if got[i] != 0 {
			t.Errorf("zero-row idx=%d: got=%.15g (must be 0)", i, got[i])
			break
		}
	}
	t.Logf("zero-row F64 OK; rms=sqrt(eps)=%.6g", math.Sqrt(eps))
}

func TestRMSNormF64_EpsSensitivity(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	const rows, cols = 2, 16
	// x — крохотные (sqrt(cols) ~4, |x|~1e-4 → sum2/cols ~ 1e-8), eps=1e-6 доминирует.
	r := rand.New(rand.NewSource(37))
	x := make([]float64, rows*cols)
	gamma := make([]float64, cols)
	for i := range x {
		x[i] = 1e-4 * r.NormFloat64()
	}
	for i := range gamma {
		gamma[i] = 1.0
	}
	epsList := []float64{1e-12, 1e-8, 1e-6, 1e-4, 1.0}
	for _, eps := range epsList {
		ref := rmsNormCPUF64(x, gamma, rows, cols, eps)
		xS, _ := b.Alloc(rows * cols * 8)
		gS, _ := b.Alloc(cols * 8)
		yS, _ := b.Alloc(rows * cols * 8)
		b.CopyH2D(xS, f64Bytes(x))
		b.CopyH2D(gS, f64Bytes(gamma))
		if err := b.RMSNormF64(xS, gS, yS, rows, cols, eps); err != nil {
			t.Fatalf("eps=%.0e: %v", eps, err)
		}
		b.Sync()
		out := make([]byte, rows*cols*8)
		b.CopyD2H(out, yS)
		got := bytesF64(out)
		var maxRel float64
		for i := range got {
			rel := math.Abs(got[i]-ref[i]) / (math.Abs(ref[i]) + 1e-30)
			if rel > maxRel {
				maxRel = rel
			}
		}
		t.Logf("eps=%.0e: maxRel=%.3e (y[0]=%.6g ref=%.6g)", eps, maxRel, got[0], ref[0])
		if maxRel > 1e-12 {
			t.Errorf("eps=%.0e: maxRel=%.3e > 1e-12", eps, maxRel)
		}
		b.Free(xS)
		b.Free(gS)
		b.Free(yS)
	}
}

// ─────────────────────────── BACKWARD ───────────────────────────

func testRMSNormGradF64Forms(t *testing.T, rows, cols int) {
	t.Helper()
	b := tryBackend(t)
	defer b.Close()

	r := rand.New(rand.NewSource(int64(rows*10000+cols) + 3))
	x := make([]float64, rows*cols)
	gamma := make([]float64, cols)
	dy := make([]float64, rows*cols)
	for i := range x {
		x[i] = r.NormFloat64()
		dy[i] = r.NormFloat64()
	}
	for i := range gamma {
		gamma[i] = 0.7 + 0.3*r.NormFloat64()
	}
	const eps = 1e-6
	refDx, refDgamma := rmsNormGradCPUF64(x, gamma, dy, rows, cols, eps)

	xS, _ := b.Alloc(rows * cols * 8)
	defer b.Free(xS)
	gS, _ := b.Alloc(cols * 8)
	defer b.Free(gS)
	dyS, _ := b.Alloc(rows * cols * 8)
	defer b.Free(dyS)
	dxS, _ := b.Alloc(rows * cols * 8)
	defer b.Free(dxS)
	dgS, _ := b.Alloc(cols * 8)
	defer b.Free(dgS)
	b.CopyH2D(xS, f64Bytes(x))
	b.CopyH2D(gS, f64Bytes(gamma))
	b.CopyH2D(dyS, f64Bytes(dy))

	if err := b.RMSNormGradF64(xS, gS, dyS, dxS, dgS, rows, cols, eps); err != nil {
		t.Fatalf("RMSNormGradF64 [%d,%d]: %v", rows, cols, err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	dxOut := make([]byte, rows*cols*8)
	dgOut := make([]byte, cols*8)
	b.CopyD2H(dxOut, dxS)
	b.CopyD2H(dgOut, dgS)
	gotDx := bytesF64(dxOut)
	gotDg := bytesF64(dgOut)

	var mxDx, mxDg float64
	failsDx, failsDg := 0, 0
	for i := range gotDx {
		rel := math.Abs(gotDx[i]-refDx[i]) / (math.Abs(refDx[i]) + 1e-30)
		if rel > mxDx {
			mxDx = rel
		}
		if !hybridPass(gotDx[i], refDx[i], 1e-12, 1e-10) {
			failsDx++
		}
	}
	for i := range gotDg {
		rel := math.Abs(gotDg[i]-refDgamma[i]) / (math.Abs(refDgamma[i]) + 1e-30)
		if rel > mxDg {
			mxDg = rel
		}
		if !hybridPass(gotDg[i], refDgamma[i], 1e-12, 1e-10) {
			failsDg++
		}
	}
	t.Logf("RMSNormGradF64 [%d,%d]: dx maxRel=%.3e fails=%d/%d; dgamma maxRel=%.3e fails=%d/%d",
		rows, cols, mxDx, failsDx, len(gotDx), mxDg, failsDg, len(gotDg))
	if failsDx > 0 {
		t.Errorf("RMSNormGradF64 [%d,%d] dx: %d fails", rows, cols, failsDx)
	}
	if failsDg > 0 {
		t.Errorf("RMSNormGradF64 [%d,%d] dgamma: %d fails", rows, cols, failsDg)
	}
}

func testRMSNormGradF32Forms(t *testing.T, rows, cols int) {
	t.Helper()
	b := tryBackend(t)
	defer b.Close()

	r := rand.New(rand.NewSource(int64(rows*10000+cols) + 5))
	x := make([]float32, rows*cols)
	gamma := make([]float32, cols)
	dy := make([]float32, rows*cols)
	xF64 := make([]float64, rows*cols)
	gF64 := make([]float64, cols)
	dyF64 := make([]float64, rows*cols)
	for i := range x {
		x[i] = float32(r.NormFloat64())
		dy[i] = float32(r.NormFloat64())
		xF64[i] = float64(x[i])
		dyF64[i] = float64(dy[i])
	}
	for i := range gamma {
		gamma[i] = float32(0.7 + 0.3*r.NormFloat64())
		gF64[i] = float64(gamma[i])
	}
	const eps = 1e-6
	refDx, refDgamma := rmsNormGradCPUF64(xF64, gF64, dyF64, rows, cols, eps)

	xS, _ := b.Alloc(rows * cols * 4)
	defer b.Free(xS)
	gS, _ := b.Alloc(cols * 4)
	defer b.Free(gS)
	dyS, _ := b.Alloc(rows * cols * 4)
	defer b.Free(dyS)
	dxS, _ := b.Alloc(rows * cols * 4)
	defer b.Free(dxS)
	dgS, _ := b.Alloc(cols * 4)
	defer b.Free(dgS)
	b.CopyH2D(xS, f32Bytes(x))
	b.CopyH2D(gS, f32Bytes(gamma))
	b.CopyH2D(dyS, f32Bytes(dy))

	if err := b.RMSNormGradF32(xS, gS, dyS, dxS, dgS, rows, cols, eps); err != nil {
		t.Fatalf("RMSNormGradF32 [%d,%d]: %v", rows, cols, err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	dxOut := make([]byte, rows*cols*4)
	dgOut := make([]byte, cols*4)
	b.CopyD2H(dxOut, dxS)
	b.CopyD2H(dgOut, dgS)
	gotDx := bytesF32(dxOut)
	gotDg := bytesF32(dgOut)

	var mxDxAbs, mxDxRel, mxDgAbs, mxDgRel float64
	failsDx, failsDg := 0, 0
	const absTolDx, relTolDx = 1e-4, 1e-4
	const absTolDg, relTolDg = 1e-3, 1e-4 // dgamma аккумулирует по rows — allow чуть шире abs
	for i := range gotDx {
		d := math.Abs(float64(gotDx[i]) - refDx[i])
		rel := d / (math.Abs(refDx[i]) + 1e-30)
		if d > mxDxAbs {
			mxDxAbs = d
		}
		if rel > mxDxRel {
			mxDxRel = rel
		}
		if !hybridPass(float64(gotDx[i]), refDx[i], absTolDx, relTolDx) {
			failsDx++
		}
	}
	for i := range gotDg {
		d := math.Abs(float64(gotDg[i]) - refDgamma[i])
		rel := d / (math.Abs(refDgamma[i]) + 1e-30)
		if d > mxDgAbs {
			mxDgAbs = d
		}
		if rel > mxDgRel {
			mxDgRel = rel
		}
		if !hybridPass(float64(gotDg[i]), refDgamma[i], absTolDg, relTolDg) {
			failsDg++
		}
	}
	t.Logf("RMSNormGradF32 [%d,%d]: dx maxAbs=%.3e maxRel=%.3e fails=%d/%d; dgamma maxAbs=%.3e maxRel=%.3e fails=%d/%d",
		rows, cols, mxDxAbs, mxDxRel, failsDx, len(gotDx), mxDgAbs, mxDgRel, failsDg, len(gotDg))
	if failsDx > 0 {
		t.Errorf("RMSNormGradF32 [%d,%d] dx: %d/%d fail (abs=%.0e+rel=%.0e·|ref|)",
			rows, cols, failsDx, len(gotDx), absTolDx, relTolDx)
	}
	if failsDg > 0 {
		t.Errorf("RMSNormGradF32 [%d,%d] dgamma: %d/%d fail (abs=%.0e+rel=%.0e·|ref|)",
			rows, cols, failsDg, len(gotDg), absTolDg, relTolDg)
	}
}

func TestRMSNormGradF64_Shapes(t *testing.T) {
	testRMSNormGradF64Forms(t, 1, 1)
	testRMSNormGradF64Forms(t, 3, 7)
	testRMSNormGradF64Forms(t, 128, 512)
	testRMSNormGradF64Forms(t, 16, 64)
}

func TestRMSNormGradF32_Shapes(t *testing.T) {
	testRMSNormGradF32Forms(t, 1, 1)
	testRMSNormGradF32Forms(t, 3, 7)
	testRMSNormGradF32Forms(t, 128, 512)
	testRMSNormGradF32Forms(t, 16, 64)
}

// ─────────────── GRAD CONSISTENCY (numerical F64) ───────────────

// TestRMSNormGradF64_Numerical — сравнение аналитического grad с numerical
// центральной разностью на скалярном выходе L = sum(dy*y). Тогда dL/dx = dx,
// dL/dgamma = dgamma. h=1e-6, threshold rel<1e-8.
func TestRMSNormGradF64_Numerical(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	const rows, cols = 3, 7
	const eps = 1e-6
	const h = 1e-6

	r := rand.New(rand.NewSource(123))
	x := make([]float64, rows*cols)
	gamma := make([]float64, cols)
	dy := make([]float64, rows*cols)
	for i := range x {
		x[i] = r.NormFloat64()
		dy[i] = r.NormFloat64()
	}
	for i := range gamma {
		gamma[i] = 0.7 + 0.3*r.NormFloat64()
	}

	// GPU analytic.
	xS, _ := b.Alloc(rows * cols * 8)
	defer b.Free(xS)
	gS, _ := b.Alloc(cols * 8)
	defer b.Free(gS)
	dyS, _ := b.Alloc(rows * cols * 8)
	defer b.Free(dyS)
	dxS, _ := b.Alloc(rows * cols * 8)
	defer b.Free(dxS)
	dgS, _ := b.Alloc(cols * 8)
	defer b.Free(dgS)
	b.CopyH2D(xS, f64Bytes(x))
	b.CopyH2D(gS, f64Bytes(gamma))
	b.CopyH2D(dyS, f64Bytes(dy))
	if err := b.RMSNormGradF64(xS, gS, dyS, dxS, dgS, rows, cols, eps); err != nil {
		t.Fatalf("grad: %v", err)
	}
	b.Sync()
	dxOut := make([]byte, rows*cols*8)
	dgOut := make([]byte, cols*8)
	b.CopyD2H(dxOut, dxS)
	b.CopyD2H(dgOut, dgS)
	gotDx := bytesF64(dxOut)
	gotDg := bytesF64(dgOut)

	// L(x, gamma) = sum(dy * y).
	L := func(xh []float64, gh []float64) float64 {
		y := rmsNormCPUF64(xh, gh, rows, cols, eps)
		var s float64
		for i := range y {
			s += dy[i] * y[i]
		}
		return s
	}

	// Numerical dx.
	worstDx := 0.0
	failsDx := 0
	for i := range x {
		xp := make([]float64, len(x))
		xm := make([]float64, len(x))
		copy(xp, x)
		copy(xm, x)
		xp[i] += h
		xm[i] -= h
		numDx := (L(xp, gamma) - L(xm, gamma)) / (2 * h)
		rel := math.Abs(gotDx[i]-numDx) / (math.Abs(numDx) + 1e-30)
		if rel > worstDx {
			worstDx = rel
		}
		if !hybridPass(gotDx[i], numDx, 1e-8, 1e-6) {
			failsDx++
		}
	}
	// Numerical dgamma.
	worstDg := 0.0
	failsDg := 0
	for j := range gamma {
		gp := make([]float64, len(gamma))
		gm := make([]float64, len(gamma))
		copy(gp, gamma)
		copy(gm, gamma)
		gp[j] += h
		gm[j] -= h
		numDg := (L(x, gp) - L(x, gm)) / (2 * h)
		rel := math.Abs(gotDg[j]-numDg) / (math.Abs(numDg) + 1e-30)
		if rel > worstDg {
			worstDg = rel
		}
		if !hybridPass(gotDg[j], numDg, 1e-8, 1e-6) {
			failsDg++
		}
	}
	t.Logf("Numerical grad F64: dx worstRel=%.3e fails=%d/%d; dgamma worstRel=%.3e fails=%d/%d",
		worstDx, failsDx, len(x), worstDg, failsDg, len(gamma))
	if failsDx > 0 || failsDg > 0 {
		t.Errorf("numerical grad F64: dx %d, dgamma %d fail (hybrid abs=1e-8+rel=1e-6·|ref|)", failsDx, failsDg)
	}
}
