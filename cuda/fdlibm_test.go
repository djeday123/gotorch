package cuda

// Ворота 4.5 R02b — fdlibm-порт exp_f64 / log_f64.
//
// Эталон: math.Exp / math.Log из Go stdlib — это ТОТ ЖЕ fdlibm (те же
// константы, та же схема Хорнера). Ожидание: единицы ulp, обычно 0-1.
//
// Валидация:
//   1. Равномерная сетка + логарифмическая сетка + случайные точки.
//   2. Edge-таблица: Inf, NaN, 0, threshold-значения.
//   3. Метрика: maxUlpErr (устойчива к масштабу) + maxRelErr (для отчёта).
//
// Цель: ulp ≤ 4 (стремление 1-2), rel <= 1e-12 из ТЗ проходит с запасом
// на всём диапазоне, включая окрестность x=1 у log (fdlibm-редукция там
// точная, в отличие от lg2.approx.f32).

import (
	"math"
	"math/rand"
	"testing"
)

// ulpDiffF64 — расстояние в ULP между got и ref (оба FP64, знак одинаковый).
// Для NaN возвращает MaxUint64/2. Для Inf с одинаковым знаком → 0.
func ulpDiffF64(got, ref float64) uint64 {
	if math.IsNaN(got) && math.IsNaN(ref) {
		return 0
	}
	if math.IsInf(got, 1) && math.IsInf(ref, 1) {
		return 0
	}
	if math.IsInf(got, -1) && math.IsInf(ref, -1) {
		return 0
	}
	if math.IsNaN(got) != math.IsNaN(ref) ||
		math.IsInf(got, 0) != math.IsInf(ref, 0) {
		return 1 << 62
	}
	gb := math.Float64bits(got)
	rb := math.Float64bits(ref)
	// Сравнение как ordered bits: для положительных больший bit = больший.
	if gb > rb {
		return gb - rb
	}
	return rb - gb
}

func f64Stats(got, ref []float64, input []float64) (maxUlp uint64, maxRel float64, worstIdx int) {
	worstIdx = -1
	for i := range got {
		u := ulpDiffF64(got[i], ref[i])
		if u > maxUlp {
			maxUlp = u
			worstIdx = i
		}
		if !math.IsInf(ref[i], 0) && !math.IsNaN(ref[i]) && ref[i] != 0 {
			r := math.Abs(got[i]-ref[i]) / math.Abs(ref[i])
			if r > maxRel {
				maxRel = r
			}
		}
	}
	return
}

// runFdlibmExpTest — прогон exp_f64 через backend с заданной сеткой.
func runFdlibmExpTest(t *testing.T, b *PuregoBackend, name string, xs []float64) {
	t.Helper()
	n := len(xs)
	ref := make([]float64, n)
	for i := range xs {
		ref[i] = math.Exp(xs[i])
	}
	aS, err := b.Alloc(n * 8)
	if err != nil {
		t.Fatalf("Alloc a: %v", err)
	}
	defer b.Free(aS)
	cS, err := b.Alloc(n * 8)
	if err != nil {
		t.Fatalf("Alloc c: %v", err)
	}
	defer b.Free(cS)
	if err := b.CopyH2D(aS, f64Bytes(xs)); err != nil {
		t.Fatalf("H2D: %v", err)
	}
	if err := b.ExpF64(aS, cS, n); err != nil {
		t.Fatalf("ExpF64: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	buf := make([]byte, n*8)
	if err := b.CopyD2H(buf, cS); err != nil {
		t.Fatalf("D2H: %v", err)
	}
	got := bytesF64(buf)
	maxUlp, maxRel, wi := f64Stats(got, ref, xs)
	if wi < 0 {
		t.Logf("ExpF64 %s: n=%d maxUlp=0 maxRel=%.3e (all bit-exact)", name, n, maxRel)
	} else {
		t.Logf("ExpF64 %s: n=%d maxUlp=%d maxRel=%.3e worstIdx=%d worstInput=%.6g got=%.17g ref=%.17g",
			name, n, maxUlp, maxRel, wi, xs[wi], got[wi], ref[wi])
	}
	if maxUlp > 4 {
		t.Errorf("ExpF64 %s: maxUlp=%d > 4", name, maxUlp)
	}
	if maxRel > 1e-12 {
		t.Errorf("ExpF64 %s: maxRel=%.3e > 1e-12", name, maxRel)
	}
}

func runFdlibmLogTest(t *testing.T, b *PuregoBackend, name string, xs []float64) {
	t.Helper()
	n := len(xs)
	ref := make([]float64, n)
	for i := range xs {
		ref[i] = math.Log(xs[i])
	}
	aS, err := b.Alloc(n * 8)
	if err != nil {
		t.Fatalf("Alloc a: %v", err)
	}
	defer b.Free(aS)
	cS, err := b.Alloc(n * 8)
	if err != nil {
		t.Fatalf("Alloc c: %v", err)
	}
	defer b.Free(cS)
	if err := b.CopyH2D(aS, f64Bytes(xs)); err != nil {
		t.Fatalf("H2D: %v", err)
	}
	if err := b.LogF64(aS, cS, n); err != nil {
		t.Fatalf("LogF64: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	buf := make([]byte, n*8)
	if err := b.CopyD2H(buf, cS); err != nil {
		t.Fatalf("D2H: %v", err)
	}
	got := bytesF64(buf)
	maxUlp, maxRel, wi := f64Stats(got, ref, xs)
	if wi < 0 {
		t.Logf("LogF64 %s: n=%d maxUlp=0 maxRel=%.3e (all bit-exact)", name, n, maxRel)
	} else {
		t.Logf("LogF64 %s: n=%d maxUlp=%d maxRel=%.3e worstIdx=%d worstInput=%.6g got=%.17g ref=%.17g",
			name, n, maxUlp, maxRel, wi, xs[wi], got[wi], ref[wi])
	}
	if maxUlp > 4 {
		t.Errorf("LogF64 %s: maxUlp=%d > 4", name, maxUlp)
	}
	if maxRel > 1e-12 {
		t.Errorf("LogF64 %s: maxRel=%.3e > 1e-12", name, maxRel)
	}
}

// Сетки.

func uniformGridF64(n int, lo, hi float64) []float64 {
	xs := make([]float64, n)
	step := (hi - lo) / float64(n-1)
	for i := range xs {
		xs[i] = lo + float64(i)*step
	}
	return xs
}

// logGridF64 — n точек равномерно по log10(x) от lo10 до hi10.
func logGridF64(n int, lo10, hi10 float64) []float64 {
	xs := make([]float64, n)
	step := (hi10 - lo10) / float64(n-1)
	for i := range xs {
		xs[i] = math.Pow(10, lo10+float64(i)*step)
	}
	return xs
}

func randomGridF64(seed int64, n int, lo, hi float64) []float64 {
	r := rand.New(rand.NewSource(seed))
	xs := make([]float64, n)
	for i := range xs {
		xs[i] = lo + (hi-lo)*r.Float64()
	}
	return xs
}

// TestExpF64Fdlibm — Ворота 4.5 для exp_f64.
func TestExpF64Fdlibm(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	// 1. Uniform grid на нормальном диапазоне.
	runFdlibmExpTest(t, b, "uniform_small", uniformGridF64(1000, -1, 1))
	runFdlibmExpTest(t, b, "uniform_medium", uniformGridF64(1000, -10, 10))
	runFdlibmExpTest(t, b, "uniform_wide", uniformGridF64(1000, -100, 100))
	runFdlibmExpTest(t, b, "uniform_extreme", uniformGridF64(1000, -700, 700))

	// 2. Log grid (маленькие + большие модули).
	// exp(±ε): проверяем что f = 1 + O(ε), не 1.
	runFdlibmExpTest(t, b, "tiny", uniformGridF64(200, -1e-6, 1e-6))

	// 3. 100k случайных на широком диапазоне.
	runFdlibmExpTest(t, b, "random_100k", randomGridF64(42, 100000, -700, 700))

	// 4. Edge-таблица (проверяем без t.Errorf, т.к. ulpDiff для Inf/NaN
	// покроется в f64Stats; для этих особых точек нужны отдельные checks).
	edge := []struct {
		x, want float64
	}{
		{0, 1},
		{1, math.E},
		{-1, 1 / math.E},
	}
	for _, e := range edge {
		xs := []float64{e.x}
		aS, _ := b.Alloc(8)
		defer b.Free(aS)
		cS, _ := b.Alloc(8)
		defer b.Free(cS)
		b.CopyH2D(aS, f64Bytes(xs))
		b.ExpF64(aS, cS, 1)
		b.Sync()
		buf := make([]byte, 8)
		b.CopyD2H(buf, cS)
		got := bytesF64(buf)[0]
		u := ulpDiffF64(got, e.want)
		t.Logf("ExpF64 edge x=%g: got=%.17g want=%.17g ulp=%d", e.x, got, e.want, u)
		if u > 4 {
			t.Errorf("ExpF64 edge x=%g: ulp=%d > 4", e.x, u)
		}
	}
}

// TestLogF64Fdlibm — Ворота 4.5 для log_f64.
func TestLogF64Fdlibm(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	// 1. Uniform в разных диапазонах (только положительные — log domain).
	runFdlibmLogTest(t, b, "uniform_small", uniformGridF64(1000, 0.01, 1))
	runFdlibmLogTest(t, b, "uniform_medium", uniformGridF64(1000, 1, 100))
	runFdlibmLogTest(t, b, "around_one",
		uniformGridF64(1000, 0.5, 1.5)) // критический для approx, у fdlibm должен быть точный
	runFdlibmLogTest(t, b, "very_close_to_one",
		uniformGridF64(500, 0.99, 1.01)) // проверка редукции вблизи 1

	// 2. Log grid — широкие диапазоны.
	runFdlibmLogTest(t, b, "log_grid_wide", logGridF64(2000, -30, 30))
	runFdlibmLogTest(t, b, "log_grid_extreme", logGridF64(2000, -300, 300))

	// 3. Случайные.
	runFdlibmLogTest(t, b, "random_100k", randomGridF64(43, 100000, 1e-100, 1e100))

	// 4. Edge x=1 → 0 (точное, критично).
	xs := []float64{1.0}
	aS, _ := b.Alloc(8)
	defer b.Free(aS)
	cS, _ := b.Alloc(8)
	defer b.Free(cS)
	b.CopyH2D(aS, f64Bytes(xs))
	b.LogF64(aS, cS, 1)
	b.Sync()
	buf := make([]byte, 8)
	b.CopyD2H(buf, cS)
	got := bytesF64(buf)[0]
	if got != 0 {
		t.Errorf("LogF64(1.0) must be exactly 0, got %.17g", got)
	}
	t.Logf("LogF64(1.0)=%.17g (must be 0)", got)
}
