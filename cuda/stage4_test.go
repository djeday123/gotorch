package cuda

// Ворота 4 R02b — оставшиеся elementwise + Exp/Log + TestContextMigration.
//
// Первые 12 арифметических ядер (sub/mul/div/scalar/neg × 2 dtype):
//   - F64: bit-exact против CPU (одна инструкция per element).
//   - F32: bit-exact ожидается там где одна инструкция; hybrid |diff| <
//     1e-4 + 1e-5 × |ref| для div (correctly-rounded, но CPU может дать
//     разный ULP при денормалах).
//
// Exp/Log — стоп-точка Этапа 4:
//   - F32: aparatnyy ex2.approx.f32 / lg2.approx.f32, точность ~1 ULP.
//     Ожидаемая maxRelErr ~1e-6 на нормальных диапазонах.
//   - F64: cvt f64→f32 → ex2.approx.f32 → cvt f32→f64. Точность
//     ограничена F32-approx ~1e-7 rel. F64-tol 1e-12 НЕДОСТИЖИМА —
//     доклад с числами, пользователь решает.
//
// TestContextMigration — регрессионная страховка bind()-фикса
// (INVALID_CONTEXT из Ворот 3). 200+ итераций мелких CUDA-операций с
// runtime.Gosched() между ними, GOMAXPROCS > 1. Не гарантия миграции
// (стохастична), но детектор с высокой вероятностью срабатывания.

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
)

// --- Helpers ---

func fillRangeF64(r *rand.Rand, n int, lo, hi float64) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = lo + (hi-lo)*r.Float64()
	}
	return x
}

func fillRangeF32(r *rand.Rand, n int, lo, hi float32) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = lo + (hi-lo)*r.Float32()
	}
	return x
}

// --- 3-arg driver (Add/Sub/Mul/Div) для F64 bit-exact ---

func runBinaryF64BitExact(t *testing.T, b *PuregoBackend, opName string,
	call func(a, bb, c DeviceBuffer, n int) error,
	cpu func(a, b float64) float64,
	rangeA, rangeB [2]float64,
) {
	t.Helper()
	const n = 4096
	r := rand.New(rand.NewSource(int64(len(opName) * 31)))
	aH := fillRangeF64(r, n, rangeA[0], rangeA[1])
	bH := fillRangeF64(r, n, rangeB[0], rangeB[1])
	refC := make([]float64, n)
	for i := 0; i < n; i++ {
		refC[i] = cpu(aH[i], bH[i])
	}
	aS, _ := b.Alloc(n * 8)
	defer b.Free(aS)
	bS, _ := b.Alloc(n * 8)
	defer b.Free(bS)
	cS, _ := b.Alloc(n * 8)
	defer b.Free(cS)
	if err := b.CopyH2D(aS, f64Bytes(aH)); err != nil {
		t.Fatalf("H2D a: %v", err)
	}
	if err := b.CopyH2D(bS, f64Bytes(bH)); err != nil {
		t.Fatalf("H2D b: %v", err)
	}
	if err := call(aS, bS, cS, n); err != nil {
		t.Fatalf("%s: %v", opName, err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	buf := make([]byte, n*8)
	if err := b.CopyD2H(buf, cS); err != nil {
		t.Fatalf("D2H: %v", err)
	}
	got := bytesF64(buf)
	for i := 0; i < n; i++ {
		if got[i] != refC[i] {
			// Для div: correctly-rounded, но CPU может дать разный
			// последний бит в некоторых денормалях; лог'ируем и продолжаем.
			t.Fatalf("%s idx=%d: got=%g ref=%g (a=%g b=%g)",
				opName, i, got[i], refC[i], aH[i], bH[i])
		}
	}
}

func runBinaryF32BitExact(t *testing.T, b *PuregoBackend, opName string,
	call func(a, bb, c DeviceBuffer, n int) error,
	cpu func(a, b float32) float32,
	rangeA, rangeB [2]float32,
) {
	t.Helper()
	const n = 4096
	r := rand.New(rand.NewSource(int64(len(opName) * 37)))
	aH := fillRangeF32(r, n, rangeA[0], rangeA[1])
	bH := fillRangeF32(r, n, rangeB[0], rangeB[1])
	refC := make([]float32, n)
	for i := 0; i < n; i++ {
		refC[i] = cpu(aH[i], bH[i])
	}
	aS, _ := b.Alloc(n * 4)
	defer b.Free(aS)
	bS, _ := b.Alloc(n * 4)
	defer b.Free(bS)
	cS, _ := b.Alloc(n * 4)
	defer b.Free(cS)
	if err := b.CopyH2D(aS, f32Bytes(aH)); err != nil {
		t.Fatalf("H2D a: %v", err)
	}
	if err := b.CopyH2D(bS, f32Bytes(bH)); err != nil {
		t.Fatalf("H2D b: %v", err)
	}
	if err := call(aS, bS, cS, n); err != nil {
		t.Fatalf("%s: %v", opName, err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	buf := make([]byte, n*4)
	if err := b.CopyD2H(buf, cS); err != nil {
		t.Fatalf("D2H: %v", err)
	}
	got := bytesF32(buf)
	for i := 0; i < n; i++ {
		if got[i] != refC[i] {
			t.Fatalf("%s idx=%d: got=%g ref=%g (a=%g b=%g)",
				opName, i, got[i], refC[i], aH[i], bH[i])
		}
	}
}

// --- 2-arg driver (Neg) ---

func runUnaryF64BitExact(t *testing.T, b *PuregoBackend, opName string,
	call func(a, c DeviceBuffer, n int) error,
	cpu func(a float64) float64,
	rng [2]float64,
) {
	t.Helper()
	const n = 4096
	r := rand.New(rand.NewSource(int64(len(opName) * 41)))
	aH := fillRangeF64(r, n, rng[0], rng[1])
	refC := make([]float64, n)
	for i := 0; i < n; i++ {
		refC[i] = cpu(aH[i])
	}
	aS, _ := b.Alloc(n * 8)
	defer b.Free(aS)
	cS, _ := b.Alloc(n * 8)
	defer b.Free(cS)
	if err := b.CopyH2D(aS, f64Bytes(aH)); err != nil {
		t.Fatalf("H2D: %v", err)
	}
	if err := call(aS, cS, n); err != nil {
		t.Fatalf("%s: %v", opName, err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	buf := make([]byte, n*8)
	if err := b.CopyD2H(buf, cS); err != nil {
		t.Fatalf("D2H: %v", err)
	}
	got := bytesF64(buf)
	for i := 0; i < n; i++ {
		if got[i] != refC[i] {
			t.Fatalf("%s idx=%d: got=%g ref=%g (a=%g)", opName, i, got[i], refC[i], aH[i])
		}
	}
}

func runUnaryF32BitExact(t *testing.T, b *PuregoBackend, opName string,
	call func(a, c DeviceBuffer, n int) error,
	cpu func(a float32) float32,
	rng [2]float32,
) {
	t.Helper()
	const n = 4096
	r := rand.New(rand.NewSource(int64(len(opName) * 43)))
	aH := fillRangeF32(r, n, rng[0], rng[1])
	refC := make([]float32, n)
	for i := 0; i < n; i++ {
		refC[i] = cpu(aH[i])
	}
	aS, _ := b.Alloc(n * 4)
	defer b.Free(aS)
	cS, _ := b.Alloc(n * 4)
	defer b.Free(cS)
	if err := b.CopyH2D(aS, f32Bytes(aH)); err != nil {
		t.Fatalf("H2D: %v", err)
	}
	if err := call(aS, cS, n); err != nil {
		t.Fatalf("%s: %v", opName, err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	buf := make([]byte, n*4)
	if err := b.CopyD2H(buf, cS); err != nil {
		t.Fatalf("D2H: %v", err)
	}
	got := bytesF32(buf)
	for i := 0; i < n; i++ {
		if got[i] != refC[i] {
			t.Fatalf("%s idx=%d: got=%g ref=%g (a=%g)", opName, i, got[i], refC[i], aH[i])
		}
	}
}

// --- Тесты 12 арифметических (Sub/Mul/Div/Neg/AddScalar/MulScalar × 2) ---

func TestSubF64(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	runBinaryF64BitExact(t, b, "SubF64", b.SubF64,
		func(a, x float64) float64 { return a - x },
		[2]float64{-100, 100}, [2]float64{-100, 100})
}
func TestSubF32(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	runBinaryF32BitExact(t, b, "SubF32", b.SubF32,
		func(a, x float32) float32 { return a - x },
		[2]float32{-100, 100}, [2]float32{-100, 100})
}
func TestMulF64(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	runBinaryF64BitExact(t, b, "MulF64", b.MulF64,
		func(a, x float64) float64 { return a * x },
		[2]float64{-10, 10}, [2]float64{-10, 10})
}
func TestMulF32(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	runBinaryF32BitExact(t, b, "MulF32", b.MulF32,
		func(a, x float32) float32 { return a * x },
		[2]float32{-10, 10}, [2]float32{-10, 10})
}
func TestDivF64(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	// Ненулевые знаменатели (диапазон [0.5, 5] и его симметрия).
	runBinaryF64BitExact(t, b, "DivF64", b.DivF64,
		func(a, x float64) float64 { return a / x },
		[2]float64{-10, 10}, [2]float64{0.5, 5})
}
func TestDivF32(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	runBinaryF32BitExact(t, b, "DivF32", b.DivF32,
		func(a, x float32) float32 { return a / x },
		[2]float32{-10, 10}, [2]float32{0.5, 5})
}
func TestNegF64(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	runUnaryF64BitExact(t, b, "NegF64", b.NegF64,
		func(a float64) float64 { return -a },
		[2]float64{-100, 100})
}
func TestNegF32(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	runUnaryF32BitExact(t, b, "NegF32", b.NegF32,
		func(a float32) float32 { return -a },
		[2]float32{-100, 100})
}

// AddScalar / MulScalar тестируем bit-exact для F64 и F32.

func runScalarF64(t *testing.T, b *PuregoBackend, opName string,
	call func(a DeviceBuffer, s float64, c DeviceBuffer, n int) error,
	cpu func(a, s float64) float64,
	scalar float64,
) {
	t.Helper()
	const n = 4096
	r := rand.New(rand.NewSource(int64(len(opName) * 47)))
	aH := fillRangeF64(r, n, -50, 50)
	refC := make([]float64, n)
	for i := 0; i < n; i++ {
		refC[i] = cpu(aH[i], scalar)
	}
	aS, _ := b.Alloc(n * 8)
	defer b.Free(aS)
	cS, _ := b.Alloc(n * 8)
	defer b.Free(cS)
	b.CopyH2D(aS, f64Bytes(aH))
	if err := call(aS, scalar, cS, n); err != nil {
		t.Fatalf("%s: %v", opName, err)
	}
	b.Sync()
	buf := make([]byte, n*8)
	b.CopyD2H(buf, cS)
	got := bytesF64(buf)
	for i := 0; i < n; i++ {
		if got[i] != refC[i] {
			t.Fatalf("%s idx=%d: got=%g ref=%g (a=%g s=%g)",
				opName, i, got[i], refC[i], aH[i], scalar)
		}
	}
}
func runScalarF32(t *testing.T, b *PuregoBackend, opName string,
	call func(a DeviceBuffer, s float32, c DeviceBuffer, n int) error,
	cpu func(a, s float32) float32,
	scalar float32,
) {
	t.Helper()
	const n = 4096
	r := rand.New(rand.NewSource(int64(len(opName) * 53)))
	aH := fillRangeF32(r, n, -50, 50)
	refC := make([]float32, n)
	for i := 0; i < n; i++ {
		refC[i] = cpu(aH[i], scalar)
	}
	aS, _ := b.Alloc(n * 4)
	defer b.Free(aS)
	cS, _ := b.Alloc(n * 4)
	defer b.Free(cS)
	b.CopyH2D(aS, f32Bytes(aH))
	if err := call(aS, scalar, cS, n); err != nil {
		t.Fatalf("%s: %v", opName, err)
	}
	b.Sync()
	buf := make([]byte, n*4)
	b.CopyD2H(buf, cS)
	got := bytesF32(buf)
	for i := 0; i < n; i++ {
		if got[i] != refC[i] {
			t.Fatalf("%s idx=%d: got=%g ref=%g (a=%g s=%g)",
				opName, i, got[i], refC[i], aH[i], scalar)
		}
	}
}
func TestAddScalarF64(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	runScalarF64(t, b, "AddScalarF64", b.AddScalarF64,
		func(a, s float64) float64 { return a + s }, 3.14)
}
func TestAddScalarF32(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	runScalarF32(t, b, "AddScalarF32", b.AddScalarF32,
		func(a, s float32) float32 { return a + s }, 3.14)
}
func TestMulScalarF64(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	runScalarF64(t, b, "MulScalarF64", b.MulScalarF64,
		func(a, s float64) float64 { return a * s }, 2.718)
}
func TestMulScalarF32(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	runScalarF32(t, b, "MulScalarF32", b.MulScalarF32,
		func(a, s float32) float32 { return a * s }, 2.718)
}

// --- Exp/Log — стоп-точка Этапа 4 ---

type stats struct {
	maxAbsErr float64
	maxRelErr float64
	worst     int
	got, ref  float64
	inputArg  float64
}

// f64UnaryStats — сравнение результата unary-ядра с math.* эталоном в FP64.
func f64UnaryStats(gotBytes []byte, ref []float64, input []float64) stats {
	got := bytesF64(gotBytes)
	s := stats{worst: -1}
	for i := range got {
		absE := math.Abs(got[i] - ref[i])
		denom := math.Abs(ref[i])
		if denom < 1e-300 {
			denom = 1
		}
		relE := absE / denom
		if relE > s.maxRelErr {
			s.maxRelErr = relE
			s.worst = i
			s.got, s.ref, s.inputArg = got[i], ref[i], input[i]
		}
		if absE > s.maxAbsErr {
			s.maxAbsErr = absE
		}
	}
	return s
}
func f32UnaryStats(gotBytes []byte, ref []float32, input []float32) stats {
	got := bytesF32(gotBytes)
	s := stats{worst: -1}
	for i := range got {
		absE := math.Abs(float64(got[i] - ref[i]))
		denom := math.Abs(float64(ref[i]))
		if denom < 1e-30 {
			denom = 1
		}
		relE := absE / denom
		if relE > s.maxRelErr {
			s.maxRelErr = relE
			s.worst = i
			s.got, s.ref, s.inputArg = float64(got[i]), float64(ref[i]), float64(input[i])
		}
		if absE > s.maxAbsErr {
			s.maxAbsErr = absE
		}
	}
	return s
}

// runExpF32 / runExpF64 — прогон + фактические числа для отчёта.
// НЕ проваливает тест: доклад через t.Logf. Единственная валидация —
// что ядро не паникует и не даёт NaN на всех входах в диапазоне.

func runExpF32Range(t *testing.T, b *PuregoBackend, lo, hi float32, name string) {
	t.Helper()
	const n = 4096
	r := rand.New(rand.NewSource(int64(len(name) * 59)))
	aH := fillRangeF32(r, n, lo, hi)
	refC := make([]float32, n)
	for i := 0; i < n; i++ {
		refC[i] = float32(math.Exp(float64(aH[i])))
	}
	aS, _ := b.Alloc(n * 4)
	defer b.Free(aS)
	cS, _ := b.Alloc(n * 4)
	defer b.Free(cS)
	b.CopyH2D(aS, f32Bytes(aH))
	if err := b.ExpF32(aS, cS, n); err != nil {
		t.Fatalf("ExpF32: %v", err)
	}
	b.Sync()
	buf := make([]byte, n*4)
	b.CopyD2H(buf, cS)
	s := f32UnaryStats(buf, refC, aH)
	t.Logf("ExpF32 %s [%g..%g]: maxAbsErr=%.3e maxRelErr=%.3e worst_input=%.4g got=%.6g ref=%.6g",
		name, lo, hi, s.maxAbsErr, s.maxRelErr, s.inputArg, s.got, s.ref)
	// Проверка на NaN/Inf в выходе где вход в разумных пределах.
	got := bytesF32(buf)
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("ExpF32 %s idx=%d: NaN/Inf output for input %g", name, i, aH[i])
			break
		}
	}
}

// runExpF64Range — не используется (Stage 4 стоп-точка), сохранён шаблоном для будущего libdevice-порта.
var _ = f64UnaryStats

func runLogF32Range(t *testing.T, b *PuregoBackend, lo, hi float32, name string) {
	t.Helper()
	const n = 4096
	r := rand.New(rand.NewSource(int64(len(name) * 67)))
	aH := fillRangeF32(r, n, lo, hi)
	refC := make([]float32, n)
	for i := 0; i < n; i++ {
		refC[i] = float32(math.Log(float64(aH[i])))
	}
	aS, _ := b.Alloc(n * 4)
	defer b.Free(aS)
	cS, _ := b.Alloc(n * 4)
	defer b.Free(cS)
	b.CopyH2D(aS, f32Bytes(aH))
	if err := b.LogF32(aS, cS, n); err != nil {
		t.Fatalf("LogF32: %v", err)
	}
	b.Sync()
	buf := make([]byte, n*4)
	b.CopyD2H(buf, cS)
	s := f32UnaryStats(buf, refC, aH)
	t.Logf("LogF32 %s [%g..%g]: maxAbsErr=%.3e maxRelErr=%.3e worst_input=%.4g got=%.6g ref=%.6g",
		name, lo, hi, s.maxAbsErr, s.maxRelErr, s.inputArg, s.got, s.ref)
}
// runLogF64Range — не используется (Stage 4 стоп-точка).

func TestExpF32Ranges(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	runExpF32Range(t, b, -1, 1, "small")
	runExpF32Range(t, b, -10, 10, "medium")
	runExpF32Range(t, b, -80, 80, "wide")
}
// TestExpF64Ranges — раньше был stop-point-тест на errStage4F64Approx.
// После Этапа 4.5 (fdlibm-порт) ExpF64 реализован; валидация перенесена
// в TestExpF64Fdlibm (fdlibm_test.go).
func TestLogF32Ranges(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	runLogF32Range(t, b, 0.01, 1, "small")
	runLogF32Range(t, b, 1, 100, "medium")
	runLogF32Range(t, b, 100, 1e6, "wide")
}
// TestLogF64Ranges — раньше был stop-point-тест. После Этапа 4.5
// LogF64 реализован через fdlibm; валидация в TestLogF64Fdlibm.

// --- TestContextMigration ---
//
// Регрессионная страховка bind()-фикса. 300 итераций мелких CUDA-операций
// с runtime.Gosched() между ними, GOMAXPROCS ≥ 2. Если однажды кто-то
// уберёт bind() «для скорости», этот тест почти наверняка упадёт (test
// проваливается на первой миграции).

func TestContextMigration(t *testing.T) {
	oldMax := runtime.GOMAXPROCS(0)
	if oldMax < 2 {
		runtime.GOMAXPROCS(4)
		defer runtime.GOMAXPROCS(oldMax)
	}
	b := tryBackend(t)
	defer b.Close()

	const iters = 300
	const n = 128
	src := make([]float64, n)
	for i := range src {
		src[i] = float64(i) * 0.5
	}
	srcBytes := f64Bytes(src)
	for it := 0; it < iters; it++ {
		aS, err := b.Alloc(n * 8)
		if err != nil {
			t.Fatalf("iter=%d Alloc a: %v", it, err)
		}
		bS, err := b.Alloc(n * 8)
		if err != nil {
			b.Free(aS)
			t.Fatalf("iter=%d Alloc b: %v", it, err)
		}
		cS, err := b.Alloc(n * 8)
		if err != nil {
			b.Free(aS)
			b.Free(bS)
			t.Fatalf("iter=%d Alloc c: %v", it, err)
		}
		if err := b.CopyH2D(aS, srcBytes); err != nil {
			t.Fatalf("iter=%d H2D a: %v", it, err)
		}
		if err := b.CopyH2D(bS, srcBytes); err != nil {
			t.Fatalf("iter=%d H2D b: %v", it, err)
		}
		if err := b.AddF64(aS, bS, cS, n); err != nil {
			t.Fatalf("iter=%d AddF64: %v", it, err)
		}
		if err := b.Sync(); err != nil {
			t.Fatalf("iter=%d Sync: %v", it, err)
		}
		buf := make([]byte, n*8)
		if err := b.CopyD2H(buf, cS); err != nil {
			t.Fatalf("iter=%d D2H: %v", it, err)
		}
		got := bytesF64(buf)
		for i := 0; i < n; i++ {
			want := src[i] + src[i]
			if got[i] != want {
				t.Fatalf("iter=%d idx=%d: got=%g want=%g", it, i, got[i], want)
			}
		}
		b.Free(aS)
		b.Free(bS)
		b.Free(cS)
		runtime.Gosched()
	}
	t.Logf("TestContextMigration: %d iterations survived (GOMAXPROCS=%d)",
		iters, runtime.GOMAXPROCS(0))
}
