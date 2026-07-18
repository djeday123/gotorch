package cuda

// Ворота 2 R02b — cuBLAS DGEMM / SGEMM корректность.
//
// CPU-эталон для ОБОИХ dtype считает в FP64 (двойной аккумулятор). Для
// F32-сравнения final result округляется в float32 в самом конце. Это
// прямой урок нашей саги валидации 93.3% из B1: эталон обязан быть
// точнее проверяемого, иначе измеренная ошибка = шум эталона + ошибка
// проверяемого, и мы не различаем их.
//
// Допуски:
//   * F64 — element-wise rel < 1e-12 (эталон в FP64, ошибка cuBLAS DGEMM
//     на этих формах ~1e-15..1e-13; запас × 10-1000).
//   * F32 — hybrid |got - ref| < 1e-4 + 1e-5 × |ref| (BLAS/LAPACK-стандарт).
//     Element-wise rel=1e-5 для FP32 GEMM с K≥64 НЕДОСТИЖИМ из-за
//     cancellation: eps × K × max_partial ≈ 1.19e-7 × 64 × 20 ≈ 1e-4
//     абсолютной ошибки в worst-case-сумме. При малом |ref| относительная
//     ошибка на элементе взлетает; это не баг реализации, это неизбежное
//     свойство FP32-суммирования. Абсолютная компонента 1e-4 в hybrid
//     покрывает cancellation, относительная 1e-5 ловит систематические
//     ошибки (перепутанные операнды, транспоны, битый lda/ldb).
//   * НЕ ужесточай допуск F32 обратно из благих намерений — упрёшься в
//     тот же cancellation.
//
// Формы: [3×4]×[4×5], [16×16]×[16×16], [128×64]×[64×32], [1×1]×[1×1].

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// --- CPU-эталон (FP64-аккумулятор для обоих dtype) ---

func cpuMatMulF64(a, b []float64, m, n, k int) []float64 {
	c := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for l := 0; l < k; l++ {
				s += a[i*k+l] * b[l*n+j]
			}
			c[i*n+j] = s
		}
	}
	return c
}

// cpuMatMulF32 — CPU-эталон для F32 с FP64-аккумулятором.
// Входы конвертируются в float64, сумма в float64, приведение в float32
// в самом конце. Это устраняет шум эталона, оставляя измеренную ошибку
// чистой ошибкой cuBLAS SGEMM.
func cpuMatMulF32(a, b []float32, m, n, k int) []float32 {
	c := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for l := 0; l < k; l++ {
				s += float64(a[i*k+l]) * float64(b[l*n+j])
			}
			c[i*n+j] = float32(s)
		}
	}
	return c
}

// --- Fill: детерминированный seed ---

func fillF64(r *rand.Rand, n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = r.NormFloat64()
	}
	return x
}

func fillF32(r *rand.Rand, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	return x
}

// --- Errors ---

// maxRelErrF64 — worst-element relative error, tolerance для F64.
func maxRelErrF64(got, ref []float64) (relErr float64, worstIdx int) {
	if len(got) != len(ref) {
		return math.Inf(1), -1
	}
	for i := range got {
		denom := math.Abs(ref[i])
		if denom < 1e-300 {
			denom = 1
		}
		e := math.Abs(got[i]-ref[i]) / denom
		if e > relErr {
			relErr, worstIdx = e, i
		}
	}
	return
}

// f32Stats — обе метрики для отчёта: worst abs и worst rel.
type f32Stats struct {
	maxAbsErr, maxRelErr float64
	worstAbsIdx          int
	worstRelIdx          int
}

func f32Compare(got, ref []float32) f32Stats {
	s := f32Stats{worstAbsIdx: -1, worstRelIdx: -1}
	if len(got) != len(ref) {
		s.maxAbsErr = math.Inf(1)
		s.maxRelErr = math.Inf(1)
		return s
	}
	for i := range got {
		absE := math.Abs(float64(got[i] - ref[i]))
		if absE > s.maxAbsErr {
			s.maxAbsErr, s.worstAbsIdx = absE, i
		}
		denom := math.Abs(float64(ref[i]))
		if denom < 1e-30 {
			denom = 1
		}
		relE := absE / denom
		if relE > s.maxRelErr {
			s.maxRelErr, s.worstRelIdx = relE, i
		}
	}
	return s
}

// hybridPassF32 — BLAS/LAPACK стандарт: |got - ref| < abs_tol + rel_tol × |ref|.
// Абсолютная компонента покрывает cancellation, относительная ловит
// систематические ошибки.
func hybridPassF32(got, ref []float32, absTol, relTol float64) (fail bool, worstAbs, worstRel, worstBound float64, worstIdx int) {
	worstIdx = -1
	for i := range got {
		absE := math.Abs(float64(got[i] - ref[i]))
		bound := absTol + relTol*math.Abs(float64(ref[i]))
		if absE > bound {
			fail = true
			if absE > worstAbs {
				worstAbs, worstIdx = absE, i
				worstBound = bound
				denom := math.Abs(float64(ref[i]))
				if denom < 1e-30 {
					denom = 1
				}
				worstRel = absE / denom
			}
		}
	}
	return
}

// --- Общий driver: одна форма F64 ---

func runMatMulShapeF64(t *testing.T, b *PuregoBackend, m, n, k int, tol float64) {
	t.Helper()
	r := rand.New(rand.NewSource(int64(m*1000 + n*10 + k)))
	aH := fillF64(r, m*k)
	bH := fillF64(r, k*n)
	refC := cpuMatMulF64(aH, bH, m, n, k)

	aBytes := f64Bytes(aH)
	bBytes := f64Bytes(bH)
	cBytes := make([]byte, len(refC)*8)

	aS, err := b.Alloc(len(aBytes))
	if err != nil {
		t.Fatalf("Alloc A [%dx%d]: %v", m, k, err)
	}
	defer b.Free(aS)
	bS, err := b.Alloc(len(bBytes))
	if err != nil {
		t.Fatalf("Alloc B [%dx%d]: %v", k, n, err)
	}
	defer b.Free(bS)
	cS, err := b.Alloc(len(cBytes))
	if err != nil {
		t.Fatalf("Alloc C [%dx%d]: %v", m, n, err)
	}
	defer b.Free(cS)

	if err := b.CopyH2D(aS, aBytes); err != nil {
		t.Fatalf("H2D A: %v", err)
	}
	if err := b.CopyH2D(bS, bBytes); err != nil {
		t.Fatalf("H2D B: %v", err)
	}
	if err := b.MatMulF64(aS, bS, cS, m, n, k); err != nil {
		t.Fatalf("MatMulF64: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	if err := b.CopyD2H(cBytes, cS); err != nil {
		t.Fatalf("D2H C: %v", err)
	}
	gotC := bytesF64(cBytes)
	relErr, wi := maxRelErrF64(gotC, refC)
	t.Logf("F64 [%dx%dx%d]: maxRelErr=%.3e (tol=%.0e)", m, k, n, relErr, tol)
	if relErr > tol {
		t.Errorf("F64 [%dx%dx%d]: maxRelErr=%.3e > tol=%.0e; worst idx=%d ref=%g got=%g",
			m, k, n, relErr, tol, wi, refC[wi], gotC[wi])
	}
}

// --- Общий driver: одна форма F32 (hybrid tolerance) ---

func runMatMulShapeF32(t *testing.T, b *PuregoBackend, m, n, k int, absTol, relTol float64) {
	t.Helper()
	r := rand.New(rand.NewSource(int64(m*1000 + n*10 + k)))
	aH := fillF32(r, m*k)
	bH := fillF32(r, k*n)
	refC := cpuMatMulF32(aH, bH, m, n, k)

	aBytes := f32Bytes(aH)
	bBytes := f32Bytes(bH)
	cBytes := make([]byte, len(refC)*4)

	aS, err := b.Alloc(len(aBytes))
	if err != nil {
		t.Fatalf("Alloc A [%dx%d]: %v", m, k, err)
	}
	defer b.Free(aS)
	bS, err := b.Alloc(len(bBytes))
	if err != nil {
		t.Fatalf("Alloc B [%dx%d]: %v", k, n, err)
	}
	defer b.Free(bS)
	cS, err := b.Alloc(len(cBytes))
	if err != nil {
		t.Fatalf("Alloc C [%dx%d]: %v", m, n, err)
	}
	defer b.Free(cS)

	if err := b.CopyH2D(aS, aBytes); err != nil {
		t.Fatalf("H2D A: %v", err)
	}
	if err := b.CopyH2D(bS, bBytes); err != nil {
		t.Fatalf("H2D B: %v", err)
	}
	if err := b.MatMulF32(aS, bS, cS, m, n, k); err != nil {
		t.Fatalf("MatMulF32: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	if err := b.CopyD2H(cBytes, cS); err != nil {
		t.Fatalf("D2H C: %v", err)
	}
	gotC := bytesF32(cBytes)
	stats := f32Compare(gotC, refC)
	t.Logf("F32 [%dx%dx%d]: maxAbsErr=%.3e maxRelErr=%.3e (hybrid abs=%.0e rel=%.0e)",
		m, k, n, stats.maxAbsErr, stats.maxRelErr, absTol, relTol)
	fail, wAbs, wRel, wBound, wi := hybridPassF32(gotC, refC, absTol, relTol)
	if fail {
		t.Errorf("F32 [%dx%dx%d]: hybrid FAIL — worst idx=%d abs=%.3e rel=%.3e bound=%.3e ref=%g got=%g",
			m, k, n, wi, wAbs, wRel, wBound, refC[wi], gotC[wi])
	}
}

// --- Ворота 2 форма 1..4 ---

var matMulShapes = []struct{ M, N, K int }{
	{3, 5, 4},     // [3x4] × [4x5]
	{16, 16, 16},  // [16x16] × [16x16]
	{128, 32, 64}, // [128x64] × [64x32]
	{1, 1, 1},     // [1x1] × [1x1]
}

func TestMatMulF64(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	const tol = 1e-12
	for _, sh := range matMulShapes {
		sh := sh
		t.Run(fmt.Sprintf("%dx%dx%d", sh.M, sh.K, sh.N), func(t *testing.T) {
			runMatMulShapeF64(t, b, sh.M, sh.N, sh.K, tol)
		})
	}
}

func TestMatMulF32(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	const absTol, relTol = 1e-4, 1e-5 // BLAS/LAPACK hybrid: покрывает cancellation
	for _, sh := range matMulShapes {
		sh := sh
		t.Run(fmt.Sprintf("%dx%dx%d", sh.M, sh.K, sh.N), func(t *testing.T) {
			runMatMulShapeF32(t, b, sh.M, sh.N, sh.K, absTol, relTol)
		})
	}
}

// --- ForeignStorage roundtrip: MatMul через дверь входа ---

func TestMatMulForeign(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	const M, N, K = 32, 24, 16
	const tol = 1e-12

	r := rand.New(rand.NewSource(777))
	aH := fillF64(r, M*K)
	bH := fillF64(r, K*N)
	refC := cpuMatMulF64(aH, bH, M, N, K)

	aBytes := f64Bytes(aH)
	bBytes := f64Bytes(bH)

	aS, err := b.Alloc(len(aBytes))
	if err != nil {
		t.Fatalf("Alloc A: %v", err)
	}
	defer b.Free(aS)
	bS, err := b.Alloc(len(bBytes))
	if err != nil {
		t.Fatalf("Alloc B: %v", err)
	}
	defer b.Free(bS)
	cS, err := b.Alloc(M*N*8)
	if err != nil {
		t.Fatalf("Alloc C: %v", err)
	}
	defer b.Free(cS)

	if err := b.CopyH2D(aS, aBytes); err != nil {
		t.Fatalf("H2D A: %v", err)
	}
	if err := b.CopyH2D(bS, bBytes); err != nil {
		t.Fatalf("H2D B: %v", err)
	}

	// Оборачиваем A как ForeignStorage через дверь входа/выхода.
	aPtr := UnsafeExtractDevicePtr(aS)
	if aPtr == nil {
		t.Fatal("UnsafeExtractDevicePtr(aS) is nil")
	}
	aForeign := WrapDevicePtr(aPtr, aS.SizeBytes(), aS.Device())
	// Проверяем что foreign реализует DeviceBuffer.
	var _ DeviceBuffer = aForeign

	if err := b.MatMulF64(aForeign, bS, cS, M, N, K); err != nil {
		t.Fatalf("MatMulF64 via foreign: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	cBytes := make([]byte, M*N*8)
	if err := b.CopyD2H(cBytes, cS); err != nil {
		t.Fatalf("D2H C: %v", err)
	}
	gotC := bytesF64(cBytes)
	e, wi := maxRelErrF64(gotC, refC)
	if e > tol {
		t.Fatalf("MatMulF64 via ForeignStorage: maxRelErr=%.3e > tol=%.0e; worst idx=%d ref=%g got=%g",
			e, tol, wi, refC[wi], gotC[wi])
	}
	t.Logf("F64 foreign [%dx%dx%d]: maxRelErr=%.3e (tol=%.0e)", M, K, N, e, tol)

	// suppress unused import guard
	_ = unsafe.Sizeof(ForeignStorage{})
}
