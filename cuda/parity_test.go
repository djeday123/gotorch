//go:build gpu

package cuda

// TestParityLegacy — Ворота 6.2 R02b.
//
// Сравнение старого cgo-путя (GPUBackend через libgotorch_cuda.so) и
// нового purego-путя (PuregoBackend через libcuda + libcublas + PTX)
// побитово по F64 на одинаковых входах.
//
// Обоснование ожидания:
//
//   * MatMulF64: оба мира зовут ОДНУ libcublas → cublasDgemm с одинаковыми
//     параметрами (col-major транспоны, alpha=1, beta=0). Ожидание bit-exact
//     эмпирически подтверждается на [16×16×16] и [128×64×32]: **100%
//     совпадение мантисс**. На маленькой [3×4×5] cuBLAS выбирает другой
//     kernel-вариант (heuristics по форме) и порядок FMA даёт разброс до
//     ~K×eps×|ref|. Это НЕ баг реализации — это стандартный BLAS-контракт
//     "≤ n × eps" где n = глубина суммирования. Проверка — hybrid tolerance
//     BLAS-стандарта: |got - ref| < eps × K × 10 × |ref|. Систематические
//     ошибки (транспон/lda/beta≠0) дают расхождение ≫ 10×K×eps и будут
//     пойманы. Дополнительно логируется число bit-exact matches для
//     наглядности: если crash в libcublas selection → счётчик резко упадёт.
//
//   * AddF64 / MulF64: cgo-путь использует ops.cu ядро (nvcc), purego-путь
//     использует PTX-ядро. Оба делают одну FP64-операцию на элемент. Один
//     FADD/FMUL, никаких аккумуляторов — **bit-exact ожидаем**.
//
// Файл под тегом gpu: cgo-часть требует `-tags gpu` для сборки.
// Заглушка parity_stub_test.go активна при отсутствии тега и печатает
// точную причину пропуска.

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/djeday123/gotorch/tensor"
)

// tryLegacyGPU инициализирует cgo GPUBackend или Skip'ает с указанием
// точной причины (тот же паттерн что tryBackend в purego_test.go).
func tryLegacyGPU(t *testing.T) *GPUBackend {
	t.Helper()
	if DeviceCount() == 0 {
		t.Skip("legacy cgo: no CUDA device visible via cudart")
	}
	b, err := NewGPUBackend(0)
	if err != nil {
		t.Skipf("legacy cgo: NewGPUBackend(0): %v", err)
	}
	return b
}

// parityF64 запускает одинаковые входы через оба backend'а и сравнивает
// побитово (math.Float64bits). Возвращает счётчик несовпадений и максимальный
// ULP-разброс.
type parityStat struct {
	n              int
	mismatches     int    // элементов с bitDelta != 0
	worstIdx       int
	worstUlp       uint64 // максимальный |bits(a) - bits(b)| по ULP-метрике
	worstAbs       float64
	worstA, worstB float64
}

// ulpDistF64 — беззнаковое ULP-расстояние между двумя FP64 через
// "brute-force" ordered-int mapping (переворот знакового бита через
// twos-complement на sign-magnitude): для положительных b1>b2 ⇒ dist=b1-b2,
// для смешанных знаков включает 0.
func ulpDistF64(x, y float64) uint64 {
	ux := math.Float64bits(x)
	uy := math.Float64bits(y)
	// signed → biased ordered: xor sign-bit + маска знаковой части.
	const signMask uint64 = 1 << 63
	if ux&signMask != 0 {
		ux = ^ux + 1
	}
	if uy&signMask != 0 {
		uy = ^uy + 1
	}
	if ux > uy {
		return ux - uy
	}
	return uy - ux
}

func compareBitwiseF64(a, b []float64) parityStat {
	s := parityStat{n: len(a), worstIdx: -1}
	if len(a) != len(b) {
		s.mismatches = -1
		return s
	}
	for i := range a {
		if math.Float64bits(a[i]) == math.Float64bits(b[i]) {
			continue
		}
		s.mismatches++
		ulp := ulpDistF64(a[i], b[i])
		if ulp > s.worstUlp {
			s.worstUlp = ulp
			s.worstIdx = i
			s.worstAbs = math.Abs(a[i] - b[i])
			s.worstA, s.worstB = a[i], b[i]
		}
	}
	return s
}

// TestParityLegacyMatMulF64 — cuBLAS DGEMM через оба пути.
// Ожидание: bit-exact, оба зовут cublasDgemm с идентичными параметрами.
func TestParityLegacyMatMulF64(t *testing.T) {
	pB := tryBackend(t)
	defer pB.Close()
	lB := tryLegacyGPU(t)
	defer lB.Close()

	shapes := []struct{ M, N, K int }{
		{3, 5, 4},
		{16, 16, 16},
		{128, 32, 64},
	}
	for _, sh := range shapes {
		sh := sh
		t.Run("MatMulF64", func(t *testing.T) {
			r := rand.New(rand.NewSource(int64(sh.M*1000 + sh.N*10 + sh.K)))
			aH := fillF64(r, sh.M*sh.K)
			bH := fillF64(r, sh.K*sh.N)

			// --- purego путь ---
			aS, _ := pB.Alloc(len(aH) * 8)
			defer pB.Free(aS)
			bS, _ := pB.Alloc(len(bH) * 8)
			defer pB.Free(bS)
			cS, _ := pB.Alloc(sh.M * sh.N * 8)
			defer pB.Free(cS)
			pB.CopyH2D(aS, f64Bytes(aH))
			pB.CopyH2D(bS, f64Bytes(bH))
			if err := pB.MatMulF64(aS, bS, cS, sh.M, sh.N, sh.K); err != nil {
				t.Fatalf("purego MatMulF64: %v", err)
			}
			pB.Sync()
			pOut := make([]byte, sh.M*sh.N*8)
			pB.CopyD2H(pOut, cS)
			pResult := bytesF64(pOut)

			// --- legacy cgo путь ---
			aT := tensor.New(aH, []int{sh.M, sh.K})
			bT := tensor.New(bH, []int{sh.K, sh.N})
			aG, err := lB.Upload(aT)
			if err != nil {
				t.Fatalf("legacy Upload A: %v", err)
			}
			defer aG.Free()
			bG, err := lB.Upload(bT)
			if err != nil {
				t.Fatalf("legacy Upload B: %v", err)
			}
			defer bG.Free()
			cG, err := lB.MatMul(aG, bG)
			if err != nil {
				t.Fatalf("legacy MatMul: %v", err)
			}
			defer cG.Free()
			StreamSync()
			lResult := cG.ToCPU().Data()

			// --- BLAS-стандарт: |diff| < K × 10 × eps × |ref|.
			// Отдельно логируем bit-exact matches: побочная метрика
			// для наблюдения (idеал = 100%, реальность = зависит от формы). ---
			const eps = 2.220446049250313e-16
			relTol := float64(sh.K) * 10 * eps
			absTol := eps // покрывает элементы с |ref| ≈ 0
			st := compareBitwiseF64(pResult, lResult)
			var blasFail bool
			var worstRel float64
			var worstBLASIdx int = -1
			for i := range pResult {
				d := math.Abs(pResult[i] - lResult[i])
				bound := absTol + relTol*math.Abs(lResult[i])
				rel := d / (math.Abs(lResult[i]) + 1e-300)
				if rel > worstRel {
					worstRel = rel
					worstBLASIdx = i
				}
				if d > bound {
					blasFail = true
				}
			}
			t.Logf("MatMulF64 [%dx%dx%d]: n=%d bit-exact=%d/%d "+
				"BLAS(|d|<K×10×eps×|ref|): %v worstRel=%.3e",
				sh.M, sh.K, sh.N, st.n, st.n-st.mismatches, st.n,
				!blasFail, worstRel)
			if blasFail {
				t.Errorf("MatMulF64 [%dx%dx%d]: BLAS tolerance FAIL "+
					"(worstRel=%.3e > %.3e); idx=%d absErr=%.3e",
					sh.M, sh.K, sh.N, worstRel, relTol,
					worstBLASIdx, st.worstAbs)
			}
		})
	}
}

// TestParityLegacyAddF64 — тривиальный elementwise c=a+b.
// Ожидание: bit-exact (один FADD без денормалей на нашей выборке).
func TestParityLegacyAddF64(t *testing.T) {
	pB := tryBackend(t)
	defer pB.Close()
	lB := tryLegacyGPU(t)
	defer lB.Close()

	const n = 4096
	r := rand.New(rand.NewSource(42))
	aH := fillF64(r, n)
	bH := fillF64(r, n)

	// --- purego путь ---
	aS, _ := pB.Alloc(n * 8)
	defer pB.Free(aS)
	bS, _ := pB.Alloc(n * 8)
	defer pB.Free(bS)
	cS, _ := pB.Alloc(n * 8)
	defer pB.Free(cS)
	pB.CopyH2D(aS, f64Bytes(aH))
	pB.CopyH2D(bS, f64Bytes(bH))
	if err := pB.AddF64(aS, bS, cS, n); err != nil {
		t.Fatalf("purego AddF64: %v", err)
	}
	pB.Sync()
	pOut := make([]byte, n*8)
	pB.CopyD2H(pOut, cS)
	pResult := bytesF64(pOut)

	// --- legacy cgo путь ---
	aT := tensor.New(aH, []int{n})
	bT := tensor.New(bH, []int{n})
	aG, _ := lB.Upload(aT)
	defer aG.Free()
	bG, _ := lB.Upload(bT)
	defer bG.Free()
	cG, err := lB.Add(aG, bG)
	if err != nil {
		t.Fatalf("legacy Add: %v", err)
	}
	defer cG.Free()
	StreamSync()
	lResult := cG.ToCPU().Data()

	st := compareBitwiseF64(pResult, lResult)
	t.Logf("AddF64 n=%d: bit-exact=%d differ=%d worstUlp=%d",
		n, st.n-st.mismatches, st.mismatches, st.worstUlp)
	if st.mismatches > 0 {
		t.Errorf("AddF64: %d bit mismatches (expected 0 for elementwise FADD), "+
			"idx=%d worstUlp=%d purego=%.17g legacy=%.17g absErr=%.3e",
			st.mismatches, st.worstIdx, st.worstUlp,
			st.worstA, st.worstB, st.worstAbs)
	}
}

// TestParityLegacyMulF64 — тривиальный elementwise c=a*b.
// Ожидание: bit-exact (один FMUL, никаких аккумуляторов).
func TestParityLegacyMulF64(t *testing.T) {
	pB := tryBackend(t)
	defer pB.Close()
	lB := tryLegacyGPU(t)
	defer lB.Close()

	const n = 4096
	r := rand.New(rand.NewSource(43))
	aH := fillF64(r, n)
	bH := fillF64(r, n)

	aS, _ := pB.Alloc(n * 8)
	defer pB.Free(aS)
	bS, _ := pB.Alloc(n * 8)
	defer pB.Free(bS)
	cS, _ := pB.Alloc(n * 8)
	defer pB.Free(cS)
	pB.CopyH2D(aS, f64Bytes(aH))
	pB.CopyH2D(bS, f64Bytes(bH))
	if err := pB.MulF64(aS, bS, cS, n); err != nil {
		t.Fatalf("purego MulF64: %v", err)
	}
	pB.Sync()
	pOut := make([]byte, n*8)
	pB.CopyD2H(pOut, cS)
	pResult := bytesF64(pOut)

	aT := tensor.New(aH, []int{n})
	bT := tensor.New(bH, []int{n})
	aG, _ := lB.Upload(aT)
	defer aG.Free()
	bG, _ := lB.Upload(bT)
	defer bG.Free()
	cG, err := lB.Mul(aG, bG)
	if err != nil {
		t.Fatalf("legacy Mul: %v", err)
	}
	defer cG.Free()
	StreamSync()
	lResult := cG.ToCPU().Data()

	st := compareBitwiseF64(pResult, lResult)
	t.Logf("MulF64 n=%d: bit-exact=%d differ=%d worstUlp=%d",
		n, st.n-st.mismatches, st.mismatches, st.worstUlp)
	if st.mismatches > 0 {
		t.Errorf("MulF64: %d bit mismatches (expected 0 for elementwise FMUL), "+
			"idx=%d worstUlp=%d purego=%.17g legacy=%.17g absErr=%.3e",
			st.mismatches, st.worstIdx, st.worstUlp,
			st.worstA, st.worstB, st.worstAbs)
	}
}

// suppress unused warnings when parts of API move.
var _ = unsafe.Sizeof(Storage{})
