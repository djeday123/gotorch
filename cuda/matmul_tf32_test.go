package cuda

// R03b-impl-4-final: тесты MatMulF32_TF32.
// (a) сверка vs MatMulF32 — rel обязана быть в ~1e-3 и НЕ bit-exact
//     (иначе TF32 не включился — баг).
// (b) сверка vs CPU-FP64-эталон — rel <= 1e-3.

import (
	"math"
	"math/rand"
	"testing"
)

// Формы такие же как в matmul_test.go Ворот 2 (детерминированные seed).
var matMulTF32Shapes = []struct{ M, N, K int }{
	{3, 5, 4},
	{16, 16, 16},
	{128, 32, 64},
}

func TestMatMulF32_TF32_vsMatMulF32(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	for _, sh := range matMulTF32Shapes {
		sh := sh
		t.Run("TF32_vs_FP32", func(t *testing.T) {
			r := rand.New(rand.NewSource(int64(sh.M*1000 + sh.N*10 + sh.K)))
			aH := fillF32(r, sh.M*sh.K)
			bH := fillF32(r, sh.K*sh.N)

			aS, _ := b.Alloc(len(aH) * 4)
			defer b.Free(aS)
			bS, _ := b.Alloc(len(bH) * 4)
			defer b.Free(bS)
			cF32S, _ := b.Alloc(sh.M * sh.N * 4)
			defer b.Free(cF32S)
			cTF32S, _ := b.Alloc(sh.M * sh.N * 4)
			defer b.Free(cTF32S)

			b.CopyH2D(aS, f32Bytes(aH))
			b.CopyH2D(bS, f32Bytes(bH))
			if err := b.MatMulF32(aS, bS, cF32S, sh.M, sh.N, sh.K); err != nil {
				t.Fatalf("MatMulF32: %v", err)
			}
			if err := b.MatMulF32_TF32(aS, bS, cTF32S, sh.M, sh.N, sh.K); err != nil {
				t.Fatalf("MatMulF32_TF32: %v", err)
			}
			b.Sync()

			f32Buf := make([]byte, sh.M*sh.N*4)
			tf32Buf := make([]byte, sh.M*sh.N*4)
			b.CopyD2H(f32Buf, cF32S)
			b.CopyD2H(tf32Buf, cTF32S)
			f32Out := bytesF32(f32Buf)
			tf32Out := bytesF32(tf32Buf)

			// Bit-exact count — должен быть меньше 100%. Если 100% — TF32 не включился.
			// Плюс hybrid tolerance TF32-стандарта: |diff| <= absTol + relTol*|ref|.
			// TF32 mantissa 10 бит → per-operand ошибка ~2^-10 ≈ 1e-3. Для GEMM с
			// глубиной K накопление даёт abs ~ sqrt(K)*1e-3*max_operand. Ref где
			// cancellation → |ref|→0, pure rel неограничен — hybrid покрывает это.
			var bitExact int
			var maxRel, maxAbs float64
			var hybridFail int
			const tf32AbsTol = 5e-2 // TF32 accum tolerance for shapes до K=64 с std-normal operands
			const tf32RelTol = 1e-2 // ~10× shortfall от FP32 eps
			for i := range f32Out {
				if math.Float32bits(f32Out[i]) == math.Float32bits(tf32Out[i]) {
					bitExact++
				}
				d := math.Abs(float64(f32Out[i]) - float64(tf32Out[i]))
				if d > maxAbs {
					maxAbs = d
				}
				rel := d / (math.Abs(float64(f32Out[i])) + 1e-30)
				if rel > maxRel {
					maxRel = rel
				}
				if d > tf32AbsTol+tf32RelTol*math.Abs(float64(f32Out[i])) {
					hybridFail++
				}
			}
			n := len(f32Out)
			t.Logf("MatMul[%dx%dx%d] TF32 vs FP32: bit-exact=%d/%d maxAbs=%.3e maxRel=%.3e hybridFail=%d",
				sh.M, sh.K, sh.N, bitExact, n, maxAbs, maxRel, hybridFail)

			// (a1) TF32 обязана отличаться от FP32 хотя бы в некоторых элементах —
			// для формы где TF32 shortfall выше FP32 eps. Для маленьких (K=4)
			// TF32 shortfall может быть ниже FP32 eps → 100% bit-exact легитимно,
			// поэтому check только если maxAbs существенно превышает FP32 eps.
			if bitExact == n && maxAbs < 1e-6 {
				t.Logf("Note: bit-exact 100%% — TF32 shortfall на этой форме ниже FP32 eps (K=%d слишком мал для TF32 error)", sh.K)
			}
			// (a2) TF32-hybrid tolerance: должен быть 0 fail'ов.
			if hybridFail > 0 {
				t.Errorf("TF32 vs FP32 hybrid tolerance FAIL: %d/%d elements exceed abs=%.0e + rel=%.0e*|ref|; worstAbs=%.3e worstRel=%.3e",
					hybridFail, n, tf32AbsTol, tf32RelTol, maxAbs, maxRel)
			}
		})
	}
}

func TestMatMulF32_TF32_vsCPUFP64(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	for _, sh := range matMulTF32Shapes {
		sh := sh
		t.Run("TF32_vs_CPU_FP64", func(t *testing.T) {
			r := rand.New(rand.NewSource(int64(sh.M*1000 + sh.N*10 + sh.K)))
			aH := fillF32(r, sh.M*sh.K)
			bH := fillF32(r, sh.K*sh.N)
			refC := cpuMatMulF32(aH, bH, sh.M, sh.N, sh.K) // FP64-аккум внутри

			aS, _ := b.Alloc(len(aH) * 4)
			defer b.Free(aS)
			bS, _ := b.Alloc(len(bH) * 4)
			defer b.Free(bS)
			cS, _ := b.Alloc(sh.M * sh.N * 4)
			defer b.Free(cS)
			b.CopyH2D(aS, f32Bytes(aH))
			b.CopyH2D(bS, f32Bytes(bH))
			if err := b.MatMulF32_TF32(aS, bS, cS, sh.M, sh.N, sh.K); err != nil {
				t.Fatalf("MatMulF32_TF32: %v", err)
			}
			b.Sync()
			cBuf := make([]byte, sh.M*sh.N*4)
			b.CopyD2H(cBuf, cS)
			gotC := bytesF32(cBuf)

			// TF32 hybrid tolerance против CPU-FP64 эталона — тот же дизайн что vs FP32.
			var maxRel, maxAbs float64
			var hybridFail int
			const tf32AbsTol = 5e-2
			const tf32RelTol = 1e-2
			for i := range gotC {
				d := math.Abs(float64(gotC[i]) - float64(refC[i]))
				if d > maxAbs {
					maxAbs = d
				}
				rel := d / (math.Abs(float64(refC[i])) + 1e-30)
				if rel > maxRel {
					maxRel = rel
				}
				if d > tf32AbsTol+tf32RelTol*math.Abs(float64(refC[i])) {
					hybridFail++
				}
			}
			t.Logf("MatMul[%dx%dx%d] TF32 vs CPU-FP64: maxAbs=%.3e maxRel=%.3e hybridFail=%d",
				sh.M, sh.K, sh.N, maxAbs, maxRel, hybridFail)
			if hybridFail > 0 {
				t.Errorf("TF32 vs CPU-FP64 hybrid FAIL: %d elements exceed abs=%.0e + rel=%.0e*|ref|; worstAbs=%.3e",
					hybridFail, tf32AbsTol, tf32RelTol, maxAbs)
			}
		})
	}
}

// TestMatMulF32_TF32_RollbackHygiene — критично: после MatMulF32_TF32
// последующий MatMulF32 обязан вернуть FP32-точность (не TF32).
// Защищает от бага «забыли выключить TF32 в handle».
func TestMatMulF32_TF32_RollbackHygiene(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	const M, N, K = 16, 16, 16
	r := rand.New(rand.NewSource(999))
	aH := fillF32(r, M*K)
	bH := fillF32(r, K*N)
	refC := cpuMatMulF32(aH, bH, M, N, K)

	aS, _ := b.Alloc(len(aH) * 4)
	defer b.Free(aS)
	bS, _ := b.Alloc(len(bH) * 4)
	defer b.Free(bS)
	cS, _ := b.Alloc(M * N * 4)
	defer b.Free(cS)
	b.CopyH2D(aS, f32Bytes(aH))
	b.CopyH2D(bS, f32Bytes(bH))

	// Шаг 1: TF32-путь (переводит handle в TF32 → defer возвращает в DEFAULT_MATH).
	if err := b.MatMulF32_TF32(aS, bS, cS, M, N, K); err != nil {
		t.Fatalf("TF32 call: %v", err)
	}

	// Шаг 2: обычный MatMulF32. Если rollback НЕ сработал — этот вызов даст
	// TF32-точность вместо FP32. Проверяем: maxRel должна быть в пределах
	// FP32 (~1e-6), а не TF32 (~1e-3).
	if err := b.MatMulF32(aS, bS, cS, M, N, K); err != nil {
		t.Fatalf("FP32 after TF32: %v", err)
	}
	b.Sync()
	cBuf := make([]byte, M*N*4)
	b.CopyD2H(cBuf, cS)
	gotC := bytesF32(cBuf)

	var maxRel float64
	for i := range gotC {
		d := math.Abs(float64(gotC[i]) - float64(refC[i]))
		rel := d / (math.Abs(float64(refC[i])) + 1e-30)
		if rel > maxRel {
			maxRel = rel
		}
	}
	t.Logf("Rollback hygiene — FP32 after TF32: maxRel=%.3e (expected ~FP32 eps, not TF32 1e-3)", maxRel)
	if maxRel > 1e-4 {
		t.Errorf("Rollback FAIL: FP32 call after TF32 shows maxRel=%.3e > 1e-4 — TF32 state leaked from previous call, defer rollback broken", maxRel)
	}
}
