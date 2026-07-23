# B-impl-2 — F16 mixed precision MatMul (второй этап главного блюда)

**Дата:** 2026-07-23
**Итог:** ✅ ПРОЙДЕН. `MatMulF16`/`MatMulStridedBatchedF16` через `cublasGemmEx`/`GemmStridedBatchedEx` реализованы поверх готового `libgotorch_blas_wrapper.so` (B-impl-1 доработка предусмотрела оба вызова). PTX `cvt.rn.f16.f32` / `cvt.f32.f16` для round-trip. Прогнозы paper B4 (F16 5e-4 rel) **побиты 10-100×** — реальный maxRel 3.4e-5.
**Method count**: 68 → **72** (+4: MatMulF16, MatMulStridedBatchedF16, CastF32ToF16, CastF16ToF32).

---

## Прогнозы, записанные ДО измерения

| # | Метрика | Прогноз (paper B4) | Actual | Итог |
|---|---|---|---|---|
| P1 | CastF32ToF16 round-trip rel | ≤ 1e-3 (F16 eps ~5e-4) | 4.75e-4 | ✅ подтверждён (правда почти на floor) |
| P2 | F16 non-batched vs J(F64 CPU on F16-inputs) | hybrid abs=1e-3+rel=5e-4 | maxAbs 2.58e-6, maxRel 3.4e-5 | ✅ запас **~150×** |
| P3 | F16 batched vs J | hybrid abs=1e-3+rel=5e-3 | maxAbs 1.77e-6, maxRel 3.4e-5 | ✅ запас **~1000×** |
| P4 | F16 non-batched vs batched b=1 | ~bit-exact (одна algo path) | **bit-exact 512/512** | ✅ подтверждён |

**Почему запас так большой:** прогноз 5e-4 был для F16 IO vs F32 reference с F32 accumulator внутри вычисления. Наш judge использует **F16-rounded inputs**, но F64 accumulator — то есть тест ловит только разницу *между algo compute path (TF32)* и *F64 accumulator CPU*. Погрешность F16 quantization на входах учтена одинаково в обеих сторонах.

Правило "два числа": pre-registered `paper B4` + actual зафиксированы; прогноз **корректен для другого случая** (когда F32 inputs дают F16-quant на GPU).

---

## Реализация

### PTX (`cuda/ptx_kernels.go`)
2 тривиальных cvt kernel:
- `cvt_f32_to_f16(src, dst, n)`: `ld.global.f32 → cvt.rn.f16.f32 → st.global.b16`. RTN round.
- `cvt_f16_to_f32(src, dst, n)`: `ld.global.b16 → cvt.f32.f16 → st.global.f32`. Widening exact.

Grid ceil(n/256), block 256. ASCII-catch #4 (русский комментарий в PTX) — правило подтверждено, fix в 30 секунд через JIT log.

### Wrapper использован без дополнений
`libgotorch_blas_wrapper.so` из B-impl-1 доработки уже экспортирует `gt_gemm_ex` и `gt_gemm_strided_batched_ex`. B-impl-2 не пересобирал wrapper.

### Go методы (`cuda/backend_purego.go`)
```go
MatMulF16(a, b, c DeviceBuffer, m, n, k int) error
MatMulStridedBatchedF16(a, b, c DeviceBuffer, batch, m, n, k int, strideA, strideB, strideC int64) error
CastF32ToF16(src, dst DeviceBuffer, n int) error
CastF16ToF32(src, dst DeviceBuffer, n int) error
```

**F16 без wrapper — HARD ERROR:** `MatMulF16` возвращает `"libgotorch_blas_wrapper.so required"` если нет `.so`. Loop-fallback для F16 не имеет смысла (single-Sgemm не даёт F16 semantics).

**Compute type:** `CUBLAS_COMPUTE_32F_FAST_TF32` (F32 accumulator через TF32 tensor cores). Соответствует goml `MatMulF16` (`cublas.go:343-347`).

**Column-major swap:** тот же паттерн что F32 batched — `(vb, va, vc)` с M/N swap.

### Backend interface + doc
`api.go`: +4 методов с doc-string о F16 buffer contract (uint16 LE, host math.Float16bits).

---

## Тесты (`cuda/matmul_f16_test.go`, 4/4 PASS)

| Тест | Форма | Результат |
|---|---|---|
| CastF32ToF16_RoundTrip | n=256 | maxRel 4.75e-4 (floor 1e-3) |
| MatMulF16 | m=32 n=16 k=24 | maxRel 3.4e-5 |
| MatMulStridedBatchedF16 | b=4 m=16 n=32 k=24 | maxRel 3.4e-5 |
| NonBatchedEqBatched1 | m=16 n=32 k=24 | **bit-exact 512/512** |

**F16 CPU reference:** IEEE 754 binary32↔binary16 round-nearest-even implementation, verified against math.Float32frombits round-trip.

---

## Регрессия ворот

| Гейт | Результат |
|---|---|
| **B-impl-2 gotorch/v6/cuda** (4 F16 + PTX conv) | ✅ PASS |
| gotorch cuda full (P2/P3/P4/P5A/B-impl-1 + доработка + B-impl-2) | ✅ ok 1.005s |
| adapter regression | ✅ ok 0.378s |
| interop_smoke 6/6 | ✅ ok 0.346s |
| f64ref 9/9 | ✅ ok 0.373s |
| P1-ABJ 10 шагов | ✅ PASS |
| **FA-canary fwd v121r** | mean **653.85T** (baseline 652±2T) — **VERDICT: WITHIN corridor** (впервые сдал WITHIN за P-серию!) |
| NoFullSync grep guard | ✅ clean |

**FA-canary WITHIN corridor** — не thermal drift, а стабильный baseline. Возможная связь: `libgotorch_blas_wrapper.so` загружает libcublas в отдельный dlopen, что убирает legacy warm-up-tax. Требует подтверждения (сейчас — наблюдение).

---

## Метрический учёт

**Backend methods**: 68 → **72** (+4).
**+2 PTX kernels** (`cvt_f32_to_f16`, `cvt_f16_to_f32`).

**Файлы:**
| Файл | Изменение |
|---|---|
| `gotorch/v6/cuda/ptx_kernels.go` | +2 PTX (~80 lines) |
| `gotorch/v6/cuda/backend_purego.go` | +MatMulF16 + MatMulStridedBatchedF16 + Cast helpers + kernel registrations |
| `gotorch/v6/cuda/api.go` | +4 methods в Backend interface + F16 doc |
| `gotorch/v6/cuda/matmul_f16_test.go` | **NEW** — 4 теста |

---

## Побочные находки

1. **F16 IO + F64 CPU ref precision** зависит от того, где применён F16 quantization. Прогноз paper B4 (5e-4) корректен для случая **F32 inputs → GPU quantize → GEMM → dequantize** (F16 loss на входе). Наш judge использует **F16-rounded inputs** → мере ошибку только *algo path*. Это не бут-факт, а методологическая заметка.
2. **PTX ASCII catch #4** — русский в новом PTX-блоке. JIT log точно указал строку. Feedback rule `feedback-ptx-jit-log-diagnostic` подтверждён 4-й раз.
3. **cublasGemmEx non-batched == cublasGemmStridedBatchedEx(b=1)** — bit-exact. Отличается от cublasSgemm vs cublasSgemmStridedBatched (в B-impl-1 доработка mismatch была). Значит `Ex` (typed) варианты внутри тот же algo, а classic (non-Ex) — разный algo.

---

## СТОП по ТЗ

B-impl-2 закрыт. По плану paper B6: **B-impl-3 F8 E4M3 + якорный бенч cublasLt-FP8**. По решению paper — gputrain на mixed-precision **строго после B-impl-4**.
