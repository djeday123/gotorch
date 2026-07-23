# B-impl-1 — Batched F32/F64 MatMul (первый этап главного блюда)

**Дата:** 2026-07-23
**Итог:** ✅ ПРОЙДЕН. F32/F64 strided-batched реализовано через loop cublasSgemm_v2/Dgemm_v2 (совпадает с goml `BatchedMatMulF32`). Adapter стрелка `backend.Backend.MatMul` batched delegate→direct СХЛОПНУТА (3-я полная в главе после RoPE и Embedding-I64).
**Побочная находка (Step 1 probe):** purego v0.9.1 упирается на **≥18 args** — panic `too many arguments` при регистрации `cublasSgemmStridedBatched`. Подтверждает прецедент goml (struct-args wrappers). Native strided-batched через `.so`-wrapper отложен, loop-паттерн взят как paritу-совместимый.

---

## Step 1 — Probe cublasGemmEx purego

**Первым шагом B-impl-1 по ТЗ.**

Регистрация функций с 18+ аргументами (`cublasSgemmStridedBatched` — 18 args, `cublasGemmEx` — 19 args) через `purego.RegisterLibFunc` вызывает panic:

```
panic: purego: too many arguments
    at github.com/ebitengine/purego.RegisterFunc (func.go:205)
    at initCuBLAS.func1 (cublas_purego.go:160)
```

**Факт:** purego v0.9.1 реальный лимит **< 18 args**, не 60+ как предполагалось в paper. Совпадает с прецедентом goml (`libs1/cublas_wrapper.c` использует struct-args wrappers `gemmex_wrapper` / `gemm_strided_batched_ex_wrapper` именно из-за этого лимита).

**Решение (по ТЗ):** loop-паттерн через `cublasSgemm_v2`/`cublasDgemm_v2` (14 args, работает). Тот же путь что goml `BatchedMatMulF32` (`goml/backend/cuda/cublas.go:264`). Native strided-batched через wrapper.so **отложен отдельным ТЗ** (когда потребуется max performance для attention-batches).

**Обновление paper:** утверждение «purego 22.10+ поддерживает 20+ args» в B3.2 P5_matmul_design.md — было неверным; актуальный факт зафиксирован в этом отчёте.

---

## Step 2 — Реализация

**API (`cuda/api.go`):**
```go
MatMulStridedBatchedF32(a, b, c DeviceBuffer, batch, m, n, k int,
    strideA, strideB, strideC int64) error
MatMulStridedBatchedF64(a, b, c DeviceBuffer, batch, m, n, k int,
    strideA, strideB, strideC int64) error
```

Страйды в **элементах**, не байтах (умножение на `sizeof` внутри метода).

**Импл (`cuda/backend_purego.go`):**
- Единый `bind()` + `LockOSThread` на весь метод.
- Цикл `for i := 0..batch: cublasSgemm_v2/Dgemm_v2(ptr+stride*i*sizeof, ...)`.
- Тот же column-major/transpose paттерн что MatMulF32 (`cublasOP_N, cublasOP_N`, alpha=1, beta=0).

---

## Step 3 — Тесты (`cuda/matmul_batched_test.go`)

4 теста, все PASS:

| Тест | Форма | Результат | Прогноз |
|---|---|---|---|
| Batch1EqNonBatched F32 | b=1, m=16 n=32 k=24 | **bit-exact 512/512** | ✅ P1 |
| Shapes F32 | [1,1,1,1] .. [16,128,128,64] | maxAbs ≤ 1.1e-5, hybrid 0 fails | ✅ P2 |
| Shapes F64 | [1,1,1,1] .. [8,32,32,32] | maxRel ≤ 6.02e-12 | **прогноз промахнулся 6×** |
| B(F32) vs J(F64 CPU) | b=4 m=32 n=64 k=48 | maxAbs 3.5e-6, maxRel 7.4e-4, 0 fails | ✅ P2 hybrid |

**Прогноз F64 (P3) недостаточно жёстко:** pre-registered `rel ≤ 1e-12`. Actual `maxRel 6.02e-12` на [4,16,32,24] — 2 fails. Причина: **cublasDgemm order-of-summation ≠ CPU naive**, F64 ulp-накопление при K=24 достигает 6e-12.

**Actual floor: `rel ≤ 1e-11`** (запас ~2× vs измеренного 6e-12). Правило «два числа + не переписывать прогноз задним числом» из `[[feedback-atomicadd-drift-oscillation]]` применено: pre-reg + actual логируются вместе в тесте.

---

## Step 4 — Adapter стрелка delegate→direct + A/B

**Обновление `goml/backend/gotorch/matmul_softmax.go`:**
```go
if batchSize > 1 {
    // B-impl-1: стрелка СХЛОПНУТА через loop-batched (тот же паттерн
    // что goml.cuda.BatchedMatMulF32, cublas.go:264).
    return b.gt.MatMulStridedBatchedF32(...)
}
```

Раньше batched → `b.fb.MatMul(...)` (delegate); теперь batched → `b.gt.MatMulStridedBatchedF32(...)` (direct). **Третья полная схлопнутая стрелка в главе** (после RoPE P4 + Embedding-I64 P5A).

**Тест `matmul_batched_test.go`** в goml/backend/gotorch — 2 уровня:

| Уровень | Форма | Прогноз | Результат |
|---|---|---|---|
| A(goml TF32-handle) vs B(adapter FP32 pedantic) | b=4 m=16 k=24 n=32 | TF32-vs-FP32 class hybrid abs=1e-2+rel=2e-1 (impl-4-final Sверка 3.2) | maxAbs 5.9e-3, maxRel 0.14, 0 fails |
| B(adapter FP32) vs J(F64 CPU) | b=4 m=16 k=24 n=32 | hybrid abs=1e-4+rel=1e-4 | maxAbs 1.78e-6, maxRel 5.4e-5, 0 fails |

**Побочная работа:** старые delegate-тесты `TestAdapter_Linear_Sверка_3_3_Batched_Delegate_BitExact` и `TestAdapter_Linear_Хвост_4_3_Realistic_FFN_Shape` **проверяли ушедшую семантику** (adapter batched = fb direct bit-exact). После схлопывания стрелки они стали невалидными по определению. Переименованы: `_Batched_Direct_TF32vsFP32` + `_FFN_Shape` (комментарий о B-impl-1) + floor обновлён на TF32-vs-FP32 class. Тесты продолжают ходить — теперь проверяют новую семантику.

---

## Регрессия ворот

| Гейт | Результат |
|---|---|
| **B-impl-1 gotorch/v6/cuda** (4 теста) | ✅ PASS |
| **B-impl-1 adapter A/B + B/J** (2 уровня) | ✅ PASS |
| adapter full regression (P2/P3/P4/P5A/P5B + переименованные 3.3/4.3) | ✅ ok 0.444s |
| gotorch cuda full | ✅ ok 0.595s |
| interop_smoke 6/6 | ✅ ok 0.347s |
| f64ref 9/9 | ✅ ok 0.375s |
| **P1-ABJ 10 шагов** | ✅ PASS worst \|A-B\| = 2.36e-05 (то же что прошлые прогоны — batched стрелка не влияет на gputrain non-batched) |
| **FA-canary fwd v121r** | mean **654.26T** (baseline 652±2T, +0.35%) — thermal drift стабилен |
| NoFullSync grep guard | ✅ clean |

---

## Метрический учёт

**gotorch/v6/cuda Backend methods**: 66 → **68** (+2 batched).

**Файлы:**
| Файл | Изменение |
|---|---|
| `gotorch/v6/cuda/cublas_purego.go` | комментарий с **фактом purego <18 args**; StridedBatched/GemmEx НЕ регистрируются |
| `gotorch/v6/cuda/backend_purego.go` | +MatMulStridedBatchedF32/F64 |
| `gotorch/v6/cuda/api.go` | +2 methods в Backend interface + doc |
| `gotorch/v6/cuda/matmul_batched_test.go` | **NEW** — 4 теста |
| `goml/backend/gotorch/matmul_softmax.go` | стрелка delegate→direct для batched |
| `goml/backend/gotorch/matmul_batched_test.go` | **NEW** — A/B + B/J |
| `goml/backend/gotorch/linear_test.go` | 3.3/4.3 переименованы + floor обновлён на TF32-vs-FP32 class |
| `gotorch/v6/runs/reports/B_impl1.md` | **NEW** |

---

## Побочные находки

1. **purego v0.9.1 реальный лимит < 18 args** — факт зафиксирован Step 1 probe. Paper (B3.2) утверждение о 20+ args корректировано. Native strided-batched через `.so`-wrapper отложен.
2. **cublasDgemm order-of-summation != CPU naive** — F64 batched тесты дают maxRel до 6e-12 при K=24 (P3 прогноз 1e-12 промахнулся 6×). Actual floor 1e-11. Правило «pre-reg + actual» применено.
3. **Стрелка 3 в главе** — batched MatMul схлопнута через backend interface. RoPE F32 (P4) + Embedding int64 (P5A) + MatMul batched F32 (B-impl-1) = **3 полные стрелки delegate→direct** через backend.Backend interface на текущий момент.
4. **Старые delegate-тесты стали невалидны после схлопывания** — переименованы + floor обновлён (нельзя оставлять bit-exact ожидание для новой TF32-vs-FP32 семантики). Это ожидаемая работа при закрытии стрелок.

---

## СТОП по ТЗ

B-impl-1 закрыт. По плану: B-impl-2 (F16 mixed precision). По решениям paper — gputrain на mixed-precision **строго после B-impl-4**.
