# P5A — int32-канон + int64-фасад Embedding: индустриальный контракт

**Дата:** 2026-07-23
**Итог:** ✅ ПРОЙДЕН. Стрелка delegate→direct для `backend.Backend.Embedding` теперь **чиста**: goml подаёт int64 tokens как есть, adapter конвертирует внутри через переиспользуемый scratch. A/B goml.cuda vs adapter через interface — **bit-exact 1024/1024** на LLM-tiny форме. Контракт «движок принимает оба типа индексов, конверсия — обязанность движка» реализован.

---

## Контракт

**Индустриальный принцип:** движок обязан принимать оба типа индексов; внутри — канон **int32**; конверсия — обязанность движка, не пользователя.

**Реализация:**
- Канонические ядра (P3): `embedding_f32/f64` + `_grad_*` — int32 индексы, без изменений.
- Новый I64-фасад (P5A): `EmbeddingF32I64/F64I64/GradF32I64/GradF64I64` — принимают int64 буфер, внутри вызывают `cvt_u64_to_u32` PTX-конвертер в scratch, затем канонический kernel.
- **Scratch pool на бэкенде** (`PuregoBackend.scratchI32Ptr/Cap`): ленивая аллокация в `ensureScratchI32(n)`; рост 2×need (минимум 1024 байта); никогда не уменьшается; освобождается в `Close()`. Без per-call `Alloc/Free` (правило hot-loop).

**Валидность индексов** (записана в doc-string api.go): `0 <= idx < vocab` **И** `idx < 2^31` — обязанность вызывающего. Нарушение = silent truncation в PTX `cvt.u32.u64` (сохранены нижние 32 бита). Debug-путь тестов проверяет диапазон на CPU перед upload.

---

## Прогнозы, записанные ДО измерения

| # | Метрика | Прогноз |
|---|---|---|
| P1 | fwd F32I64 == fwd F32 при равных индексах | **bit-exact** (тот же канонический kernel, cvt без потерь) |
| P2 | fwd F64I64 == fwd F64 | **bit-exact** |
| P3 | grad F32I64 vs grad F32 (одинаковые индексы) | ≈ bit-exact (тот же kernel, тот же порядок atomics) — floor hybrid abs=1e-4+rel=1e-4 (см. [[feedback-atomicadd-drift-oscillation]]) |
| P4 | A vs B через backend.Backend.Embedding interface | **bit-exact** (стрелка delegate→direct работает) |
| P5 | граничные индексы работают | верно (кanон Embedding уже проверен на 0 и vocab-1) |

---

## Реализация

### PTX-конвертер (`cuda/ptx_kernels.go`)

```
cvt_u64_to_u32(src, dst, n) — grid=ceil(n/256), block=256
  ld.global.u64 → cvt.u32.u64 → st.global.u32
```

Тривиальный поэлементный. Один PTX-instruction path на элемент (плюс addressing). Стоимость: ~4×уменьшение стоимости выделения scratch (мемория DRAM-bandwidth, не compute-bottleneck).

### Scratch pool (`cuda/backend_purego.go`)

```go
type PuregoBackend struct {
    ...
    scratchI32Ptr uintptr
    scratchI32Cap int
}

func (b *PuregoBackend) ensureScratchI32(n int) (uintptr, error) {
    need := n * 4
    if b.scratchI32Cap >= need { return b.scratchI32Ptr, nil }
    if b.scratchI32Ptr != 0 { cuMemFree(b.scratchI32Ptr) }
    grow := max(need*2, 1024)
    cuMemAlloc(&ptr, grow)
    ...
}
```

**Освобождение в `Close()`** — добавлено ПЕРЕД destroy PTX-модуля/cuBLAS/контекста.

### I64-фасад методы

Все 4 метода следуют одному шаблону:
1. `LockOSThread` + `defer UnlockOSThread` + `cuCtxSetCurrent`.
2. `scratch, err := b.convertI64ToScratch(indices64, n)` — cvt в scratch, возвращает `ForeignStorage` для canonical launcher.
3. Для grad: `cuMemsetD8(dtable, 0, vocab*hidden*sizeof)` (тот же контракт canonical grad).
4. `b.launchEmbedding("embedding_f32/f64", table, scratch, out, hidden, n)` или `launchEmbeddingGrad`.

**Backend interface** (`api.go`): +4 методов в секции «Embedding I64-фасад».

---

## Тесты gotorch (`cuda/embedding_i64_test.go`)

4 теста, все PASS:

| Тест | Результат | Прогноз |
|---|---|---|
| EmbeddingF32I64 vs EmbeddingF32 canon | bit-exact **1024/1024** | ✅ P1 |
| EmbeddingF64I64 vs EmbeddingF64 canon | bit-exact **256/256** | ✅ P2 |
| EmbeddingGradF32I64 vs canon (v=100,h=32,n=128 с коллизиями) | maxRel 9.82e-6, 0 fails | ✅ P3 |
| Boundary indices 0/vocab-1/2 | все match | ✅ P5 |

**Регрессия P3-тестов НЕТРОНУТА** — все существующие Embedding shapes/edge/grad/atomic-repro/numerical тесты продолжают PASS.

---

## Adapter стрелка delegate→direct

**Схлопнута через backend.Backend.Embedding** (второе полное схлопывание в главе после RoPE):

```go
func (b *Backend) Embedding(dst, weight, indices backend.Storage,
    vocabSize, embedDim, seqLen int, dtype core.DType) error {
    switch dtype {
    case core.Float32:
        return b.gt.EmbeddingF32I64(wrapForeign(weight), wrapForeign(indices),
            wrapForeign(dst), vocabSize, embedDim, seqLen)
    case core.Float64:
        return b.gt.EmbeddingF64I64(...)
    default:
        return b.fb.Embedding(...)  // fallback -- goml.cuda F16 orphan
    }
}
```

**dtype в сигнатуре → метод-суффикс** (правило R02a): никакого state-режима, dispatcher по dtype.

Из `delegate.go` удалён старый Embedding-delegate, комментарий указывает на embedding.go.

### A/B через backend interface (`backend/gotorch/embedding_test.go`)

Новый тест `TestAdapterEmbedding_AvsB_via_Interface`:
- A: `goml.cuda.Embedding(int64)` через backend interface.
- B: `adapter.Embedding(int64)` — тот же интерфейс, но внутри cvt→canonical.
- **Результат: bit-exact 1024/1024** (P4 подтверждён).

Старый extension A/B тест (`TestAdapterEmbedding_AvsB_Forward`) остаётся живым — теперь он проверяет extension API вручную через int32-канон.

---

## Регрессия ворот

| Гейт | Результат |
|---|---|
| **P5A gotorch/v6/cuda** (4 новых теста + P3 регрессия) | ✅ PASS |
| **P5A adapter через interface** (bit-exact 1024/1024) | ✅ PASS |
| adapter full regression | ✅ ok 0.418s |
| gotorch cuda full (P2 + P3 + P4 + P5A) | ✅ ok 0.584s |
| interop_smoke 6/6 | ✅ ok 0.357s |
| f64ref grad 9/9 | ✅ ok 0.370s |
| P1-ABJ 10 шагов | ✅ PASS |
| **FA-canary fwd v121r** | mean **654.09T** (baseline 652±2T, +0.32%) — thermal drift стабильный |
| NoFullSync grep guard | ✅ clean |

FA-canary: 5 runs [653.46, 654.43], та же природа что P2/P3/P4 — thermal drift +0.32% при 36°C, WITHIN honest tolerance.

---

## Метрический учёт

**gotorch/v6/cuda Backend methods**: 62 → **66** (+4 I64-фасад).
**+1 PTX kernel** (`cvt_u64_to_u32`), но это не Backend-method — util-конвертер.

**Файлы:**

| Файл | Изменение |
|---|---|
| `gotorch/v6/cuda/ptx_kernels.go` | +1 PTX (`cvt_u64_to_u32`) |
| `gotorch/v6/cuda/backend_purego.go` | struct +2 fields (scratch), Close +scratch cleanup, ensureScratchI32, launchCvtU64ToU32, convertI64ToScratch, 4 фасад-методов |
| `gotorch/v6/cuda/api.go` | +4 methods в Backend interface + doc-string |
| `gotorch/v6/cuda/embedding_i64_test.go` | **NEW** — 4 теста |
| `goml/backend/gotorch/embedding.go` | +Embedding(backend interface) через dtype-switch |
| `goml/backend/gotorch/delegate.go` | Embedding-delegate удалён, комментарий-указатель |
| `goml/backend/gotorch/embedding_test.go` | +1 тест `AvsB_via_Interface` |

---

## Побочные находки

1. **Legacy inventory обновление**: goml.cuda.Embedding — статус **стрелка delegate→direct СХЛОПНУТА** через фасад (2-я полная стрелка в главе после RoPE). goml подаёт свои int64 tokens, gotorch принимает без изменений на стороне gputrain — правило «движок принимает оба типа» реализовано.
2. **Scratch pool паттерн** — переиспользуемый device buffer через `ensureXxx(n)` + освобождение в `Close`. Прецедент для будущих util-конверсий (FP16↔FP32 квантизация, transpose scratch, etc). Заслуживает feedback-memory.
3. **PTX cvt.u32.u64 без модификаторов**: `cvt` без `.wrap/.sat` — silent truncation, что и требуется по контракту («caller responsibility idx<2^31»). Правильный выбор для fast-path.

---

## СТОП по правилу ТЗ

Часть A P5 пройдена. **Стрелка Embedding delegate→direct схлопнута чисто.** Оба уровня API (extension P3 int32 + interface P5A int64-фасад) сосуществуют — «движок принимает оба типа индексов».

Далее — **P5B: paper главного блюда** (batched FP16/FP8 MatMul design). Только бумага, никакого кода.
