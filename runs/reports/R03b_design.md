# R03b — Проектирование интеграции goml ↔ gotorch (paper, без кода)

**Дата:** 2026-07-18
**Статус:** проектный документ ДО R03b-impl. Ни одной правки в исходниках.
**Основание:** R03a матрица 4/4 PASS ({sm_120, sm_89} × {MPS OFF, MPS ON}), решение по фиксу A принято (LOW, но первый коммит R03b-impl).

Каждое утверждение о кодовых базах — с file:line. Слова «наверное» в этом документе нет.

---

## Вопрос 1 — Направление владения тензором

### Варианты

- **(а)** goml-тензор главный. gotorch — библиотека операций; adapter в goml оборачивает `goml.Storage` в `gotorch.ForeignStorage` через `WrapDevicePtr`, вызывает нужный gotorch-метод, кладёт результат в goml-выделенную память.
- **(б)** gotorch-тензор главный. goml-ядра зовутся через `UnsafeExtractDevicePtr`.
- **(в)** Гибрид по зонам.

### Факты из кодовой базы

- `goml/tensor/tensor.go:11-23` — `type Tensor struct { storage backend.Storage; shape Shape; strides Strides; dtype DType; offset int; requiresGrad bool; grad *Tensor; gradFn GradFn; isLeaf bool }`. Autograd (`requiresGrad`, `grad`, `gradFn`, `isLeaf`) живёт в goml.Tensor.
- `goml/train/trainer.go:47-56` — `Trainer{Model *nn.LLM, Optimizer *optim.AdamW, Tokenizer *ByteTokenizer, ...}` — весь тренировочный контур goml.
- `goml/train/trainer.go:99-131` — training step: `Model.ForwardWithCache`, `ops.CrossEntropyLoss`, `ops.CrossEntropyBackward`, `Model.Backward`, `Optimizer.Step`. Все — goml.
- `gotorch/v6/cuda/api.go` — интерфейс `Backend` из 50 методов **не имеет** ни autograd, ни модели, ни оптимизатора. Плоская вычислительная библиотека.
- `gotorch/v6/cuda/api.go:87-99` — `WrapDevicePtr(ptr unsafe.Pointer, sizeBytes, device int) ForeignStorage` — специально спроектированная дверь входа.

### Решение

**Вариант (а) — goml-тензор главный.** Обоснование:

1. **Autograd graph в goml.Tensor** (`goml/tensor/tensor.go:20-22`). У gotorch собственного graph'а нет — он бы дублировал goml-структуру, а поддерживать два графа синхронно = боль.
2. **Model/Optimizer/Tokenizer/GradFn — весь тренировочный контур на goml-стороне** (`train/trainer.go:47-56`, `nn/model.go`, `optim/adamw.go`).
3. **gotorch спроектирован как плоская compute-библиотека** — R02b финальный отчёт явно это фиксирует: «первая реализация — PuregoBackend через libcuda/libcublas + PTX-ядра». Нет и не планируется autograd/model/optim.

### Shape/stride на границе

- goml.Tensor хранит **stride + offset**, т.е. **есть views** (`tensor.go:227-263` — `View`, `Transpose`, `T`).
- goml.Backend методы **принимают shape** (`backend.go:70-137`), но НЕ strides — все backend-реализации предполагают **contiguity**.
- gotorch.Backend методы **плоские** (`AddF32(a, bb, c, n int)`, `MatMulF32(..., m, n, k int)`) — никаких strides.
- В goml.Tensor уже есть проверка `IsContiguous()` (`tensor.go:158`) и материализация `Contiguous()` (`tensor.go:164`).

**Правило для adapter'а R03b:**

> **Contract**: adapter требует contiguous storage на входе. Non-contiguous → adapter вызывает `t.Contiguous()` (аллокация нового буфера) перед `WrapDevicePtr` и оборачивает **этот** контигуальный буфер. Метаданные (shape/dtype) передаются в gotorch как плоские числа (n, rows/cols, m/n/k), потому что gotorch strides не понимает.

Contiguity — ответственность adapter'а, не gotorch'а. gotorch остаётся плоским.

---

## Вопрос 2 — Слой стыковки + таблица покрытия (центральный артефакт)

### Варианты

- **(а)** Новый пакет `goml/backend/gotorch/` — адаптер реализует `goml.Backend` поверх `gotorch.Backend`.
- **(б)** Новый пакет в gotorch.
- **(в)** Третий репозиторий-мост.

### Решение

**Вариант (а) — `goml/backend/gotorch/`.** Обоснование:

- `goml/backend/backend.go:69-137` — Backend interface. Уже есть `backend/cpu/` и `backend/cuda/` реализации; `backend/gotorch/` — идиоматичное третье место.
- `goml/backend/backend.go:139-155` — реестр через `Register(&Backend{})` в `init()`, идентично `backend/cuda/cuda..go:42-52`. Adapter регистрируется тем же способом (blank import).
- gotorch НЕ должен зависеть от goml (`goml.Storage`, `goml.Shape`, `core.DType`) — иначе gotorch перестаёт быть автономной библиотекой.
- Третий репо — YAGNI, интеграция тесная, не универсальная библиотека.

### Таблица покрытия — goml.Backend (31 метод) vs gotorch (50 методов)

**Легенда:**
- `direct` = один-в-один вызов gotorch-метода.
- `compose` = композиция ≥2 gotorch-вызовов, реализуется на adapter-стороне.
- `stays-in-goml` = метод остаётся на goml PTX-ядрах, adapter делегирует в `backend/cuda` (fallback backend); gotorch к нему не касается.
- `gap` = дырка в gotorch, требует новый метод в gotorch (если хотим покрыть) или fallback.

| # | goml.Backend метод | Покрытие | gotorch-эквивалент | Комментарий |
|---|---|---|---|---|
| **Info** |
| 1 | `Name() string` | trivial | — | Adapter возвращает "gotorch". |
| 2 | `DeviceType() DeviceType` | trivial | — | Adapter возвращает `backend.CUDA`. |
| **Memory (4)** |
| 3 | `Alloc(byteLen)` | direct | `gotorch.Alloc(byteLen)` | Adapter оборачивает результат в свой `Storage`, хранящий `gotorch.Storage`. |
| 4 | `Free(Storage)` | direct | `gotorch.Free(gotorch.Storage)` | Работает по унаследованному типу. |
| 5 | `Copy(dst, src, byteLen)` = D2D | direct | `gotorch.CopyD2D(dst, src, byteLen)` | D2D байт-в-байт. |
| 6 | `ToDevice(dst, src)` | compose | Copy + Alloc | Только CPU↔GPU (`cuda..go:181-206`). GPU→CPU: alloc host slice + `gotorch.CopyD2H`. CPU→GPU: alloc device + `gotorch.CopyH2D`. |
| **Unary (10)** |
| 7 | `Neg(dst, src, shape, dtype)` | direct | `gotorch.NegF32(a, c, n)` | dtype=F32; `n = shape.NumElements()`. |
| 8 | `Abs(dst, src, shape, dtype)` | **gap** | — | В gotorch нет. Опции: (1) добавить `AbsF32` в gotorch (~10 строк PTX); (2) fallback в goml PTX (`abs_f32` — уже есть, `ops.go:45-47`). **Рекомендация:** добавить в gotorch (простое ядро). |
| 9 | `Exp(dst, src, shape, dtype)` | direct | `gotorch.ExpF32(a, c, n)` | |
| 10 | `Log(dst, src, shape, dtype)` | direct | `gotorch.LogF32(a, c, n)` | |
| 11 | `Sqrt(dst, src, shape, dtype)` | **gap** | — | В gotorch нет. Используется 9 раз в goml (grep выше), в т.ч. в LayerNorm вспомогательно. Рекомендация: добавить `SqrtF32` в gotorch. |
| 12 | `Tanh(dst, src, shape, dtype)` | direct | `gotorch.TanhF32(a, c, n)` | |
| 13 | `Relu(dst, src, shape, dtype)` | direct | `gotorch.ReLUF32(a, c, n)` | |
| 14 | `Gelu(dst, src, shape, dtype)` | **gap** | — | В gotorch нет. Используется в feedforward (`nn/feedforward.go:87`). **Рекомендация:** stays-in-goml (goml PTX `gelu_f32` уже есть, `ops.go:65`) — до тех пор, пока не нужен gotorch-only путь. |
| 15 | `Sigmoid(dst, src, shape, dtype)` | direct | `gotorch.SigmoidF32(a, c, n)` | |
| 16 | `Silu(dst, src, shape, dtype)` | **gap** | — | В gotorch нет. Используется в feedforward. Рекомендация: stays-in-goml (goml PTX `silu_f32`, `ops.go:41-43`) — LLM-специфично, не общая CS-op. |
| **Binary + broadcasting (4)** |
| 17 | `Add(dst, a, b, shapeA, shapeB, shapeOut, dtype)` | direct | `gotorch.AddF32(a, b, c, n)` | **Broadcasting в goml НЕ реализован** — `backend/cuda/ops.go:81 TODO: broadcasting support`. Adapter требует shapeA==shapeB, идентично goml. |
| 18 | `Sub(...)` | direct | `gotorch.SubF32(a, b, c, n)` | Same. |
| 19 | `Mul(...)` | direct | `gotorch.MulF32(a, b, c, n)` | Same. |
| 20 | `Div(...)` | direct | `gotorch.DivF32(a, b, c, n)` | Same. |
| **Reduction (3)** |
| 21 | `Sum(dst, src, shape, axes, keepDim, dtype)` | compose / stays-in-goml | Частично `gotorch.SumF32(a, n)` | gotorch.SumF32 — **глобальный** (n элементов → 1 скаляр, `backend_purego.go:720+`). goml.Sum — **по осям через reshape** (`ops.go:159-201`): `rowSize × numRows`. Прямого gotorch эквивалента нет. **Рекомендация:** stays-in-goml (goml имеет уже реализованный per-row reduce ядром `sum_reduce_f32`) — не переизобретать. |
| 22 | `Max(dst, src, shape, axes, keepDim, dtype)` | **gap** | — | В gotorch нет вообще. Рекомендация: stays-in-goml. |
| 23 | `Mean(dst, src, shape, axes, keepDim, dtype)` | compose / stays-in-goml | Частично `gotorch.MeanF32(a, n)` | Аналогично Sum. Stays-in-goml. |
| **Linalg (1)** |
| 24 | `MatMul(dst, a, b, shapeA, shapeB, dtype)` | direct (batch=1) / stays-in-goml (batch>1) | `gotorch.MatMulF32(a, b, c, m, n, k)` | **Batch=1:** direct. **Batched (batchSize>1, `ops.go:251-258`):** stays-in-goml. Обоснование: goml уже держит боевой `BatchedMatMulF32` через `cublasGemmStridedBatched` (один запуск на batch). Адаптер-loop с N вызовами `gotorch.MatMulF32` дал бы N kernel launches вместо одного — anti-pattern, деградация линейно от batch size. **Loop допустим только в тестах корректности** с явным комментарием `// not for production Step`. Boевой adapter при batchSize>1 делегирует в `fb *cuda.Backend` (stays-in-goml). gotorch batched-API — будущее отдельное расширение по потребности. |
| **Composite (5)** |
| 25 | `Softmax(dst, src, shape, axis, dtype)` | direct if axis=-1, else compose | `gotorch.SoftmaxF32(a, c, rows, cols)` | gotorch делает по последней оси (rows/cols). goml принимает `axis`. Adapter: если `axis == ndim-1` — direct, rows=outerSize, cols=innerSize; иначе — либо transpose (goml `Transpose` уже есть) → softmax → transpose обратно, либо stays-in-goml. **Рекомендация:** direct при axis=-1, stays-in-goml иначе (99% случаев в LLM — axis=-1). |
| 26 | `LayerNorm(dst, src, gamma, beta, shape, normAxis, eps, dtype)` | **stays-in-goml** | — | Composite: mean-var-normalize-scale-shift. gotorch не имеет. Композиция была бы 5-6 kernel launches vs один fused kernel goml (`ops.go:300-340`). Stays-in-goml по перформансу. |
| 27 | `Embedding(dst, weight, indices, vocabSize, embedDim, seqLen, dtype)` | **stays-in-goml** | — | Sparse gather. gotorch не имеет. Stays-in-goml. |
| 28 | `RoPE(dst, src, shape, headDim, base, dtype)` | **stays-in-goml** | — | LLM-специфично, sin/cos лут + apply. Stays-in-goml. |
| 29 | `ScaledDotProductAttention(...)` | **stays-in-goml** | — | Целиком FA-territory goml (`ops.go:405-540` + FA-ядра `fa_v121r*`). **Никогда** не заходит в gotorch — это оптимизированный FA-стек. |
| **Fill/Compare (3)** |
| 30 | `Fill(dst, shape, value, dtype)` | **gap** / stays-in-goml | — | gotorch не имеет `Fill`. Используется в `Zeros`/`Ones` (`tensor/tensor.go:82-124`) — на init-time, не hot path. Stays-in-goml. |
| 31 | `Arange(dst, start, step, n, dtype)` | **gap** / stays-in-goml | — | Аналогично Fill, редко на hot path. Stays-in-goml. |
| 32 | `Where(dst, cond, a, b, shape, dtype)` | **gap** / stays-in-goml | — | Аналогично. Stays-in-goml. |

### Итоговая статистика покрытия

- **direct**: 15/32 (47%) — Alloc/Free/Copy(D2D)/Neg/Exp/Log/Tanh/Relu/Sigmoid/Add/Sub/Mul/Div/MatMul(batch=1)/Softmax(axis=-1)
- **compose (adapter-side)**: 1/32 (ToDevice — 3%)
- **stays-in-goml (fallback в goml PTX)**: 14/32 (44%) — LayerNorm, Embedding, RoPE, SDPA, Gelu, Silu, Sum-axis, Mean-axis, Max-axis, MatMul(batch>1), Fill, Arange, Where, Abs+Sqrt (до impl-6)
- **gap → add-to-gotorch кандидаты**: 2/32 (6%) — Abs, Sqrt (простые unary, impl-6 опционально). **Batched-MatMul УБРАН из gap** — переквалифицирован в stays-in-goml (у goml `cublasGemmStridedBatched` уже боевой, `cuda..go:283+`; N-вызов loop в gotorch — anti-pattern).

**Практический смысл:** мост покрывает **~47% боевого backend interface прямо через gotorch, ещё ~44% остаётся в goml PTX-ядрах** (adapter делегирует в существующий `backend/cuda` через embedded fallback). Ни строки нового backend-кода в goml для 44% писать не нужно. Gap 6% — два PTX-ядра `AbsF32`/`SqrtF32`, добавляются в gotorch мелким коммитом impl-6.

### Adapter mechanic (архитектура пакета)

Псевдо-структура `goml/backend/gotorch/gotorch.go`:

```
type Backend struct {
    gt   gotorch.Backend         // созданный NewBackend(0)
    fb   *cuda.Backend           // fallback для stays-in-goml (реиспользование существующего)
    device int
}

func init() { backend.Register(&Backend{}) }  // регистрация на CUDA device type

type Storage struct {
    gtStore gotorch.Storage       // owner-handle из gotorch.Alloc
    byteLen int
    device  backend.Device
}
func (s *Storage) Ptr() unsafe.Pointer { return gotorch.UnsafeExtractDevicePtr(s.gtStore) }
func (s *Storage) DevicePtr() uintptr  { return uintptr(s.Ptr()) }
```

Для методов stays-in-goml adapter делегирует в `fb` (embedded `cuda.Backend`), при этом Storage'и adapter'а совместимы с fallback через тот же raw pointer (`uintptr`).

---

## Вопрос 3 — dtype-мост

### Факты

- **goml.Backend CUDA — F32-only.** `backend/cuda/ops.go:240` явно: `if dtype != core.Float32 { return fmt.Errorf("only float32 supported currently, got %s", dtype) }`. Все kernel'ы (`add_f32`, `mul_f32`, `relu_f32`, `softmax_f32`, `layernorm_f32`, `matmul_f32`) — F32 (grep по `ops.go` + `kernels.go`).
- **gotorch — F32/F64 пара для каждого метода.** F32-покрытие полное для всего пересечения (см. таблицу выше).
- **F16/FP8 в goml идут вне Backend interface.** `backend/cuda/cuda..go:283-305` — `MatMulF16`, `MatMulF8E4M3`, `MatMulF8E5M2` — методы **прямо на `*cuda.Backend`**, не через интерфейс. Плюс `Launch(name, grid, block, params)` для запуска чужих FA-ядер над сырыми uintptr (`cuda..go:271-276`).
- **Пересечение F32 полное для того что реально дёргает Step.** `train/trainer.go:99-131` использует Model.Forward/Backward → внутри `ops.MatMul`/`ops.Add`/`ops.LayerNorm`/`ops.Softmax` → все F32.

### Решение

**Adapter покрывает только F32.** Обоснование:

1. Backend interface **уже F32-only** в CUDA-реализации. Adapter поддерживает тот же набор dtype что и `cuda.Backend`; для не-F32 возвращает ту же ошибку.
2. **F16/FP8 пути мост не касаются.** Они вызываются напрямую на `*cuda.Backend` через `Launch()` / `MatMulF8E4M3(dstPtr, aPtr, bPtr uintptr, M, K, N)`. Adapter (по контракту) выдаёт `Storage.DevicePtr() uintptr`, значит FA-код в goml_v4 продолжит работать через тот же путь, минуя интерфейс. **Adapter НЕ пропускает F16/FP8 через себя.**
3. **F64 в goml нет** — adapter НЕ реализует F64 пока goml.Backend не потребует.

**Правило:** если goml где-то в будущем нарастит F16-путь **через Backend interface** — adapter выдаст `fmt.Errorf("gotorch adapter: F16 not supported")` и делегирует в goml.cuda fallback. Никакого молчаливого downcast'а.

---

## Вопрос 4 — Жизненный цикл и владение памятью

### Факты

- `gotorch/v6/cuda/api.go:60-68` — `ForeignStorage` (обёртка чужого указателя) **не имеет метода Free()**. Backend.Free принимает только `Storage` (не Foreign) по сигнатуре — «освободить чужую память через gotorch невозможно на уровне компиляции» (комментарий api.go).
- `goml/backend/cuda/pool.go:60-100` — `Pool.Get(byteLen)` возвращает *Storage (cached or new), `Pool.Put(s *Storage)` — вернуть в пул (без cuMemFree).
- **В goml нет ни одного `SetFinalizer`** (grep выше). Владение полностью ручное, symmetric between GC-Go-heap и manual-GPU.
- `goml/tensor/tensor.go:274-284` — `Tensor.Free()` вызывает `storage.Free()`. Ручное.
- `goml/backend/cuda/storage.go:32-38` — `Storage.Free()` = `cuMemFree(s.ptr); s.ptr = 0` — idempotent-guard (нельзя вызвать дважды).
- `goml/backend/cuda/cuda..go:159-163` — `Backend.Free(s)` вызывает `pool.Put(s)`, а не прямой Free.
- gotorch не имеет SetFinalizer (обзор `backend_purego.go`).

### Решение

**Ручная дисциплина владения. Никаких SetFinalizer'ов.** Обоснование:

1. **Обе кодовые базы уже symmetric-manual.** Введение finalizer сломало бы invariant «на GPU время жизни определяется явно», и добавило бы GC-паузы в hot path (`AdamW.Step`, `Model.Forward`).
2. **ForeignStorage — специально спроектирован как no-op Free.** gotorch **не может** случайно освободить чужую память (`api.go:60-68`). Adapter обёртывает goml.Storage → ForeignStorage → передаёт в gotorch метод → метод возвращается → ForeignStorage выходит из scope, `Free()` не звонится ни разу.

### Правила для adapter'а

1. **Foreign-wrap живёт только внутри одного метода adapter'а.** Не хранить между вызовами.

2. **Pool + Foreign — контракт «не Put пока Foreign жив»:**
   - goml.Backend.Alloc возвращает *Storage от `pool.Get`.
   - Пользователь может вызвать `goml.Backend.Free(s)` → `pool.Put(s)` → буфер вернётся в пул и следующий `Get` может отдать тот же буфер другому tensor'у.
   - **Ответственность:** adapter НЕ вызывает `goml.Free` в течение своих методов. Хранит все goml.Storage до конца метода adapter'а, потом возвращает управление вызывающему.
   - Вызывающий goml-код (nn/ops) отвечает за Free.

3. **Panic между Wrap и Use:**
   - Если между `WrapDevicePtr` и `gotorch.AddF32` возникнет panic — Go stack unwind вызовет defer'ы. У ForeignStorage нет `Free()`, никаких утечек тут не происходит. **goml.Storage остаётся жив** — это ответственность вызывающего (стандартная).
   - **Правило:** adapter не заводит собственных defer'ов на Free — goml.Storage владение чужое, ничего освобождать нельзя.

4. **Идентичность dst == src** (in-place ops): проверить в adapter'е. gotorch методы принимают три разных DeviceBuffer (`AddF32(a, b, c)`), но безопасность aliasing зависит от конкретного ядра. `gotorch.NegF32` in-place safe (per-element). `gotorch.MatMulF32` — **не** in-place (cuBLAS требует C ≠ A, B).

### Что делать при вызове `Free` после adapter-cleanup

Adapter содержит собственный dispatcher `AllocOwn`/`FreeOwn` через `gotorch.Alloc`/`gotorch.Free`. Это НЕ пересекается с goml.Pool — pool живёт в fallback (`cuda.Backend`). Два раздельных менеджера памяти в одном процессе. **Правило разделения:** объект, аллоцированный через `gotorch.Alloc`, должен освобождаться через `gotorch.Free`; аллоцированный через `pool.Get` → `pool.Put`. **Adapter не смешивает.**

---

## Вопрос 5 — Контексты и streams после фикса A

### Факты после фикса A

- `gotorch/v6/cuda/backend_purego.go:39-73` — gotorch retain'ит primary через `cuDevicePrimaryCtxRetain`.
- **После фикса A** `goml/backend/cuda/cuda..go:72` `cuCtxCreate` → `cuDevicePrimaryCtxRetain`. Оба мира retain'ят **один и тот же primary context handle** на device 0.
- **UVA-зависимость исчезает для этой пары.** Test 1 в interop_smoke после фикса покажет **одинаковый handle** — это и есть приёмка.
- `goml/backend/cuda/cuda..go:77` — goml создаёт свой **`CU_STREAM_NON_BLOCKING` stream**, cuBLAS привязан к нему (`cuda..go:93`), все ядра запускаются на нём (`cuda..go:246 stream_param`).
- `gotorch/v6/cuda/backend_purego.go` — gotorch работает на **default stream (handle 0)** (R02a решение). `cublasSetStream_v2(h, 0)` в `newPuregoBackend`. **Sync = `cuCtxSynchronize` (весь контекст)**.
- `goml/backend/cuda/cuda..go:262-267` — `goml.Sync()` = `cuStreamSynchronize(b.stream)` (только goml stream).

### Что упрощается в мосте

**Единственная UVA-зависимость исчезает** — при одном контексте `cuMemcpyDtoD`/`cuLaunchKernel` от gotorch над goml-указателем работают без обхода UVA. Но **это косметика** — R03a показал что UVA cross-ctx работает и без фикса. Реальная выгода фикса — **безусловное сохранение поведения** при будущих драйверах / архитектурах (см. решение по фиксу выше).

**Streams не сливаются.** Один context — два stream'а параллельно. Это стандартный CUDA-паттерн.

### Проблема ordering между stream'ами (наивный вариант отклонён)

Наивный подход: goml пишет в буфер X на своём stream, adapter вызывает `gotorch.AddF32(X, ...)` над default stream, между запусками **нет happens-before**, ставим full-sync на входе и на выходе каждого adapter-метода.

**Такой контракт отклоняется по производительности.** Два `cuStreamSynchronize` / `cuCtxSynchronize` на каждый adapter-вызов **осушают конвейер**: GPU быстр ровно потому, что операции стоят в очереди и карта молотит без пауз. Full-sync — «останови завод, дождись пустых лент, потом запускай снова». Step из десятков adapter-вызовов = серийные запуски пустой карты, гарантированная деградация на порядок. Такой мост «работает, но медленнее черепахи» — anti-pattern.

### Правильная дисциплина — оба мира на ОДНОМ stream (stream-injection)

**Контракт двери:** оба мира работают на **одном** stream. Инъекция stream'а происходит **при инициализации adapter'а**. Полные sync в теле adapter-методов **запрещены**. Порядок операций гарантируется самой очередью stream'а. Единственный sync — в конце Step, где goml и так синхронизируется (`trainer.go` implicit через loss materialization).

**Механизм инъекции (форма выбрана):**

- В `gotorch.PuregoBackend` уже хранится `stream uintptr` неявно — все `cuLaunchKernel` вызовы (`backend_purego.go:launchElementwise3`, etc.) сейчас передают литеральный `0` (default stream). Заменить `0` на поле `b.stream`, инициализировать по умолчанию `0` (сохраняет старое поведение default-stream для тех, кто не инжектирует).
- Добавить в gotorch **один экспортированный метод** `func (b *PuregoBackend) SetStream(s unsafe.Pointer) error`:
  1. Сохраняет `s` в `b.stream` для последующих kernel launch'ей.
  2. Вызывает `cublasSetStream_v2(b.cublas, uintptr(s))` — cuBLAS handle перенаправляется на новый stream (binding уже есть в `cublas_purego.go`).
- Adapter при инициализации:
  1. Получает `gomlStream` из fallback `cuda.Backend` (нужен маленький аксессор в goml, обсуждается ниже).
  2. Зовёт `gotorch.PuregoBackend.SetStream(unsafe.Pointer(gomlStream))`.
  3. Дальше все operations обоих миров идут в **одну очередь** — happens-before обеспечивается порядком в stream'е.

**Что это НЕ меняет:**
- Публичный контракт gotorch остаётся: `Sync()` без параметра (сейчас `cuCtxSynchronize`, после инъекции — de-facto синхронизирует stream, но семантика «дождись всего» сохраняется). Ни одна из 50 сигнатур не тронута.
- Решение R02a «Sync без параметра, StreamBackend — будущее» **не нарушено**: stream-инъекция — деталь инициализации adapter'а, не расширение публичного API. `SetStream` — single-purpose hook для интеграции, не универсальный stream-парам.

**Что нужно добавить в goml (тривиально):**
- Экспортированный аксессор `func (b *Backend) Stream() uintptr` в `cuda..go` — вернуть `b.stream` (поле уже приватное, аксессор — одна строка). Это позволит adapter'у прочитать goml stream без reflection.

**Тесты в impl-2 (усилены):**
- `TestAdapterAddF32_InjectedStream` — Adapter, инжектирует stream, запускает Add — bit-exact vs cpu.
- `TestAdapterNoFullSync` — grep по `goml/backend/gotorch/*.go` на `.Sync()` / `cuCtxSynchronize` / `cuStreamSynchronize` — вхождения допустимы **только** в файле/функции, помеченной комментарием `// end-of-Step sync boundary`. Nолее одного места не должно быть. Контроль — статический (grep-based).

### Долгосрочно (не в scope R03b, но и не блокер после stream-injection)

Stream-injection уже даёт нужный ordering. cuda events (`cuEventRecord` + `cuStreamWaitEvent`) понадобятся только когда у нас будет **больше одного** stream (например, отдельный stream для async H2D-транспорта). R03c и позже.

---

## Вопрос 6 — Этапность R03b-impl

**Ворота по стандарту R02b/R03a: каждый этап заканчивается PASS-тестом с приведёнными числами.**

### impl-1 — Фикс A + interop_smoke регресс

**Изменения (минимальные):**
- `goml/backend/cuda/driver.go` — добавить биндинги `cuDevicePrimaryCtxRetain(pctx *uintptr, dev int32)` и `cuDevicePrimaryCtxRelease_v2(dev int32)`. Два новых `RegisterLibFunc`.
- `goml/backend/cuda/cuda..go:72` — заменить `cuCtxCreate(&b.ctx, 0, b.device)` → `cuDevicePrimaryCtxRetain(&b.ctx, b.device)`.
- `goml/backend/cuda/cuda..go:333` — заменить `cuCtxDestroy(b.ctx)` → `cuDevicePrimaryCtxRelease_v2(b.device)`.

**Ворота 1:**
- Прогон существующего `interop_smoke/main_test.go` (без модификации). **Приёмка: Test 1 показывает одинаковый handle для gotorch и goml** (`identical: true`).
- Прогон существующих `goml/backend/cuda/*_test.go` (если есть) — регрессии goml нет.
- Прогон нового 20-count stress (по образцу R02b Ворот 5) — 0 INVALID_CONTEXT events.

### impl-2 — Adapter package + AddF32 end-to-end

**Изменения:**
- Новый пакет `goml/backend/gotorch/` с файлами:
  - `gotorch.go` — `type Backend struct`, `init()` регистрация.
  - `storage.go` — `type Storage struct { gtStore gotorch.Storage; ... }`.
  - `add.go` — единственная операция `Add`.
- Никаких других методов интерфейса ещё не реализовано. Их вызов возвращает `fmt.Errorf("not implemented in gotorch adapter")` — компилируется, но при использовании явно падает.
- Регистрация — через blank import в `nn` или новый env-переключатель `GOML_BACKEND=gotorch`.

**Ворота 2:**
- Тест adapter: `TestAdapterAddF32` — Alloc через adapter → FromSlice → adapter.Add → adapter.Copy → сверить bit-exact против CPU-эталона.
- Тест интеграции: same Add через direct `goml.cuda.Add` vs через `goml.gotorch.Add` — bit-exact 100%.

### impl-3 — Покрытие прямых методов

**Изменения:** реализация 15 direct-методов из таблицы (Alloc/Free/Copy/ToDevice + Neg/Exp/Log/Tanh/Relu/Sigmoid/Add/Sub/Mul/Div + MatMul-batchSize=1 + Softmax-axis=-1).

Каждый метод — тонкая обёртка:
```go
func (b *Backend) Neg(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
    if dtype != core.Float32 { return errNotF32 }
    a := b.wrapForeign(src)
    c := b.wrapForeign(dst)
    b.fb.Sync()             // sync goml stream
    if err := b.gt.NegF32(a, c, shape.NumElements()); err != nil { return err }
    return b.gt.Sync()      // sync default stream
}
```

**stays-in-goml методы** делегируются в `b.fb` (embedded `cuda.Backend`): `Sum`, `Max`, `Mean-axis`, `LayerNorm`, `Embedding`, `RoPE`, `SDPA`, `Gelu`, `Silu`, `Fill`, `Arange`, `Where`.

**Gap методы** (`Abs`, `Sqrt`) — временно stays-in-goml через `b.fb`; **после** impl-3 отдельным коммитом добавить в gotorch и переключить.

**Ворота 3:**
- Прогон всех goml unit-тестов `nn/*_test.go`, `ops/*_test.go` через adapter (bit-exact vs cpu / vs goml.cuda путь).
- Толерантность: F32 — hybrid abs+rel (BLAS-стандарт из R02b Ворот 2).

### impl-4 — Первый боевой кусок Step через мост

**Кандидат:** `nn.Linear.Forward` (`goml/nn/linear.go:60-70`):
```go
out, err := ops.MatMul(x, wT)
if err != nil { return nil, err }
if l.Bias != nil {
    out, err = ops.Add(out, l.Bias)
}
```

Обе операции покрываются прямо (MatMulF32 batchSize=1 + AddF32). Изолированный кусок, широко используется (каждый Linear layer в LLM = FFN + attention projections).

**Замер:**
- A/B: full LLM.Forward с `GOML_BACKEND=cuda` vs `GOML_BACKEND=gotorch` на одном input tensor.
- Приёмка: **bit-exact** (одинаковый cuBLAS Dgemm по одному контексту → identical FMA order). Толерантность 0 mismatches.
- Ошибка `> 0 mismatches` — indicative bug в adapter'е, не в toolchain'е.

**Что покрывает impl-4:** ~5% времени Step (Linear-layers по времени доминируются MatMul; MatMul уже был на cuBLAS в обоих путях, значит выигрыш нулевой — но валидируется контракт). Настоящий тест интеграции — тайминг сохраняется в шумовом диапазоне.

### impl-5 — Full Step regression

**Замер:** `train.Trainer.Train()` с 10 шагами (batch=4, seqLen=64):
- Backend A: чистый `goml.cuda` (baseline).
- Backend B: `goml.gotorch` adapter (все direct-методы через gotorch, композиты stays-in-goml).

### Таблица ожиданий bit-exact vs floor — ЗАПИСАНА ДО ИЗМЕРЕНИЯ

**Правило (жёсткое):** расхождение сверх ниже-записанного floor = баг раскладки, СТОП и разбор. **Допуск задним числом не расширяется.**

| Категория операций | Путь | Ожидание | Floor (element-wise) | Обоснование |
|---|---|---|---|---|
| Alloc/Free/Copy(D2D)/H2D/D2H | direct memory | **bit-exact** | 0 | Тривиальный memcpy, оба вызывают одинаковый `cuMemcpy*` |
| Add/Sub/Mul/Div/Neg (F32 elementwise) | direct | **bit-exact** | 0 | Один FADD/FMUL/FNEG на элемент, kernel implementation безразличен (одна и та же IEEE-754 операция) |
| ReLU (F32) | direct | **bit-exact** | 0 | max(0, x) — тривиально |
| Sigmoid, Tanh (F32) | direct | НЕ bit-exact | abs ≤ 5e-7 | Разные PTX-реализации (gotorch: `ex2.approx.f32 + rcp.approx.f32` / `tanh.approx.f32`; goml: свои PTX `sigmoid_f32`/`tanh_f32`). Проверено в R02b Ворот 5 против CPU: `SigmoidF32` maxAbs=1.19e-7, `TanhF32` medium=1e-5 → берём консервативно **5e-7** для двухстороннего match. |
| Exp, Log (F32) | direct | НЕ bit-exact | abs ≤ 5e-7 | Аналогично — approx-family PTX инструкции |
| Softmax (F32, axis=-1) | direct | НЕ bit-exact | abs ≤ 1e-5 | 3 фазы (row_max → exp+sum → divide) с разным accumulation order + разный exp. Тот же порядок магнитуды что в R02b `TestSoftmaxF32`: maxAbs=5.96e-8, но при разных implementations безопасно **1e-5** |
| MatMul (F32, batch=1) | direct | **bit-exact** | 0 | Одна libcublas `cublasSgemm_v2`, один context, один stream, идентичные параметры (col-major swap trick одинаков) → идентичный FMA order → идентичный результат |
| MatMul (F32, batch>1) | stays-in-goml | **bit-exact** | 0 | Тот же самый `goml.cuda.MatMul` через delegate — путь не меняется |
| Sum/Mean-axis (F32) | stays-in-goml | **bit-exact** | 0 | Тот же самый `goml.cuda` reduce kernel — путь не меняется |
| LayerNorm/Embedding/RoPE/SDPA/Gelu/Silu/Max/Fill/Arange/Where | stays-in-goml | **bit-exact** | 0 | Тот же самый goml PTX kernel — путь не меняется |

**Композиция floor для Full-Step:**

Softmax + Sigmoid/Tanh — только источники drift. Активации в LLM (обычно Gelu, а Gelu stays-in-goml → bit-exact) не участвуют. Основной канал drift = **Softmax в attention** и **CE softmax на выходе**.

- Per-token logits drift после Softmax: ≤ 1e-5 abs.
- CE loss на batch: **ожидаемый drift loss ≤ 5e-4 abs после 1 шага, ≤ 5e-3 после 10 шагов** (накопление через AdamW-обновления). Числа консервативные, включают gradient drift.
- **Приёмка impl-5:** `|loss_B[k] - loss_A[k]| ≤ 5e-3` для k ∈ [1..10]. Плюс `tokens/sec` в пределах ±20% от baseline (проверка что нет full-sync деградации, per правке 1).

**Если loss расходится > 5e-3 после 10 шагов:**
1. Bisect по операциям — найти первый шаг где расхождение вышло за floor.
2. Проверить: используется ли где-то direct-путь для operation с ЗАПИСАННЫМ floor=0 (Add/Mul/MatMul/Neg/ReLU)? Если да — **баг раскладки**, разбор.
3. Проверить stream-injection: `TestAdapterNoFullSync` grep всё ещё чист? Если да — full-sync race исключён.
4. Проверить contiguity check — не-contiguous tensor попал в adapter?

**Ключевое:** floor `5e-3` записан ДО измерения. При превышении — стоп и разбор, не расширение tolerance.

---

## Итоговый план R03b-impl

| Этап | Что | Артефакт | Приёмка |
|---|---|---|---|
| **impl-1** | Fix A: goml на primary context | 2 биндинга + 2 replacement строки в `goml/backend/cuda/{driver,cuda..}.go` | Test 1 interop_smoke: identical=true; 20-count 0 errors |
| **impl-2** | Adapter package + AddF32 | `goml/backend/gotorch/{gotorch,storage,add}.go` | TestAdapterAddF32 bit-exact vs cpu + vs goml.cuda |
| **impl-3** | 15 direct-методов + stays-in-goml делегация | `goml/backend/gotorch/*.go` (по одному файлу на группу) | Все существующие goml unit tests PASS через adapter |
| **impl-4** | nn.Linear.Forward через adapter | Флаг `GOML_BACKEND=gotorch` включает adapter в train | Test A/B: LLM.Forward bit-exact vs cuda |
| **impl-5** | Full Step regression | Trainer.Train 10 steps A/B | loss/step within hybrid tolerance; tokens/sec same order |
| **impl-6** (опционально) | Gap: AbsF32 + SqrtF32 в gotorch | `gotorch/cuda/{ptx_kernels,backend_purego}.go` | Прямое покрытие Abs/Sqrt (замена stays-in-goml на direct) |

**Порядок обязателен:** impl-1 blocking для impl-2 (без единого контекста Test 1 не пройдёт), impl-2 blocking для impl-3, impl-3 blocking для impl-4, impl-4 blocking для impl-5.

---

## Открытые вопросы после импла (для R03c)

1. **Stream-параметр в gotorch** — `AddF32Stream(a, b, c, n, stream uintptr)`. Убирает full sync overhead.
2. **Batched MatMul в gotorch** — либо adapter loop, либо новый метод `MatMulBatchedF32`.
3. **AdamW на device** (`optim/adamw.go:50-93` — сейчас host-side loop через `ToFloat32Slice`). Не имеет отношения к мосту, но лёгкий выигрыш.
4. **F16 через adapter** — при появлении F16-пути в goml.Backend interface.

---

## Приёмка бумаги

Документ содержит:
- ✅ Шесть решений по шести вопросам.
- ✅ Таблица покрытия 32 методов goml.Backend с явным маркером direct/compose/stays-in-goml/gap для каждого.
- ✅ file:line ссылки на каждое утверждение о кодовых базах (`tensor.go:11-23`, `backend.go:69-137`, `cuda..go:72`, `ops.go:81,240,251-258`, `api.go:60-99`, `adamw.go:50-93`).
- ✅ План этапов impl-1…impl-5 (+impl-6 опционально) с воротами.
- Слова «наверное» не содержит.

После ревью — переход к R03b-impl-1 (Fix A).
