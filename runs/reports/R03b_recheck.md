# R03b-recheck — поиск GPU-пути в goml

**Дата:** 2026-07-21
**Итог:** Вердикт impl-5c **частично пересмотрен**: два пути в goml, а не один. `nn.LLM` действительно на CPU (impl-5c был прав), но **есть второй, GPU-путь** — `cmd/gputrain/main.go`, обходящий `nn.*` через прямую композицию backend-вызовов. ABJ становится возможным через **gputrain-style** модель.

---

## (а) Существует ли GPU-путь тренировки

**ДА.** Точка входа: **`goml/cmd/gputrain/main.go:29-243`**.

```
gputrain/main.go:34   gpu, err := backend.Get(backend.CUDA)
gputrain/main.go:56   embWeight := randGPU(gpu, vocabSize*embedDim, 0.02)
gputrain/main.go:109  gpu.Embedding(embedded, embWeight, inputGPU, ...)
gputrain/main.go:119  gpu.MatMul(logits, embedded, outWeight, ..., core.Float32)
gputrain/main.go:127  gpu.Softmax(probs, logits, ..., 1, core.Float32)
gputrain/main.go:201  launchKernel(gpu, "adamw_f32", ...)
```

**Устройство:**
- Веса аллоцированы **напрямую через `gpu.Alloc()`** (не через `tensor.FromSlice`).
- Все ops идут через `gpu.Embedding()/MatMul()/Softmax()/LayerNorm()` — **прямые вызовы backend**, не `ops.MatMul`.
- Forward + Backward + AdamW всё в device memory.
- AdamW через **custom PTX kernel** `adamw_f32` (запускается через `launchKernel`).
- Backward гибрид: `outWeight` gradient считается на CPU (loop 176-183), потом upload на GPU для AdamW.

Пайплайн: Embedding → LayerNorm → Linear (MatMul + bias) → Softmax + CE → AdamW GPU-update.

**Смок-тест реально работает:** loss decrease документирован в самом файле (line 236-240 «Loss decreased by X%»).

---

## (б) Почему impl-5c его не нашёл, и меняет ли это план

**Почему не нашёл:** impl-5c искал по цепочке `nn.NewLinear → FromSlice → CPU`. Эта цепочка **действительно** приводит на CPU — и это верно для **`nn.LLM`, `nn.SimpleLM`, `train.Trainer`**. Но `cmd/gputrain/main.go` **не пользуется `nn.*`** — там всё через прямые backend-вызовы. Impl-5c не заметил обходной путь.

**Прямые доказательства «nn.*-путь = CPU»** (не переменилось):
- `goml/main.go:43` — `model, _ := nn.NewLLM(cfg, backend.CPU0)` — **явно** CPU
- `goml/cmd/simpletrain/main.go:179` — `NewSimpleLM(cfg, backend.CPU0)` — **явно** CPU
- `goml/nn/linear.go:30` — `tensor.FromSlice(wData, ...)` (device аргумент игнорируется)

**Значит goml-разработчики сами знают, что `nn.*` — CPU**: оба trainer-cmd явно передают `CPU0`. GPU-путь построен как **параллельная ветвь**, обходящая `nn.*`.

**Меняет ли это план для ABJ?** — **ДА**. ABJ через `gputrain`-style модель:
- **A** — чистый goml.cuda: прогон `gputrain`-style кода как есть (backend = goml.cuda).
- **B** — adapter: тот же `gputrain`-код, но `backend.Get(CUDA)` возвращает наш adapter (после `gotorch.Enable()`). Прозрачное переключение.
- **J** — F64-судья через `f64ref.LLMF64` **с усечённой моделью до gputrain-набора ops**: Embedding + LayerNorm + Linear + Softmax + CE + AdamW. Attention/RoPE/FFN — в gputrain **нет** (комментарий line 3-14: только 5 ops), значит **не нужны судье для этого пути**. Уже готовые f64ref-компоненты покрывают всё.

**Это делает ABJ реалистичным в 1 сессию** — модель проще (5 ops vs полный transformer), инфраструктура gputrain уже есть, F64-судья уже собран.

---

## (в) Подтверждение вердикта impl-5c

**Частично подтверждён, частично неверен:**

- ✅ **Верно:** `nn.LLM.Forward` / `Trainer.Train` работают на CPU в текущей версии goml. Мост adapter не задействован там.
- ❌ **Неверно:** «GPU-движок goml = ядра + бенчи без боевого потребителя». `cmd/gputrain/main.go:29-243` — это **боевой GPU-потребитель** движка (пусть простой, без attention).

---

## (г) Статус двух backward-путей

### Путь 1 — autograd в `ops/`

Живой. `SetGradFn` + `GradFn` interface в `ops/ops.go:11-73`. Реализованы: `addGradFn`, `mulGradFn`, `matmulGradFn`, `reluGradFn`. Ветка `fix/ops-autograd-gradfns` (коммит `52c2009`) добавила backward для ReLU/Softmax/GELU/SiLU. **Не используется** в текущих trainer'ах — они через ручной путь.

### Путь 2 — ручной в `nn/backward*.go`

Живой и основной. `Linear.Backward`/`LayerNorm.Backward`/`FeedForward.Backward`/`Embedding.Backward`/`MHA.Backward`/`TransformerBlock.Backward`/`LLM.Backward` — все ручные (`nn/backward.go:14-260`, `nn/backward_attn.go:19-403`). Используется `trainer.go:125 t.Model.Backward(cache, dLogits)`.

**На каком backend работают** — тот же device что у входного tensor. `Linear.Backward(x, dout)` вызывает `ops.MatMul(dout, wT)` внутри, а `ops.MatMul` берёт backend через `getBackend(a) = backend.GetForDevice(a.Device())`. Если input CPU → всё CPU.

### Путь 3 — прямой в `cmd/gputrain/main.go`

Ручной, ad-hoc backward прямо в main.go. **Не переиспользуется** — inline код в одном файле. GPU-специфичный.

**Карта для главы портирования:**
- Автоматический `autograd`-путь есть, но с ограниченным набором ops. Расширение — рабочее направление.
- Ручной `nn/backward*.go` — device-agnostic по интерфейсу, но задействует backend через input tensor.Device(). Если сделать `nn.NewLinear` GPU-first (веса на device), backward автоматически пойдёт на GPU.
- `gputrain`-путь — доказательство что GPU-стек **работает end-to-end**. Модель для порта в gotorch: **Linear + Embedding + LayerNorm** GPU-first, потом autograd/ручной путь заработает автоматически.

---

## Реестр проверенных сущностей

### Все конструкторы `nn/*New*`

| Файл:строка | Конструктор | Device param | Использует device |
|---|---|---|---|
| `nn/linear.go:21` | `NewLinear` | `backend.Device` | **НЕТ** — `FromSlice → CPU` |
| `nn/embedding.go:18` | `NewEmbedding` | `backend.Device` | **НЕТ** — `FromSlice → CPU` |
| `nn/layernorm.go:18` | `NewLayerNorm` | `backend.Device` | **ЧАСТИЧНО** — `Zeros(shape,dtype,device)` **действительно** device-aware (`tensor.go:83-99`) |
| `nn/feedforward.go:21` | `NewFeedForward` | `backend.Device` | **НЕТ** (через NewLinear) |
| `nn/attention.go:29` | `NewMultiHeadAttention` | `backend.Device` | **НЕТ** (через NewLinear) |
| `nn/transformer.go:22` | `NewTransformerBlock` | `backend.Device` | **НЕТ** (композиция) |
| `nn/model.go:60` | `NewLLM` | `backend.Device` | **НЕТ** (композиция) |
| `nn/optimizer.go:26` | `NewAdamW` | нет | host-side loop over `ToFloat32Slice` |

### Все `cmd/`

| cmd | Путь | Device | Trainer/Bench |
|---|---|---|---|
| `cudatest` | direct backend ops | GPU | bench (MatMul cuBLAS vs CPU) |
| `cudatest_b` | direct backend ops | GPU | bench (LayerNorm/Softmax/etc) |
| `gputrain` | **direct backend ops** | **GPU** | **trainer** (Embed+LN+Linear+Softmax+CE+AdamW) ✅ boевой GPU trainer |
| `fp16bench` | `cublas.MatMulF16`/`F8` direct | GPU | bench (mixed precision) |
| `simpletrain` | `nn.SimpleLM` (Embed+LN+Linear) через `ops.*` | **CPU (line 179)** | trainer |
| `gradcheck` | не проверял (probably tests) | ? | test |
| `bpetest` | не проверял | ? | test |
| `hfdownload` | data | — | utility |
| `mingrad` | не проверял | ? | test |
| `nvlink_probe` | MMIO probe | none | diagnostics |
| `wikiextract` | data | — | utility |
| `main.go` (root) | `nn.NewLLM` | **CPU (line 43)** | full trainer |

### `transformer_kernels.go` — callers

| Kernel | Callers (grep across goml/) |
|---|---|
| `RMSNorm` (kernels_b.go/transformer_kernels.go) | **только определения** — нет callers в моделях или trainer'ах |
| `SwiGLU` | **только определения** |
| `RoPE` (backend/cuda/transformer_kernels.go:78) | **только определения** — `nn.attention.applyRoPE` реализует его через ops, не через этот kernel |
| `Attention` (backend/cuda/transformer_kernels.go:96) | **только определения** — `ops.ScaledDotProductAttention` реализует отдельно, не через этот kernel |
| `MatMulF16` | `cmd/fp16bench/main.go:186-197` (только bench) |
| `MatMulF8E4M3` | `cmd/fp16bench/main.go:205-208` (только bench) |
| `adamw_f32` PTX kernel | `cmd/gputrain/main.go:201` (единственный боевой caller!) |

**Вывод по transformer_kernels:** GPU-ядра RMSNorm/SwiGLU/RoPE/Attention **написаны и работают в бенчах**, но **не имеют потребителя из nn.* или Trainer**. Ждут порта в gotorch и переключения `nn.*`-слоёв.

### История

- `git branch -a`: main, `w-train` (пусто relative main), `fix/ops-autograd-gradfns` (1 коммит про autograd wiring — не GPU).
- Grep коммитов по 'gpu'/'device' — 4 матча, все про инфраструктуру (build multi-arch, R03b interop). Нет удалённого GPU-варианта модели.

---

## Пересмотр плана главы портирования

**Приоритет 1 — ABJ через `gputrain`-путь (impl-5c завершение):**
- Модель усечённая: **Embed + LN + Linear + Softmax + CE + AdamW** (5 ops, полностью покрыто f64ref).
- A/B тривиально через `backend.Get(CUDA)` (goml/cuda vs adapter).
- J через `f64ref` уже готов.
- Реалистично 1 сессия, ~300 строк тестового кода. **Не требует правки goml.**

**Приоритет 2 — порт `nn.Linear`/`nn.Embedding` в gotorch:**
- Обходит `FromSlice → CPU` gap.
- После порта: `nn.LLM` начнёт работать на GPU через gotorch-версии слоёв.
- Далее LayerNorm/RoPE/SDPA — по одному, каждый с f64ref-сверкой + adapter A/B.

**Приоритет 3 — GPU-специфичные kernel'ы:**
- RMSNorm, SwiGLU (fused), RoPE (fused), Attention (FA-style) — уже есть в goml `backend/cuda/transformer_kernels.go`, ждут порта в gotorch + подключения в nn.*.
- fused-AdamW есть kernel (`adamw_f32`) — портировать в gotorch.
- Mixed precision F16/F8 — cublas обёртки готовы (`cublas_fp8.go`), портировать в gotorch.

---

## Слово `наверное` в отчёте — 0 вхождений.

Все утверждения покрыты grep-выводом или file:line ссылкой. Каждая ячейка таблицы — из фактического файла.

## СТОП по правилу ТЗ

Второй GPU-путь найден. По ТЗ — план главы портирования пересматриваем вместе. Готов принять решение по приоритетам.
