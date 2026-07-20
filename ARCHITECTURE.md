# gotorch — Архитектура и целевая роль в стеке

**Статус:** 2026-07-18 (R03b era). Актуально до пересмотра.

---

## Целевая картина стека (3 продукта)

**gotorch** — фреймворк и **ЕДИНСТВЕННЫЙ целевой источник истины** для всех переиспользуемых вычислительных компонентов ML-стека: тензоры, слои, оптимизаторы, GPU-backend. Сейчас реализовано: тензорная модель + autograd, F64/F32 pure-Go CUDA backend (50 методов через purego, R02b). В планах: F16/FP8-путь (batched MatMul через `cublasGemmEx` / `cublasGemmStridedBatched`), трансформерные ядра (RMSNorm, RoPE, embedding, softmax-по-осям, fused-AdamW), memory pool.

Каждый портируемый кусок — по образцу боевого кода из [goml](https://github.com/djeday123/goml) (та реализация выстрадана оптимизациями `sm_120a` / `sm_89`, воспроизводить с нуля бессмысленно). Верификация — A/B против goml-эталона через мост-адаптер (`goml/backend/gotorch/`).

**fa-blackwell-fp8** — отдельная библиотека FlashAttention-ядер для sm_120a Blackwell FP8. C-ABI + Go/Python-биндинги. **Не** часть gotorch (специфична по архитектуре и dtype-миксу); подключается опционально когда нужен FA-путь.

**goml** — референс-приложение тренировки (модель LLM, трейнер, токенизатор, данные) + исторически боевой GPU-движок `backend/cuda`. Движок остаётся в силе и обеспечивает текущий тренинг; **не развивается** как переиспользуемая библиотека. Его функциональность поэтапно появляется в gotorch (порт с goml-эталона, A/B-верификация через adapter-мост), после чего goml-трейн переключается на gotorch-версии метод за методом.

**Уборка** goml-копий — отдельное решение **после** доказанной полноты gotorch. До этого моста два, но живой источник истины — один (gotorch); goml/backend/cuda замораживается как read-only эталон.

## Правило переходного периода

**goml/backend/cuda — read-only эталон.** Фиксы и развитие вычислительной логики идут **только в gotorch**. Никаких «заодно поправим в goml» — иначе источников истины снова два и синхронизация становится проблемой на всю жизнь стека.

Единственные разрешённые правки в goml: (1) корректировка публичных аксессоров, необходимых мосту (напр. `Backend.Stream()` в R03b-impl-2); (2) регистрация новых зависимостей в `go.mod`; (3) чинить критические регрессии, если они блокируют переходный период. Логика PTX-ядер, cuBLAS-обёрток, LayerNorm/RoPE/SDPA/etc — не трогается.

## Текущее состояние моста (R03b)

- Один CUDA-контекст: **primary** (после R03b-impl-1 Fix A в `goml/backend/cuda/cuda..go:72`).
- Один stream: **injection** через `gotorch.PuregoBackend.SetStream(...)` при инициализации адаптера `goml/backend/gotorch/`.
- Adapter покрытие по [R03b_design.md](runs/reports/R03b_design.md):
  - **47% direct** — Alloc/Free/Copy/Add/Sub/Mul/Div/Neg/Exp/Log/Tanh/Relu/Sigmoid/MatMul(batch=1)/Softmax(axis=-1) — уже реализованы или будут в impl-3.
  - **44% stays-in-goml** — LayerNorm/Embedding/RoPE/SDPA/Gelu/Silu/Sum-axis/Mean-axis/Max-axis/MatMul(batch>1)/Fill/Arange/Where/Abs+Sqrt (до impl-6). **Эта вторая цифра — карта будущих портов в gotorch.**
  - **6% gap** — Abs, Sqrt (два простых PTX-ядра, добавляются в gotorch мелким коммитом impl-6).

## Как ими пользоваться

Человеку, строящему свой ML-трейн:

```
go get github.com/djeday123/gotorch
# + опционально для FA:
go get github.com/djeday123/fa-blackwell-fp8/gofa
```

goml используется целиком **только** если нужен весь референс-стек (модель + токенизатор + trainer). Иначе — конструируется свой трейн поверх gotorch.

## После R03b-impl-5

Следующая большая серия — **не «интеграция»**, а достройка gotorch до полноты. По одному куску за итерацию:

1. Порт с goml-эталона (не изобретать заново).
2. Тесты корректности через мост: **gotorch-версия vs goml-эталон в одном процессе** (bit-exact или документированный floor).
3. Переключение goml-трейн-пути на gotorch-версию через adapter.
4. Метод в списке stays-in-goml сокращается на единицу.

Когда список стал 0 — фаза «уборки goml/backend/cuda» становится безопасной.

---

# ENGLISH SECTION

## Target stack picture (3 products)

**gotorch** — the framework and **THE SINGLE canonical source of truth** for all reusable compute components of the ML stack: tensors, layers, optimizers, GPU backend. Currently: tensor model + autograd, F64/F32 pure-Go CUDA backend (50 methods via purego, R02b). Planned: F16/FP8 path (batched MatMul via cuBLAS GemmEx / StridedBatched), transformer kernels (RMSNorm, RoPE, embedding, softmax-along-axis, fused-AdamW), memory pool.

Each portable piece is written by porting battle-hardened code from [goml](https://github.com/djeday123/goml) — that implementation is already optimized for sm_120a Blackwell / sm_89 Ada, reinventing from scratch would be waste. Verification: A/B against the goml reference through the bridge adapter (`goml/backend/gotorch/`).

**fa-blackwell-fp8** — a separate FlashAttention kernels library for sm_120a Blackwell FP8. C ABI + Go/Python bindings. **Not** part of gotorch (architecture- and dtype-specific); pulled in optionally when the FA path is needed.

**goml** — a reference training application (LLM model, trainer, tokenizer, data) + historically a battle GPU engine in `backend/cuda`. The engine stays in production and powers current training runs; it is **no longer developed** as a reusable library. Its functionality lands in gotorch piece by piece (port from the goml reference, A/B-verified via the bridge), and the goml trainer switches to the gotorch versions method by method.

**Cleanup** of the goml duplicates is a separate decision **after** gotorch's completeness is proven. Until then, there are two bridges but only one live source of truth (gotorch); `goml/backend/cuda` is frozen as a read-only reference.

## Transition-period rule

**`goml/backend/cuda` is a read-only reference.** Fixes and development of compute logic land **in gotorch only**. No "let's patch goml while we're here" — otherwise you get two sources of truth again and synchronizing them becomes a life-of-the-stack problem.

The only permitted changes to goml: (1) tiny public accessors the bridge needs (e.g. `Backend.Stream()` in R03b-impl-2); (2) `go.mod` dependency registration; (3) critical regression fixes if they block the transition. PTX kernel logic, cuBLAS wrappers, LayerNorm/RoPE/SDPA/etc are untouched.

## Bridge state (R03b)

- One CUDA context: **primary** (after R03b-impl-1 Fix A in `goml/backend/cuda/cuda..go:72`).
- One stream: **injection** via `gotorch.PuregoBackend.SetStream(...)` at adapter init in `goml/backend/gotorch/`.
- Adapter coverage per [R03b_design.md](runs/reports/R03b_design.md):
  - **47% direct** — Alloc/Free/Copy/Add/Sub/Mul/Div/Neg/Exp/Log/Tanh/Relu/Sigmoid/MatMul(batch=1)/Softmax(axis=-1) — done or done in impl-3.
  - **44% stays-in-goml** — LayerNorm/Embedding/RoPE/SDPA/Gelu/Silu/Sum-axis/Mean-axis/Max-axis/MatMul(batch>1)/Fill/Arange/Where/Abs+Sqrt (until impl-6). **This second number is the map of future ports into gotorch.**
  - **6% gap** — Abs, Sqrt (two trivial PTX kernels, added to gotorch in a small impl-6 commit).

## How to use

For someone building their own ML trainer:

```
go get github.com/djeday123/gotorch
# + optional FA path:
go get github.com/djeday123/fa-blackwell-fp8/gofa
```

goml as a whole is used **only** if you want the full reference stack (model + tokenizer + trainer). Otherwise, construct your own trainer on top of gotorch.

## After R03b-impl-5

The next big series is **not "integration"** but building gotorch to completeness. One piece per iteration:

1. Port from the goml reference (do not reinvent).
2. Correctness tests through the bridge: **gotorch version vs goml reference in one process** (bit-exact or documented floor).
3. Switch the goml trainer path to the gotorch version through the adapter.
4. One item drops out of the stays-in-goml list.

When that list hits 0, the goml/backend/cuda cleanup phase becomes safe.
