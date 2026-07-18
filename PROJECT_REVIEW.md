# gotorch v6.0.0 — Project Review

**Reviewer briefing.** Полный обзор проекта: что построено, как это устроено, где сильные и слабые места, куда стоит смотреть внимательнее. Цель — помочь ревьюеру быстро войти в контекст и провести содержательную критическую оценку.

- **Repo**: https://github.com/djeday123/gotorch
- **Latest tag**: v6.0.0 (2026-05-24)
- **Language**: Go 1.22+, чистый Go для CPU + опциональный CUDA backend через CGo
- **License**: MIT
- **Author**: djeday123
- **Codebase size**: ~16 800 строк Go (33 файла кода + 32 test-файла)
- **Test count**: 349 тестов, `go test -race` clean
- **Positioning**: PyTorch-like deep learning framework, research / educational

---

## 1. Executive summary

gotorch — **чистый Go порт PyTorch-подобного API** с полноценным reverse-mode autograd, набором nn-слоёв (включая полный Transformer, LSTM/GRU с BPTT), оптимизаторами семьи Adam, DataLoader с горутинным prefetch, и опциональным CUDA backend через CGo. Проект **явно позиционируется** как research/educational (см. README), не как production-competitive.

**Главная особенность v6.0.0** — это первый релиз, в котором **autograd действительно математически корректен**. Автор публично признаёт, что версии v1-v5 (Levels 1-8 фичей) содержали **10 критических тихих багов в backward passes** (Sum, Mean, Softmax, LayerNorm, BatchNorm1d, GroupNorm, MultiHeadAttention, LSTM/GRU). Багфиксы задокументированы поштучно в CHANGELOG.md с ссылками на конкретные коммиты и с новым `nn/gradcheck_test.go` который проверяет каждый исправленный градиент через central-difference numerical gradient.

**Это делает v6.0.0 фундаментально другим релизом**, чем предшествующие. Ревью должно оценивать **v6.0.0 отдельно от истории проекта**, потому что предыдущие релизы обучали модели неверными градиентами (сходились случайно на простых задачах).

---

## 2. Architecture

```
gotorch/
├── tensor/     ~4 000 строк — N-мерные тензоры, broadcasting, dtypes, sparse COO, memory pool
├── autograd/   ~700 строк  — reverse-mode autodiff, computation graph, NoGrad ctx
├── nn/         ~7 200 строк — все слои, включая полный Transformer (encoder+decoder)
├── optim/      ~800 строк  — SGD/Adam/AdamW/RMSprop + LR schedulers + checkpoint
├── data/       ~200 строк  — DataLoader с горутинным prefetch (race-safe после v6.0.0)
├── amp/        ~150 строк  — GradScaler для mixed-precision
├── export/     ~470 строк  — ONNX export (base opset, forward-only graph)
└── cuda/       ~1 200 строк — CGo bridge + cuBLAS для базовых ops (опциональный)
```

**Design principles** (наблюдаемые из кода):
- **Eager / imperative** — как PyTorch, не как Gorgonia (граф-first / Theano-подобный)
- **Reverse-mode autograd через `Variable`** — тензор + grad + reference на GradFn, топологическая сортировка при backward
- **Все NN-слои — `Module` interface** (`Forward(*Variable) *Variable`, `Parameters() []*Variable`, `ZeroGrad()`)
- **CPU-first**: `cuda/` backend узкий, покрывает только elementwise + MatMul (cuBLAS DGEMM), не полный training path
- **Внутренняя аккумуляция в Float64** для стабильности, Float32 preserved на входе (v6.0.0)

---

## 3. Coverage — что реализовано

### Tensors
- N-мерные с broadcasting, advanced indexing (`Select`, `Narrow`, `Gather`, `ScatterAdd`, `Where`, `MaskedSelect`)
- Dtypes: Float64 (default), Float32, Float16, BFloat16, Int8
- Linalg: MatMul, BatchMatMul, Dot, Outer (Float32-preserving с v6.0.0)
- Structural: Cumsum, Cumprod, Tril, Triu, RepeatInterleave, TopK (O(n log k) heap)
- Sparse COO (базовые: ToDense, MatMulDense)
- Memory pool с `sync.Pool` (power-of-two buckets 64→16M)
- **In-place ops** (`AddInPlace`, ...): 49× быстрее аллокирующих версий, 0 аллокаций

### Autograd
- Reverse-mode через `Variable + engine.go` (топ. сортировка + backward pass)
- Все дифференцируемые ops: Add/Sub/Mul/Div/MatMul, MulScalar/PowScalar, Neg/Exp/Log, ReLU/Sigmoid/Tanh/Softmax, Mean/Sum/SumDim
- `NoGrad()` / `EnableGrad()` — depth counter (v6.0.0, race-safe)
- Numerical gradient check infrastructure в `nn/gradcheck_test.go`

### NN layers
- **Linear** — стандартный `y = xW^T + b`
- **Свёртки**: Conv1d, Conv2d (im2col), ConvTranspose2d (per-batch matmul + col2im — v6.0.0)
- **Pooling**: MaxPool2d, AdaptiveAvgPool2d
- **Активации**: ReLU/Sigmoid/Tanh/GELU/LeakyReLU/ELU/SiLU/Softplus
- **Normalization** (**все с честным backward после v6.0.0**): BatchNorm1d/2d, LayerNorm (per-group std stored), GroupNorm (per-(n,group) std)
- **Sequence**:
  - `MultiheadAttention` — **v6.0.0 fix**: правильный chain rule для Q/K/V (было: одинаковый grad во все три)
  - `LSTM`/`GRU` — **v6.0.0 fix**: proper BPTT через все gates (было: raw `[]float64` вне graph)
  - `StackedLSTM`/`StackedGRU` — multi-layer с dropout
- **Positional encoding**: SinusoidalPE, PositionalEmbedding
- **Transformer (полный)**:
  - `NewTransformer(dModel, nHead, numEnc, numDec, dFF, dropout)`
  - Отдельные `TransformerEncoderLayer/Encoder`, `TransformerDecoderLayer/Decoder`
- **Quantization**: `QLinear` (int8 inference, post-training)
- **DataParallel** (multi-GPU distribution)
- **Loss functions**: MSE, BCE, CE, NLL, L1, Huber, KLDiv
- **Контейнеры**: Sequential, ModuleList, ModuleDict
- **Model utils**: Summary/PrintSummary, Save/Load

### Optim
- SGD (+ momentum), Adam, AdamW, RMSprop, Adadelta
- LR schedulers: StepLR, CosineAnnealingLR, LinearWarmup
- Gradient clipping: ClipGradNorm, ClipGradValue
- **Checkpoint**: SaveCheckpoint/LoadCheckpoint — веса + moments + step count в одном `.gob`

### Data / AMP / Export / CUDA
- **DataLoader**: goroutine prefetch, mutex-serialised, `-race` clean (v6.0.0)
- **AMP**: GradScaler c grow/backoff/inf-nan detection
- **Export**: ONNX (base opset, forward-only)
- **CUDA**: elementwise + activations + cuBLAS MatMul + pinned memory — **узкий, не покрывает training path**

---

## 4. Correctness story — самая важная часть ревью

### Как признание в багах

Автор в CHANGELOG.md и RELEASE_NOTES_v6.0.0.md **явно и подробно признаёт**, что v1-v5 shipping'и содержали неверные gradient computations. Ключевая цитата из v6.0.0 release:

> "This is the first release of gotorch where the autograd is actually correct. Earlier v1.0.0 … v5.0.0 builds shipped silent bugs in backward passes for LSTM, GRU, MultiHeadAttention, LayerNorm, BatchNorm1d, GroupNorm, Sum, Mean, and Softmax. Training with those releases produced wrong gradients that just happened to converge for simple problems."

Это **научно честно** и я бы отметил в ревью как позитивный сигнал зрелости.

### 10 багфиксов в v6.0.0 (commit 28284e8)

Каждый описан в CHANGELOG с формулой:

| Слой | Что было сломано | Что стало |
|---|---|---|
| `Sum.Backward` | возвращал `Ones(shape)`, игнорировал upstream grad | `MulScalar(Ones, grad.Item())` |
| `Mean.Backward` | тот же баг | `MulScalar(Ones, grad.Item() / n)` |
| `Softmax.Backward` | identity passthrough | Реальный Jacobian `dx = s · (g − sum(g · s, dim))` |
| `LayerNorm.Backward` | без фактора `1/std` (forward не хранил std) | Forward пишет per-group std; backward: `(γ · dy − sumDy/M − xn · sumDyXn/M) / std` |
| `BatchNorm1d` | `requiresGrad=false`, gradient не тёк | Полный backward для train (per-batch stats) и eval (running stats) |
| `GroupNorm` | тот же broken-flow bug | Per-(n, group) std backward |
| `MultiHeadAttention.Backward` | одинаковый grad в Q/K/V | Правильный chain rule: `dVh = attnᵀ · dHead`, `dScore = softmax_backward(dAttn, attn)`, `dQh = dScore · Kh · scale`, `dKh = dScoreᵀ · Qh · scale` |
| `LSTM`/`GRU` gates | raw `[]float64` вне autograd graph → BPTT никогда не доходил до gate weights | Sequence-level autograd op с явным BPTT через каждый gate на каждом timestep |
| `Div.Backward` | NaN/Inf при `b=0` | Smooth reciprocal `b / (b² + 1e-12)` |

### Verification infrastructure

`nn/gradcheck_test.go` содержит **central-difference numerical gradient check**:

```go
func numericalGrad(fn func(*tensor.Tensor) *tensor.Tensor, x *tensor.Tensor, eps float64) *tensor.Tensor {
    // (f(x+eps) - f(x-eps)) / (2*eps) для каждой координаты
}
```

Каждый исправленный backward проверяется отдельным тестом. **Это хороший подход**, но ревьюеру стоит посмотреть:

**Что проверить в gradcheck_test.go**:
- Насколько тугой tolerance (eps выбор для central diff)
- Покрыты ли все 10 багфиксов
- Проверяются ли **все компоненты** для MultiHeadAttention (dQ/dK/dV отдельно)
- BPTT для LSTM/GRU — проверяется ли grad wrt gate weights **на каждом timestep**, или только на выходе
- Hidden state gradients в LSTM/GRU — проверяются ли отдельно от output grads

**Красный флаг для ревьюера**: если gradcheck использует **fp32** для probe-forward, малые градиенты будут в шуме FP32 (~1e-7 relative), и мутации <0.1% индистинктибельны. У автора есть параллельный проект `fa-blackwell-fp8` где это ошибка была явно детектирована и починена (FP64 probe + hybrid tolerance). **Стоит проверить не наступил ли gotorch на те же грабли.**

---

## 5. Test coverage

**349 тестов + бенчмарки**, все зелёные, `go test -race` clean.

Распределение (из sample):
- `nn/level*_test.go` — по уровням фич (Level 2-8): 92 теста
- `nn/gradcheck_test.go` — 6 тестов для v6.0.0 фиксов
- `nn/nn_test.go` — 13 базовых
- `nn/parallel_test.go` — 10 (DataParallel)
- `nn/quantization_test.go` — 12 (QLinear)
- `tensor/tensor_test.go` — 23, `level*_test.go` — 48
- `optim/optim_test.go` + level tests + checkpoint_test — 27
- `autograd/autograd_test.go` — 15
- Race-safety: `no_grad_concurrent_test.go`, dedicated

**Что ревьюеру проверить**:
- **Покрытие backward** — 349 тестов это много, но сколько тестируют **именно gradient correctness** vs shape/forward-value?
- **End-to-end training** — есть ли тест "обучили MLP на XOR, loss упал"? Или только unit-tests изолированных слоёв?
- **PyTorch parity check** — есть `bench_pytorch.py` в корне; сверяются ли численно с PyTorch на реальном training loop?

---

## 6. Engineering decisions worth calling out

### Positive

1. **Честное позиционирование**: README прямо говорит "Research / Educational", "~7% PyTorch API surface", "40-50% practical usefulness". Нет overclaims.
2. **CHANGELOG честный**: явно перечисляет known limitations и что не поддерживается.
3. **`go test -race` clean** — на языке типа Go это структурная гарантия для DataLoader/NoGrad.
4. **In-place ops с честным benchmark** ("49× faster, 0 allocations on `[1024]`") — реальные числа.
5. **Numerical stability aware**: Welford variance для `Var` (стабильна при больших offsets), smooth reciprocal в Div, cache-tiled MatMul в ConvTranspose2d.
6. **Memory pool** правильно спроектирован (`sync.Pool` + power-of-two buckets, opt-in via `Release()`).

### Negative / concerning

1. **Внутренняя аккумуляция в Float64** для всех операций (даже когда inputs Float32) — это трата 2× памяти на промежуточных суммах ради стабильности. В README явно написано что "Native Float32 compute path is not yet present". Для обучения больших моделей это ограничение.
2. **CUDA backend узкий**: только elementwise + MatMul (cuBLAS DGEMM = Float64!). Не покрывает Conv2d, softmax MMA, attention. Реального multi-GPU training path нет (несмотря на `DataParallel` в API).
3. **DataParallel** есть в nn/, но CUDA backend не поддерживает разделение по devices полноценно — стоит проверить не декоративный ли это API.
4. **ONNX export** — "base opset, forward-only graph" — не production ready для трансформеров с dynamic shapes.
5. **Отсутствует `torch.compile`-like graph capture**, JIT, second-order autograd (vmap, jacobian, hessian) — явно указано в "Not yet supported".
6. **Sparse tensors** ограничены COO, few ops (только `ToDense`, `MatMulDense`) — nowhere close to `torch.sparse`.

---

## 7. Code quality observations

### Что видно из structure

- **Плоская структура пакетов** (`tensor/`, `nn/`, ...), без лишней подкатегоризации. Легко навигировать.
- **Разделение по уровням фич**: `level2_test.go`, `level3_test.go`, ... — chronological grouping тестов. Помогает истории, но может усложнить refactoring.
- **CHANGELOG + RELEASE_NOTES** — оба есть, синхронизированы. Хорошо.
- **`gradcheck_test.go` — новый файл в v6.0.0** — правильный шаг, но покрытие требует ревью.

### Наблюдения по размерам файлов

- `nn/rnn.go` — 739 строк (LSTM + GRU + Stacked* + BPTT)
- `nn/norm.go` — 694 строки (BatchNorm1d/2d + LayerNorm + GroupNorm со stored std)
- Оба **относительно крупные**, но обоснованы — там живёт вся сложная backward-математика.

### Стиль

Из sample (`nn/gradcheck_test.go`): чистый идиоматический Go, без sysops-магии, комментарии на английском, тесты в стандартном `testing.T` стиле. Быстрое чтение.

---

## 8. Comparison to alternatives

| Framework | Language | Model | vs gotorch |
|---|---|---|---|
| **PyTorch** | Python + C++ | Eager | Reference, гораздо шире, gotorch честно признаёт ~7% coverage |
| **Gorgonia** | Go | Graph-first (Theano-like) | Другая парадигма; gotorch ближе к PyTorch API |
| **GoLearn** | Go | Classical ML | Не neural nets, не сравнимо |
| **GoMLX** | Go + XLA | XLA-backed | Другой стек; больше про production ML в Go via XLA |

**Ниша gotorch**: PyTorch-eager стиль в Go без CGo dependency на libtorch. Учебный / research инструмент. Не конкурирует с production frameworks.

---

## 9. Suggested reviewer focus areas

Приоритизировано по важности:

### High priority

1. **`nn/gradcheck_test.go`** — критичная часть v6.0.0 claim. Что проверить:
   - Tolerance/eps подобраны корректно
   - Покрыты все 10 багфиксов
   - Проверяется ли BPTT для LSTM/GRU не только на выходе, но на каждом timestep
   - Есть ли gradient проверка для attention Q/K/V **раздельно**
   - Не используется ли pure FP32 probe (риск маскировать баги в шуме)

2. **`nn/attention.go`** — MultiheadAttention с новым chain rule. Проверить:
   - Правильный порядок операций (scale × K → S → softmax → attn × V)
   - Backward пути dV, dScore, dQ, dK — соответствуют формулам из CHANGELOG
   - Multi-head splitting/concat корректный

3. **`nn/rnn.go`** — LSTM/GRU BPTT. Проверить:
   - `rowSliceBackward` op — упомянут в CHANGELOG но требует внимания
   - Gate gradients (input, forget, output, cell) — все компоненты присутствуют
   - Numerical stability на длинных последовательностях

4. **`nn/norm.go`** — все три Norm слоя. Проверить:
   - Forward действительно хранит per-group std (не пересчитывает в backward — потенциальный источник расхождения)
   - Backward формула соответствует стандартной derivation
   - BatchNorm1d train vs eval mode переключение

### Medium priority

5. **`autograd/engine.go`** — топологическая сортировка backward. Проверить корректность обхода графа (особенно с shared parameters, in-place ops)

6. **`autograd/no_grad.go`** — race-safe depth counter. Проверить что nested NoGrad из разных горутин не создаёт race conditions

7. **`tensor/pool.go`** — memory pool. Проверить что `Release()` не позволяет использовать storage после возврата в пул (use-after-free class risk)

8. **`data/dataloader.go`** — race-safe после v6.0.0. Проверить channel rotation при Reset

### Low priority (optional)

9. **`export/onnx.go`** — ONNX serialization. Проверить соответствие opset (без production претензий, но структурная корректность важна)

10. **`nn/quantization.go` (QLinear)** — post-training int8. Проверить scale/zero-point правильно применяются в forward

---

## 10. Risks and honest concerns

### Numerical fidelity risk
Как я упомянул выше — если `gradcheck_test.go` использует **FP32 finite-difference с fixed eps**, small gradients будут в шуме и мутации <0.1% не будут детектироваться. Это **точно та же ловушка** в которую попал автор в параллельном FA-проекте (fa-blackwell-fp8 → см. B1-FIX-EXTRA log где путь 93.3% → 100% agreement достигался через FP64 probe + hybrid tolerance). **Стоит проверить не наступил ли gotorch на те же грабли.**

### Backward correctness claim requires deep audit
"10 backward-bugs fixed" — сильное заявление. Полное подтверждение потребует **независимого сверения gradients с PyTorch на нескольких формах** — не только formal formula check.

### CUDA path decoration
Если DataParallel API не покрыт реальным multi-device execution в CUDA backend — это **декоративный API**, вводящий пользователя в заблуждение. Стоит проверить есть ли real test с 2+ devices.

### Performance claims
"49× faster" для in-place ops — reasonable для zero-alloc. "10-50× для ConvTranspose2d" — не benchmarked конкретно в CHANGELOG, только "expected". Стоит убедиться что бенчмарки в тестах реально измеряют это.

### Legacy code paths
v1-v5 API surface **не изменился** (v6.0.0 CHANGELOG утверждает). Значит если пользователь мигрирует старый код, он получит **другие численные результаты** (правильные) — что может сломать integration tests, model checkpoints, etc. Это заявлено в CHANGELOG, но стоит проверить upgrade guide.

---

## 11. Overall verdict

**Strengths:**
- Честное позиционирование как research/educational tool
- Открытое признание прошлых багов + систематическая работа над их исправлением
- Хорошая инфраструктура (numerical gradient check, race-safe DataLoader/NoGrad)
- Разумные engineering choices (memory pool, in-place ops, Welford variance)
- Полное покрытие PyTorch-подобного API для core обучающих сценариев

**Weaknesses:**
- Внутренняя Float64 аккумуляция (2× память на промежуточных суммах)
- Узкий CUDA backend, не покрывающий training path
- Отсутствие FP64 probe в gradient check (потенциально)
- ONNX export не production ready
- Отсутствует `torch.compile`-like оптимизация

**Recommendation for reviewer:**
Провести focused audit по разделу "Suggested reviewer focus areas" (High priority первые 4 пункта). Особенно **валидировать gradcheck на FP64** и **cross-check с PyTorch** на 2-3 нетривиальных архитектурах (LSTM sequence, Transformer с causal mask, ResNet-block с BatchNorm).

Если это подтвердится — **v6.0.0 это качественный образовательный / research инструмент**, вполне юзабельный для обучения нейросетей до среднего размера в pure-Go окружении. Если найдут проблему в gradient check — issue-tracker.

---

## 12. Файлы для быстрого входа

**Для понимания архитектуры**:
1. `README.md` — верхнеуровневый обзор + примеры
2. `CHANGELOG.md` — история фичей и багов
3. `RELEASE_NOTES_v6.0.0.md` — детальное описание v6.0.0 fixes
4. `ROADMAP.md` — план развития
5. `DOCS.md` — 37 KB API документации
6. `TESTS.md` — детальный breakdown тестов

**Для code review**:
7. `nn/gradcheck_test.go` — numerical gradient verification (**критичный**)
8. `nn/attention.go`, `nn/rnn.go`, `nn/norm.go` — где живут v6.0.0 fixes
9. `autograd/engine.go` — backward pass engine
10. `autograd/functions.go` — все дифференцируемые ops

**Примеры для sanity check**:
11. `examples/01_xor` — минимальный training loop
12. `examples/04_lstm_lm` — LSTM с fixed BPTT
13. `examples/05_transformer` — encoder Transformer

---

**Готовность к принятию**: v6.0.0 — первый релиз, к которому эти вопросы вообще имеет смысл задавать (предыдущие содержали известные баги). Для v6.0.0 честный вердикт: **сильный кандидат на "correct research tool"**, требует independent audit gradcheck infrastructure перед trust'ом на production ML tasks.
