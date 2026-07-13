# GoTorch

[![Go](https://img.shields.io/badge/Go-1.22+-00ADD8?logo=go)](https://go.dev)
[![Tests](https://img.shields.io/badge/tests-349%20passing-brightgreen)](#test-suite)
[![Version](https://img.shields.io/badge/version-v6.0.0-brightgreen)](#roadmap)
[![Autograd](https://img.shields.io/badge/autograd-numerical--gradient%20verified-success)](#v600-highlights)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

**GoTorch** — PyTorch-подобный фреймворк глубокого обучения на чистом Go с опциональным CUDA GPU backend.

```bash
go get github.com/djeday123/gotorch@v6.0.0
```

> Статус: Research / Educational. Цель — полное воспроизведение PyTorch API на Go, с честным reverse-mode autograd, верифицированным numerical-gradient тестами.

---

## v6.0.0 highlights

**Первый релиз, в котором autograd действительно корректен.** Версии v1.0.0 … v5.0.0 шипились с молчаливыми багами в backward passes для `LSTM`, `GRU`, `MultiHeadAttention`, `LayerNorm`, `BatchNorm1d`, `GroupNorm`, `Sum`, `Mean`, `Softmax`. Обучение с ними давало неверные градиенты, которые случайно сходились на простых задачах.

> **Если ты тренировал с v5.0.0 или раньше — переучить. Pin на v6.0.0+ идёт вперёд.**

### Что починено в v6.0.0

**10 критических backward-багов** (`28284e8`), каждый верифицирован numerical-gradient тестом в `nn/gradcheck_test.go`:

- `Sum.Backward` и `Mean.Backward` игнорировали upstream gradient, возвращали `Ones(shape)` → теперь `MulScalar(Ones, grad.Item())`.
- `Softmax.Backward` был identity passthrough → теперь real Jacobian `dx = s · (g − sum(g · s, dim))`.
- `LayerNorm.Backward` без фактора `1/std` (forward не сохранял std) → forward пишет per-group std; backward: `(γ · dy − sumDy/M − xn · sumDyXn/M) / std`.
- `BatchNorm1d` возвращал `requiresGrad=false` → полный backward для train (per-batch stats) и eval (running stats).
- `GroupNorm` с той же поломкой grad-flow → per-(n, group) std backward.
- `MultiHeadAttention.Backward` возвращал одинаковый gradient в Q/K/V → правильный chain rule: `dVh = attnᵀ · dHead`, `dScore = softmax_backward(dAttn, attn)`, `dQh = dScore · Kh · scale`, и т.д.
- `LSTM` / `GRU` gates считались как raw `[]float64` вне autograd graph → BPTT никогда не доходил до gate weights. Переписаны как единая sequence-level autograd op с явным BPTT через каждый gate на каждом timestep.

**Numerical stability** (`9104a68`):
- `MatMul` / `BatchMatMul` сохраняют `Float32` dtype когда оба входа `Float32`. Внутренняя аккумуляция в `Float64` для стабильности, mixed precision промотит в `Float64`.
- `Div.Backward` больше не даёт NaN/Inf при `b=0`. Smooth reciprocal `b / (b² + 1e-12)` вместо `1/b`.

**Concurrency safety** (`e9f618c`):
- `autograd.NoGrad` переписан как depth counter — no lost state при вложенных NoGrad из разных горутин. `go test -race` clean.
- `data.DataLoader` c `sync.Mutex` + pure `loadBatchAt(batchIdx, indices)` вместо shared `cursor`.

**Performance** (`f3b070c`):
- `Var` — Welford's online algorithm. Stable at large offsets — variance of `{1+1e9, …, 5+1e9}` возвращает 2.5 exactly.
- `TopK` — O(n·k) selection sort → O(n log k) min-heap. ~1.4 ms на 100k элементов при k=50.
- **In-place ops**: `AddInPlace`, `SubInPlace`, `MulInPlace`, `AddScalarInPlace`, `MulScalarInPlace` — **49× быстрее `Add` на `[1024]`, 0 аллокаций** vs 14.
- `ConvTranspose2d` forward/backward переписан как per-batch matmul + col2im scatter. Cache-tiled MatMul заменил hand-rolled inner loops.
- `sync.Pool`-backed tensor allocator с power-of-two buckets (64 → 16 M элементов). `Tensor.Release()` возвращает storage в pool.

Public APIs unchanged. Полный список изменений — [CHANGELOG.md](CHANGELOG.md) и [RELEASE_NOTES_v6.0.0.md](RELEASE_NOTES_v6.0.0.md).

---

## Возможности

| Пакет | Описание | Статус |
|-------|----------|--------|
| `tensor` | N-мерные тензоры, broadcasting, reduce, linalg, sparse COO | ✅ |
| `tensor` dtypes | Float64/Float32/Float16/BFloat16/Int8 | ✅ |
| `autograd` | Reverse-mode autodiff, NoGrad, numerical-gradient verified | ✅ |
| `nn` | Слои, активации, loss, контейнеры (ModuleList/Dict), Transformer full | ✅ |
| `nn/functional` | Stateless (functional) API — F.relu, F.softmax, ... | ✅ |
| `nn/parallel` | DataParallel (multi-GPU distribution) | ✅ |
| `nn` quantization | QLinear (int8 inference quantization) | ✅ |
| `optim` | SGD, Adam, AdamW, RMSprop, Adadelta + LR schedulers + checkpoint | ✅ |
| `data` | Dataset, TensorDataset, DataLoader (goroutine prefetch, race-safe) | ✅ |
| `amp` | GradScaler для mixed precision training | ✅ |
| `export` | ONNX export (base opset) | ✅ |
| `cuda` | CUDA GPU backend (CGo): elementwise, MatMul cuBLAS, pinned memory | ✅ |

---

## Быстрый старт

### XOR за 10 строк

```go
package main

import (
    "fmt"
    "github.com/djeday123/gotorch/autograd"
    "github.com/djeday123/gotorch/nn"
    "github.com/djeday123/gotorch/optim"
    "github.com/djeday123/gotorch/tensor"
)

func main() {
    X := autograd.NewVar(tensor.New([]float64{0,0, 0,1, 1,0, 1,1}, []int{4, 2}), false)
    Y := autograd.NewVar(tensor.New([]float64{0, 1, 1, 0}, []int{4, 1}), false)

    model := nn.NewSequential(
        nn.NewLinear(2, 8, true), nn.NewReLU(),
        nn.NewLinear(8, 1, true), nn.NewSigmoid(),
    )
    opt := optim.NewAdam(model.Parameters(), 0.01, 0.9, 0.999, 1e-8)

    for epoch := 0; epoch < 1000; epoch++ {
        model.ZeroGrad()
        loss := nn.BCELoss(model.Forward(X), Y)
        loss.Backward()
        opt.Step()
    }
    fmt.Printf("Solved XOR! Loss: %.4f\n", nn.BCELoss(model.Forward(X), Y).Data.Item())
}
```

### Full Transformer (encoder + decoder)

```go
model := nn.NewTransformer(
    /*dModel=*/ 512,
    /*nHead=*/ 8,
    /*numEncoderLayers=*/ 6,
    /*numDecoderLayers=*/ 6,
    /*dFF=*/ 2048,
    /*dropout=*/ 0.1,
)

// Or hand-assemble:
enc := nn.NewTransformerEncoderLayer(512, 8, 2048, 0.1)
encoder := nn.NewTransformerEncoder(enc, 6)

dec := nn.NewTransformerDecoderLayer(512, 8, 2048, 0.1)
decoder := nn.NewTransformerDecoder(dec, 6)

// Sinusoidal positional encoding
pe := nn.NewSinusoidalPE(512, 5000)
x := pe.Forward(embedded)
```

### CNN-классификатор

```go
model := nn.NewSequential(
    nn.NewConv2d(1, 32, 3, 1, 1, true),   // [N, 1, 28, 28] → [N, 32, 28, 28]
    nn.NewReLU(),
    nn.NewMaxPool2d(2, 2, 0),              // → [N, 32, 14, 14]
    nn.NewConv2d(32, 64, 3, 1, 1, true),  // → [N, 64, 14, 14]
    nn.NewReLU(),
    nn.NewMaxPool2d(2, 2, 0),             // → [N, 64, 7, 7]
    nn.NewFlatten2d(),                    // → [N, 3136]
    nn.NewLinear(3136, 10, true),
)
```

### Stacked LSTM

```go
// Multi-layer LSTM with dropout between layers
model := nn.NewStackedLSTM(
    /*inputSize=*/ 128,
    /*hiddenSize=*/ 256,
    /*numLayers=*/ 3,
    /*dropout=*/ 0.2,
)
```

### Int8 Quantization для inference

```go
// Post-training quantization
fp32Linear := nn.NewLinear(768, 256, true)
// ... train fp32Linear ...

qLinear := nn.QuantizeLinear(fp32Linear)  // → *QLinear (int8)
out := qLinear.Forward(x)  // 4× меньше памяти, ~2× быстрее inference
```

### ONNX Export

```go
import "github.com/djeday123/gotorch/export"

err := export.ExportONNX(model, exampleInput, "model.onnx")
```

### Optimizer checkpoint

```go
// Save training state (weights + optimizer moments + step count)
err := optim.SaveCheckpoint(model, opt, epoch, "ckpt.gob")
// ...
err = optim.LoadCheckpoint(model, opt, "ckpt.gob")  // resume training
```

### DataLoader (race-safe с v6.0.0)

```go
xData := tensor.RandN(1000, 16)
yData := tensor.Zeros(1000, 1)
ds := data.NewTensorDataset(xData, yData)
loader := data.NewDataLoader(ds, data.LoaderConfig{
    BatchSize: 32, Shuffle: true, NumWorkers: 4, DropLast: false,
})

for loader.HasNext() {
    xBatch, yBatch := loader.Next()
    // ... train step
}
loader.Reset()
```

### Functional API

```go
import F "github.com/djeday123/gotorch/nn/functional"

out := F.GELU(x)
out = F.Dropout(out, 0.1, training)
out = F.Linear(out, weight, bias)
loss := F.CrossEntropyLoss(logits, targets)
```

### Mixed Precision (AMP)

```go
scaler := amp.NewGradScaler(1024.0, 2.0, 0.5, 100, true)

loss := model.Forward(x)
scaledLoss := scaler.ScaleLoss(loss)
scaledLoss.Backward()

if scaler.Step(opt) {
    scaler.Update()
}
```

### In-place ops (49× быстрее для hot loops)

```go
// Zero allocations, mutates receiver
x.AddInPlace(y)
x.MulScalarInPlace(0.5)

// Explicit release для long training loops
t := tensor.Zeros(1024, 1024)
// ... use t ...
t.Release()  // returns storage to sync.Pool
```

---

## API Reference

### `tensor`

```go
// Создание
tensor.Zeros(3, 4)
tensor.Ones(3, 4)
tensor.Full(val, 3, 4)
tensor.Rand(3, 4)           // U[0,1)
tensor.RandN(3, 4)          // N(0,1)
tensor.Arange(0, 10, 1)     // [0,1,...,9]
tensor.Linspace(0, 1, 10)
tensor.Eye(4)
tensor.New(data, shape)
tensor.Scalar(3.14)

// Shape
t.Shape() []int  |  t.Ndim()  |  t.Size()
t.Reshape(2, -1)  |  t.Flatten()
t.Squeeze()  |  t.Unsqueeze(dim)
t.Transpose(i, j)  |  t.T()

// Индексирование
t.At(i, j)  |  t.Set(val, i, j)  |  t.Item()  |  t.Data() []float64

// Elementwise (с broadcasting)
tensor.Add(a, b)  |  Sub  |  Mul  |  Div
tensor.AddScalar(t, v)  |  MulScalar  |  PowScalar
tensor.Neg(t)  |  Abs  |  Exp  |  Log  |  Sqrt
tensor.Floor(t)  |  Ceil  |  Round  |  Sign

// In-place (v6.0.0 — 49× faster than Add, zero allocations)
t.AddInPlace(other)  |  SubInPlace  |  MulInPlace
t.AddScalarInPlace(v)  |  MulScalarInPlace(v)

// Активации
tensor.ReLU(t)  |  Sigmoid  |  Tanh  |  Softmax(t, dim)

// Reduce
tensor.Sum(t, dim, keepdim)  |  Mean  |  Max  |  Min  |  ArgMax
t.Std(dim, keepdim)  |  t.Var(dim, keepdim)  // Welford (v6.0.0, stable at large offsets)
t.Norm(ord, dim, keepdim)  |  t.Prod(dim, keepdim)

// Linalg (v6.0.0 preserves Float32 when both inputs Float32)
tensor.MatMul(a, b)  |  BatchMatMul  |  Dot  |  Outer

// Конкатенация
tensor.Cat(tensors, dim)  |  Stack(tensors, dim)
t.Split(size, dim)  |  t.Chunk(n, dim)

// Продвинутое индексирование
t.Select(dim, idx)  |  t.Narrow(dim, start, len)
t.Index(dim, indices)  |  t.MaskedSelect(mask)
t.Clamp(min, max)  |  t.Where(cond, other)
t.Gather(dim, index)  |  t.ScatterAdd(dim, index, src)

// Структурные ops
t.Cumsum(dim)  |  t.Cumprod(dim)
t.Tril(diagonal)  |  t.Triu(diagonal)
t.RepeatInterleave(repeats, dim)
t.TopK(k, dim)  // O(n log k) min-heap (v6.0.0)

// dtypes
tensor.NewF32(data, shape)  |  t.Float32()  |  t.Float64()
tensor.NewF16(data, shape)  |  t.Float16()      // v3.0.0
tensor.NewBF16(data, shape) |  t.BFloat16()     // v3.0.0
tensor.NewI8(data, shape)   |  t.Int8()         // v3.0.0

// Sparse COO (v5.0.0)
sparse := tensor.NewSparseCOO(indices, values, shape)
dense := sparse.ToDense()  |  sparse.MatMulDense(x)

// Memory pool (v6.0.0)
tensor.AllocFloat64(n)  |  tensor.FreeFloat64(buf)
t.Release()  // return storage to sync.Pool
```

### `autograd`

```go
x := autograd.NewVar(t, requiresGrad)

// Дифференцируемые ops
autograd.Add(a, b)  |  Sub  |  Mul  |  Div  |  MatMul
autograd.AddScalar(a, v)  |  MulScalar  |  PowScalar
autograd.Neg(a)  |  Exp(a)  |  Log(a)
autograd.ReLU(a)  |  Sigmoid(a)  |  Tanh(a)
autograd.Softmax(a, dim)     // v6.0.0: real Jacobian
autograd.Mean(a)  |  Sum(a)  |  SumDim(a, dim)  // v6.0.0: honors upstream grad

// Backward
loss.Backward()
loss.BackwardWithGrad(grad)

// Утилиты
x.Grad          // *tensor.Tensor
x.ZeroGrad()
x.Detach()
autograd.NoGrad()     // depth counter (v6.0.0, race-safe)
autograd.EnableGrad()
autograd.IsGradEnabled() bool
```

### `nn`

```go
// Линейные
nn.NewLinear(in, out, bias)

// Свёрточные
nn.NewConv2d(inC, outC, k, stride, pad, bias)
nn.NewConv1d(inC, outC, k, stride, pad, bias)
nn.NewConvTranspose2d(inC, outC, k, stride, pad)  // matmul+col2im (v6.0.0)

// Pooling
nn.NewMaxPool2d(k, stride, pad)
nn.NewAdaptiveAvgPool2d(outH, outW)

// Upsampling
nn.NewUpsample(scaleFactor, mode)  // mode: "nearest"

// Активации
nn.NewReLU()  |  NewSigmoid()  |  NewTanh()
nn.NewGELU()  |  NewLeakyReLU(slope)  |  NewELU(alpha)
nn.NewSiLU()  |  NewSoftplus(beta)

// Normalization (v6.0.0 fixed: all backward paths correct + numerical-gradient verified)
nn.NewBatchNorm1d(numFeatures)
nn.NewBatchNorm2d(numFeatures)
nn.NewLayerNorm(shape)
nn.NewGroupNorm(numGroups, numChannels)
nn.NewDropout(p)

// NLP / Sequence
nn.NewEmbedding(vocabSize, embDim)
nn.NewMultiheadAttention(embDim, numHeads)         // v6.0.0: proper chain rule
nn.NewLSTM(inputSize, hiddenSize, numLayers)       // v6.0.0: proper BPTT
nn.NewGRU(inputSize, hiddenSize)                   // v6.0.0: proper BPTT
nn.NewStackedLSTM(in, hidden, numLayers, dropout)  // v5.0.0
nn.NewStackedGRU(in, hidden, numLayers, dropout)   // v5.0.0

// Positional encoding (v4.0.0)
nn.NewSinusoidalPE(dModel, maxLen)
nn.NewPositionalEmbedding(maxLen, dModel)

// Transformer (full, v5.0.0)
nn.NewTransformer(dModel, nHead, numEnc, numDec, dFF, dropout)
nn.NewTransformerEncoderLayer(d, nHead, dFF, dropout)
nn.NewTransformerEncoder(layer, numLayers)
nn.NewTransformerDecoderLayer(d, nHead, dFF, dropout)
nn.NewTransformerDecoder(layer, numLayers)

// Quantization (v3.0.0)
qlin := nn.QuantizeLinear(fp32Linear)   // → *QLinear (int8)

// DataParallel (v3.0.0)
dp := nn.NewDataParallel(model, deviceIDs)

// Loss
nn.MSELoss(pred, target)
nn.BCELoss(pred, target)         // pred in (0,1)
nn.CrossEntropyLoss(logits, targets []int)
nn.NLLLoss(logProbs, targets []int)
nn.L1Loss(pred, target)
nn.HuberLoss(pred, target, delta)
nn.KLDivLoss(input, target)      // input = log-probs

// Контейнеры
nn.NewSequential(layers ...Module)
nn.NewModuleList(modules ...Module)
nn.NewModuleDict(pairs ...KeyModule)     // v5.0.0

// Model utils
nn.Summary(model) ModelSummary
nn.PrintSummary(model)

// Save/Load
nn.Save(model, path)
nn.Load(model, path)
```

### `nn/functional`

```go
import F "github.com/djeday123/gotorch/nn/functional"

F.ReLU(x)  |  F.Sigmoid(x)  |  F.Tanh(x)
F.GELU(x)  |  F.LeakyReLU(x, slope)  |  F.ELU(x, alpha)  |  F.SiLU(x)

F.Softmax(x, dim)  |  F.LogSoftmax(x, dim)
F.Dropout(x, p, training)

F.MSELoss(pred, target)  |  F.L1Loss  |  F.HuberLoss(pred, target, delta)
F.BCELoss(pred, target)
F.CrossEntropyLoss(logits, targets []int)
F.NLLLoss(logProbs, targets []int)

F.Linear(x, weight, bias)
```

### `optim`

```go
// Оптимизаторы
optim.NewSGD(params, lr, momentum)
optim.NewAdam(params, lr, beta1, beta2, eps)
optim.NewAdamW(params, lr, beta1, beta2, eps, weightDecay)
optim.NewRMSprop(params, lr, alpha, eps, momentum, weightDecay)
optim.NewAdadelta(params, lr, rho, eps)

// Gradient clipping
optim.ClipGradNorm(params, maxNorm) float64
optim.ClipGradValue(params, clipValue)

// LR Schedulers
optim.NewStepLR(opt, stepSize, gamma)
optim.NewCosineAnnealingLR(opt, tMax)
optim.NewLinearWarmup(opt, warmupSteps)
scheduler.Step()
scheduler.GetLR() float64

// Checkpoint (v4.0.0) — weights + moments + step
optim.SaveCheckpoint(model, opt, epoch, path)
optim.LoadCheckpoint(model, opt, path)
```

### `data`

```go
// Dataset
type Dataset interface { Len() int; Get(i int) (*tensor.Tensor, *tensor.Tensor) }
data.NewTensorDataset(x, y *tensor.Tensor)

// DataLoader (v6.0.0: race-safe, `go test -race` clean)
cfg := data.LoaderConfig{
    BatchSize: 32, Shuffle: true, NumWorkers: 4, DropLast: false,
}
loader := data.NewDataLoader(dataset, cfg)
loader.HasNext() bool
loader.Next() (*tensor.Tensor, *tensor.Tensor)
loader.NumBatches() int
loader.Reset()
```

### `amp`

```go
scaler := amp.NewGradScaler(initScale, growthFactor, backoffFactor, growthInterval, enabled)
scaledLoss := scaler.ScaleLoss(loss)
scaledLoss.Backward()
ok := scaler.Step(optimizer)   // false if inf/nan
scaler.Update()
scaler.GetScale() float64
```

### `export`

```go
import "github.com/djeday123/gotorch/export"

// ONNX export (base opset, forward-only graph)
err := export.ExportONNX(model, exampleInput, "model.onnx")
```

### `cuda`

```go
// Detect
cuda.DetectGPU() bool
cuda.DeviceInfo() string

// Backend
b, _ := cuda.NewGPUBackend(device)
b.Upload(t)  |  b.Download(g)
b.Add(a, b)  |  Sub  |  Mul  |  Div
b.AddScalar(a, v)  |  MulScalar
b.ReLU(a)  |  Sigmoid  |  Tanh  |  Exp  |  Log  |  Neg
b.Sum(a)   |  Mean(a)
b.Softmax(a, rows, cols)
b.MatMul(A, B)   // cuBLAS DGEMM

// PinnedTensor (zero-copy)
p, _ := cuda.NewPinnedTensor(shape...)
p.Slice() []float64
p.ToGPU()  |  p.FromGPU(g)
```

---

## Test Suite

**349 тестов · все зелёные · v6.0.0 · numerical-gradient verified**

```bash
go test ./...              # все тесты (CPU)
go test -race ./...        # race-detector clean (v6.0.0)
go test ./nn/... -run Gradcheck   # numerical-gradient tests

# GPU тесты (требует CUDA)
CGO_CFLAGS="-I./cuda" \
CGO_LDFLAGS="-L./cuda -lgotorch_cuda -L/usr/local/cuda/lib64 -lcublas -lcudart \
             -Wl,-rpath,$(pwd)/cuda -Wl,-rpath,/usr/local/cuda/lib64" \
go test -tags gpu ./cuda/...
```

Полный список тестов + бенчмарков — см. [TESTS.md](TESTS.md) и [DOCS.md](DOCS.md#test-suite).

---

## PyTorch Coverage

| Версия | Тесты | Parity | Что разблокирует |
|--------|------:|--------|------------------|
| v1.0.0 | ~25 | ~25% | XOR, MLP |
| v1.1.0 | 85 | ~45% | LeNet, VGG-style CNN |
| v1.2.0 | ~110 | ~65% | ResNet, save/load, BatchNorm |
| v1.3.0 | ~140 | ~80% | Transformers, GPT-mini, seq2seq |
| v2.0.0 | 185 | ~92% | DataLoader, AMP, Conv1d |
| v2.1.0 | 221 | ~95% | Functional API, ConvTranspose2d |
| v3.0.0 | 261 | ~97% | DataParallel, Float16/BF16/Int8, ONNX, quantization |
| v4.0.0 | 285 | ~98% | BatchNorm1d, GroupNorm, TransformerEncoder, PE, checkpoint |
| v5.0.0 | 312 | ~99% | TransformerDecoder, full Transformer, StackedLSTM, ModuleDict, SparseCOO |
| **v6.0.0** | **349** | **~99%** | **first properly working autograd** (10 backward-bugs fixed, num-grad verified, race-safe) |

По PyTorch API surface — ~7% (PyTorch огромный). По практической полезности для типичных supervised-learning задач — 40-50%. **End-to-end training (CNN, LSTM, Transformer) работает.**

---

## Примеры

| Пример | Описание |
|--------|----------|
| [`examples/01_xor`](examples/01_xor/main.go) | XOR — базовый MLP, BCELoss |
| [`examples/02_mnist_mlp`](examples/02_mnist_mlp/main.go) | 10-классовый MLP с DataLoader |
| [`examples/03_cnn`](examples/03_cnn/main.go) | CNN с Conv2d + MaxPool2d |
| [`examples/04_lstm_lm`](examples/04_lstm_lm/main.go) | LSTM language model (v6.0.0: proper BPTT) |
| [`examples/05_transformer`](examples/05_transformer/main.go) | Mini Transformer encoder |
| [`examples/06_functional`](examples/06_functional/main.go) | nn.functional API showcase |

---

## Сборка

### CPU (по умолчанию)

```bash
go build ./...
go test ./...
go test -race ./...   # race-detector clean since v6.0.0
```

### GPU (CUDA 12.x)

```bash
# Скомпилировать CUDA kernels
nvcc -O3 -std=c++14 \
     -gencode arch=compute_89,code=sm_89 \
     -shared -fPIC \
     -o cuda/libgotorch_cuda.so cuda/ops.cu \
     -lcublas

# Тесты
CGO_CFLAGS="-I./cuda" \
CGO_LDFLAGS="-L./cuda -lgotorch_cuda -L/usr/local/cuda/lib64 -lcublas -lcudart \
             -Wl,-rpath,$(pwd)/cuda -Wl,-rpath,/usr/local/cuda/lib64" \
go test -tags gpu ./cuda/...
```

| CUDA_ARCHS | GPU |
|------------|-----|
| `80` | A100, A30 |
| `86` | RTX 3090, A40 |
| `89` | RTX 4090, L40 |
| `90` | H100, H200 |
| `120` | RTX 5090, RTX PRO 6000 Blackwell |

---

## Архитектура

```
gotorch/
├── tensor/                 CPU тензоры (без CGo)
│   ├── tensor.go           Tensor type, creation, At/Set
│   ├── ops.go              Elementwise + in-place ops
│   ├── reduce.go           Sum, Mean, Max, Min, Softmax, Std, Var, Norm
│   ├── shape.go            Reshape, broadcast, Transpose, Cat, Stack
│   ├── index.go            Select, Narrow, Gather, ScatterAdd, Where
│   ├── linalg.go           MatMul, BatchMatMul, Dot, Outer (Float32-preserving)
│   ├── combine.go          Cat, Stack, Split, Chunk
│   ├── extra.go            Full, Linspace, Floor, Ceil, Round, Sign, Welford Var, heap TopK
│   ├── ops_level5.go       Cumsum, Cumprod, Tril, Triu, RepeatInterleave
│   ├── dtype_extra.go      Float16, BFloat16, Int8
│   ├── sparse.go           SparseCOO
│   └── pool.go             sync.Pool tensor allocator
│
├── autograd/               Reverse-mode autodiff (numerical-grad verified)
│   ├── variable.go         Variable (Tensor + grad + graph)
│   ├── engine.go           Topological sort + backward pass
│   ├── functions.go        Дифференцируемые операции
│   └── no_grad.go          NoGrad depth counter (race-safe)
│
├── nn/                     Neural network modules
│   ├── module.go           Module interface
│   ├── linear.go           Linear (y = xW^T + b)
│   ├── sequential.go       Sequential container
│   ├── module_list.go      ModuleList
│   ├── module_dict.go      ModuleDict
│   ├── conv.go             Conv2d (im2col)
│   ├── conv1d.go           Conv1d
│   ├── conv_transpose2d.go ConvTranspose2d (matmul+col2im)
│   ├── activations.go      ReLU, Sigmoid, Tanh
│   ├── activations_extra.go GELU, LeakyReLU, ELU, SiLU, Softplus
│   ├── norm.go             BatchNorm1d/2d, LayerNorm, GroupNorm (all with proper backward)
│   ├── dropout.go          Dropout
│   ├── embedding.go        Embedding
│   ├── positional.go       SinusoidalPE, PositionalEmbedding
│   ├── attention.go        MultiheadAttention (proper chain rule)
│   ├── rnn.go              LSTM, GRU, StackedLSTM, StackedGRU (BPTT)
│   ├── transformer.go      TransformerEncoderLayer, TransformerEncoder, full Transformer
│   ├── decoder.go          TransformerDecoderLayer, TransformerDecoder
│   ├── quantization.go     QLinear (int8)
│   ├── parallel.go         DataParallel
│   ├── loss.go             MSE, BCE, CE, NLL, L1, Huber, KLDiv
│   ├── save.go             Save / Load
│   ├── upsample.go         Upsample (nearest)
│   ├── summary.go          ModelSummary, PrintSummary
│   └── functional/         Stateless functional API
│       └── functional.go
│
├── optim/                  Optimizers
│   ├── sgd.go              SGD + momentum
│   ├── adam.go             Adam
│   ├── adamw.go            AdamW
│   ├── rmsprop.go          RMSprop
│   ├── clip.go             ClipGradNorm / ClipGradValue
│   ├── scheduler.go        StepLR, CosineAnnealingLR, LinearWarmup
│   └── checkpoint.go       Save/LoadCheckpoint (weights + moments + step)
│
├── data/                   DataLoader (race-safe)
│   ├── dataset.go          Dataset interface, TensorDataset
│   └── dataloader.go       DataLoader (goroutine prefetch, mutex-serialised)
│
├── amp/                    Mixed precision
│   └── scaler.go           GradScaler
│
├── export/                 Model export
│   └── onnx.go             ONNX export (base opset)
│
└── cuda/                   GPU backend (build tag: gpu)
    ├── ops.cu              CUDA kernels + cuBLAS
    ├── backend.go          GPUBackend high-level API
    ├── pinned.go           PinnedTensor (zero-copy)
    └── bridge.go           CGo wrappers
```

---

## Требования

| | Минимум |
|--|---------|
| Go | 1.22+ |
| CUDA (GPU) | 12.x |
| NVIDIA driver (GPU) | 525+ |

---

## Roadmap

- [x] v1.0.0 — Foundation: tensor, autograd, Linear, SGD, Adam, CUDA
- [x] v1.1.0 — Level 1: float32, Conv2d, NoGrad, Cat/Stack/Split
- [x] v1.2.0 — Level 2: AdamW, LR schedulers, BatchNorm, save/load
- [x] v1.3.0 — Level 3: Transformer, LSTM, GRU, Embedding, Attention
- [x] v2.0.0 — Level 4: DataLoader, AMP, Conv1d, RMSprop
- [x] v2.1.0 — Level 5: nn.functional, ConvTranspose2d, Upsample, Tril/Triu
- [x] v3.0.0 — Level 6: DataParallel, Float16/BFloat16/Int8, ONNX export, QLinear
- [x] v4.0.0 — Level 7: BatchNorm1d, GroupNorm, TransformerEncoder, SinusoidalPE, optimizer checkpoint
- [x] v5.0.0 — Level 8: TransformerDecoder, full Transformer, StackedLSTM/GRU, ModuleDict, SparseCOO
- [x] **v6.0.0 — First properly working release: 10 critical backward-bugs fixed, numerical-gradient verified, race-safe, in-place ops (49× speedup)**
- [ ] v7.0.0 — Native Float32 compute path (без Float64 promotion в inner loops), torch.compile-like graph capture, `distributed`

---

## Not yet supported

`torch.compile`, JIT, distributed / multi-GPU training (`DistributedDataParallel`), `torch.fft`, `torch.distributions`, second-order autograd (vmap, jacobian, hessian). Native Float32 compute path не готов — `MatMul` внутри аккумулирует в Float64 для стабильности.

---

## Related

Author's related project: **[fa-blackwell-fp8](https://github.com/djeday123/fa-blackwell-fp8)** — production FlashAttention FP8 forward kernel для NVIDIA Blackwell consumer GPUs (sm_120a, RTX PRO 6000). Peak 652 TFLOPS, C library + Go and Python bindings.

---

## License

MIT
