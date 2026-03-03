# GoTorch

[![Go](https://img.shields.io/badge/Go-1.22+-00ADD8?logo=go)](https://go.dev)
[![Tests](https://img.shields.io/badge/tests-221%20passing-brightgreen)](#test-suite)
[![Coverage](https://img.shields.io/badge/PyTorch%20parity-%7E95%25-blue)](#coverage)
[![Version](https://img.shields.io/badge/version-v2.1.0-orange)](#roadmap)

**GoTorch** — PyTorch-подобный фреймворк глубокого обучения на чистом Go с опциональным CUDA GPU backend.

```bash
go get github.com/djeday123/gotorch
```

> Статус: Research / Educational. Цель — полное воспроизведение PyTorch API на Go.

---

## Возможности

| Пакет | Описание | Статус |
|-------|----------|--------|
| `tensor` | N-мерные тензоры, broadcasting, reshape, reduce, linalg | ✅ |
| `autograd` | Automatic differentiation (reverse-mode backprop) | ✅ |
| `nn` | Слои, активации, loss-функции, контейнеры | ✅ |
| `nn/functional` | Stateless (functional) API — F.relu, F.softmax, … | ✅ |
| `optim` | SGD, Adam, AdamW, RMSprop, Adadelta + LR schedulers | ✅ |
| `data` | Dataset interface, TensorDataset, DataLoader (горутины) | ✅ |
| `amp` | GradScaler для mixed precision training | ✅ |
| `cuda` | CUDA GPU backend (CGo): elementwise ops, MatMul cuBLAS, pinned memory | ✅ |

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

### DataLoader

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
tensor.Arange(0, 10, 1)    // [0,1,...,9]
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

// Активации
tensor.ReLU(t)  |  Sigmoid  |  Tanh  |  Softmax(t, dim)

// Reduce
tensor.Sum(t, dim, keepdim)  |  Mean  |  Max  |  Min  |  ArgMax
t.Std(dim, keepdim)  |  t.Var(dim, keepdim)
t.Norm(ord, dim, keepdim)  |  t.Prod(dim, keepdim)

// Linalg
tensor.MatMul(a, b)  |  BatchMatMul  |  Dot  |  Outer

// Конкатенация
tensor.Cat(tensors, dim)  |  Stack(tensors, dim)
t.Split(size, dim)  |  t.Chunk(n, dim)

// Продвинутое индексирование
t.Select(dim, idx)  |  t.Narrow(dim, start, len)
t.Index(dim, indices)  |  t.MaskedSelect(mask)
t.Clamp(min, max)  |  t.Where(cond, other)
t.Gather(dim, index)  |  t.ScatterAdd(dim, index, src)

// Структурные ops (v2.1)
t.Cumsum(dim)  |  t.Cumprod(dim)
t.Tril(diagonal)  |  t.Triu(diagonal)
t.RepeatInterleave(repeats, dim)
t.TopK(k, dim)

// dtype
tensor.NewF32(data, shape)  |  t.Float32()  |  t.Float64()
```

### `autograd`

```go
x := autograd.NewVar(t, requiresGrad)

// Дифференцируемые ops
autograd.Add(a, b)  |  Sub  |  Mul  |  Div  |  MatMul
autograd.AddScalar(a, v)  |  MulScalar  |  PowScalar
autograd.Neg(a)  |  Exp(a)  |  Log(a)
autograd.ReLU(a)  |  Sigmoid(a)  |  Tanh(a)
autograd.Softmax(a, dim)
autograd.Mean(a)  |  Sum(a)  |  SumDim(a, dim)

// Backward
loss.Backward()
loss.BackwardWithGrad(grad)

// Утилиты
x.Grad          // *tensor.Tensor
x.ZeroGrad()
x.Detach()
autograd.NoGrad()     // отключить grad tracking
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
nn.NewConvTranspose2d(inC, outC, k, stride, pad)   // transposed conv

// Pooling
nn.NewMaxPool2d(k, stride, pad)
nn.NewAdaptiveAvgPool2d(outH, outW)

// Upsampling
nn.NewUpsample(scaleFactor, mode)  // mode: "nearest"

// Активации
nn.NewReLU()  |  NewSigmoid()  |  NewTanh()
nn.NewGELU()  |  NewLeakyReLU(slope)  |  NewELU(alpha)
nn.NewSiLU()  |  NewSoftplus(beta)

// Normalization
nn.NewBatchNorm2d(numFeatures)
nn.NewLayerNorm(shape)
nn.NewDropout(p)

// NLP
nn.NewEmbedding(vocabSize, embDim)
nn.NewMultiheadAttention(embDim, numHeads)
nn.NewLSTM(inputSize, hiddenSize, numLayers)
nn.NewGRU(inputSize, hiddenSize)
nn.NewTransformerEncoderLayer(d, nHead, dFF, dropout)
nn.NewTransformerEncoder(layer, numLayers)

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

// Активации
F.ReLU(x)  |  F.Sigmoid(x)  |  F.Tanh(x)
F.GELU(x)  |  F.LeakyReLU(x, slope)  |  F.ELU(x, alpha)  |  F.SiLU(x)

// Нормализация
F.Softmax(x, dim)  |  F.LogSoftmax(x, dim)
F.Dropout(x, p, training)

// Loss
F.MSELoss(pred, target)  |  F.L1Loss  |  F.HuberLoss(pred, target, delta)
F.BCELoss(pred, target)
F.CrossEntropyLoss(logits, targets []int)
F.NLLLoss(logProbs, targets []int)

// Linear
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
```

### `data`

```go
// Dataset
type Dataset interface { Len() int; Get(i int) (*tensor.Tensor, *tensor.Tensor) }
data.NewTensorDataset(x, y *tensor.Tensor)

// DataLoader
cfg := data.LoaderConfig{
    BatchSize: 32, Shuffle: true, NumWorkers: 4, DropLast: false,
}
loader := data.NewDataLoader(dataset, cfg)
loader.HasNext() bool
loader.Next() (*tensor.Tensor, *tensor.Tensor)
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

**221 тест · все зелёные · v2.1.0**

```bash
go test ./...            # все тесты (CPU)
go test ./tensor/...
go test ./autograd/...
go test ./nn/...
go test ./optim/...
go test ./data/...
go test ./amp/...

# GPU тесты (требует CUDA)
CGO_CFLAGS="-I./cuda" \
CGO_LDFLAGS="-L./cuda -lgotorch_cuda -L/usr/local/cuda/lib64 -lcublas -lcudart \
             -Wl,-rpath,$(pwd)/cuda -Wl,-rpath,/usr/local/cuda/lib64" \
go test -tags gpu ./cuda/...
```

| Пакет | Тестов | Что проверяется |
|-------|--------|-----------------|
| `tensor` | 52 | ops, shape, reduce, linalg, advanced indexing, cumsum, tril/triu, gather |
| `autograd` | 18 | chain rule, grad correctness, NoGrad ctx, numerical grad check |
| `nn` | 99 | все слои: Linear, Conv2d/1d, Transformer, LSTM, BatchNorm, losses, functional, summary |
| `optim` | 29 | SGD/Adam/AdamW/RMSprop/Adadelta, grad clipping, LR schedulers |
| `data` | 7 | Dataset, DataLoader batching/shuffle/prefetch/reset |
| `amp` | 10 | GradScaler scaling, inf/nan detection, backoff, growth |
| `cuda` (GPU) | 16 | upload/download, all GPU ops, MatMul accuracy, pinned memory |

Полный список всех 221 теста — см. [DOCS.md](DOCS.md#test-suite).

---

## PyTorch Coverage

| Версия | Тесты | Parity | Что разблокирует |
|--------|-------|--------|-----------------|
| v1.0.0 | ~25 | ~25% | XOR, MLP |
| v1.1.0 | 85 | ~45% | LeNet, VGG-style CNN |
| v1.2.0 | ~110 | ~65% | ResNet, save/load, BatchNorm |
| v1.3.0 | ~140 | ~80% | Transformers, GPT-mini, seq2seq |
| v2.0.0 | 185 | ~92% | DataLoader, AMP, Conv1d |
| **v2.1.0** | **221** | **~95%** | Functional API, ConvTranspose2d, KL/Huber loss |

---

## Примеры

| Пример | Описание |
|--------|----------|
| [`examples/01_xor`](examples/01_xor/main.go) | XOR — базовый MLP, BCELoss |
| [`examples/02_mnist_mlp`](examples/02_mnist_mlp/main.go) | 10-классовый MLP с DataLoader |
| [`examples/03_cnn`](examples/03_cnn/main.go) | CNN с Conv2d + MaxPool2d |
| [`examples/04_lstm_lm`](examples/04_lstm_lm/main.go) | LSTM language model |
| [`examples/05_transformer`](examples/05_transformer/main.go) | Mini Transformer encoder |
| [`examples/06_functional`](examples/06_functional/main.go) | nn.functional API showcase |

---

## Сборка

### CPU (по умолчанию)

```bash
go build ./...
go test ./...
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

---

## Архитектура

```
gotorch/
├── tensor/           CPU тензоры (без CGo)
│   ├── tensor.go     Tensor type, creation, At/Set
│   ├── ops.go        Elementwise ops, activations
│   ├── reduce.go     Sum, Mean, Max, Min, Softmax, Std, Var, Norm
│   ├── shape.go      Reshape, broadcast, Transpose, Cat, Stack
│   ├── index.go      Select, Narrow, Gather, ScatterAdd, Where
│   ├── linalg.go     MatMul, BatchMatMul, Dot, Outer
│   ├── combine.go    Cat, Stack, Split, Chunk
│   ├── extra.go      Full, Linspace, Floor, Ceil, Round, Sign
│   └── ops_level5.go Cumsum, Cumprod, Tril, Triu, RepeatInterleave
│
├── autograd/         Reverse-mode autodiff
│   ├── variable.go   Variable (Tensor + grad + graph)
│   ├── engine.go     Topological sort + backward pass
│   ├── functions.go  Дифференцируемые операции
│   └── no_grad.go    NoGrad / EnableGrad ctx
│
├── nn/               Neural network modules
│   ├── module.go     Module interface
│   ├── linear.go     Linear (y = xW^T + b)
│   ├── sequential.go Sequential container
│   ├── conv.go       Conv2d (im2col)
│   ├── conv1d.go     Conv1d
│   ├── conv_transpose2d.go ConvTranspose2d
│   ├── activations.go      ReLU, Sigmoid, Tanh
│   ├── activations_extra.go GELU, LeakyReLU, ELU, SiLU, Softplus
│   ├── norm.go       BatchNorm2d, LayerNorm
│   ├── dropout.go    Dropout
│   ├── embedding.go  Embedding
│   ├── attention.go  MultiheadAttention
│   ├── rnn.go        LSTM, GRU
│   ├── loss.go       MSE, BCE, CE, NLL, L1, Huber, KLDiv
│   ├── save.go       Save / Load
│   ├── upsample.go   Upsample (nearest)
│   ├── module_list.go ModuleList
│   ├── summary.go    ModelSummary, PrintSummary
│   └── functional/   Stateless functional API
│       └── functional.go
│
├── optim/            Optimizers
│   ├── sgd.go        SGD + momentum
│   ├── adam.go       Adam
│   ├── adamw.go      AdamW
│   ├── rmsprop.go    RMSprop
│   ├── clip.go       ClipGradNorm / ClipGradValue
│   └── scheduler.go  StepLR, CosineAnnealingLR, LinearWarmup
│
├── data/             DataLoader
│   ├── dataset.go    Dataset interface, TensorDataset
│   └── dataloader.go DataLoader (goroutine prefetch)
│
├── amp/              Mixed precision
│   └── scaler.go     GradScaler
│
└── cuda/             GPU backend (build tag: gpu)
    ├── ops.cu        CUDA kernels + cuBLAS
    ├── backend.go    GPUBackend high-level API
    ├── pinned.go     PinnedTensor (zero-copy)
    └── bridge.go     CGo wrappers
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
- [ ] v3.0.0 — Level 6: Multi-GPU DataParallel, float16/bfloat16, int8, ONNX

---

## License

MIT
