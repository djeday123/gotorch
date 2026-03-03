# GoTorch — Полная документация

**v2.1.0 · 221 тест · ~95% PyTorch parity**

---

## Содержание

1. [Установка и сборка](#установка-и-сборка)
2. [Пакет `tensor`](#пакет-tensor)
3. [Пакет `autograd`](#пакет-autograd)
4. [Пакет `nn`](#пакет-nn)
5. [Пакет `nn/functional`](#пакет-nnfunctional)
6. [Пакет `optim`](#пакет-optim)
7. [Пакет `data`](#пакет-data)
8. [Пакет `amp`](#пакет-amp)
9. [Пакет `cuda`](#пакет-cuda)
10. [Test Suite — все 221 теста](#test-suite)

---

## Установка и сборка

```bash
go get github.com/djeday123/gotorch
```

### CPU-only

```bash
go test ./...
```

### GPU (CUDA 12.x + nvcc)

```bash
# 1. Скомпилировать CUDA .so
nvcc -O3 -std=c++14 \
     -gencode arch=compute_89,code=sm_89 \
     -shared -fPIC \
     -o cuda/libgotorch_cuda.so cuda/ops.cu \
     -lcublas

# 2. Тесты GPU
CGO_CFLAGS="-I./cuda" \
CGO_LDFLAGS="-L./cuda -lgotorch_cuda -L/usr/local/cuda/lib64 -lcublas -lcudart \
             -Wl,-rpath,$(pwd)/cuda -Wl,-rpath,/usr/local/cuda/lib64" \
go test -tags gpu ./cuda/...
```

---

## Пакет `tensor`

Чистый Go, без CGo. Реализует N-мерные тензоры с float64 (и float32) хранением.

### Внутреннее устройство

```go
type Tensor struct {
    data    []float64  // flat array (F64)
    f32     []float32  // flat array (F32, если dtype == Float32)
    dtype   DType      // Float64 | Float32
    shape   []int      // размеры по каждому измерению
    strides []int      // шаги (row-major / C-order)
    offset  int        // смещение (для views)
}
```

Strides: для shape `[3, 4]` strides = `[4, 1]`. Элемент `t.At(i, j) = data[offset + i*4 + j]`.

Broadcasting реализован через `Unsqueeze` + итератор — без аллокаций лишних данных.

### Создание тензоров

```go
tensor.Zeros(3, 4)                          // [[0,0,0,0], [0,0,0,0], [0,0,0,0]]
tensor.Ones(3, 4)
tensor.Full(2.5, 2, 3)                      // [[2.5, 2.5, 2.5], ...]
tensor.Rand(3, 4)                           // U[0, 1)
tensor.RandN(3, 4)                          // N(0, 1)
tensor.Arange(0, 5, 1)                      // [0, 1, 2, 3, 4]
tensor.Linspace(0, 1, 5)                    // [0, 0.25, 0.5, 0.75, 1]
tensor.Eye(3)                               // 3×3 identity matrix
tensor.New([]float64{1,2,3,4}, []int{2,2}) // from data
tensor.Scalar(3.14)                         // 0-dim tensor

// Float32
tensor.NewF32([]float32{...}, shape)
tensor.ZerosF32(3, 4)
tensor.OnesF32(3, 4)
tensor.RandNF32(3, 4)
```

### Shape ops

```go
t.Shape() []int                // копия shape
t.Ndim() int                   // len(shape)
t.Size() int                   // total elements
t.At(i, j, ...)  float64      // элемент по индексам
t.Set(val, i, j, ...)          // установить элемент
t.Item() float64               // scalar → float64
t.Data() []float64             // flat copy (логический порядок)

t.Reshape(2, -1)               // -1 = infer; panica если incompatible
t.Flatten() *Tensor            // -> [N]
t.Squeeze() *Tensor            // убрать единичные размеры
t.Unsqueeze(dim) *Tensor       // добавить единичное измерение
t.Transpose(dim0, dim1)        // swap axes
t.T() *Tensor                  // 2D transpose
t.ContiguousCopy() *Tensor     // сделать contiguous (без views)

// Конкатенация
tensor.Cat([a, b, c], dim)     // объединить по dim
tensor.Stack([a, b, c], dim)   // добавить новое измерение
t.Split(size, dim)             // → []Tensor, каждый size по dim
t.Chunk(n, dim)               // → n равных частей
```

### Elementwise ops

```go
// Бинарные (с broadcasting)
tensor.Add(a, b)  |  Sub  |  Mul  |  Div

// Scalar ops
tensor.AddScalar(t, 2.0)
tensor.MulScalar(t, 0.5)
tensor.PowScalar(t, 2.0)        // t^2

// Унарные
tensor.Neg(t)     // -t
tensor.Abs(t)     // |t|
tensor.Exp(t)     // e^t
tensor.Log(t)     // ln(t)
tensor.Sqrt(t)    // √t

// Округление (v2.1)
tensor.Floor(t)
tensor.Ceil(t)
tensor.Round(t)
tensor.Sign(t)    // -1, 0, +1

// Активации
tensor.ReLU(t)
tensor.Sigmoid(t)
tensor.Tanh(t)
tensor.Softmax(t, dim)
tensor.LogSoftmax(t, dim)  // числово стабильный

// Clamp / Where
t.Clamp(min, max float64)
t.Where(cond *Tensor, other *Tensor)  // cond != 0 ? self : other
```

### Reduce ops

```go
tensor.Sum(t, dim, keepdim)    // dim=-1 = все элементы; keepdim сохраняет dim=1
tensor.Mean(t, dim, keepdim)
tensor.Max(t, dim, keepdim)
tensor.Min(t, dim, keepdim)
tensor.ArgMax(t, dim)

// v2.x
t.Std(dim, keepdim)            // стандартное отклонение (несмещённое)
t.Var(dim, keepdim)            // дисперсия
t.Norm(ord, dim, keepdim)      // L1/L2/Linf нормы
t.Prod(dim, keepdim)           // произведение
t.TopK(k, dim)                 // top-k значений
```

### Linalg

```go
tensor.MatMul(a, b)            // [M,K] × [K,N] → [M,N]
tensor.BatchMatMul(a, b)       // [B,M,K] × [B,K,N] → [B,M,N]
tensor.Dot(a, b)               // 1D dot product
tensor.Outer(a, b)             // [M] × [N] → [M,N] outer product
```

### Продвинутое индексирование

```go
t.Select(dim, idx)             // выбрать срез по dim[idx], убрать измерение
t.Narrow(dim, start, len)      // t[..., start:start+len, ...]
t.Index(dim, []int)            // выбрать несколько индексов по dim
t.MaskedSelect(mask)           // выбрать элементы где mask != 0 → [K]

// Gather / Scatter (v2.1)
t.Gather(dim, index)           // out[i,j] = t[index[i,j], j] (для dim=0)
t.ScatterAdd(dim, index, src)  // out := copy(t); out[..., index[i], ...] += src[i]

// Cumulative (v2.1)
t.Cumsum(dim)                  // cumulative sum
t.Cumprod(dim)                 // cumulative product

// Triangular (v2.1)
t.Tril(diagonal)               // нижний треугольник (только 2D)
t.Triu(diagonal)               // верхний треугольник (только 2D)

// Repeat
t.RepeatInterleave(n, dim)     // [1,2,3] × 2 → [1,1,2,2,3,3]
```

---

## Пакет `autograd`

Reverse-mode automatic differentiation. Строит граф вычислений и делает backprop.

### Основные типы

```go
type Variable struct {
    Data         *tensor.Tensor
    Grad         *tensor.Tensor  // накопленный градиент (nil до .Backward())
    RequiresGrad bool
}

type GradFn interface {
    Apply(upstreamGrad *tensor.Tensor) []*tensor.Tensor
}
```

### Создание

```go
x := autograd.NewVar(t, true)   // leaf, требует grad
y := autograd.NewVar(t, false)  // leaf, без grad
```

### Операции (все дифференцируемы)

```go
autograd.Add(a, b)      |  Sub  |  Mul  |  Div  |  MatMul
autograd.AddScalar(a, s)  |  MulScalar(a, s)  |  PowScalar(a, p)
autograd.Neg(a)
autograd.Exp(a)         // e^a, grad = e^a
autograd.Log(a)         // ln(a), grad = 1/a
autograd.ReLU(a)        // max(0,a), grad = 1 if a>0 else 0
autograd.Sigmoid(a)     // σ(a), grad = σ*(1-σ)
autograd.Tanh(a)        // tanh(a), grad = 1-tanh²
autograd.Softmax(a, dim) // grad = identity (используй в loss, не дифференцируй)
autograd.Mean(a)         // mean(a), grad = 1/N
autograd.Sum(a)          // sum(a), grad = 1
autograd.SumDim(a, dim)  // sum along dim
```

### Backward

```go
loss.Backward()                   // scalar loss → заполняет .Grad у всех leaf
loss.BackwardWithGrad(upGrad)     // custom upstream gradient

x.Grad                            // *tensor.Tensor или nil
x.ZeroGrad()                      // x.Grad = nil
x.Detach()                        // → leaf без grad tracking
```

### NoGrad контекст

```go
autograd.NoGrad()          // отключить накопление графа
defer autograd.EnableGrad()

// Или:
autograd.IsGradEnabled() bool
```

### Как пишутся custom операции

```go
type myOpBackward struct {
    savedInput *tensor.Tensor
}

func (f *myOpBackward) Apply(grad *tensor.Tensor) []*tensor.Tensor {
    // вернуть градиент для каждого child Variable
    dX := ... // вычислить dL/dX из grad
    return []*tensor.Tensor{dX}
}

func MyOp(x *autograd.Variable) *autograd.Variable {
    out := ... // вычислить forward
    return autograd.NewResult(out, &myOpBackward{x.Data}, x)
}
```

---

## Пакет `nn`

### Module interface

```go
type Module interface {
    Forward(x *Variable) *Variable
    Parameters() []*Variable
    ZeroGrad()
}

nn.ZeroGradAll(model)   // обходной ZeroGrad для всех параметров
```

### Линейные слои

```go
// Linear: y = x @ W^T + b
// Weight: [outF, inF], Bias: [outF]
l := nn.NewLinear(inFeatures, outFeatures int, bias bool)
l.Weight  // *autograd.Variable
l.Bias    // *autograd.Variable

out := l.Forward(x)  // x: [N, inF] → [N, outF]
```

### Свёрточные слои

```go
// Conv2d (im2col + matmul)
// Input: [N, inC, H, W] → Output: [N, outC, oH, oW]
// oH = (H + 2*pad - kernel) / stride + 1
c := nn.NewConv2d(inC, outC, kernelSize, stride, padding int, bias bool)
c.Weight  // [outC, inC, kH, kW]
c.Bias    // [outC]

// Conv1d (1D convolution)
// Input: [N, inC, L] → Output: [N, outC, oL]
c1 := nn.NewConv1d(inC, outC, kernelSize, stride, padding int, bias bool)

// ConvTranspose2d (transposed convolution / deconvolution)
// Input: [N, inC, H, W] → Output: [N, outC, oH, oW]
// oH = (H-1)*stride - 2*padding + kernelSize
ct := nn.NewConvTranspose2d(inC, outC, kernelSize, stride, padding int)
ct.Weight  // [inC, outC, kH, kW]
```

### Pooling

```go
// MaxPool2d
// oH = (H + 2*pad - kernel) / stride + 1
mp := nn.NewMaxPool2d(kernelSize, stride, padding int)

// AdaptiveAvgPool2d — всегда производит outH × outW вывод
ap := nn.NewAdaptiveAvgPool2d(outH, outW int)
```

### Upsampling

```go
// Upsample: [N, C, H, W] → [N, C, H*scale, W*scale]
up := nn.NewUpsample(scaleFactor int, mode string)
// mode: "nearest" (поддерживается), "bilinear" (TODO)
```

### Активации

```go
nn.NewReLU()
nn.NewSigmoid()
nn.NewTanh()
nn.NewGELU()                      // Gaussian Error Linear Unit
nn.NewLeakyReLU(negativeSlope)    // slope при x<0
nn.NewELU(alpha)                  // alpha*(exp(x)-1) при x<0
nn.NewSiLU()                      // x * sigmoid(x)  (Swish)
nn.NewSoftplus(beta)              // (1/beta)*log(1+exp(beta*x))
```

### Normalization

```go
// BatchNorm2d: нормализует по [N, H, W] для каждого канала
// Поддерживает train/eval режим с running mean/var
bn := nn.NewBatchNorm2d(numFeatures int)
bn.Train()  |  bn.Eval()

// LayerNorm: нормализует по normalizedShape
ln := nn.NewLayerNorm(normalizedShape []int)

// Dropout
d := nn.NewDropout(p float64)   // p = вероятность зануления
d.Train()  |  d.Eval()
```

### NLP

```go
// Embedding: lookup table [vocabSize, embDim]
emb := nn.NewEmbedding(vocabSize, embDim int)
// Input: []int токены → Output: [len(tokens), embDim]

// MultiheadAttention: scaled dot-product attention
mha := nn.NewMultiheadAttention(embedDim, numHeads int)
// Input: query [N, seqLen, embDim], key, value — same shape
out := mha.Forward(query)  // (Q=K=V=query для self-attention)

// LSTM
lstm := nn.NewLSTM(inputSize, hiddenSize, numLayers int)
// Input: [seqLen, batchSize, inputSize]
// Output: [seqLen, batchSize, hiddenSize]

// GRU
gru := nn.NewGRU(inputSize, hiddenSize int)

// TransformerEncoderLayer
tel := nn.NewTransformerEncoderLayer(d_model, nHead, dim_feedforward, dropout)

// TransformerEncoder
te := nn.NewTransformerEncoder(layer, numLayers)
```

### Loss функции

```go
// MSELoss: mean((pred-target)^2)
nn.MSELoss(pred, target *Variable) *Variable

// BCELoss: -mean(t*log(p) + (1-t)*log(1-p))
// pred должен быть в (0,1) — применить Sigmoid перед этим
nn.BCELoss(pred, target *Variable) *Variable

// CrossEntropyLoss: LogSoftmax + NLLLoss (численно стабильный)
// logits: [N, C], targets: []int длиной N
nn.CrossEntropyLoss(logits *Variable, targets []int) *Variable

// NLLLoss: negative log-likelihood
// logProbs: [N, C] (вывод LogSoftmax), targets: []int
nn.NLLLoss(logProbs *Variable, targets []int) *Variable

// L1Loss: mean(|pred-target|)  (MAE)
nn.L1Loss(pred, target *Variable) *Variable

// HuberLoss (Smooth L1):
// |x|<=delta: 0.5*x^2/delta;  |x|>delta: |x| - 0.5*delta
nn.HuberLoss(pred, target *Variable, delta float64) *Variable

// KLDivLoss: mean(target*(log(target)-input))
// input = log-probs (из LogSoftmax), target = вероятности
nn.KLDivLoss(input, target *Variable) *Variable
```

### Контейнеры

```go
// Sequential: цепочка слоёв
model := nn.NewSequential(
    nn.NewLinear(128, 64, true),
    nn.NewReLU(),
    nn.NewLinear(64, 10, true),
)
out := model.Forward(x)
params := model.Parameters()

// ModuleList: список модулей с доступом по индексу
ml := nn.NewModuleList(layer1, layer2, layer3)
ml.Append(layer4)
m := ml.Get(2)        // Module по индексу
ml.Len()
ml.Forward(x)          // последовательный forward
ml.Parameters()        // все параметры
```

### Save / Load

```go
err := nn.Save(model, "model.json")
err = nn.Load(model, "model.json")  // восстанавливает веса (shapes должны совпадать)
```

### Model Summary

```go
s := nn.Summary(model)
// s.TotalParams     int
// s.TrainableParams int
// s.Layers          []LayerInfo{{Name, Type, ParamCount}}

nn.PrintSummary(model)
// Name                           Type                           Params
// ─────────────────────────────────────────────────────────────────────
// model                          *nn.Sequential                    ...
```

---

## Пакет `nn/functional`

Stateless функции — аналог `torch.nn.functional`.  
Не имеют состояния (нет обучаемых параметров), работают напрямую с `*autograd.Variable`.

```go
import F "github.com/djeday123/gotorch/nn/functional"
```

### Активации

```go
F.ReLU(x)
F.Sigmoid(x)
F.Tanh(x)
F.GELU(x)
F.LeakyReLU(x, negativeSlope float64)
F.ELU(x, alpha float64)
F.SiLU(x)
```

### Нормализация

```go
F.Softmax(x, dim int)          // вероятности, sum=1 по dim
F.LogSoftmax(x, dim int)       // log(softmax), численно стабильный
F.Dropout(x, p float64, training bool)
```

### Linear

```go
// weight: [outF, inF], bias: [outF] или nil
F.Linear(x, weight, bias *Variable) *Variable
// эквивалент: x @ weight^T + bias
```

### Loss функции

```go
F.MSELoss(pred, target)
F.L1Loss(pred, target)
F.HuberLoss(pred, target, delta float64)
F.BCELoss(pred, target)
F.CrossEntropyLoss(logits, targets []int)
F.NLLLoss(logProbs, targets []int)
```

---

## Пакет `optim`

### SGD

```go
opt := optim.NewSGD(params []*Variable, lr, momentum float64)
// momentum=0 → vanilla SGD
// update: v = momentum*v - lr*grad; w += v
```

### Adam

```go
opt := optim.NewAdam(params, lr, beta1, beta2, eps float64)
// стандартные значения: lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8
```

### AdamW

```go
opt := optim.NewAdamW(params, lr, beta1, beta2, eps, weightDecay float64)
// decoupled weight decay (Loshchilov & Hutter 2017)
// weightDecay обычно 0.01 или 0.1
```

### RMSprop

```go
opt := optim.NewRMSprop(params, lr, alpha, eps, momentum, weightDecay float64)
// alpha=0.99, eps=1e-8, momentum=0, weightDecay=0
```

### Adadelta

```go
opt := optim.NewAdadelta(params, lr, rho, eps float64)
// rho=0.9, eps=1e-6
```

### Gradient Clipping

```go
norm := optim.ClipGradNorm(params []*Variable, maxNorm float64) float64
// обрезает L2-норму всех градиентов до maxNorm, возвращает норму до клиппинга

optim.ClipGradValue(params []*Variable, clipValue float64)
// клиппирует каждый градиент в [-clipValue, +clipValue]
```

### LR Schedulers

```go
// StepLR: lr *= gamma каждые stepSize шагов
s := optim.NewStepLR(optimizer, stepSize int, gamma float64)

// CosineAnnealingLR: lr = lrMin + 0.5*(lrMax-lrMin)*(1+cos(pi*t/T))
s := optim.NewCosineAnnealingLR(optimizer, tMax int)

// LinearWarmup: линейный разогрев за warmupSteps шагов
s := optim.NewLinearWarmup(optimizer, warmupSteps int)

s.Step()         // вызывать каждую эпоху (или каждый шаг для warmup)
s.GetLR()        // текущий lr
s.SetLR(lr)      // вручную установить lr
```

---

## Пакет `data`

### Dataset interface

```go
type Dataset interface {
    Len() int
    Get(i int) (*tensor.Tensor, *tensor.Tensor)  // (features, label)
}
```

### TensorDataset

```go
ds := data.NewTensorDataset(x, y *tensor.Tensor)
// x: [N, ...], y: [N, ...] или nil
ds.Len()          // N
xI, yI := ds.Get(i)
```

### DataLoader

```go
cfg := data.LoaderConfig{
    BatchSize:  32,
    Shuffle:    true,    // перемешивать каждую эпоху
    NumWorkers: 4,       // горутины для prefetch
    DropLast:   false,   // отбросить последний неполный батч
}
loader := data.NewDataLoader(ds, cfg)

for loader.HasNext() {
    xBatch, yBatch := loader.Next()
    // xBatch: [BatchSize, ...], yBatch: [BatchSize, ...]
}
loader.Reset()  // перемотать на начало (+ перемешать если Shuffle)
```

---

## Пакет `amp`

GradScaler для mixed precision training. Масштабирует loss перед backward, чтобы предотвратить underflow в float16.

```go
scaler := amp.NewGradScaler(
    initScale      float64,  // начальный scale factor (напр. 1024)
    growthFactor   float64,  // умножить scale на этот после growthInterval успешных шагов (2.0)
    backoffFactor  float64,  // делить scale на это при inf/nan (0.5)
    growthInterval int,      // шагов без inf/nan до роста scale (100)
    enabled        bool,     // включён ли scaler
)

// Training loop
scaledLoss := scaler.ScaleLoss(loss)    // loss * scale
scaledLoss.Backward()

if scaler.Step(optimizer) {             // false если inf/nan в gradients
    // обновление произошло
}
scaler.Update()                         // обновить scale factor

// State
scaler.GetScale() float64
scaler.IsEnabled() bool
```

---

## Пакет `cuda`

CGo-обёртка над CUDA runtime и cuBLAS. Компилируется только с тегом `gpu`.

### Detect GPU

```go
// DetectGPU() доступен всегда (без тега gpu)
if cuda.DetectGPU() {
    fmt.Println(cuda.DeviceInfo())
    // "NVIDIA GeForce RTX 4090 (sm_89), 24.8/25.3 GB"
}
```

### GPUBackend

```go
b, err := cuda.NewGPUBackend(device int)  // device=0 обычно
defer b.Close()

// Transfer
gA, err := b.Upload(t *tensor.Tensor)   // CPU → GPU
tCPU := b.Download(gA)                  // GPU → CPU
defer gA.Free()

// Elementwise ops (результат — новый GPUTensor)
gC, _ := b.Add(gA, gB)
gC, _ := b.Sub(gA, gB)
gC, _ := b.Mul(gA, gB)
gC, _ := b.Div(gA, gB)
gC, _ := b.AddScalar(gA, 2.0)
gC, _ := b.MulScalar(gA, 0.5)
gC, _ := b.Neg(gA)
gC, _ := b.ReLU(gA)
gC, _ := b.Sigmoid(gA)
gC, _ := b.Tanh(gA)
gC, _ := b.Exp(gA)
gC, _ := b.Log(gA)

// Reduce
sum, _  := b.Sum(gA)    // float64
mean, _ := b.Mean(gA)   // float64

// Softmax
gS, _ := b.Softmax(gA, rows, cols int)

// Matrix multiplication (cuBLAS DGEMM)
gOut, _ := b.MatMul(gA, gB)  // [M,K] × [K,N] → [M,N]
```

### PinnedTensor (zero-copy)

Page-locked память — CPU и GPU работают с одним буфером без staging copy.

```go
p, err := cuda.NewPinnedTensor(1024, 1024)  // pinned CPU alloc
defer p.Free()

// Записать данные (прямо в pinned buffer)
copy(p.Slice(), myData)

// Async DMA → GPU
gT, err := p.ToGPU()
defer gT.Free()

// ... GPU compute ...

// Async DMA ← GPU
err = p.FromGPU(gT)
result := p.Slice()  // читать напрямую

// Конвертировать в обычный Tensor
t := p.ToCPUTensor()
```

**Производительность (RTX 4090, 8 MB):**

| Операция | Скорость | vs PyTorch |
|----------|----------|------------|
| Pinned H2D | ~24.6 GB/s | идентично |
| Pageable H2D | ~21.7 GB/s | идентично |
| Pinned D2H | ~25.7 GB/s | идентично |
| Pageable D2H | ~16.0 GB/s | идентично |

---

## Test Suite

**Все 221 тест — упорядочены по пакетам**

### `autograd` — 18 тестов

| Тест | Что проверяется |
|------|-----------------|
| `TestAddGrad` | d/dx(x+y)=1 для обоих входов |
| `TestSubGrad` | d/dx(x-y)=1, d/dy=-1 |
| `TestMulGrad` | произведение правило: d/dx(x·y)=y |
| `TestMatMulGrad` | форма градиентов матричного умножения |
| `TestNegGrad` | d/dx(-x)=-1 |
| `TestExpGrad` | d/dx(exp(x))=exp(x) |
| `TestLogGrad` | d/dx(log(x))=1/x |
| `TestReLUGrad` | grad=1 при x>0, grad=0 при x≤0 |
| `TestSigmoidGrad` | d/dx σ(x) = σ(x)*(1-σ(x)) |
| `TestTanhGrad` | d/dx tanh(x) = 1 - tanh²(x) |
| `TestPowScalarGrad` | d/dx x^n = n*x^(n-1) |
| `TestMeanGrad` | d/dx mean(x) = 1/N для каждого элемента |
| `TestChainRule` | цепочка: d/dx sigmoid(x²+1) |
| `TestNoGradLeaf` | leaf без grad не накапливает .Grad |
| `TestMatMulGradNumerical` | числовая проверка градиента MatMul |
| `TestNoGrad` | NoGrad() → leaf без grad_fn |
| `TestNoGradNested` | вложенные NoGrad/EnableGrad восстанавливают состояние |
| `TestNoGradNoBackward` | данные корректны внутри NoGrad, нет паники |

### `tensor` — 52 теста

| Тест | Что проверяется |
|------|-----------------|
| `TestZeros` | форма и значения tensor.Zeros |
| `TestOnes` | форма и значения tensor.Ones |
| `TestArange` | корректные шаги и границы |
| `TestEye` | единичная матрица |
| `TestFull` | заполнение константой |
| `TestLinspace` | равномерное распределение (включая конечную точку) |
| `TestLinspaceSingle` | n=1 → только start |
| `TestAtSet` | At() и Set() по многомерным индексам |
| `TestReshape` | shape изменяется, данные не меняются |
| `TestReshapeInferDim` | -1 в shape корректно выводится |
| `TestTranspose` | swap осей, stride корректен |
| `TestFlatten` | → одномерный тензор |
| `TestSqueeze` | удаляет единичные размеры |
| `TestUnsqueeze` | добавляет единичный размер |
| `TestContiguousCopy` | правильная копия для non-contiguous |
| `TestAddBroadcast` | broadcasting по разным размерам |
| `TestAddScalar` | скалярное прибавление |
| `TestMatMul` | матричное умножение (форма и значения) |
| `TestDot` | 1D dot product |
| `TestOuter` | outer product |
| `TestSum` | sum по dim и всем элементам |
| `TestMean` | mean значение |
| `TestSoftmax` | сумма по dim = 1 |
| `TestSoftmaxNumericalStability` | нет overflow при больших входах |
| `TestReLU` | max(0,x) |
| `TestArgMax` | индекс максимума |
| `TestFloat32Creation` | NewF32, ZerosF32 |
| `TestFloat32Item` | Item() для Float32 |
| `TestFloat32Ops` | операции сохраняют float32 dtype |
| `TestFloat32Cast` | Float32() / Float64() конвертация |
| `TestFloat32RandN` | RandNF32 правильная форма |
| `TestCat1D` | concat по dim=0 для 1D |
| `TestCat2DDim0` | concat по dim=0 для 2D |
| `TestCat2DDim1` | concat по dim=1 для 2D |
| `TestStack` | stack добавляет новое измерение |
| `TestStackDim1` | stack по dim=1 |
| `TestSplit` | разбить на равные части |
| `TestChunk` | chunk с остатком |
| `TestSelect` | срез убирает измерение |
| `TestNarrow` | подтензор без копии |
| `TestIndex` | выборка нескольких индексов |
| `TestMaskedSelect` | элементы по boolean mask |
| `TestClamp` | зажать в [min, max] |
| `TestWhere` | условный выбор |
| `TestFloor` | поэлементный floor |
| `TestCeil` | поэлементный ceil |
| `TestRound` | поэлементный round |
| `TestSign` | знак элемента (-1,0,+1) |
| `TestStd` | std несмещённый |
| `TestVar` | var несмещённый |
| `TestVarBiased` | var смещённый (ddof=0) |
| `TestNormL1` | L1 норма |
| `TestNormL2` | L2 норма (Frobenius) |
| `TestNormInf` | L∞ норма |
| `TestNormDim` | норма по dimension |
| `TestProdAll` | произведение всех элементов |
| `TestProdDim` | произведение по dim |
| `TestTopK` | top-k значений и индексов |
| `TestCumsum` | cumulative sum 1D |
| `TestCumsumMatrix` | cumulative sum 2D |
| `TestCumsumDim1` | cumsum по dim=1 |
| `TestCumsumDim0` | cumsum по dim=0 |
| `TestCumprodDim1` | cumulative product |
| `TestGatherDim0` | gather по dim=0 |
| `TestGatherDim1` | gather по dim=1 |
| `TestScatterAdd` | scatter-add в правильные позиции |
| `TestTril` | нижний треугольник (main diag) |
| `TestTrilPositiveDiag` | нижний треугольник (+1 offset) |
| `TestTriu` | верхний треугольник |
| `TestRepeatInterleaveDim0` | [1,2,3]→[1,1,2,2,3,3] |
| `TestRepeatInterleave2D` | по dim=1 для 2D |

### `nn` — 99 тестов

**Линейные**

| Тест | Что проверяется |
|------|-----------------|
| `TestLinearForwardShape` | [N,in] → [N,out] |
| `TestLinearParameters` | возвращает weight + bias (2 params) |
| `TestLinearNoBias` | без bias — 1 param |
| `TestLinearGrad` | градиенты текут через Linear |
| `TestLinearZeroGrad` | ZeroGrad очищает накопленные grad |

**Sequential**

| Тест | Что проверяется |
|------|-----------------|
| `TestSequentialForward` | многослойный forward pass |
| `TestSequentialParameters` | собирает params из всех слоёв |

**Loss**

| Тест | Что проверяется |
|------|-----------------|
| `TestMSELoss` | MSELoss строит grad граф |
| `TestMSELossValue` | MSE = mean((pred-target)²) |
| `TestBCELoss` | binary cross-entropy значение |
| `TestCrossEntropyLoss` | softmax + NLL, правильный класс ↓ loss |
| `TestL1LossLayer` | L1Loss = mean(|pred-target|), grad sign |
| `TestHuberLossLayer` | HuberLoss: значение + backward |
| `TestNLLLossLayer` | NLLLoss = -mean(logProbs[target]) |
| `TestKLDivLossLayer` | KL(p\|\|p)=0; KL разных > 0 |

**Conv**

| Тест | Что проверяется |
|------|-----------------|
| `TestConv2dShape` | правильная форма вывода |
| `TestConv2dShapeWithPadding` | с padding форма не уменьшается |
| `TestConv2dNoBias` | без bias: 1 параметр |
| `TestConv2dGradNumerical` | численная проверка backward |
| `TestConv1dShape` | [N,C,L] → правильный оL |
| `TestConv1dWithPadding` | padding сохраняет длину |
| `TestConv1dParameters` | weight + bias |
| `TestConv1dBackward` | граф не паникует |
| `TestConvTranspose2dShape` | oH = (H-1)*s - 2*p + k |
| `TestConvTranspose2dBackward` | grad на input и weight |
| `TestConvTranspose2dStrided` | stride=2 → правильный размер |

**Pooling**

| Тест | Что проверяется |
|------|-----------------|
| `TestMaxPool2dShape` | правильный вывод |
| `TestMaxPool2dValues` | max в каждом окне |
| `TestMaxPool2dGrad` | grad через max-маску |
| `TestAdaptiveAvgPool2dShape` | всегда [N, C, outH, outW] |
| `TestAdaptiveAvgPool2dValues` | среднее по пулинг-регионам |
| `TestAdaptiveAvgPool2dBackward` | grad текут |

**Upsample**

| Тест | Что проверяется |
|------|-----------------|
| `TestUpsampleNearest` | форма = input * scale, значения дублируются |
| `TestUpsampleBackward` | grad на input |

**Активации**

| Тест | Что проверяется |
|------|-----------------|
| `TestActivationModules` | ReLU/Sigmoid/Tanh как Modules (нет params) |
| `TestGELUShape` | форма сохраняется |
| `TestGELUValues` | GELU(0)=0, GELU(1)≈0.841 |
| `TestGELUBackward` | grad не nil |
| `TestLeakyReLUPositive` | x>0 → x |
| `TestLeakyReLUNegative` | x<0 → slope*x |
| `TestLeakyReLUBackward` | правильный grad |
| `TestELUPositive` | x>0 → x |
| `TestELUNegative` | x<0 → alpha*(exp(x)-1) |
| `TestELUBackward` | grad не nil |
| `TestSiLUZero` | SiLU(0)=0 |
| `TestSiLUPositive` | SiLU(2)≈1.762 |
| `TestSiLUBackward` | grad не nil |
| `TestSoftplusPositive` | Softplus(x)>0 для всех x |
| `TestSoftplusApproxReLU` | Softplus(x) ≈ ReLU(x) при больших x |

**Normalization**

| Тест | Что проверяется |
|------|-----------------|
| `TestBatchNorm2dShape` | форма не меняется |
| `TestBatchNorm2dNormalized` | mean≈0, std≈1 после BN |
| `TestBatchNorm2dRunningStats` | running_mean обновляется в train |
| `TestLayerNormShape` | форма не меняется |
| `TestLayerNormNormalized` | нормализация по последним dims |
| `TestDropoutShape` | форма сохраняется |
| `TestDropoutScaling` | surviving элементы масштабируются на 1/(1-p) |
| `TestDropoutEvalPassthrough` | в eval режиме → identity |

**NLP**

| Тест | Что проверяется |
|------|-----------------|
| `TestEmbeddingShape` | [N, seqLen, embDim] |
| `TestEmbeddingValues` | правильные строки из lookup table |
| `TestEmbeddingParameters` | 1 параметр (weight) |
| `TestMHAShape` | [N, seq, embDim] → [N, seq, embDim] |
| `TestMHAParameters` | 4 weight матрицы |
| `TestCausalMask` | треугольная маска для авторегрессии |
| `TestLSTMShape` | [seqLen, B, hidden] |
| `TestLSTMStateCarried` | hidden state переносится |
| `TestGRUShape` | [seqLen, B, hidden] |
| `TestGRUParameters` | корректное число параметров |
| `TestTransformerEncoderLayerShape` | [N, seq, d] → [N, seq, d] |
| `TestTransformerEncoderLayerParameters` | params > 0 |

**Misc**

| Тест | Что проверяется |
|------|-----------------|
| `TestFlatten2d` | [N,C,H,W] → [N, C*H*W] |
| `TestSaveLoad` | веса сохраняются и восстанавливаются |
| `TestLoadShapeMismatch` | паника при несовпадении shape |
| `TestXOR` | обучение XOR до loss < 0.01 за ≤1000 эпох |
| `TestModuleList` | Parameters, Forward, Len |
| `TestModuleListAppend` | добавление модуля |
| `TestModelSummary` | TotalParams совпадает с ожидаемым |
| `TestModelSummaryPrint` | нет паники |

**Functional**

| Тест | Что проверяется |
|------|-----------------|
| `TestFunctionalReLU` | F.ReLU forward + grad |
| `TestFunctionalGELU` | F.GELU значения |
| `TestFunctionalLeakyReLU` | F.LeakyReLU ветки |
| `TestFunctionalSiLU` | F.SiLU значения |
| `TestFunctionalSoftmax` | сумма = 1 по строкам |
| `TestFunctionalLogSoftmax` | exp(out) суммируется в 1 |
| `TestFunctionalDropout` | eval mode = identity |
| `TestFunctionalMSELoss` | совпадает с nn.MSELoss |
| `TestFunctionalL1Loss` | = mean(|pred-target|) |
| `TestFunctionalHuberLoss` | L2-режим близко, L1-режим далеко |
| `TestFunctionalCrossEntropy` | совпадает с nn.CrossEntropyLoss |
| `TestFunctionalNLLLoss` | = -mean(logP[target]) |

### `optim` — 29 тестов

| Тест | Что проверяется |
|------|-----------------|
| `TestSGDStep` | вес меняется в направлении -lr*grad |
| `TestSGDMomentum` | momentum ускоряет шаг |
| `TestSGDZeroGrad` | ZeroGrad обнуляет grad |
| `TestAdamStep` | вес обновляется с Adam |
| `TestAdamMultipleSteps` | несколько шагов — grad обнуляется |
| `TestAdamWStep` | AdamW шаг с weight decay |
| `TestAdamWWeightDecay` | weight decay уменьшает норму |
| `TestRMSpropStep` | RMSprop обновляет вес |
| `TestRMSpropDecreasing` | loss убывает на простой задаче |
| `TestRMSpropMomentum` | momentum вариант |
| `TestRMSpropWeightDecay` | weight decay работает |
| `TestRMSpropZeroGrad` | ZeroGrad |
| `TestRMSpropGetSetLR` | GetLR / SetLR |
| `TestAdadeltaStep` | Adadelta обновляет вес |
| `TestAdadeltaDecreasing` | loss убывает |
| `TestAdadeltaZeroGrad` | ZeroGrad |
| `TestAdadeltaGetSetLR` | GetLR / SetLR |
| `TestClipGradNorm` | нормы градиентов обрезаются |
| `TestClipGradNormBelowThreshold` | если norm < maxNorm → без изменений |
| `TestClipGradValue` | каждый элемент в [-clip, clip] |
| `TestStepLR` | lr уменьшается каждые stepSize шагов |
| `TestCosineAnnealingLR` | lr следует косинусному расписанию |
| `TestLinearWarmup` | lr линейно растёт до базового значения |

### `data` — 7 тестов

| Тест | Что проверяется |
|------|-----------------|
| `TestTensorDatasetLen` | Len() = N |
| `TestTensorDatasetGet` | Get(i) возвращает правильный элемент |
| `TestTensorDatasetGetBounds` | паника при i >= Len() |
| `TestTensorDatasetNilTarget` | работает без y (nil) |
| `TestDataLoaderBatches` | правильные batch размеры |
| `TestDataLoaderNumBatches` | число батчей = ceil(N/BatchSize) |
| `TestDataLoaderShuffle` | перемешивает индексы |
| `TestDataLoaderDropLast` | отбрасывает последний неполный батч |
| `TestDataLoaderPrefetch` | prefetch с NumWorkers > 0 |
| `TestDataLoaderReset` | Reset() позволяет итерировать снова |

### `amp` — 10 тестов

| Тест | Что проверяется |
|------|-----------------|
| `TestGradScalerGetScale` | GetScale() возвращает initScale |
| `TestGradScalerIsEnabled` | IsEnabled() = enabled flag |
| `TestGradScalerScaleLoss` | loss * scale |
| `TestGradScalerStepFinite` | Step() = true при конечных grad |
| `TestGradScalerStepInf` | Step() = false при inf в grad |
| `TestGradScalerStepNaN` | Step() = false при nan в grad |
| `TestGradScalerGrowthOnGoodIters` | scale растёт после growthInterval шагов |
| `TestGradScalerBackoffOnInf` | scale делится на backoffFactor при inf |
| `TestGradScalerDisabled` | disabled → Step() всегда true, scale=1 |
| `TestGradScalerEndToEnd` | полный training loop с scaler |

### `cuda` — 16 тестов (требуют тег `gpu`)

| Тест | Что проверяется |
|------|-----------------|
| `TestGPUDetect` | DetectGPU() = true |
| `TestGPUDeviceInfo` | строка содержит "NVIDIA" |
| `TestGPUUploadDownload` | round-trip: CPU→GPU→CPU без потерь |
| `TestGPUAdd` | GPU Add совпадает с CPU |
| `TestGPUSub` | GPU Sub |
| `TestGPUMul` | GPU Mul |
| `TestGPUDiv` | GPU Div |
| `TestGPUAddScalar` | GPU scalar add |
| `TestGPUMulScalar` | GPU scalar mul |
| `TestGPUReLU` | GPU ReLU |
| `TestGPUSigmoid` | GPU Sigmoid |
| `TestGPUSoftmax` | GPU Softmax, row sums = 1 |
| `TestGPUMatMul` | cuBLAS DGEMM vs CPU (< 1e-6 err) |
| `TestGPUSumMean` | GPU Sum и Mean |
| `TestBenchmarkPinnedH2D` | Pinned Transfer: > 10 GB/s |
| `TestBenchmarkPageableH2D` | Pageable Transfer: > 5 GB/s |
| `TestBenchmarkPinnedD2H` | Pinned D2H: > 10 GB/s |
| `TestBenchmarkPageableD2H` | Pageable D2H: > 5 GB/s |

---

*Последнее обновление: v2.1.0 — 2026-03-03*
