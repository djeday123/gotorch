# GoTorch

A PyTorch-inspired deep learning framework written in pure Go, with an optional CUDA GPU backend.

> **Status:** Research / Educational — not production-ready. Built to explore deep learning primitives in Go.

```
go get github.com/your-org/gotorch_v1
```

---

## Features

| Module | Description |
|---|---|
| `tensor` | N-dimensional float64 tensors with broadcasting, reshape, reduce |
| `autograd` | Automatic differentiation — reverse-mode (backprop) |
| `nn` | Neural network layers, activations, loss functions |
| `optim` | SGD (with momentum), Adam |
| `cuda` | CUDA GPU backend via CGo — elementwise ops, MatMul (cuBLAS), pinned memory |

---

## Quick Start

### CPU — XOR in 10 lines

```go
package main

import (
    "fmt"
    "gotorch_v1/autograd"
    "gotorch_v1/nn"
    "gotorch_v1/optim"
    "gotorch_v1/tensor"
)

func main() {
    // Dataset
    X := autograd.NewVar(tensor.New([]float64{0,0, 0,1, 1,0, 1,1}, []int{4, 2}), false)
    Y := autograd.NewVar(tensor.New([]float64{0, 1, 1, 0}, []int{4, 1}), false)

    // Model: 2 → 8 → 1
    model := nn.NewSequential(
        nn.NewLinear(2, 8, true),
        nn.NewReLU(),
        nn.NewLinear(8, 1, true),
        nn.NewSigmoid(),
    )
    opt := optim.NewAdam(model.Parameters(), 0.01, 0.9, 0.999, 1e-8)

    // Train
    for epoch := 0; epoch < 1000; epoch++ {
        model.ZeroGrad()
        pred := model.Forward(X)
        loss := nn.BCELoss(pred, Y)
        loss.Backward()
        opt.Step()
        if epoch%100 == 0 {
            fmt.Printf("epoch %4d  loss %.4f\n", epoch, loss.Data().Item())
        }
    }
}
```

### GPU — upload tensor and run MatMul

```go
//go:build gpu

package main

import (
    "fmt"
    "gotorch_v1/cuda"
    "gotorch_v1/tensor"
)

func main() {
    fmt.Println(cuda.DeviceInfo())

    b, _ := cuda.NewGPUBackend(0)
    defer b.Close()

    a := tensor.Rand(512, 512)
    x := tensor.Rand(512, 512)

    ga, _ := b.Upload(a)
    gx, _ := b.Upload(x)
    defer ga.Free()
    defer gx.Free()

    out, _ := b.MatMul(ga, gx)
    defer out.Free()

    result := b.Download(out)
    fmt.Println("result shape:", result.Shape())
}
```

### GPU — zero-copy with PinnedTensor

```go
//go:build gpu

// Allocate page-locked (pinned) CPU memory.
// The same buffer is accessible by both CPU and GPU — no staging copy.
p, _ := cuda.NewPinnedTensor(1024, 1024)
defer p.Free()

// Write via Go slice — this IS the pinned buffer, no copy
copy(p.Slice(), myData)

// Async DMA to GPU — CPU is not involved in the transfer
g, _ := p.ToGPU()
defer g.Free()

// ... compute on GPU ...

// Async DMA back
p.FromGPU(g)
result := p.Slice() // read result directly
```

---

## Build

### CPU only (default)

```bash
go build ./...
go test ./...
```

### GPU (requires CUDA 12.x + nvcc)

```bash
# 1. Compile CUDA kernels
make build-gpu CUDA_ARCHS="89"   # sm_89 = RTX 4090 / L40

# 2. Build & test
CGO_CFLAGS="-I./cuda" \
CGO_LDFLAGS="-L./cuda -lgotorch_cuda -L/usr/local/cuda/lib64 -lcublas -lcudart \
             -Wl,-rpath,$(pwd)/cuda -Wl,-rpath,/usr/local/cuda/lib64" \
go test -tags gpu ./cuda/...
```

**Supported GPU architectures:**

| CUDA_ARCHS | GPUs |
|---|---|
| `80` | A100, A30 |
| `86` | RTX 3090, RTX 3080, A40 |
| `89` | RTX 4090, L40, L40S |
| `90` | H100, H200 |

---

## API Reference

### `tensor` — Core Tensor

```go
// Creation
tensor.Zeros(3, 4)
tensor.Ones(3, 4)
tensor.Rand(3, 4)       // uniform [0, 1)
tensor.RandN(3, 4)      // normal(0, 1)
tensor.Arange(0, 10, 1) // [0, 1, ..., 9]
tensor.Eye(4)
tensor.New(data []float64, shape []int)
tensor.Scalar(3.14)

// Shape
t.Shape()               // []int
t.Ndim()
t.Size()                // total elements
t.Reshape(2, -1)        // -1 = infer
t.Flatten()
t.Squeeze()
t.Unsqueeze(0)
t.Transpose(0, 1)
t.T()                   // 2D transpose

// Indexing
t.At(i, j)
t.Set(val, i, j)
t.Item()                // scalar → float64
t.Data()                // []float64 flat view

// Elementwise (with broadcasting)
tensor.Add(a, b)
tensor.Sub(a, b)
tensor.Mul(a, b)
tensor.Div(a, b)
tensor.AddScalar(t, 1.0)
tensor.MulScalar(t, 2.0)
tensor.PowScalar(t, 2.0)
tensor.Neg(t)
tensor.Abs(t)
tensor.Exp(t)
tensor.Log(t)
tensor.Sqrt(t)

// Activations
tensor.ReLU(t)
tensor.Sigmoid(t)
tensor.Tanh(t)
tensor.Softmax(t, dim)
tensor.LogSoftmax(t, dim)

// Reduction
tensor.Sum(t, dim, keepdim)    // dim=-1 = all
tensor.Mean(t, dim, keepdim)
tensor.Max(t, dim, keepdim)
tensor.Min(t, dim, keepdim)
tensor.ArgMax(t, dim)

// Linear algebra
tensor.MatMul(a, b)            // 2D matrix multiply
tensor.BatchMatMul(a, b)       // [B, M, K] × [B, K, N]
tensor.Dot(a, b)               // 1D dot product
tensor.Outer(a, b)             // outer product
```

### `autograd` — Automatic Differentiation

```go
// Create variables
x := autograd.NewVar(t, true)     // requiresGrad = true
y := autograd.NewVar(t, false)

// Differentiable ops
z := autograd.Add(x, y)
z := autograd.Sub(x, y)
z := autograd.Mul(x, y)
z := autograd.Div(x, y)
z := autograd.MatMul(x, y)
z := autograd.Neg(x)
z := autograd.Exp(x)
z := autograd.Log(x)
z := autograd.ReLU(x)
z := autograd.Sigmoid(x)
z := autograd.Tanh(x)
z := autograd.PowScalar(x, 2.0)
z := autograd.Mean(x)
z := autograd.Sum(x)

// Backward pass
loss.Backward()              // accumulates .Grad on all leaf Variables
loss.BackwardWithGrad(g)     // custom upstream gradient

// Utilities
x.Grad                       // *tensor.Tensor, nil if not computed
x.Detach()                   // stop gradient
x.ZeroGrad()
```

### `nn` — Layers & Losses

```go
// Layers
nn.NewLinear(inFeatures, outFeatures int, bias bool)
nn.NewSequential(layers ...nn.Module)

// Activations (as Modules)
nn.NewReLU()
nn.NewSigmoid()
nn.NewTanh()

// Module interface
type Module interface {
    Forward(x *autograd.Variable) *autograd.Variable
    Parameters() []*autograd.Variable
    ZeroGrad()
}

// Loss functions
nn.MSELoss(pred, target *autograd.Variable) *autograd.Variable
nn.BCELoss(pred, target *autograd.Variable) *autograd.Variable
nn.CrossEntropyLoss(logits *autograd.Variable, targets []int) *autograd.Variable
```

### `optim` — Optimizers

```go
// SGD
opt := optim.NewSGD(model.Parameters(), lr, momentum)
// lr=0.01, momentum=0.9

// Adam
opt := optim.NewAdam(model.Parameters(), lr, beta1, beta2, eps)
// lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8

// Training loop
opt.ZeroGrad()
loss.Backward()
opt.Step()
```

### `cuda` — GPU Backend

```go
// Init
cuda.DetectGPU() bool
cuda.DeviceInfo() string   // "NVIDIA GeForce RTX 4090 (sm_89), 24.8/25.3 GB"

// Backend
b, err := cuda.NewGPUBackend(device int)
b.Upload(t *tensor.Tensor) (*GPUTensor, error)   // CPU → GPU
b.Download(g *GPUTensor) *tensor.Tensor          // GPU → CPU

// GPU ops
b.Add(a, x *GPUTensor) (*GPUTensor, error)
b.Sub(a, x *GPUTensor) (*GPUTensor, error)
b.Mul(a, x *GPUTensor) (*GPUTensor, error)
b.Div(a, x *GPUTensor) (*GPUTensor, error)
b.AddScalar(a *GPUTensor, s float64) (*GPUTensor, error)
b.MulScalar(a *GPUTensor, s float64) (*GPUTensor, error)
b.Neg(a *GPUTensor) (*GPUTensor, error)
b.ReLU(a *GPUTensor) (*GPUTensor, error)
b.Sigmoid(a *GPUTensor) (*GPUTensor, error)
b.Tanh(a *GPUTensor) (*GPUTensor, error)
b.Exp(a *GPUTensor) (*GPUTensor, error)
b.Log(a *GPUTensor) (*GPUTensor, error)
b.Sum(a *GPUTensor) (float64, error)
b.Mean(a *GPUTensor) (float64, error)
b.Softmax(a *GPUTensor, rows, cols int) (*GPUTensor, error)
b.MatMul(A, B *GPUTensor) (*GPUTensor, error)    // cuBLAS DGEMM

// PinnedTensor (zero-copy)
p, err := cuda.NewPinnedTensor(shape ...int)
p.Slice() []float64          // direct Go slice over pinned memory
p.ToGPU() (*GPUTensor, error)
p.FromGPU(g *GPUTensor) error
p.ToCPUTensor() *tensor.Tensor
p.Free()
```

---

## Performance

Benchmarks on **RTX 4090 + Intel Core Ultra 7 265K** — 1M float64 (8 MB):

```
Transfer (H2D CPU → GPU)
  PinnedTensor.ToGPU()      324 µs   24,661 MB/s
  GPUBackend.Upload()       368 µs   21,717 MB/s    (pageable memcpy)

Transfer (D2H GPU → CPU)
  PinnedTensor.FromGPU()    312 µs   25,668 MB/s
  Download to []float64     499 µs   16,042 MB/s
```

**vs PyTorch 2.10** (same hardware, same data size):

```
  Pinned H2D    GoTorch: 324 µs   PyTorch: 325 µs   → identical
  Pageable H2D  GoTorch: 368 µs   PyTorch: 378 µs   → identical
  Pinned D2H    GoTorch: 312 µs   PyTorch: 311 µs   → identical
  Pageable D2H  GoTorch: 499 µs   PyTorch: 500 µs   → identical
```

GoTorch's CUDA transfer performance is **within measurement noise** of PyTorch — both call the same CUDA runtime functions.

---

## Architecture

```
gotorch_v1/
├── tensor/          Pure Go CPU tensors (no CGo)
│   ├── tensor.go    Tensor type, creation, indexing
│   ├── ops.go       Elementwise ops, activations
│   ├── reduce.go    Sum, Mean, Max, Min, ArgMax, Softmax
│   ├── shape.go     Reshape, broadcast, transpose, strides
│   └── linalg.go    MatMul, BatchMatMul, Dot, Outer
│
├── autograd/        Reverse-mode automatic differentiation
│   ├── variable.go  Variable (wraps Tensor + grad)
│   ├── engine.go    Topological sort backward pass
│   └── functions.go Differentiable op implementations
│
├── nn/              Neural network modules
│   ├── linear.go    Linear layer (y = xW^T + b)
│   ├── sequential.go Sequential container
│   ├── activations.go ReLU, Sigmoid, Tanh as Modules
│   └── loss.go      MSELoss, BCELoss, CrossEntropyLoss
│
├── optim/           Optimizers
│   ├── sgd.go       SGD + momentum
│   └── adam.go      Adam (Kingma & Ba 2014)
│
└── cuda/            GPU backend (build tag: gpu)
    ├── ops.cu        CUDA kernels + cuBLAS MatMul (nvcc → .so)
    ├── cuda.h        C API declarations
    ├── bridge.go     CGo wrappers
    ├── tensor_gpu.go GPUTensor type
    ├── backend.go    High-level GPUBackend API
    ├── pinned.go     PinnedTensor — zero-copy CPU↔GPU
    └── detect_*.go   DetectGPU() — always available, no build tag needed
```

---

## PyTorch Coverage

GoTorch covers the **core primitives** most used in practice:

| Area | Covered | Notes |
|---|---|---|
| Tensor creation | ✅ zeros, ones, rand, randn, eye, arange | missing: full, linspace, from_numpy, load/save |
| Elementwise ops | ✅ +, -, *, /, pow, exp, log, sqrt, neg, abs | missing: floor, ceil, round, clamp, where |
| Activations | ✅ relu, sigmoid, tanh, softmax, log_softmax | missing: gelu, leaky_relu, selu, elu |
| Reductions | ✅ sum, mean, max, min, argmax | missing: std, var, norm, prod |
| Linear algebra | ✅ matmul, bmm, dot, outer | missing: conv2d, conv1d, svd, eig |
| Shape ops | ✅ reshape, flatten, squeeze, unsqueeze, transpose, T | missing: cat, stack, split, chunk, index, slice |
| Autograd | ✅ reverse-mode, topological backward | missing: higher-order grads, no_grad ctx, Jacobian |
| nn.Linear | ✅ | |
| nn.Sequential | ✅ | |
| nn activations | ✅ ReLU, Sigmoid, Tanh | missing: ~40 modules |
| Loss functions | ✅ MSE, BCE, CrossEntropy | missing: NLL, KL, Huber, etc. |
| SGD | ✅ with momentum | missing: weight_decay, Nesterov |
| Adam | ✅ | missing: AdamW, eps_hat |
| CUDA backend | ✅ elementwise, MatMul (cuBLAS), pinned memory | missing: conv, BN, multi-GPU |
| dtype support | ❌ float64 only | PyTorch: float16/32/64, int, bool, complex |
| Serialization | ❌ | no save/load |
| DataLoader | ❌ | |

**Overall: ~20–25% of PyTorch's surface area**, focusing on the primitives needed to build and train simple feed-forward networks. The GPU transfer performance is identical to PyTorch.

---

## Roadmap

- [ ] `float32` dtype support (cuts memory in half, unlocks Tensor Cores)
- [ ] `cat`, `stack`, `split` — tensor concatenation
- [ ] `nn.Conv2d` — convolution (CPU + CUDA)
- [ ] `nn.BatchNorm` — batch normalization
- [ ] `nn.Dropout`
- [ ] `torch.no_grad()` context
- [ ] Model serialization (save/load weights)
- [ ] DataLoader with shuffling
- [ ] Multi-GPU support

---

## Requirements

| Component | Minimum |
|---|---|
| Go | 1.22+ |
| CUDA Toolkit (GPU only) | 12.x |
| NVIDIA driver (GPU only) | 525+ |
| GPU architectures | sm_80 / sm_86 / sm_89 / sm_90 |

---

## License

MIT
