# gotorch_v1 — GPU Backend

This package implements a CUDA GPU backend for gotorch_v1.
It is only compiled when the `gpu` build tag is set.

---

## Requirements

| Dependency        | Minimum version |
|-------------------|----------------|
| CUDA Toolkit      | 12.x           |
| NVIDIA driver     | 525+           |
| cuBLAS            | included in Toolkit |
| nvcc              | in `$PATH`     |
| Go                | 1.22+          |

---

## Build

### CPU-only (default, no CUDA needed)

```bash
make          # build
make test     # test
```

### GPU

```bash
make build-gpu                        # default: sm_80/86/89/90 + PTX fallback
make build-gpu CUDA_ARCHS="80 86 89"  # custom arch list
make test-gpu                         # runs GPU tests (needs CUDA device)
```

Manual build without Make:

```bash
# 1. Compile shared lib
nvcc -O3 -std=c++14 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_90,code=compute_90 \
  --compiler-options -fPIC \
  -shared -o cuda/libgotorch_cuda.so cuda/ops.cu

# 2. Build Go with gpu tag
CGO_CFLAGS="-I./cuda" \
CGO_LDFLAGS="-L./cuda -lgotorch_cuda -lcublas -lcudart -Wl,-rpath,./cuda" \
go build -tags gpu ./...
```

---

## Supported GPU Architectures

| SM version | GPU examples                     | Generation      |
|------------|----------------------------------|-----------------|
| sm_80      | A100, A30                        | Ampere          |
| sm_86      | RTX 3090, RTX 3080, A40          | Ampere          |
| sm_89      | **RTX 4090**, L40, L40S          | Ada Lovelace    |
| sm_90      | H100, H200                       | Hopper          |
| PTX (90)   | Future GPUs (JIT compiled)       | Forward-compat  |

### PTX Fallback Explained

When you compile with `-gencode arch=compute_90,code=compute_90`, nvcc
embeds portable PTX bytecode in addition to native SASS kernels.

If a user runs the binary on a GPU that is **newer** than any embedded
native kernel (e.g., sm_100 Blackwell), the CUDA JIT compiler will
automatically compile the PTX at first launch. This happens once and
is cached — subsequent runs are as fast as native.

**Result:** a binary compiled today works on GPUs that don't exist yet.

---

## Architecture

```
tensor/          — pure Go CPU tensors (no build tag)
cuda/
  cuda.h         — C API declarations
  ops.cu         — CUDA kernels + cuBLAS matmul (nvcc compiled → .so)
  bridge.go      — CGo wrappers (//go:build gpu)
  tensor_gpu.go  — GPUTensor type: upload/download (//go:build gpu)
  backend.go     — High-level GPUBackend API (//go:build gpu)
  detect_cpu.go  — Stubs: DetectGPU()=false (//go:build !gpu)
  detect_gpu.go  — Real: DetectGPU(), DeviceInfo() (//go:build gpu)
  ops_test.go    — GPU tests (//go:build gpu)
```

---

## Usage Example

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

    b, err := cuda.NewGPUBackend(0)
    if err != nil {
        panic(err)
    }
    defer b.Close()

    // Upload
    a := tensor.Rand(512, 512)
    x := tensor.Rand(512, 512)

    ga, _ := b.Upload(a)
    defer ga.Free()
    gx, _ := b.Upload(x)
    defer gx.Free()

    // MatMul on GPU
    out, _ := b.MatMul(ga, gx)
    defer out.Free()

    // Download result
    result := b.Download(out)
    fmt.Println("Result shape:", result.Shape())
}
```

---

## Ops Implemented

| Category      | Functions                                              |
|---------------|--------------------------------------------------------|
| Memory        | Upload, Download, Malloc, Free, H2D, D2H               |
| Elementwise   | Add, Sub, Mul, Div, AddScalar, MulScalar, Neg          |
| Activations   | ReLU, Sigmoid, Tanh, Exp, Log                         |
| Grad helpers  | ReLUGrad, SigmoidGrad, TanhGrad                       |
| Reduction     | Sum, Mean                                              |
| Softmax       | Softmax (row-wise, numerically stable)                 |
| Linear alg.   | MatMul (cuBLAS DGEMM, row-major)                       |

---

## Runtime Detection (no build tag required)

```go
import "gotorch_v1/cuda"

if cuda.DetectGPU() {
    fmt.Println(cuda.DeviceInfo())
    // use GPU backend
} else {
    // fall back to CPU
}
```

`DetectGPU()` and `DeviceInfo()` are always available. In CPU-only
builds they return `false` / `"no GPU (CPU-only build)"` with zero overhead.

---

## Troubleshooting

**`libgotorch_cuda.so: cannot open shared object file`**
→ Add `./cuda` to `LD_LIBRARY_PATH`, or install the `.so` to `/usr/local/lib`.

**`CUDA error: no kernel image available`**
→ The binary wasn't compiled for this GPU's SM version.
Re-run `make build-gpu CUDA_ARCHS="XX"` adding your SM version,
or check that PTX fallback was included.

**`cublasCreate failed`**
→ cuBLAS not installed. Install full CUDA Toolkit (not just the runtime).

**Driver version mismatch**
→ CUDA 12.x requires driver ≥ 525. Check with `nvidia-smi`.
