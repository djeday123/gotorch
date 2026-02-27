//go:build gpu

// Package cuda provides a GPU backend for gotorch_v1.
//
// Usage:
//
//	b, err := cuda.NewGPUBackend(0)   // device 0
//	defer b.Close()
//
//	gt, _ := b.Upload(cpuTensor)
//	defer gt.Free()
//
//	result, _ := b.MatMul(a, b)       // stays on GPU
//	cpu := result.ToCPU()             // pull back when needed
package cuda

import (
	"fmt"
	"gotorch_v1/tensor"
)

// GPUBackend manages a CUDA device and exposes high-level tensor ops.
// All ops accept and return *GPUTensor (device memory).
// Upload / Download handle CPU↔GPU transfers.
type GPUBackend struct {
	device int
}

// NewGPUBackend initializes the GPU backend on the given device index.
func NewGPUBackend(device int) (*GPUBackend, error) {
	d, err := Init(device)
	if err != nil {
		return nil, fmt.Errorf("GPUBackend: %w", err)
	}
	return &GPUBackend{device: d}, nil
}

// Close releases the cuBLAS handle and resets the device.
// Call via defer after NewGPUBackend.
func (b *GPUBackend) Close() {
	// cuBLAS handle is global; nothing to do here for now.
	// Future: cublasDestroy when we expose handle to Go.
}

// Device returns the device index.
func (b *GPUBackend) Device() int { return b.device }

// Upload copies a CPU tensor to GPU memory.
func (b *GPUBackend) Upload(t *tensor.Tensor) (*GPUTensor, error) {
	return NewGPUTensor(t)
}

// Download copies a GPU tensor back to CPU.
func (b *GPUBackend) Download(g *GPUTensor) *tensor.Tensor {
	return g.ToCPU()
}

// ---------------------------------------------------------------------------
// Binary elementwise ops
// ---------------------------------------------------------------------------

func (b *GPUBackend) Add(a, x *GPUTensor) (*GPUTensor, error) {
	if err := checkSameSize(a, x); err != nil {
		return nil, err
	}
	out, err := NewGPUTensorEmpty(a.shape...)
	if err != nil {
		return nil, err
	}
	GPUAdd(a.ptr, x.ptr, out.ptr, a.size)
	return out, nil
}

func (b *GPUBackend) Sub(a, x *GPUTensor) (*GPUTensor, error) {
	if err := checkSameSize(a, x); err != nil {
		return nil, err
	}
	out, err := NewGPUTensorEmpty(a.shape...)
	if err != nil {
		return nil, err
	}
	GPUSub(a.ptr, x.ptr, out.ptr, a.size)
	return out, nil
}

func (b *GPUBackend) Mul(a, x *GPUTensor) (*GPUTensor, error) {
	if err := checkSameSize(a, x); err != nil {
		return nil, err
	}
	out, err := NewGPUTensorEmpty(a.shape...)
	if err != nil {
		return nil, err
	}
	GPUMul(a.ptr, x.ptr, out.ptr, a.size)
	return out, nil
}

func (b *GPUBackend) Div(a, x *GPUTensor) (*GPUTensor, error) {
	if err := checkSameSize(a, x); err != nil {
		return nil, err
	}
	out, err := NewGPUTensorEmpty(a.shape...)
	if err != nil {
		return nil, err
	}
	GPUDiv(a.ptr, x.ptr, out.ptr, a.size)
	return out, nil
}

// ---------------------------------------------------------------------------
// Scalar ops
// ---------------------------------------------------------------------------

func (b *GPUBackend) AddScalar(a *GPUTensor, s float64) (*GPUTensor, error) {
	out, err := NewGPUTensorEmpty(a.shape...)
	if err != nil {
		return nil, err
	}
	GPUAddScalar(a.ptr, s, out.ptr, a.size)
	return out, nil
}

func (b *GPUBackend) MulScalar(a *GPUTensor, s float64) (*GPUTensor, error) {
	out, err := NewGPUTensorEmpty(a.shape...)
	if err != nil {
		return nil, err
	}
	GPUMulScalar(a.ptr, s, out.ptr, a.size)
	return out, nil
}

// ---------------------------------------------------------------------------
// Activation ops
// ---------------------------------------------------------------------------

func (b *GPUBackend) ReLU(a *GPUTensor) (*GPUTensor, error) {
	out, err := NewGPUTensorEmpty(a.shape...)
	if err != nil {
		return nil, err
	}
	GPUReLU(a.ptr, out.ptr, a.size)
	return out, nil
}

func (b *GPUBackend) Sigmoid(a *GPUTensor) (*GPUTensor, error) {
	out, err := NewGPUTensorEmpty(a.shape...)
	if err != nil {
		return nil, err
	}
	GPUSigmoid(a.ptr, out.ptr, a.size)
	return out, nil
}

func (b *GPUBackend) Tanh(a *GPUTensor) (*GPUTensor, error) {
	out, err := NewGPUTensorEmpty(a.shape...)
	if err != nil {
		return nil, err
	}
	GPUTanh(a.ptr, out.ptr, a.size)
	return out, nil
}

func (b *GPUBackend) Exp(a *GPUTensor) (*GPUTensor, error) {
	out, err := NewGPUTensorEmpty(a.shape...)
	if err != nil {
		return nil, err
	}
	GPUExp(a.ptr, out.ptr, a.size)
	return out, nil
}

func (b *GPUBackend) Log(a *GPUTensor) (*GPUTensor, error) {
	out, err := NewGPUTensorEmpty(a.shape...)
	if err != nil {
		return nil, err
	}
	GPULog(a.ptr, out.ptr, a.size)
	return out, nil
}

func (b *GPUBackend) Neg(a *GPUTensor) (*GPUTensor, error) {
	out, err := NewGPUTensorEmpty(a.shape...)
	if err != nil {
		return nil, err
	}
	GPUNeg(a.ptr, out.ptr, a.size)
	return out, nil
}

// ---------------------------------------------------------------------------
// Gradient ops (used during backward pass)
// ---------------------------------------------------------------------------

func (b *GPUBackend) ReLUGrad(input, grad *GPUTensor) (*GPUTensor, error) {
	if err := checkSameSize(input, grad); err != nil {
		return nil, err
	}
	out, err := NewGPUTensorEmpty(input.shape...)
	if err != nil {
		return nil, err
	}
	GPUReLUGrad(input.ptr, grad.ptr, out.ptr, input.size)
	return out, nil
}

func (b *GPUBackend) SigmoidGrad(sigOutput, grad *GPUTensor) (*GPUTensor, error) {
	if err := checkSameSize(sigOutput, grad); err != nil {
		return nil, err
	}
	out, err := NewGPUTensorEmpty(sigOutput.shape...)
	if err != nil {
		return nil, err
	}
	GPUSigmoidGrad(sigOutput.ptr, grad.ptr, out.ptr, sigOutput.size)
	return out, nil
}

func (b *GPUBackend) TanhGrad(tanhOutput, grad *GPUTensor) (*GPUTensor, error) {
	if err := checkSameSize(tanhOutput, grad); err != nil {
		return nil, err
	}
	out, err := NewGPUTensorEmpty(tanhOutput.shape...)
	if err != nil {
		return nil, err
	}
	GPUTanhGrad(tanhOutput.ptr, grad.ptr, out.ptr, tanhOutput.size)
	return out, nil
}

// ---------------------------------------------------------------------------
// Reduction ops
// ---------------------------------------------------------------------------

func (b *GPUBackend) Sum(a *GPUTensor) float64 {
	return GPUSum(a.ptr, a.size)
}

func (b *GPUBackend) Mean(a *GPUTensor) float64 {
	return GPUMean(a.ptr, a.size)
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

// Softmax applies softmax over the last dimension.
// a must be 2-D: [rows x cols].
func (b *GPUBackend) Softmax(a *GPUTensor) (*GPUTensor, error) {
	if len(a.shape) != 2 {
		return nil, fmt.Errorf("GPUBackend.Softmax: expected 2-D tensor, got shape %v", a.shape)
	}
	out, err := NewGPUTensorEmpty(a.shape...)
	if err != nil {
		return nil, err
	}
	GPUSoftmax(a.ptr, out.ptr, a.shape[0], a.shape[1])
	return out, nil
}

// ---------------------------------------------------------------------------
// Matrix multiply
// ---------------------------------------------------------------------------

// MatMul computes C = A @ B.
// A: [M x K], B: [K x N] => C: [M x N].
func (b *GPUBackend) MatMul(a, x *GPUTensor) (*GPUTensor, error) {
	if len(a.shape) != 2 || len(x.shape) != 2 {
		return nil, fmt.Errorf("GPUBackend.MatMul: expected 2-D tensors, got %v and %v", a.shape, x.shape)
	}
	M, K := a.shape[0], a.shape[1]
	K2, N := x.shape[0], x.shape[1]
	if K != K2 {
		return nil, fmt.Errorf("GPUBackend.MatMul: shape mismatch [%d x %d] @ [%d x %d]", M, K, K2, N)
	}
	out, err := NewGPUTensorEmpty(M, N)
	if err != nil {
		return nil, err
	}
	GPUMatMul(a.ptr, x.ptr, out.ptr, M, N, K)
	return out, nil
}

// ---------------------------------------------------------------------------
// Convenience: run a CPU tensor through a GPU op and return CPU result.
// Useful for one-off operations without managing GPUTensor lifetime.
// ---------------------------------------------------------------------------

func (b *GPUBackend) RunUnary(t *tensor.Tensor, op func(*GPUTensor) (*GPUTensor, error)) (*tensor.Tensor, error) {
	g, err := b.Upload(t)
	if err != nil {
		return nil, err
	}
	defer g.Free()
	out, err := op(g)
	if err != nil {
		return nil, err
	}
	defer out.Free()
	return b.Download(out), nil
}

func (b *GPUBackend) RunBinary(a, x *tensor.Tensor, op func(*GPUTensor, *GPUTensor) (*GPUTensor, error)) (*tensor.Tensor, error) {
	ga, err := b.Upload(a)
	if err != nil {
		return nil, err
	}
	defer ga.Free()
	gx, err := b.Upload(x)
	if err != nil {
		return nil, err
	}
	defer gx.Free()
	out, err := op(ga, gx)
	if err != nil {
		return nil, err
	}
	defer out.Free()
	return b.Download(out), nil
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func checkSameSize(a, b *GPUTensor) error {
	if a.size != b.size {
		return fmt.Errorf("cuda: tensor size mismatch: %d vs %d", a.size, b.size)
	}
	return nil
}
