//go:build gpu

package cuda

import (
	"fmt"
	"github.com/djeday123/gotorch/tensor"
	"unsafe"
)

// GPUTensor is a tensor that lives in GPU memory.
// Always call Free() when done to avoid GPU memory leaks.
type GPUTensor struct {
	ptr   unsafe.Pointer // device pointer
	shape []int
	size  int // total number of float64 elements
}

// NewGPUTensor uploads a CPU tensor to GPU memory.
func NewGPUTensor(t *tensor.Tensor) (*GPUTensor, error) {
	data := t.Data()
	n := len(data)
	if n == 0 {
		return nil, fmt.Errorf("cuda: cannot upload empty tensor")
	}
	ptr := Malloc(n * 8)
	if ptr == nil {
		return nil, fmt.Errorf("cuda: gpu_malloc failed (n=%d)", n)
	}
	H2D(ptr, data)
	shape := make([]int, len(t.Shape()))
	copy(shape, t.Shape())
	return &GPUTensor{ptr: ptr, shape: shape, size: n}, nil
}

// NewGPUTensorEmpty allocates an uninitialized GPU tensor of given shape.
func NewGPUTensorEmpty(shape ...int) (*GPUTensor, error) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	ptr := Malloc(n * 8)
	if ptr == nil {
		return nil, fmt.Errorf("cuda: gpu_malloc failed for empty tensor")
	}
	s := make([]int, len(shape))
	copy(s, shape)
	return &GPUTensor{ptr: ptr, shape: s, size: n}, nil
}

// ToCPU downloads the GPU tensor to a CPU tensor.Tensor.
func (g *GPUTensor) ToCPU() *tensor.Tensor {
	data := make([]float64, g.size)
	D2H(data, g.ptr, g.size)
	return tensor.New(data, g.shape)
}

// Free releases GPU memory. Safe to call on nil.
func (g *GPUTensor) Free() {
	if g != nil && g.ptr != nil {
		Free(g.ptr)
		g.ptr = nil
	}
}

// Shape returns the tensor shape (read-only copy).
func (g *GPUTensor) Shape() []int {
	s := make([]int, len(g.shape))
	copy(s, g.shape)
	return s
}

// Size returns the total number of elements.
func (g *GPUTensor) Size() int { return g.size }

// Ptr returns the raw device pointer (for passing to bridge ops).
func (g *GPUTensor) Ptr() unsafe.Pointer { return g.ptr }
