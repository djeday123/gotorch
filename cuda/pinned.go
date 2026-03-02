//go:build gpu

package cuda

/*
#include "cuda.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"github.com/djeday123/gotorch/tensor"
	"unsafe"
)

// PinnedTensor is a CPU tensor backed by cudaMallocHost (page-locked / pinned memory).
//
// Key property: the same memory buffer is accessible from both CPU (via Slice())
// and GPU (via ToGPU / FromGPU) without any CPU-side memcpy. The DMA transfer
// is handled asynchronously by the GPU DMA engine, freeing the CPU entirely.
//
// Ownership: caller must call Free() when done. PinnedTensor is NOT garbage
// collected — it lives in C-managed memory outside the Go heap.
type PinnedTensor struct {
	ptr   unsafe.Pointer
	shape []int
	size  int // total number of float64 elements
}

// NewPinnedTensor allocates a zeroed PinnedTensor with the given shape in
// page-locked host memory. Returns an error if CUDA allocation fails.
func NewPinnedTensor(shape ...int) (*PinnedTensor, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("pinned: shape must have at least one dimension")
	}
	size := 1
	for _, d := range shape {
		if d <= 0 {
			return nil, fmt.Errorf("pinned: invalid dimension %d", d)
		}
		size *= d
	}
	ptr := C.gpu_alloc_pinned(C.size_t(size * 8))
	if ptr == nil {
		return nil, fmt.Errorf("pinned: cudaMallocHost failed for %d bytes", size*8)
	}
	// Zero the memory via Go slice view
	sl := unsafe.Slice((*float64)(ptr), size)
	for i := range sl {
		sl[i] = 0
	}

	shapeCopy := make([]int, len(shape))
	copy(shapeCopy, shape)

	return &PinnedTensor{ptr: ptr, shape: shapeCopy, size: size}, nil
}

// Free releases the pinned host memory. Must be called exactly once.
func (p *PinnedTensor) Free() {
	if p.ptr != nil {
		C.gpu_free_pinned(p.ptr)
		p.ptr = nil
	}
}

// Slice returns a Go []float64 view directly over the pinned buffer.
// This is zero-copy: writing to the slice modifies the pinned memory
// that the GPU will read during the next ToGPU call.
// The slice is valid only while the PinnedTensor is alive (before Free()).
func (p *PinnedTensor) Slice() []float64 {
	return unsafe.Slice((*float64)(p.ptr), p.size)
}

// Shape returns the tensor dimensions.
func (p *PinnedTensor) Shape() []int {
	s := make([]int, len(p.shape))
	copy(s, p.shape)
	return s
}

// Size returns the total number of elements.
func (p *PinnedTensor) Size() int { return p.size }

// ToGPU uploads the pinned buffer to a new GPUTensor using async DMA.
// Because the source is pinned memory, the CPU is not involved in the transfer.
// Synchronizes before returning so the GPUTensor is ready to use.
func (p *PinnedTensor) ToGPU() (*GPUTensor, error) {
	g, err := NewGPUTensorEmpty(p.shape...)
	if err != nil {
		return nil, fmt.Errorf("pinned ToGPU: %w", err)
	}
	C.gpu_memcpy_h2d_async(g.ptr, p.ptr, C.size_t(p.size*8))
	C.gpu_stream_sync()
	return g, nil
}

// FromGPU downloads a GPUTensor into this pinned buffer using async DMA.
// Synchronizes before returning so Slice() reflects the GPU result.
func (p *PinnedTensor) FromGPU(g *GPUTensor) error {
	if g.size != p.size {
		return fmt.Errorf("pinned FromGPU: size mismatch: pinned=%d gpu=%d", p.size, g.size)
	}
	C.gpu_memcpy_d2h_async(p.ptr, g.ptr, C.size_t(p.size*8))
	C.gpu_stream_sync()
	return nil
}

// StreamSync blocks until all pending GPU operations are complete.
func StreamSync() {
	C.gpu_stream_sync()
}

// H2DAsync copies pinned host memory to a pre-allocated GPU tensor asynchronously.
// Synchronizes before returning. dst and src must have the same size.
func H2DAsync(dst *GPUTensor, src *PinnedTensor) error {
	if dst.size != src.size {
		return fmt.Errorf("H2DAsync: size mismatch dst=%d src=%d", dst.size, src.size)
	}
	C.gpu_memcpy_h2d_async(dst.ptr, src.ptr, C.size_t(src.size*8))
	C.gpu_stream_sync()
	return nil
}

// D2HAsync copies a GPU tensor to a pre-allocated pinned tensor asynchronously.
// Synchronizes before returning. dst and src must have the same size.
func D2HAsync(dst *PinnedTensor, src *GPUTensor) error {
	return dst.FromGPU(src)
}

// ToCPUTensor copies pinned data into a regular Go-heap tensor.Tensor.
// Use this for interoperability with CPU-only code that doesn't know about
// pinned memory. Note: this IS a copy (pinned → Go heap).
func (p *PinnedTensor) ToCPUTensor() *tensor.Tensor {
	data := make([]float64, p.size)
	copy(data, p.Slice())
	return tensor.New(data, p.shape)
}
