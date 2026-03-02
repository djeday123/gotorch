//go:build gpu

package cuda

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -lgotorch_cuda -lcublas -lcudart
#include "cuda.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Init initializes CUDA on the given device index (0-based).
// Returns the actual device index used, or an error if no GPU is available.
func Init(device int) (int, error) {
	ret := int(C.init_cuda(C.int(device)))
	if ret < 0 {
		return -1, fmt.Errorf("cuda: no CUDA-capable device found")
	}
	return ret, nil
}

// DeviceCount returns the number of available CUDA devices.
func DeviceCount() int {
	return int(C.gpu_device_count())
}

// MemoryInfo returns free and total GPU memory in bytes.
func MemoryInfo() (free, total uint64) {
	var f, t C.size_t
	C.gpu_memory_info(&f, &t)
	return uint64(f), uint64(t)
}

// DeviceName returns the device name and compute capability.
func DeviceName(device int) string {
	return C.GoString(C.gpu_device_name(C.int(device)))
}

// Malloc allocates bytes on the GPU. Caller must call Free when done.
func Malloc(bytes int) unsafe.Pointer {
	return C.gpu_malloc(C.size_t(bytes))
}

// Free releases GPU memory previously allocated with Malloc.
func Free(ptr unsafe.Pointer) {
	C.gpu_free(ptr)
}

// H2D copies len(data)*8 bytes from a []float64 slice to GPU pointer dst.
func H2D(dst unsafe.Pointer, data []float64) {
	if len(data) == 0 {
		return
	}
	C.gpu_memcpy_h2d(dst, unsafe.Pointer(&data[0]), C.size_t(len(data)*8))
}

// D2H copies n float64 values from GPU pointer src into dst slice.
// dst must have cap >= n.
func D2H(dst []float64, src unsafe.Pointer, n int) {
	if n == 0 {
		return
	}
	C.gpu_memcpy_d2h(unsafe.Pointer(&dst[0]), src, C.size_t(n*8))
}

// ---------------------------------------------------------------------------
// Raw GPU ops — operate on device pointers directly.
// All output pointers must be pre-allocated on GPU.
// ---------------------------------------------------------------------------

func GPUAdd(a, b, c unsafe.Pointer, n int) {
	C.gpu_add_f64((*C.double)(a), (*C.double)(b), (*C.double)(c), C.int(n))
}
func GPUSub(a, b, c unsafe.Pointer, n int) {
	C.gpu_sub_f64((*C.double)(a), (*C.double)(b), (*C.double)(c), C.int(n))
}
func GPUMul(a, b, c unsafe.Pointer, n int) {
	C.gpu_mul_f64((*C.double)(a), (*C.double)(b), (*C.double)(c), C.int(n))
}
func GPUDiv(a, b, c unsafe.Pointer, n int) {
	C.gpu_div_f64((*C.double)(a), (*C.double)(b), (*C.double)(c), C.int(n))
}
func GPUAddScalar(a unsafe.Pointer, scalar float64, c unsafe.Pointer, n int) {
	C.gpu_add_scalar_f64((*C.double)(a), C.double(scalar), (*C.double)(c), C.int(n))
}
func GPUMulScalar(a unsafe.Pointer, scalar float64, c unsafe.Pointer, n int) {
	C.gpu_mul_scalar_f64((*C.double)(a), C.double(scalar), (*C.double)(c), C.int(n))
}
func GPUReLU(a, c unsafe.Pointer, n int) {
	C.gpu_relu_f64((*C.double)(a), (*C.double)(c), C.int(n))
}
func GPUSigmoid(a, c unsafe.Pointer, n int) {
	C.gpu_sigmoid_f64((*C.double)(a), (*C.double)(c), C.int(n))
}
func GPUTanh(a, c unsafe.Pointer, n int) {
	C.gpu_tanh_f64((*C.double)(a), (*C.double)(c), C.int(n))
}
func GPUExp(a, c unsafe.Pointer, n int) {
	C.gpu_exp_f64((*C.double)(a), (*C.double)(c), C.int(n))
}
func GPULog(a, c unsafe.Pointer, n int) {
	C.gpu_log_f64((*C.double)(a), (*C.double)(c), C.int(n))
}
func GPUNeg(a, c unsafe.Pointer, n int) {
	C.gpu_neg_f64((*C.double)(a), (*C.double)(c), C.int(n))
}
func GPUReLUGrad(a, grad, out unsafe.Pointer, n int) {
	C.gpu_relu_grad_f64((*C.double)(a), (*C.double)(grad), (*C.double)(out), C.int(n))
}
func GPUSigmoidGrad(sig, grad, out unsafe.Pointer, n int) {
	C.gpu_sigmoid_grad_f64((*C.double)(sig), (*C.double)(grad), (*C.double)(out), C.int(n))
}
func GPUTanhGrad(tanhOut, grad, out unsafe.Pointer, n int) {
	C.gpu_tanh_grad_f64((*C.double)(tanhOut), (*C.double)(grad), (*C.double)(out), C.int(n))
}
func GPUSum(a unsafe.Pointer, n int) float64 {
	return float64(C.gpu_sum_f64((*C.double)(a), C.int(n)))
}
func GPUMean(a unsafe.Pointer, n int) float64 {
	return float64(C.gpu_mean_f64((*C.double)(a), C.int(n)))
}
func GPUSoftmax(a, c unsafe.Pointer, rows, cols int) {
	C.gpu_softmax_f64((*C.double)(a), (*C.double)(c), C.int(rows), C.int(cols))
}
func GPUMatMul(A, B, C unsafe.Pointer, M, N, K int) {
	C.gpu_matmul_f64((*C.double)(A), (*C.double)(B), (*C.double)(C), C.int(M), C.int(N), C.int(K))
}
