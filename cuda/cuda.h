#ifndef GOTORCH_CUDA_H
#define GOTORCH_CUDA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Device management
int  init_cuda(int device);
int  gpu_device_count(void);
void gpu_memory_info(size_t *free_bytes, size_t *total_bytes);
const char *gpu_device_name(int device);

// Memory management
void *gpu_malloc(size_t bytes);
void  gpu_free(void *ptr);
void  gpu_memcpy_h2d(void *dst, const void *src, size_t bytes);
void  gpu_memcpy_d2h(void *dst, const void *src, size_t bytes);

// Pinned (page-locked) host memory — zero-copy CPU↔GPU transfers
void *gpu_alloc_pinned(size_t bytes);
void  gpu_free_pinned(void *ptr);
void  gpu_memcpy_h2d_async(void *dst_gpu, const void *src_pinned, size_t bytes);
void  gpu_memcpy_d2h_async(void *dst_pinned, const void *src_gpu, size_t bytes);
void  gpu_stream_sync(void);

// Elementwise ops (double / float64)
void gpu_add_f64(const double *a, const double *b, double *c, int n);
void gpu_sub_f64(const double *a, const double *b, double *c, int n);
void gpu_mul_f64(const double *a, const double *b, double *c, int n);
void gpu_div_f64(const double *a, const double *b, double *c, int n);
void gpu_add_scalar_f64(const double *a, double scalar, double *c, int n);
void gpu_mul_scalar_f64(const double *a, double scalar, double *c, int n);
void gpu_relu_f64(const double *a, double *c, int n);
void gpu_sigmoid_f64(const double *a, double *c, int n);
void gpu_tanh_f64(const double *a, double *c, int n);
void gpu_exp_f64(const double *a, double *c, int n);
void gpu_log_f64(const double *a, double *c, int n);
void gpu_neg_f64(const double *a, double *c, int n);
void gpu_relu_grad_f64(const double *a, const double *grad, double *out, int n);
void gpu_sigmoid_grad_f64(const double *sig, const double *grad, double *out, int n);
void gpu_tanh_grad_f64(const double *tanh_out, const double *grad, double *out, int n);

// Reduction
double gpu_sum_f64(const double *a, int n);
double gpu_mean_f64(const double *a, int n);

// Matrix multiply: C = A @ B, row-major
// A: [M x K], B: [K x N], C: [M x N]
void gpu_matmul_f64(const double *A, const double *B, double *C, int M, int N, int K);

// Softmax over last dim: input [rows x cols], output [rows x cols]
void gpu_softmax_f64(const double *a, double *c, int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif // GOTORCH_CUDA_H
