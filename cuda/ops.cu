/*
 * ops.cu — CUDA kernels for gotorch_v1 GPU backend.
 *
 * Compiled with nvcc, linked as a shared library.
 * Build: see Makefile (multi-arch: sm_80/86/89/90 + PTX fallback)
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#define BLOCK 256
#define GRID(n) (((n) + BLOCK - 1) / BLOCK)

#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t e = (call);                                               \
        if (e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(e));               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                       \
        cublasStatus_t s = (call);                                            \
        if (s != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n",                      \
                    __FILE__, __LINE__, (int)s);                              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Global cuBLAS handle (initialized once)
static cublasHandle_t g_cublas = NULL;

static void ensure_cublas() {
    if (g_cublas == NULL) {
        CUBLAS_CHECK(cublasCreate(&g_cublas));
    }
}

// ---------------------------------------------------------------------------
// Device management
// ---------------------------------------------------------------------------

extern "C" int init_cuda(int device) {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count == 0) return -1;
    if (device >= count) device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    ensure_cublas();
    return device;
}

extern "C" int gpu_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

extern "C" void gpu_memory_info(size_t *free_bytes, size_t *total_bytes) {
    CUDA_CHECK(cudaMemGetInfo(free_bytes, total_bytes));
}

extern "C" const char *gpu_device_name(int device) {
    static char name[256];
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    snprintf(name, sizeof(name), "%s (sm_%d%d)",
             prop.name, prop.major, prop.minor);
    return name;
}

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------

extern "C" void *gpu_malloc(size_t bytes) {
    void *ptr = NULL;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

extern "C" void gpu_free(void *ptr) {
    if (ptr) cudaFree(ptr);
}

extern "C" void gpu_memcpy_h2d(void *dst, const void *src, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

extern "C" void gpu_memcpy_d2h(void *dst, const void *src, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// Pinned (page-locked) memory — zero-copy CPU↔GPU
// ---------------------------------------------------------------------------

extern "C" void *gpu_alloc_pinned(size_t bytes) {
    void *ptr = NULL;
    CUDA_CHECK(cudaMallocHost(&ptr, bytes));
    return ptr;
}

extern "C" void gpu_free_pinned(void *ptr) {
    if (ptr) cudaFreeHost(ptr);
}

extern "C" void gpu_memcpy_h2d_async(void *dst_gpu, const void *src_pinned, size_t bytes) {
    CUDA_CHECK(cudaMemcpyAsync(dst_gpu, src_pinned, bytes, cudaMemcpyHostToDevice, 0));
}

extern "C" void gpu_memcpy_d2h_async(void *dst_pinned, const void *src_gpu, size_t bytes) {
    CUDA_CHECK(cudaMemcpyAsync(dst_pinned, src_gpu, bytes, cudaMemcpyDeviceToHost, 0));
}

extern "C" void gpu_stream_sync(void) {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Elementwise kernels
// ---------------------------------------------------------------------------

__global__ void k_add_f64(const double *a, const double *b, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void k_sub_f64(const double *a, const double *b, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] - b[i];
}

__global__ void k_mul_f64(const double *a, const double *b, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

__global__ void k_div_f64(const double *a, const double *b, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] / b[i];
}

__global__ void k_add_scalar_f64(const double *a, double s, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + s;
}

__global__ void k_mul_scalar_f64(const double *a, double s, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * s;
}

__global__ void k_relu_f64(const double *a, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] > 0.0 ? a[i] : 0.0;
}

__global__ void k_sigmoid_f64(const double *a, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = 1.0 / (1.0 + exp(-a[i]));
}

__global__ void k_tanh_f64(const double *a, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = tanh(a[i]);
}

__global__ void k_exp_f64(const double *a, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = exp(a[i]);
}

__global__ void k_log_f64(const double *a, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = log(a[i]);
}

__global__ void k_neg_f64(const double *a, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = -a[i];
}

// Gradient kernels
__global__ void k_relu_grad_f64(const double *a, const double *grad, double *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] > 0.0 ? grad[i] : 0.0;
}

__global__ void k_sigmoid_grad_f64(const double *sig, const double *grad, double *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sig[i] * (1.0 - sig[i]) * grad[i];
}

__global__ void k_tanh_grad_f64(const double *tanh_out, const double *grad, double *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (1.0 - tanh_out[i] * tanh_out[i]) * grad[i];
}

// ---------------------------------------------------------------------------
// Extern C wrappers for elementwise ops
// ---------------------------------------------------------------------------

extern "C" void gpu_add_f64(const double *a, const double *b, double *c, int n) {
    k_add_f64<<<GRID(n), BLOCK>>>(a, b, c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" void gpu_sub_f64(const double *a, const double *b, double *c, int n) {
    k_sub_f64<<<GRID(n), BLOCK>>>(a, b, c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" void gpu_mul_f64(const double *a, const double *b, double *c, int n) {
    k_mul_f64<<<GRID(n), BLOCK>>>(a, b, c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" void gpu_div_f64(const double *a, const double *b, double *c, int n) {
    k_div_f64<<<GRID(n), BLOCK>>>(a, b, c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" void gpu_add_scalar_f64(const double *a, double scalar, double *c, int n) {
    k_add_scalar_f64<<<GRID(n), BLOCK>>>(a, scalar, c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" void gpu_mul_scalar_f64(const double *a, double scalar, double *c, int n) {
    k_mul_scalar_f64<<<GRID(n), BLOCK>>>(a, scalar, c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" void gpu_relu_f64(const double *a, double *c, int n) {
    k_relu_f64<<<GRID(n), BLOCK>>>(a, c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" void gpu_sigmoid_f64(const double *a, double *c, int n) {
    k_sigmoid_f64<<<GRID(n), BLOCK>>>(a, c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" void gpu_tanh_f64(const double *a, double *c, int n) {
    k_tanh_f64<<<GRID(n), BLOCK>>>(a, c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" void gpu_exp_f64(const double *a, double *c, int n) {
    k_exp_f64<<<GRID(n), BLOCK>>>(a, c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" void gpu_log_f64(const double *a, double *c, int n) {
    k_log_f64<<<GRID(n), BLOCK>>>(a, c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" void gpu_neg_f64(const double *a, double *c, int n) {
    k_neg_f64<<<GRID(n), BLOCK>>>(a, c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" void gpu_relu_grad_f64(const double *a, const double *grad, double *out, int n) {
    k_relu_grad_f64<<<GRID(n), BLOCK>>>(a, grad, out, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" void gpu_sigmoid_grad_f64(const double *sig, const double *grad, double *out, int n) {
    k_sigmoid_grad_f64<<<GRID(n), BLOCK>>>(sig, grad, out, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" void gpu_tanh_grad_f64(const double *tanh_out, const double *grad, double *out, int n) {
    k_tanh_grad_f64<<<GRID(n), BLOCK>>>(tanh_out, grad, out, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Parallel reduction: sum
// ---------------------------------------------------------------------------

__global__ void k_reduce_sum_f64(const double *a, double *out, int n) {
    __shared__ double sdata[BLOCK];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? a[i] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

extern "C" double gpu_sum_f64(const double *a, int n) {
    double *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(double)));
    k_reduce_sum_f64<<<GRID(n), BLOCK>>>(a, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    double result;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    return result;
}

extern "C" double gpu_mean_f64(const double *a, int n) {
    return gpu_sum_f64(a, n) / (double)n;
}

// ---------------------------------------------------------------------------
// Softmax: numerically stable, row-wise
// ---------------------------------------------------------------------------

__global__ void k_softmax_f64(const double *a, double *c, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const double *in  = a + row * cols;
    double       *out = c + row * cols;

    // max for stability
    double mx = in[0];
    for (int j = 1; j < cols; j++) if (in[j] > mx) mx = in[j];

    double s = 0.0;
    for (int j = 0; j < cols; j++) { out[j] = exp(in[j] - mx); s += out[j]; }
    for (int j = 0; j < cols; j++) out[j] /= s;
}

extern "C" void gpu_softmax_f64(const double *a, double *c, int rows, int cols) {
    k_softmax_f64<<<GRID(rows), BLOCK>>>(a, c, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// MatMul via cuBLAS: C = A @ B   (row-major, double)
// A: [M x K], B: [K x N], C: [M x N]
//
// cuBLAS is column-major. For row-major A (M×K) and B (K×N):
//   Treat A as col-major (K×M) = A^T
//   Treat B as col-major (N×K) = B^T
//   C = A @ B  =>  C^T = B^T @ A^T
//   cublasDgemm(B^T, A^T) → C^T which stored row-major = C
// ---------------------------------------------------------------------------

extern "C" void gpu_matmul_f64(const double *A, const double *B, double *C,
                                int M, int N, int K) {
    ensure_cublas();
    const double alpha = 1.0, beta = 0.0;
    // C^T (N×M col-major) = B^T (N×K col-major) @ A^T (K×M col-major)
    // cublasDgemm(handle, transB, transA, N, M, K, ...)
    CUBLAS_CHECK(cublasDgemm(
        g_cublas,
        CUBLAS_OP_N,  // op(B^T) = no-op  => B treated as N×K col-major
        CUBLAS_OP_N,  // op(A^T) = no-op  => A treated as K×M col-major
        N, M, K,
        &alpha,
        B, N,         // B: leading dim = N  (B^T col-major: N×K)
        A, K,         // A: leading dim = K  (A^T col-major: K×M)
        &beta,
        C, N          // C: leading dim = N  (C^T col-major: N×M)
    ));
    CUDA_CHECK(cudaDeviceSynchronize());
}
