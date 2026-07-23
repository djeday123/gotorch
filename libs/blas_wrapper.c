// gotorch cuBLAS wrapper — struct-args entry points to avoid purego 18+ args limit.
//
// Fact (B-impl-1 Step 1 probe): purego v0.9.1 panics at RegisterLibFunc with
// >=18-argument functions. cublasSgemmStridedBatched has 18 args, cublasGemmEx 19,
// cublasGemmStridedBatchedEx 22 -- all beyond limit.
//
// KEY LESSON (доработка diag): cublas handles do NOT transfer across dlopens.
// Handle created via purego-loaded libcublas.so IS NOT usable inside
// wrapper.so-linked libcublas.so (returns CUBLAS_STATUS_NOT_INITIALIZED, even
// though both dlopens resolve to the same on-disk libcublas.so.13 file).
// Diagnostic: cublasCreate INSIDE wrapper returns different handle value,
// which does work. Meaning cuBLAS keeps per-load state.
//
// Solution: wrapper owns its own static cublasHandle_t. Go caller passes
// STREAM instead of HANDLE; wrapper calls cublasSetStream on its local handle
// before each op. Costs one setStream per call (cheap; setStream is state
// mutation on the handle, not a driver sync).
//
// Build:
//   make -f Makefile.blas_wrapper
//
// Contract: struct field types/order MUST match Go side EXACTLY. Natural C
// alignment (no packed) -- Go 4/8-byte natural alignment matches.

#include <cublas_v2.h>
#include <stddef.h>
#include <stdio.h>

// Wrapper-owned local cublas handle. Lazy init on first call.
// Uses whatever CUDA context is current at that moment (primary ctx in our
// R02b architecture -- shared across the process).
static cublasHandle_t s_wrapper_h = NULL;

static int ensure_local_handle(void) {
    if (s_wrapper_h) return 0;
    cublasStatus_t st = cublasCreate_v2(&s_wrapper_h);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[gotorch wrapper] cublasCreate_v2 failed: %d\n", (int)st);
        return (int)st;
    }
    return 0;
}

// -----------------------------------------------------------------------------
// SgemmStridedBatched (18 native args -> 1 pointer here). F32 strided-batched.
// Field `stream` REPLACED former `handle` field: caller passes CUstream (or 0
// for default stream); wrapper calls cublasSetStream on its local handle.
// -----------------------------------------------------------------------------
typedef struct {
    void            *stream;   // CUstream (opaque handle in driver); NULL/0 = default
    int32_t          transa;
    int32_t          transb;
    int32_t          m;
    int32_t          n;
    int32_t          k;
    const float     *alpha;
    const float     *A;
    int32_t          lda;
    int64_t          strideA;
    const float     *B;
    int32_t          ldb;
    int64_t          strideB;
    const float     *beta;
    float           *C;
    int32_t          ldc;
    int64_t          strideC;
    int32_t          batchCount;
} SgemmStridedBatchedArgs;

int32_t gt_sgemm_strided_batched(SgemmStridedBatchedArgs *a) {
    int rc = ensure_local_handle();
    if (rc) return rc;
    cublasSetStream_v2(s_wrapper_h, (cudaStream_t)a->stream);
    return (int32_t)cublasSgemmStridedBatched(
        s_wrapper_h,
        (cublasOperation_t)a->transa, (cublasOperation_t)a->transb,
        a->m, a->n, a->k,
        a->alpha,
        a->A, a->lda, a->strideA,
        a->B, a->ldb, a->strideB,
        a->beta,
        a->C, a->ldc, a->strideC,
        a->batchCount);
}

// -----------------------------------------------------------------------------
// DgemmStridedBatched. F64 strided-batched.
// -----------------------------------------------------------------------------
typedef struct {
    void            *stream;
    int32_t          transa;
    int32_t          transb;
    int32_t          m;
    int32_t          n;
    int32_t          k;
    const double    *alpha;
    const double    *A;
    int32_t          lda;
    int64_t          strideA;
    const double    *B;
    int32_t          ldb;
    int64_t          strideB;
    const double    *beta;
    double          *C;
    int32_t          ldc;
    int64_t          strideC;
    int32_t          batchCount;
} DgemmStridedBatchedArgs;

int32_t gt_dgemm_strided_batched(DgemmStridedBatchedArgs *a) {
    int rc = ensure_local_handle();
    if (rc) return rc;
    cublasSetStream_v2(s_wrapper_h, (cudaStream_t)a->stream);
    return (int32_t)cublasDgemmStridedBatched(
        s_wrapper_h,
        (cublasOperation_t)a->transa, (cublasOperation_t)a->transb,
        a->m, a->n, a->k,
        a->alpha,
        a->A, a->lda, a->strideA,
        a->B, a->ldb, a->strideB,
        a->beta,
        a->C, a->ldc, a->strideC,
        a->batchCount);
}

// -----------------------------------------------------------------------------
// GemmEx (19 native args). Universal typed GEMM -- used for F16 in B-impl-2.
// alpha/beta type depends on computeType; caller passes device or host ptr per
// cublas convention.
// -----------------------------------------------------------------------------
typedef struct {
    void            *stream;
    int32_t          transa;
    int32_t          transb;
    int32_t          m;
    int32_t          n;
    int32_t          k;
    const void      *alpha;
    const void      *A;
    int32_t          aType;
    int32_t          lda;
    const void      *B;
    int32_t          bType;
    int32_t          ldb;
    const void      *beta;
    void            *C;
    int32_t          cType;
    int32_t          ldc;
    int32_t          computeType;
    int32_t          algo;
} GemmExArgs;

int32_t gt_gemm_ex(GemmExArgs *a) {
    int rc = ensure_local_handle();
    if (rc) return rc;
    cublasSetStream_v2(s_wrapper_h, (cudaStream_t)a->stream);
    return (int32_t)cublasGemmEx(
        s_wrapper_h,
        (cublasOperation_t)a->transa, (cublasOperation_t)a->transb,
        a->m, a->n, a->k,
        a->alpha,
        a->A, (cudaDataType)a->aType, a->lda,
        a->B, (cudaDataType)a->bType, a->ldb,
        a->beta,
        a->C, (cudaDataType)a->cType, a->ldc,
        (cublasComputeType_t)a->computeType,
        (cublasGemmAlgo_t)a->algo);
}

// -----------------------------------------------------------------------------
// GemmStridedBatchedEx (22 native args). Universal typed strided-batched GEMM.
// Used for F16 batched in B-impl-2.
// -----------------------------------------------------------------------------
typedef struct {
    void            *stream;
    int32_t          transa;
    int32_t          transb;
    int32_t          m;
    int32_t          n;
    int32_t          k;
    const void      *alpha;
    const void      *A;
    int32_t          aType;
    int32_t          lda;
    int64_t          strideA;
    const void      *B;
    int32_t          bType;
    int32_t          ldb;
    int64_t          strideB;
    const void      *beta;
    void            *C;
    int32_t          cType;
    int32_t          ldc;
    int64_t          strideC;
    int32_t          batchCount;
    int32_t          computeType;
    int32_t          algo;
} GemmStridedBatchedExArgs;

int32_t gt_gemm_strided_batched_ex(GemmStridedBatchedExArgs *a) {
    int rc = ensure_local_handle();
    if (rc) return rc;
    cublasSetStream_v2(s_wrapper_h, (cudaStream_t)a->stream);
    return (int32_t)cublasGemmStridedBatchedEx(
        s_wrapper_h,
        (cublasOperation_t)a->transa, (cublasOperation_t)a->transb,
        a->m, a->n, a->k,
        a->alpha,
        a->A, (cudaDataType)a->aType, a->lda, a->strideA,
        a->B, (cudaDataType)a->bType, a->ldb, a->strideB,
        a->beta,
        a->C, (cudaDataType)a->cType, a->ldc, a->strideC,
        a->batchCount,
        (cublasComputeType_t)a->computeType,
        (cublasGemmAlgo_t)a->algo);
}
