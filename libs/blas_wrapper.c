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
#include <cublasLt.h>
#include <cuda_runtime.h>
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

// =============================================================================
// B-impl-3: FP8 E4M3 MatMul via cublasLt.
//
// Discovery (Step 1): cublasGemmEx НЕ поддерживает CUDA_R_8F_E4M3 -- нужен
// cublasLtMatmul. Handle-урок из B-impl-1 доработки применён: wrapper owns
// LOCAL cublasLtHandle_t (не переносится через dlopen), Go caller передаёт
// CUstream и device scales, wrapper вызывает cublasLtMatmul через local Lt-handle.
//
// Contract:
//   A, B: FP8 E4M3 buffers (uint8).
//   C: FP32 buffer.
//   scaleA, scaleB, scaleC: device float* (per-tensor, host-computed = amax/max).
//   amaxD: optional device float* -- если non-NULL, cuBLASLt пишет absmax(D) сюда.
//
// Layout: same swap trick as F32/F16 -- caller row-major, cuBLASLt column-major
// с swap A/B и M/N.
//
// Descriptor cache: cache per (M,N,K) -- сброс при смене формы. Workspace
// растёт через ensureLtWorkspace lazy pool.
// =============================================================================

typedef struct {
    void       *stream;
    int32_t     m;
    int32_t     n;
    int32_t     k;
    int32_t     _pad;      // align next void* on 8-byte
    const void *A;         // FP8 E4M3
    const void *B;         // FP8 E4M3
    void       *C;         // FP32
    const void *alpha;     // host float32*
    const void *beta;      // host float32*
    const void *scaleA;    // device float* (per-tensor)
    const void *scaleB;    // device float*
    const void *scaleC;    // device float* (output pre-quant)
    void       *amaxD;     // device float* (optional)
} Fp8E4M3MatmulArgs;

static cublasLtHandle_t s_lt_h = NULL;

// Cached descriptors -- сбрасываем при смене формы.
static cublasLtMatmulDesc_t   s_lt_desc = NULL;
static cublasLtMatrixLayout_t s_lt_A = NULL, s_lt_B = NULL, s_lt_C = NULL;
static cublasLtMatmulAlgo_t   s_lt_algo;
static void                  *s_lt_ws = NULL;
static size_t                 s_lt_ws_size = 0;
static int32_t                s_lt_M = 0, s_lt_N = 0, s_lt_K = 0;
static void                  *s_lt_last_amaxD = NULL;

static int ensure_lt_handle(void) {
    if (s_lt_h) return 0;
    cublasStatus_t st = cublasLtCreate(&s_lt_h);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[gotorch wrapper] cublasLtCreate failed: %d\n", (int)st);
        return (int)st;
    }
    return 0;
}

static void lt_cleanup_cached(void) {
    if (s_lt_desc) { cublasLtMatmulDescDestroy(s_lt_desc); s_lt_desc = NULL; }
    if (s_lt_A)    { cublasLtMatrixLayoutDestroy(s_lt_A); s_lt_A = NULL; }
    if (s_lt_B)    { cublasLtMatrixLayoutDestroy(s_lt_B); s_lt_B = NULL; }
    if (s_lt_C)    { cublasLtMatrixLayoutDestroy(s_lt_C); s_lt_C = NULL; }
    if (s_lt_ws)   { cudaFree(s_lt_ws); s_lt_ws = NULL; }
    s_lt_ws_size = 0;
    s_lt_M = s_lt_N = s_lt_K = 0;
    s_lt_last_amaxD = NULL;
}

static int lt_setup(Fp8E4M3MatmulArgs *a) {
    lt_cleanup_cached();

    int M = a->m, N = a->n, K = a->k;

    // FP8 layout (row-major from caller view, cuBLASLt sees column-major swap):
    //   A_lt = caller's B [K,M], B_lt = caller's A [K,N], C_lt = [N,M].
    // TransA=T so cuBLASLt reads A_lt as [M,K].
    cublasLtMatrixLayoutCreate(&s_lt_A, CUDA_R_8F_E4M3, K, N, K);
    cublasLtMatrixLayoutCreate(&s_lt_B, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&s_lt_C, CUDA_R_16F, N, M, N);

    // Try compute types in preferred order. v3 goml prior art shows COMPUTE_16F
    // + scale=R_16F works on sm_89; для sm_120a пробуем сначала так же.
    cublasComputeType_t tryCompute[] = {
        CUBLAS_COMPUTE_16F,           // v3 goml successful path
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_COMPUTE_32F,
    };
    cudaDataType_t tryScale[] = {
        CUDA_R_16F,                    // matches CUBLAS_COMPUTE_16F
        CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
    };
    const char *tryNames[] = {"COMPUTE_16F+R_16F", "FAST_16F", "FAST_TF32", "32F"};
    cublasStatus_t st = CUBLAS_STATUS_NOT_SUPPORTED;
    cublasLtMatmulHeuristicResult_t heur[16];
    int found = 0;
    int chosenIdx = -1;

    for (int ci = 0; ci < 4; ci++) {
        if (s_lt_desc) {
            cublasLtMatmulDescDestroy(s_lt_desc);
            s_lt_desc = NULL;
        }
        st = cublasLtMatmulDescCreate(&s_lt_desc, tryCompute[ci], tryScale[ci]);
        if (st != CUBLAS_STATUS_SUCCESS) continue;

        cublasOperation_t opA = CUBLAS_OP_T, opB = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(s_lt_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
        cublasLtMatmulDescSetAttribute(s_lt_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
        cublasLtMatmulDescSetAttribute(s_lt_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a->scaleA, sizeof(a->scaleA));
        cublasLtMatmulDescSetAttribute(s_lt_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &a->scaleB, sizeof(a->scaleB));
        // NB: D_SCALE_POINTER NOT set for FP16 output path (v3 goml precedent).
        // Setting it triggers cublasLt to expect FP8 output which our C = FP16 layout is not.
        int8_t fastAccum = 1;
        cublasLtMatmulDescSetAttribute(s_lt_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum, sizeof(fastAccum));

        cublasLtMatmulPreference_t pref = NULL;
        cublasLtMatmulPreferenceCreate(&pref);
        size_t maxWs = 256 * 1024 * 1024;
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs));

        found = 0;
        st = cublasLtMatmulAlgoGetHeuristic(
            s_lt_h, s_lt_desc, s_lt_A, s_lt_B, s_lt_C, s_lt_C,
            pref, 16, heur, &found);
        cublasLtMatmulPreferenceDestroy(pref);

        if (st == CUBLAS_STATUS_SUCCESS && found > 0) {
            fprintf(stderr, "[gotorch wrapper] FP8 compute=%s found %d algos\n", tryNames[ci], found);
            chosenIdx = ci;
            break;
        }
        fprintf(stderr, "[gotorch wrapper] FP8 compute=%s: no algos (st=%d)\n", tryNames[ci], (int)st);
    }

    if (chosenIdx < 0) {
        lt_cleanup_cached();
        return 15;
    }

    s_lt_algo    = heur[0].algo;
    s_lt_ws_size = heur[0].workspaceSize;
    if (s_lt_ws_size > 0) {
        cudaMalloc(&s_lt_ws, s_lt_ws_size);
    }

    s_lt_M = M;
    s_lt_N = N;
    s_lt_K = K;
    s_lt_last_amaxD = a->amaxD;
    return 0;
}

int32_t gt_lt_matmul_fp8_e4m3(Fp8E4M3MatmulArgs *a) {
    int rc = ensure_lt_handle();
    if (rc) return rc;

    // Rebuild cache if shape changed or amax presence toggles.
    if (a->m != s_lt_M || a->n != s_lt_N || a->k != s_lt_K
        || (a->amaxD != NULL) != (s_lt_last_amaxD != NULL)) {
        rc = lt_setup(a);
        if (rc) return rc;
    } else {
        // Update A/B scale pointers on cached desc (D_SCALE not set for FP16 output).
        cublasLtMatmulDescSetAttribute(s_lt_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a->scaleA, sizeof(a->scaleA));
        cublasLtMatmulDescSetAttribute(s_lt_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &a->scaleB, sizeof(a->scaleB));
    }
    // amaxD is optional -- set/unset per call.
    if (a->amaxD) {
        cublasLtMatmulDescSetAttribute(s_lt_desc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &a->amaxD, sizeof(a->amaxD));
    }

    return (int32_t)cublasLtMatmul(
        s_lt_h, s_lt_desc,
        a->alpha,
        a->B, s_lt_A,      // swap: caller's B in A slot (column-major)
        a->A, s_lt_B,
        a->beta,
        a->C, s_lt_C,
        a->C, s_lt_C,
        &s_lt_algo,
        s_lt_ws, s_lt_ws_size,
        (cudaStream_t)a->stream);
}
