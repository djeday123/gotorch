package cuda

// gotorch cuBLAS wrapper loader (B-impl-1 доработка + B-impl-2 foundation).
//
// Rationale (see cublas_purego.go комментарий): purego v0.9.1 упирается на
// >=18 args. Solution -- thin C wrapper `libs/blas_wrapper.c` builds
// libgotorch_blas_wrapper.so with struct-args entry points (2 args each).
// Go biнди'нг здесь через purego.
//
// Resolution order for the .so (matches goml pattern):
//   1. $GOTORCH_LIBS_DIR/libgotorch_blas_wrapper.so
//   2. ./libgotorch_blas_wrapper.so (cwd)
//   3. libgotorch_blas_wrapper.so via ldconfig
// If not found: hasBlasWrapper=false, MatMulStridedBatched* fall back to loop.

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Struct layouts MUST match blas_wrapper.c exactly (packed).
// Go struct with mixed 32/64-bit fields normally adds padding; we control
// layout manually by ordering fields and using unsafe.Sizeof asserts at
// init time. All fields use fixed-size Go types (int32/int64/uintptr/uP).

// SgemmStridedBatchedArgs -- 18 полей SgemmStridedBatched в порядке C-структуры.
type SgemmStridedBatchedArgs struct {
	Stream  uintptr        // CUstream (0 = default stream). Wrapper owns local cublas handle.
	TransA  int32          //
	TransB  int32          //
	M       int32          //
	N       int32          //
	K       int32          //
	Alpha   unsafe.Pointer // *float32 (host)
	A       uintptr        // device
	Lda     int32          //
	StrideA int64          //
	B       uintptr        //
	Ldb     int32          //
	StrideB int64          //
	Beta    unsafe.Pointer //
	C       uintptr        //
	Ldc     int32          //
	StrideC int64          //
	Batch   int32          //
}

type DgemmStridedBatchedArgs struct {
	Stream  uintptr
	TransA  int32
	TransB  int32
	M       int32
	N       int32
	K       int32
	Alpha   unsafe.Pointer // *float64
	A       uintptr
	Lda     int32
	StrideA int64
	B       uintptr
	Ldb     int32
	StrideB int64
	Beta    unsafe.Pointer
	C       uintptr
	Ldc     int32
	StrideC int64
	Batch   int32
}

// GemmExArgs -- B-impl-2 F16.
type GemmExArgs struct {
	Stream      uintptr
	TransA      int32
	TransB      int32
	M           int32
	N           int32
	K           int32
	Alpha       unsafe.Pointer
	A           uintptr
	AType       int32
	Lda         int32
	B           uintptr
	BType       int32
	Ldb         int32
	Beta        unsafe.Pointer
	C           uintptr
	CType       int32
	Ldc         int32
	ComputeType int32
	Algo        int32
}

type GemmStridedBatchedExArgs struct {
	Stream      uintptr
	TransA      int32
	TransB      int32
	M           int32
	N           int32
	K           int32
	Alpha       unsafe.Pointer
	A           uintptr
	AType       int32
	Lda         int32
	StrideA     int64
	B           uintptr
	BType       int32
	Ldb         int32
	StrideB     int64
	Beta        unsafe.Pointer
	C           uintptr
	CType       int32
	Ldc         int32
	StrideC     int64
	Batch       int32
	ComputeType int32
	Algo        int32
}

// cudaDataType (subset used here).
const (
	CUDA_R_16F int32 = 2  // half
	CUDA_R_32F int32 = 0  // float
	CUDA_R_64F int32 = 1  // double
	CUDA_R_16BF int32 = 14 // bfloat16
	CUDA_R_8F_E4M3 int32 = 28 // FP8 E4M3 (Ada+/Hopper+)
)

// cublasComputeType_t (subset).
const (
	CUBLAS_COMPUTE_32F           int32 = 68
	CUBLAS_COMPUTE_32F_FAST_TF32 int32 = 77
	CUBLAS_COMPUTE_64F           int32 = 70
)

// cublasGemmAlgo_t default.
const (
	CUBLAS_GEMM_DEFAULT             int32 = -1
	CUBLAS_GEMM_DEFAULT_TENSOR_OP   int32 = 99 // legacy alias -- same as DEFAULT on modern cuBLAS
)

var (
	blasWrapperOnce sync.Once
	hasBlasWrapper  bool

	gtSgemmStridedBatched   func(args unsafe.Pointer) int32
	gtDgemmStridedBatched   func(args unsafe.Pointer) int32
	gtGemmEx                func(args unsafe.Pointer) int32
	gtGemmStridedBatchedEx  func(args unsafe.Pointer) int32
)

// resolveBlasWrapperPath ищет libgotorch_blas_wrapper.so в 3 местах.
func resolveBlasWrapperPath() (string, error) {
	name := "libgotorch_blas_wrapper.so"
	if dir := os.Getenv("GOTORCH_LIBS_DIR"); dir != "" {
		p := filepath.Join(dir, name)
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	// cwd
	if _, err := os.Stat(name); err == nil {
		return name, nil
	}
	// pathhinted by binary dir (repo layout gotorch/v6/libs/)
	if exe, err := os.Executable(); err == nil {
		p := filepath.Join(filepath.Dir(exe), "libs", name)
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	// Common project layout hints (repo/libs). Test binaries live in cache dir,
	// so also try walking up from cwd looking for libs/.
	// Give a single generic try -- ldconfig cache.
	return name, nil // let purego try ldconfig
}

// initBlasWrapper -- ленивая загрузка libgotorch_blas_wrapper.so.
// Missing .so => hasBlasWrapper=false, warning to stderr, MatMulStridedBatched*
// falls back to loop path.
func initBlasWrapper() {
	blasWrapperOnce.Do(func() {
		path, _ := resolveBlasWrapperPath()
		lib, err := purego.Dlopen(path, purego.RTLD_LAZY|purego.RTLD_GLOBAL)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[gotorch] libgotorch_blas_wrapper.so not found (%v); batched GEMM falls back to loop-per-batch. Build with `make -C gotorch/v6/libs -f Makefile.blas_wrapper`.\n", err)
			return
		}
		purego.RegisterLibFunc(&gtSgemmStridedBatched, lib, "gt_sgemm_strided_batched")
		purego.RegisterLibFunc(&gtDgemmStridedBatched, lib, "gt_dgemm_strided_batched")
		purego.RegisterLibFunc(&gtGemmEx, lib, "gt_gemm_ex")
		purego.RegisterLibFunc(&gtGemmStridedBatchedEx, lib, "gt_gemm_strided_batched_ex")
		hasBlasWrapper = true
	})
}

// HasBlasWrapper — доступен ли struct-args wrapper (для условной подачи в тесты).
func HasBlasWrapper() bool {
	initBlasWrapper()
	return hasBlasWrapper
}
