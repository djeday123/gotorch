package cuda

// cuBLAS Level-3 биндинги через purego. По ТЗ Этапа 2 — только 5 функций:
// Create/Destroy/SetStream + Dgemm/Sgemm. GemmEx, batched-варианты, TF32
// math mode — вне этого этапа: не тащим для чистоты и минимизации surface.
//
// TF32 умышленно НЕ включаем: без cublasSetMathMode default = FULL FP32,
// что даёт допуск ~1e-5 на F32 (заявленный в Воротах 2). TF32 срезал бы
// точность до ~1e-3, требуя ослабления допуска.
//
// uintptr в сигнатурах биндингов легален — конверсия из unsafe.Pointer
// происходит в момент вызова (unsafe.Pointer rule 4 про syscall-подобные
// вызовы), хранение в полях структур запрещено. См. bufferView в api.go.

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// cublasStatus_t — код возврата cuBLAS.
type cublasStatus int32

const (
	CUBLAS_STATUS_SUCCESS          cublasStatus = 0
	CUBLAS_STATUS_NOT_INITIALIZED  cublasStatus = 1
	CUBLAS_STATUS_ALLOC_FAILED     cublasStatus = 3
	CUBLAS_STATUS_INVALID_VALUE    cublasStatus = 7
	CUBLAS_STATUS_ARCH_MISMATCH    cublasStatus = 8
	CUBLAS_STATUS_MAPPING_ERROR    cublasStatus = 11
	CUBLAS_STATUS_EXECUTION_FAILED cublasStatus = 13
	CUBLAS_STATUS_INTERNAL_ERROR   cublasStatus = 14
	CUBLAS_STATUS_NOT_SUPPORTED    cublasStatus = 15
)

func (s cublasStatus) Error() string {
	names := map[cublasStatus]string{
		0: "SUCCESS", 1: "NOT_INITIALIZED", 3: "ALLOC_FAILED",
		7: "INVALID_VALUE", 8: "ARCH_MISMATCH", 11: "MAPPING_ERROR",
		13: "EXECUTION_FAILED", 14: "INTERNAL_ERROR", 15: "NOT_SUPPORTED",
	}
	if name, ok := names[s]; ok {
		return fmt.Sprintf("CUBLAS_STATUS_%s (%d)", name, s)
	}
	return fmt.Sprintf("CUBLAS_STATUS(%d)", s)
}

// cublasOperation_t.
type cublasOperation int32

const (
	CUBLAS_OP_N cublasOperation = 0
	CUBLAS_OP_T cublasOperation = 1
)

var (
	cublasOnce sync.Once
	cublasErr  error

	cublasCreate_v2    func(handle *uintptr) cublasStatus
	cublasDestroy_v2   func(handle uintptr) cublasStatus
	cublasSetStream_v2 func(handle uintptr, stream uintptr) cublasStatus

	cublasDgemm_v2 func(
		handle uintptr,
		transa, transb cublasOperation,
		m, n, k int32,
		alpha unsafe.Pointer,
		A uintptr, lda int32,
		B uintptr, ldb int32,
		beta unsafe.Pointer,
		C uintptr, ldc int32,
	) cublasStatus

	cublasSgemm_v2 func(
		handle uintptr,
		transa, transb cublasOperation,
		m, n, k int32,
		alpha unsafe.Pointer,
		A uintptr, lda int32,
		B uintptr, ldb int32,
		beta unsafe.Pointer,
		C uintptr, ldc int32,
	) cublasStatus
)

// initCuBLAS загружает libcublas.so.12 и регистрирует биндинги. Идемпотентна.
func initCuBLAS() error {
	cublasOnce.Do(func() {
		var lib uintptr
		lib, cublasErr = purego.Dlopen("libcublas.so.12", purego.RTLD_LAZY|purego.RTLD_GLOBAL)
		if cublasErr != nil {
			lib, cublasErr = purego.Dlopen("libcublas.so", purego.RTLD_LAZY|purego.RTLD_GLOBAL)
			if cublasErr != nil {
				cublasErr = fmt.Errorf("cuda: cannot load libcublas.so.12: %w", cublasErr)
				return
			}
		}
		purego.RegisterLibFunc(&cublasCreate_v2, lib, "cublasCreate_v2")
		purego.RegisterLibFunc(&cublasDestroy_v2, lib, "cublasDestroy_v2")
		purego.RegisterLibFunc(&cublasSetStream_v2, lib, "cublasSetStream_v2")
		purego.RegisterLibFunc(&cublasDgemm_v2, lib, "cublasDgemm_v2")
		purego.RegisterLibFunc(&cublasSgemm_v2, lib, "cublasSgemm_v2")
	})
	return cublasErr
}
