package cuda

// cuBLAS Level-3 биндинги через purego. По ТЗ Этапа 2 — только 5 функций:
// Create/Destroy/SetStream + Dgemm/Sgemm. GemmEx, batched-варианты, TF32
// math mode — вне этого этапа: не тащим для чистоты и минимизации surface.
//
// Handle-глобальный TF32-mode умышленно НЕ включаем: без cublasSetMathMode
// default = FULL FP32, что даёт допуск ~1e-5 на F32 (заявленный в Воротах 2).
// TF32-как-состояние срезал бы точность до ~1e-3, требуя ослабления допуска.
// R03b-impl-4-final: точечный per-call TF32 доступен через MatMulF32_TF32 —
// SetMathMode(TF32) внутри метода, defer возврат в DEFAULT_MATH до return.
// Философия dtype-суффиксов R02a: режим — свойство метода, не состояние
// backend'а; невозможно «забыть выключить» то, что не включается глобально.
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

// cublasMath_t (для точечного TF32 через MatMulF32_TF32).
const (
	CUBLAS_DEFAULT_MATH        int32 = 0
	CUBLAS_TF32_TENSOR_OP_MATH int32 = 3
)

var (
	cublasOnce sync.Once
	cublasErr  error

	cublasCreate_v2    func(handle *uintptr) cublasStatus
	cublasDestroy_v2   func(handle uintptr) cublasStatus
	cublasSetStream_v2 func(handle uintptr, stream uintptr) cublasStatus
	cublasSetMathMode  func(handle uintptr, mode int32) cublasStatus

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

// B-impl-1 Step 1 probe (FIXED FACT):
//   purego v0.9.1 паникует "too many arguments" при регистрации функций с
//   >=18 аргументами. Проверено экспериментально:
//     cublasSgemmStridedBatched (18 args) -- panic at RegisterLibFunc.
//     cublasGemmEx (19 args) -- то же.
//   Это подтверждает прецедент goml: `libs1/cublas_wrapper.c` использует
//   struct-args wrappers `gemmex_wrapper` / `gemm_strided_batched_ex_wrapper`
//   с 1-pointer entry именно из-за этого лимита.
//
// Решение B-impl-1: batched F32/F64 реализован через ЦИКЛ cublasSgemm_v2 /
//   cublasDgemm_v2 (14 args, работает). Это тот же паттерн что goml
//   `BatchedMatMulF32` (`goml/backend/cuda/cublas.go:264`) -- loop by batch.
//   Loop-batched < strided-batched на big-batch (kernel launch overhead), но:
//     (а) даёт bit-exact vs goml на A/B тестах (один и тот же алгоритм);
//     (б) для gputrain-shape batch=1 разницы нет;
//     (в) настоящий strided-batched через wrapper.so -- отдельный ТЗ.

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
		purego.RegisterLibFunc(&cublasSetMathMode, lib, "cublasSetMathMode")
		purego.RegisterLibFunc(&cublasDgemm_v2, lib, "cublasDgemm_v2")
		purego.RegisterLibFunc(&cublasSgemm_v2, lib, "cublasSgemm_v2")
		// B-impl-1: strided batched НЕ регистрируем -- purego v0.9.1 упирается
		// на >=18 args (проверено). Batched через loop cublasSgemm_v2. См. комментарий выше.
	})
	return cublasErr
}
