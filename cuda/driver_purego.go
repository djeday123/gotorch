package cuda

// CUDA Driver API bindings via purego — единственное место в пакете,
// где uintptr появляется в сигнатурах биндингов. Хранение uintptr в полях
// структур ЗАПРЕЩЕНО (см. TestNoUintptrInPublicAPI); конверсия из
// unsafe.Pointer выполняется в момент вызова.
//
// Загружаем libcuda.so.1 (driver API), а не libcudart (runtime). Driver API
// даёт primary-context, cuModuleLoadData для PTX JIT и явный контроль над
// stream'ами — этого хватает без cudart.

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// CUresult — код возврата driver API.
type CUresult int32

const (
	CUDA_SUCCESS                  CUresult = 0
	CUDA_ERROR_INVALID_VALUE      CUresult = 1
	CUDA_ERROR_OUT_OF_MEMORY      CUresult = 2
	CUDA_ERROR_NOT_INITIALIZED    CUresult = 3
	CUDA_ERROR_DEINITIALIZED      CUresult = 4
	CUDA_ERROR_NO_DEVICE          CUresult = 100
	CUDA_ERROR_INVALID_DEVICE     CUresult = 101
	CUDA_ERROR_INVALID_CONTEXT    CUresult = 201
	CUDA_ERROR_INVALID_HANDLE     CUresult = 400
	CUDA_ERROR_NOT_FOUND          CUresult = 500
	CUDA_ERROR_NOT_READY          CUresult = 600
	CUDA_ERROR_LAUNCH_FAILED      CUresult = 719
	CUDA_ERROR_INVALID_IMAGE      CUresult = 200
	CUDA_ERROR_INVALID_PTX        CUresult = 218
	CUDA_ERROR_NVLINK_UNCORRECT   CUresult = 220
	CUDA_ERROR_JIT_COMPILER_ERROR CUresult = 221
)

func (r CUresult) Error() string {
	if r == CUDA_SUCCESS {
		return "CUDA_SUCCESS"
	}
	names := map[CUresult]string{
		1: "INVALID_VALUE", 2: "OUT_OF_MEMORY", 3: "NOT_INITIALIZED",
		4: "DEINITIALIZED", 100: "NO_DEVICE", 101: "INVALID_DEVICE",
		200: "INVALID_IMAGE", 201: "INVALID_CONTEXT", 218: "INVALID_PTX",
		220: "NVLINK_UNCORRECT", 221: "JIT_COMPILER_ERROR",
		400: "INVALID_HANDLE", 500: "NOT_FOUND", 600: "NOT_READY",
		719: "LAUNCH_FAILED",
	}
	if name, ok := names[r]; ok {
		return fmt.Sprintf("CUDA_ERROR_%s (%d)", name, r)
	}
	return fmt.Sprintf("CUDA_ERROR(%d)", r)
}

// Флаги.
const (
	CU_STREAM_NON_BLOCKING       uint32 = 1
	CU_MEMHOSTALLOC_PORTABLE     uint32 = 1
	CU_CTX_SCHED_AUTO            uint32 = 0
	CU_LIMIT_STACK_SIZE          int32  = 0
	CU_LIMIT_PRINTF_FIFO_SIZE    int32  = 1
	CU_LIMIT_MALLOC_HEAP_SIZE    int32  = 2
	CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH int32 = 3
)

// ──────────────────────────────────────────────────────────
// Driver function pointers — заполняются один раз при initDriver()
// ──────────────────────────────────────────────────────────

var (
	driverOnce sync.Once
	driverErr  error

	// Init
	cuInit func(flags uint32) CUresult

	// Device
	cuDeviceGet          func(device *int32, ordinal int32) CUresult
	cuDeviceGetCount     func(count *int32) CUresult
	cuDeviceGetName      func(name *byte, len int32, dev int32) CUresult
	cuDeviceTotalMem     func(bytes *uint64, dev int32) CUresult
	cuDeviceGetAttribute func(pi *int32, attrib int32, dev int32) CUresult

	// Primary context (по требованию ТЗ — retain/release, не Create)
	cuDevicePrimaryCtxRetain  func(pctx *uintptr, dev int32) CUresult
	cuDevicePrimaryCtxRelease func(dev int32) CUresult
	cuCtxSetCurrent           func(ctx uintptr) CUresult
	cuCtxGetCurrent           func(pctx *uintptr) CUresult
	cuCtxSynchronize          func() CUresult

	// Memory
	cuMemAlloc     func(dptr *uintptr, bytesize uint64) CUresult
	cuMemFree      func(dptr uintptr) CUresult
	cuMemAllocHost func(pp *unsafe.Pointer, bytesize uint64) CUresult
	cuMemFreeHost  func(p unsafe.Pointer) CUresult
	cuMemGetInfo   func(free, total *uint64) CUresult
	cuMemsetD8     func(dstDevice uintptr, uc byte, n uint64) CUresult

	cuMemcpyHtoD      func(dstDevice uintptr, srcHost unsafe.Pointer, byteCount uint64) CUresult
	cuMemcpyDtoH      func(dstHost unsafe.Pointer, srcDevice uintptr, byteCount uint64) CUresult
	cuMemcpyDtoD      func(dstDevice uintptr, srcDevice uintptr, byteCount uint64) CUresult
	cuMemcpyHtoDAsync func(dstDevice uintptr, srcHost unsafe.Pointer, byteCount uint64, hStream uintptr) CUresult
	cuMemcpyDtoHAsync func(dstHost unsafe.Pointer, srcDevice uintptr, byteCount uint64, hStream uintptr) CUresult

	// Module / kernel — потребуются на этапе 3+, объявляем сразу для проверки dlsym.
	cuModuleLoadData    func(module *uintptr, image unsafe.Pointer) CUresult
	cuModuleLoadDataEx  func(module *uintptr, image unsafe.Pointer, numOptions uint32, options *int32, optionValues *unsafe.Pointer) CUresult
	cuModuleGetFunction func(hfunc *uintptr, hmod uintptr, name *byte) CUresult
	cuModuleUnload      func(hmod uintptr) CUresult
	cuLaunchKernel      func(
		f uintptr,
		gridDimX, gridDimY, gridDimZ uint32,
		blockDimX, blockDimY, blockDimZ uint32,
		sharedMemBytes uint32,
		hStream uintptr,
		kernelParams unsafe.Pointer,
		extra unsafe.Pointer,
	) CUresult

	// Stream
	cuStreamCreate      func(phStream *uintptr, flags uint32) CUresult
	cuStreamSynchronize func(hStream uintptr) CUresult
	cuStreamDestroy     func(hStream uintptr) CUresult
)

// initDriver загружает libcuda.so.1 и регистрирует все указатели функций.
// Идемпотентна — sync.Once. Ошибка кэшируется.
func initDriver() error {
	driverOnce.Do(func() {
		var lib uintptr
		lib, driverErr = purego.Dlopen("libcuda.so.1", purego.RTLD_LAZY|purego.RTLD_GLOBAL)
		if driverErr != nil {
			lib, driverErr = purego.Dlopen("libcuda.so", purego.RTLD_LAZY|purego.RTLD_GLOBAL)
			if driverErr != nil {
				driverErr = fmt.Errorf("cuda: cannot load libcuda.so.1: %w "+
					"(is NVIDIA driver installed?)", driverErr)
				return
			}
		}
		reg := func(fn any, name string) {
			purego.RegisterLibFunc(fn, lib, name)
		}
		reg(&cuInit, "cuInit")
		reg(&cuDeviceGet, "cuDeviceGet")
		reg(&cuDeviceGetCount, "cuDeviceGetCount")
		reg(&cuDeviceGetName, "cuDeviceGetName")
		reg(&cuDeviceTotalMem, "cuDeviceTotalMem_v2")
		reg(&cuDeviceGetAttribute, "cuDeviceGetAttribute")

		reg(&cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain")
		reg(&cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2")
		reg(&cuCtxSetCurrent, "cuCtxSetCurrent")
		reg(&cuCtxGetCurrent, "cuCtxGetCurrent")
		reg(&cuCtxSynchronize, "cuCtxSynchronize")

		reg(&cuMemAlloc, "cuMemAlloc_v2")
		reg(&cuMemFree, "cuMemFree_v2")
		reg(&cuMemAllocHost, "cuMemAllocHost_v2")
		reg(&cuMemFreeHost, "cuMemFreeHost")
		reg(&cuMemGetInfo, "cuMemGetInfo_v2")
		reg(&cuMemsetD8, "cuMemsetD8_v2")

		reg(&cuMemcpyHtoD, "cuMemcpyHtoD_v2")
		reg(&cuMemcpyDtoH, "cuMemcpyDtoH_v2")
		reg(&cuMemcpyDtoD, "cuMemcpyDtoD_v2")
		reg(&cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2")
		reg(&cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2")

		reg(&cuModuleLoadData, "cuModuleLoadData")
		reg(&cuModuleLoadDataEx, "cuModuleLoadDataEx")
		reg(&cuModuleGetFunction, "cuModuleGetFunction")
		reg(&cuModuleUnload, "cuModuleUnload")
		reg(&cuLaunchKernel, "cuLaunchKernel")

		reg(&cuStreamCreate, "cuStreamCreate")
		reg(&cuStreamSynchronize, "cuStreamSynchronize")
		reg(&cuStreamDestroy, "cuStreamDestroy_v2")
	})
	return driverErr
}

// check оборачивает CUresult в error с контекстом операции.
func check(r CUresult, op string) error {
	if r != CUDA_SUCCESS {
		return fmt.Errorf("%s: %s", op, r.Error())
	}
	return nil
}
