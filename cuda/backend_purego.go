package cuda

// PuregoBackend — реализация Backend через CUDA driver API + PTX-ядра.
// Живёт параллельно с legacy cgo-миром (bridge.go/backend.go). После R02b
// вторая реализация (cgo-backend) сядет рядом, оба будут удовлетворять
// один и тот же интерфейс Backend.

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"runtime"
	"unsafe"
)

// PuregoBackend — тип, реализующий Backend через driver API + cuBLAS + PTX.
type PuregoBackend struct {
	device     int
	primaryCtx uintptr
	cublas     uintptr            // cublasHandle_t (Этап 2)
	ptxModule  uintptr            // CUmodule с r02bKernelsPTX (Этап 3)
	fns        map[string]uintptr // CUfunction cache: name → handle (Этап 3+)
}

// Assertion: *PuregoBackend удовлетворяет Backend.
var _ Backend = (*PuregoBackend)(nil)

// newPuregoBackend инициализирует driver API, retain'ит primary context на
// выбранном устройстве, делает его текущим и возвращает *PuregoBackend.
//
// Обоснование primary-context (по требованию ТЗ): любой процесс, который
// использует CUDA на данном устройстве, обычно уже держит primary-context
// (PyTorch, TensorRT, cuBLAS сам создаёт primary если Create-контекст не
// был установлен, goml — то же самое). Retain'я тот же primary, мы
// гарантируем, что gotorch-purego и goml-ядра, вызванные в одном процессе,
// делят один контекст: аллокация одной библиотеки видима второй, никакого
// «wrong-context» error'а на UnsafeExtractDevicePtr → чужое ядро.
func newPuregoBackend(device int) (*PuregoBackend, error) {
	if err := initDriver(); err != nil {
		return nil, err
	}
	if err := check(cuInit(0), "cuInit"); err != nil {
		return nil, err
	}
	var dev int32
	if err := check(cuDeviceGet(&dev, int32(device)), "cuDeviceGet"); err != nil {
		return nil, err
	}
	var ctx uintptr
	if err := check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain"); err != nil {
		return nil, err
	}
	if err := check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent"); err != nil {
		cuDevicePrimaryCtxRelease(dev)
		return nil, err
	}
	// cuBLAS-handle. Инициализируем на этом же primary-context. cuBLAS сам
	// подхватывает current-context в момент cublasCreate_v2.
	if err := initCuBLAS(); err != nil {
		cuDevicePrimaryCtxRelease(dev)
		return nil, err
	}
	var h uintptr
	if s := cublasCreate_v2(&h); s != CUBLAS_STATUS_SUCCESS {
		cuDevicePrimaryCtxRelease(dev)
		return nil, fmt.Errorf("cublasCreate_v2: %s", s.Error())
	}
	// SetStream(0) = default stream. Sync() покроется cuCtxSynchronize.
	if s := cublasSetStream_v2(h, 0); s != CUBLAS_STATUS_SUCCESS {
		cublasDestroy_v2(h)
		cuDevicePrimaryCtxRelease(dev)
		return nil, fmt.Errorf("cublasSetStream_v2: %s", s.Error())
	}

	// PTX-модуль (Этап 3+). JIT-компиляция driver'ом при cuModuleLoadData.
	// Битая PTX-строка даёт CUDA_ERROR_INVALID_PTX/JIT_COMPILER_ERROR —
	// формируем внятную ошибку с полным сообщением, не панику.
	ptxBytes := append([]byte(r02bKernelsPTX), 0)
	var module uintptr
	// Use LoadDataEx with error-log buffer to get exact JIT error message.
	logBuf := make([]byte, 8192)
	const (
		CU_JIT_ERROR_LOG_BUFFER            int32 = 5
		CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES int32 = 6
	)
	logBufSize := uint32(len(logBuf))
	opts := []int32{CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER}
	optVals := []unsafe.Pointer{
		*(*unsafe.Pointer)(unsafe.Pointer(&logBufSize)),
		unsafe.Pointer(&logBuf[0]),
	}
	if r := cuModuleLoadDataEx(&module, unsafe.Pointer(&ptxBytes[0]),
		uint32(len(opts)), &opts[0], &optVals[0]); r != CUDA_SUCCESS {
		cublasDestroy_v2(h)
		cuDevicePrimaryCtxRelease(dev)
		// Trim log buffer to actual message.
		msg := string(logBuf)
		for i, c := range logBuf {
			if c == 0 {
				msg = string(logBuf[:i])
				break
			}
		}
		return nil, fmt.Errorf("cuModuleLoadDataEx: %s\nJIT log:\n%s", r.Error(), msg)
	}
	pb := &PuregoBackend{
		device: device, primaryCtx: ctx, cublas: h,
		ptxModule: module, fns: make(map[string]uintptr),
	}
	// Прогрев кэша cuFunction для всех ядер Этапов 3-4.
	kernelNames := []string{
		// Stage 3
		"add_f64", "add_f32",
		// Stage 4 arithmetic
		"sub_f64", "sub_f32",
		"mul_f64", "mul_f32",
		"div_f64", "div_f32",
		"neg_f64", "neg_f32",
		// Stage 4 scalar
		"addscalar_f64", "addscalar_f32",
		"mulscalar_f64", "mulscalar_f32",
		// Stage 4 transcendental F32 (aparatnyy approx)
		"exp_f32",
		"log_f32",
		// Stage 4.5 transcendental F64 (fdlibm port)
		"exp_f64",
		"log_f64",
		// Stage 5 non-composite activations
		"relu_f32", "relu_f64",
		"sigmoid_f32", "sigmoid_f64",
		"tanh_f32", "tanh_f64",
		"relu_grad_f32", "relu_grad_f64",
		"sigmoid_grad_f32", "sigmoid_grad_f64",
		"tanh_grad_f32", "tanh_grad_f64",
		// Stage 5 composite operations
		"sum_f64", "sum_f32",
		"softmax_f64", "softmax_f32",
	}
	for _, name := range kernelNames {
		if _, err := pb.getKernel(name); err != nil {
			cuModuleUnload(module)
			cublasDestroy_v2(h)
			cuDevicePrimaryCtxRelease(dev)
			return nil, err
		}
	}
	return pb, nil
}

// getKernel — резолвит имя PTX-ядра в CUfunction handle, кэширует.
func (b *PuregoBackend) getKernel(name string) (uintptr, error) {
	if f, ok := b.fns[name]; ok {
		return f, nil
	}
	nameBytes := append([]byte(name), 0)
	var f uintptr
	if r := cuModuleGetFunction(&f, b.ptxModule, &nameBytes[0]); r != CUDA_SUCCESS {
		return 0, fmt.Errorf("cuModuleGetFunction(%q): %s", name, r.Error())
	}
	b.fns[name] = f
	return f, nil
}

// launchGrid — общие параметры запуска: block=256, grid=ceil(n/256).
func launchGrid(n int) (grid, block uint32) {
	const bd uint32 = 256
	return uint32((n + int(bd) - 1) / int(bd)), bd
}

// launchElementwise3 — запуск 3-аргументного элементного ядра (a,b,c,n).
func (b *PuregoBackend) launchElementwise3(fnName string, a, bb, c DeviceBuffer, n int) error {
	if n <= 0 {
		return fmt.Errorf("cuda.%s: n must be > 0, got %d", fnName, n)
	}
	if err := b.bind(); err != nil {
		return err
	}
	defer runtime.UnlockOSThread()
	fn, err := b.getKernel(fnName)
	if err != nil {
		return err
	}
	va, vb, vc := a.deviceBuffer(), bb.deviceBuffer(), c.deviceBuffer()
	args := struct {
		a, b, c uintptr
		n       int32
		_       int32
	}{va.ptr, vb.ptr, vc.ptr, int32(n), 0}
	params := [4]unsafe.Pointer{
		unsafe.Pointer(&args.a),
		unsafe.Pointer(&args.b),
		unsafe.Pointer(&args.c),
		unsafe.Pointer(&args.n),
	}
	grid, block := launchGrid(n)
	if r := cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0,
		unsafe.Pointer(&params[0]), nil); r != CUDA_SUCCESS {
		return fmt.Errorf("cuLaunchKernel(%s): %s", fnName, r.Error())
	}
	return nil
}

// launchElementwise2 — 2-аргументное ядро (a,c,n) для унарных ops.
func (b *PuregoBackend) launchElementwise2(fnName string, a, c DeviceBuffer, n int) error {
	if n <= 0 {
		return fmt.Errorf("cuda.%s: n must be > 0, got %d", fnName, n)
	}
	if err := b.bind(); err != nil {
		return err
	}
	defer runtime.UnlockOSThread()
	fn, err := b.getKernel(fnName)
	if err != nil {
		return err
	}
	va, vc := a.deviceBuffer(), c.deviceBuffer()
	args := struct {
		a, c uintptr
		n    int32
		_    int32
	}{va.ptr, vc.ptr, int32(n), 0}
	params := [3]unsafe.Pointer{
		unsafe.Pointer(&args.a),
		unsafe.Pointer(&args.c),
		unsafe.Pointer(&args.n),
	}
	grid, block := launchGrid(n)
	if r := cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0,
		unsafe.Pointer(&params[0]), nil); r != CUDA_SUCCESS {
		return fmt.Errorf("cuLaunchKernel(%s): %s", fnName, r.Error())
	}
	return nil
}

// launchScalarF64 — (a, scalarF64, c, n). Scalar передан by-value через PTX .param.f64.
func (b *PuregoBackend) launchScalarF64(fnName string, a DeviceBuffer, scalar float64, c DeviceBuffer, n int) error {
	if n <= 0 {
		return fmt.Errorf("cuda.%s: n must be > 0, got %d", fnName, n)
	}
	if err := b.bind(); err != nil {
		return err
	}
	defer runtime.UnlockOSThread()
	fn, err := b.getKernel(fnName)
	if err != nil {
		return err
	}
	va, vc := a.deviceBuffer(), c.deviceBuffer()
	args := struct {
		a      uintptr
		scalar float64
		c      uintptr
		n      int32
		_      int32
	}{va.ptr, scalar, vc.ptr, int32(n), 0}
	params := [4]unsafe.Pointer{
		unsafe.Pointer(&args.a),
		unsafe.Pointer(&args.scalar),
		unsafe.Pointer(&args.c),
		unsafe.Pointer(&args.n),
	}
	grid, block := launchGrid(n)
	if r := cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0,
		unsafe.Pointer(&params[0]), nil); r != CUDA_SUCCESS {
		return fmt.Errorf("cuLaunchKernel(%s): %s", fnName, r.Error())
	}
	return nil
}

// launchScalarF32 — (a, scalarF32, c, n). Scalar передан by-value через PTX .param.f32.
func (b *PuregoBackend) launchScalarF32(fnName string, a DeviceBuffer, scalar float32, c DeviceBuffer, n int) error {
	if n <= 0 {
		return fmt.Errorf("cuda.%s: n must be > 0, got %d", fnName, n)
	}
	if err := b.bind(); err != nil {
		return err
	}
	defer runtime.UnlockOSThread()
	fn, err := b.getKernel(fnName)
	if err != nil {
		return err
	}
	va, vc := a.deviceBuffer(), c.deviceBuffer()
	args := struct {
		a      uintptr
		scalar float32
		_pad   uint32 // выравнивание указателя c на 8 байт
		c      uintptr
		n      int32
		_      int32
	}{va.ptr, scalar, 0, vc.ptr, int32(n), 0}
	params := [4]unsafe.Pointer{
		unsafe.Pointer(&args.a),
		unsafe.Pointer(&args.scalar),
		unsafe.Pointer(&args.c),
		unsafe.Pointer(&args.n),
	}
	grid, block := launchGrid(n)
	if r := cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0,
		unsafe.Pointer(&params[0]), nil); r != CUDA_SUCCESS {
		return fmt.Errorf("cuLaunchKernel(%s): %s", fnName, r.Error())
	}
	return nil
}

// ──────────────────────────────────────────────────────────
// Управление устройством
// ──────────────────────────────────────────────────────────

func (b *PuregoBackend) Device() int { return b.device }

// bind — LockOSThread + cuCtxSetCurrent. Вызывающий ОБЯЗАН сразу после
// проверки err добавить `defer runtime.UnlockOSThread()`.
// Устраняет окно миграции горутины между cuCtxSetCurrent и следующим
// cu*-вызовом. Без LockOSThread bind() покрывал 99% путей, но при 20+
// повторах полного regression давал ~1 сбой INVALID_CONTEXT из ~6000
// итераций. С LockOSThread — 0/6000 (см. отчёт stage5).
func (b *PuregoBackend) bind() error {
	runtime.LockOSThread()
	if err := check(cuCtxSetCurrent(b.primaryCtx), "cuCtxSetCurrent"); err != nil {
		runtime.UnlockOSThread()
		return err
	}
	return nil
}

func (b *PuregoBackend) Sync() error {
	if err := b.bind(); err != nil {
		return err
	}
	defer runtime.UnlockOSThread()
	return check(cuCtxSynchronize(), "cuCtxSynchronize")
}

func (b *PuregoBackend) Close() error {
	// Порядок destroy: сначала PTX-модуль (использует контекст), потом
	// cuBLAS handle, потом Release primary-context. Другие пользователи
	// primary-контекста (goml/pytorch/...) продолжают жить: Release
	// уменьшает refcount, реальный destroy случится когда refcount
	// дойдёт до нуля.
	if b.ptxModule != 0 {
		if r := cuModuleUnload(b.ptxModule); r != CUDA_SUCCESS {
			return fmt.Errorf("cuModuleUnload: %s", r.Error())
		}
		b.ptxModule = 0
	}
	if b.cublas != 0 {
		if s := cublasDestroy_v2(b.cublas); s != CUBLAS_STATUS_SUCCESS {
			return fmt.Errorf("cublasDestroy_v2: %s", s.Error())
		}
		b.cublas = 0
	}
	return check(cuDevicePrimaryCtxRelease(int32(b.device)), "cuDevicePrimaryCtxRelease")
}

// ──────────────────────────────────────────────────────────
// Аллокация
// ──────────────────────────────────────────────────────────

func (b *PuregoBackend) Alloc(sizeBytes int) (Storage, error) {
	if sizeBytes <= 0 {
		return Storage{}, fmt.Errorf("cuda.Alloc: sizeBytes must be > 0, got %d", sizeBytes)
	}
	if err := b.bind(); err != nil {
		return Storage{}, err
	}
	defer runtime.UnlockOSThread()
	var dptr uintptr
	if err := check(cuMemAlloc(&dptr, uint64(sizeBytes)), "cuMemAlloc"); err != nil {
		return Storage{}, err
	}
	return Storage{
		ptr:       dptr,
		sizeBytes: sizeBytes,
		device:    b.device,
	}, nil
}

func (b *PuregoBackend) Free(s Storage) error {
	if s.ptr == 0 {
		return nil
	}
	if err := b.bind(); err != nil {
		return err
	}
	defer runtime.UnlockOSThread()
	return check(cuMemFree(s.ptr), "cuMemFree")
}

func (b *PuregoBackend) AllocPinned(sizeBytes int) (PinnedStorage, error) {
	if sizeBytes <= 0 {
		return PinnedStorage{}, fmt.Errorf("cuda.AllocPinned: sizeBytes must be > 0, got %d", sizeBytes)
	}
	if err := b.bind(); err != nil {
		return PinnedStorage{}, err
	}
	defer runtime.UnlockOSThread()
	var pp unsafe.Pointer
	if err := check(cuMemAllocHost(&pp, uint64(sizeBytes)), "cuMemAllocHost"); err != nil {
		return PinnedStorage{}, err
	}
	return PinnedStorage{ptr: pp, sizeBytes: sizeBytes}, nil
}

func (b *PuregoBackend) FreePinned(p PinnedStorage) error {
	if p.ptr == nil {
		return nil
	}
	if err := b.bind(); err != nil {
		return err
	}
	defer runtime.UnlockOSThread()
	return check(cuMemFreeHost(p.ptr), "cuMemFreeHost")
}

// ──────────────────────────────────────────────────────────
// Копирования
// ──────────────────────────────────────────────────────────

func (b *PuregoBackend) CopyH2D(dst DeviceBuffer, src []byte) error {
	v := dst.deviceBuffer()
	if len(src) != v.sizeBytes {
		return fmt.Errorf("cuda.CopyH2D: size mismatch — src=%d dst=%d", len(src), v.sizeBytes)
	}
	if len(src) == 0 {
		return nil
	}
	if err := b.bind(); err != nil {
		return err
	}
	defer runtime.UnlockOSThread()
	return check(
		cuMemcpyHtoD(uintptr(v.ptr), unsafe.Pointer(&src[0]), uint64(len(src))),
		"cuMemcpyHtoD",
	)
}

func (b *PuregoBackend) CopyD2H(dst []byte, src DeviceBuffer) error {
	v := src.deviceBuffer()
	if len(dst) != v.sizeBytes {
		return fmt.Errorf("cuda.CopyD2H: size mismatch — src=%d dst=%d", v.sizeBytes, len(dst))
	}
	if len(dst) == 0 {
		return nil
	}
	if err := b.bind(); err != nil {
		return err
	}
	defer runtime.UnlockOSThread()
	return check(
		cuMemcpyDtoH(unsafe.Pointer(&dst[0]), uintptr(v.ptr), uint64(len(dst))),
		"cuMemcpyDtoH",
	)
}

func (b *PuregoBackend) CopyH2DAsync(dst DeviceBuffer, src PinnedStorage, sizeBytes int) error {
	v := dst.deviceBuffer()
	if sizeBytes > v.sizeBytes || sizeBytes > src.sizeBytes {
		return fmt.Errorf("cuda.CopyH2DAsync: sizeBytes=%d exceeds dst=%d or src=%d",
			sizeBytes, v.sizeBytes, src.sizeBytes)
	}
	if sizeBytes <= 0 {
		return nil
	}
	if err := b.bind(); err != nil {
		return err
	}
	defer runtime.UnlockOSThread()
	// Stream 0 — default; Sync() покрывается cuCtxSynchronize.
	return check(
		cuMemcpyHtoDAsync(uintptr(v.ptr), src.ptr, uint64(sizeBytes), 0),
		"cuMemcpyHtoDAsync",
	)
}

func (b *PuregoBackend) CopyD2HAsync(dst PinnedStorage, src DeviceBuffer, sizeBytes int) error {
	v := src.deviceBuffer()
	if sizeBytes > dst.sizeBytes || sizeBytes > v.sizeBytes {
		return fmt.Errorf("cuda.CopyD2HAsync: sizeBytes=%d exceeds dst=%d or src=%d",
			sizeBytes, dst.sizeBytes, v.sizeBytes)
	}
	if sizeBytes <= 0 {
		return nil
	}
	if err := b.bind(); err != nil {
		return err
	}
	defer runtime.UnlockOSThread()
	return check(
		cuMemcpyDtoHAsync(dst.ptr, uintptr(v.ptr), uint64(sizeBytes), 0),
		"cuMemcpyDtoHAsync",
	)
}

func (b *PuregoBackend) CopyD2D(dst, src DeviceBuffer, sizeBytes int) error {
	vd, vs := dst.deviceBuffer(), src.deviceBuffer()
	if sizeBytes > vd.sizeBytes || sizeBytes > vs.sizeBytes {
		return fmt.Errorf("cuda.CopyD2D: sizeBytes=%d exceeds dst=%d or src=%d",
			sizeBytes, vd.sizeBytes, vs.sizeBytes)
	}
	if sizeBytes <= 0 {
		return nil
	}
	if err := b.bind(); err != nil {
		return err
	}
	defer runtime.UnlockOSThread()
	return check(
		cuMemcpyDtoD(uintptr(vd.ptr), uintptr(vs.ptr), uint64(sizeBytes)),
		"cuMemcpyDtoD",
	)
}

// ──────────────────────────────────────────────────────────
// Compute-методы: STUBS для этапов 2-5.
// Каждый вернёт errStagePending пока соответствующий этап не выполнен.
// ──────────────────────────────────────────────────────────

var errStage2Pending = errors.New("cuda: cuBLAS MatMul not implemented (R02b stage 2 pending)")
var errStage3Pending = errors.New("cuda: PTX kernel not implemented (R02b stage 3 pending)")
var errStage4Pending = errors.New("cuda: elementwise op not implemented (R02b stage 4 pending)")
var errStage5Pending = errors.New("cuda: activation/reduce op not implemented (R02b stage 5 pending)")

// Elementwise (Этапы 3-4 — все реализованы)
func (b *PuregoBackend) AddF64(a, bb, c DeviceBuffer, n int) error {
	return b.launchElementwise3("add_f64", a, bb, c, n)
}
func (b *PuregoBackend) AddF32(a, bb, c DeviceBuffer, n int) error {
	return b.launchElementwise3("add_f32", a, bb, c, n)
}
func (b *PuregoBackend) SubF64(a, bb, c DeviceBuffer, n int) error {
	return b.launchElementwise3("sub_f64", a, bb, c, n)
}
func (b *PuregoBackend) SubF32(a, bb, c DeviceBuffer, n int) error {
	return b.launchElementwise3("sub_f32", a, bb, c, n)
}
func (b *PuregoBackend) MulF64(a, bb, c DeviceBuffer, n int) error {
	return b.launchElementwise3("mul_f64", a, bb, c, n)
}
func (b *PuregoBackend) MulF32(a, bb, c DeviceBuffer, n int) error {
	return b.launchElementwise3("mul_f32", a, bb, c, n)
}
func (b *PuregoBackend) DivF64(a, bb, c DeviceBuffer, n int) error {
	return b.launchElementwise3("div_f64", a, bb, c, n)
}
func (b *PuregoBackend) DivF32(a, bb, c DeviceBuffer, n int) error {
	return b.launchElementwise3("div_f32", a, bb, c, n)
}
func (b *PuregoBackend) AddScalarF64(a DeviceBuffer, s float64, c DeviceBuffer, n int) error {
	return b.launchScalarF64("addscalar_f64", a, s, c, n)
}
func (b *PuregoBackend) AddScalarF32(a DeviceBuffer, s float32, c DeviceBuffer, n int) error {
	return b.launchScalarF32("addscalar_f32", a, s, c, n)
}
func (b *PuregoBackend) MulScalarF64(a DeviceBuffer, s float64, c DeviceBuffer, n int) error {
	return b.launchScalarF64("mulscalar_f64", a, s, c, n)
}
func (b *PuregoBackend) MulScalarF32(a DeviceBuffer, s float32, c DeviceBuffer, n int) error {
	return b.launchScalarF32("mulscalar_f32", a, s, c, n)
}
// ExpF64/LogF64 реализованы через fdlibm-порт в PTX (Этап 4.5).
// Схема идентична Go math.Exp/math.Log — тот же fdlibm происхождение,
// те же константы. Ожидание совпадения: единицы ulp.

func (b *PuregoBackend) ExpF64(a, c DeviceBuffer, n int) error {
	return b.launchElementwise2("exp_f64", a, c, n)
}
func (b *PuregoBackend) ExpF32(a, c DeviceBuffer, n int) error {
	return b.launchElementwise2("exp_f32", a, c, n)
}
func (b *PuregoBackend) LogF64(a, c DeviceBuffer, n int) error {
	return b.launchElementwise2("log_f64", a, c, n)
}
func (b *PuregoBackend) LogF32(a, c DeviceBuffer, n int) error {
	return b.launchElementwise2("log_f32", a, c, n)
}
func (b *PuregoBackend) NegF64(a, c DeviceBuffer, n int) error {
	return b.launchElementwise2("neg_f64", a, c, n)
}
func (b *PuregoBackend) NegF32(a, c DeviceBuffer, n int) error {
	return b.launchElementwise2("neg_f32", a, c, n)
}

// Activations (Этап 5 — non-composite)
func (b *PuregoBackend) ReLUF64(a, c DeviceBuffer, n int) error {
	return b.launchElementwise2("relu_f64", a, c, n)
}
func (b *PuregoBackend) ReLUF32(a, c DeviceBuffer, n int) error {
	return b.launchElementwise2("relu_f32", a, c, n)
}
func (b *PuregoBackend) SigmoidF64(a, c DeviceBuffer, n int) error {
	return b.launchElementwise2("sigmoid_f64", a, c, n)
}
func (b *PuregoBackend) SigmoidF32(a, c DeviceBuffer, n int) error {
	return b.launchElementwise2("sigmoid_f32", a, c, n)
}
func (b *PuregoBackend) TanhF64(a, c DeviceBuffer, n int) error {
	return b.launchElementwise2("tanh_f64", a, c, n)
}
func (b *PuregoBackend) TanhF32(a, c DeviceBuffer, n int) error {
	return b.launchElementwise2("tanh_f32", a, c, n)
}
func (b *PuregoBackend) ReLUGradF64(input, grad, out DeviceBuffer, n int) error {
	return b.launchElementwise3("relu_grad_f64", input, grad, out, n)
}
func (b *PuregoBackend) ReLUGradF32(input, grad, out DeviceBuffer, n int) error {
	return b.launchElementwise3("relu_grad_f32", input, grad, out, n)
}
func (b *PuregoBackend) SigmoidGradF64(sigOut, grad, out DeviceBuffer, n int) error {
	return b.launchElementwise3("sigmoid_grad_f64", sigOut, grad, out, n)
}
func (b *PuregoBackend) SigmoidGradF32(sigOut, grad, out DeviceBuffer, n int) error {
	return b.launchElementwise3("sigmoid_grad_f32", sigOut, grad, out, n)
}
func (b *PuregoBackend) TanhGradF64(tanhOut, grad, out DeviceBuffer, n int) error {
	return b.launchElementwise3("tanh_grad_f64", tanhOut, grad, out, n)
}
func (b *PuregoBackend) TanhGradF32(tanhOut, grad, out DeviceBuffer, n int) error {
	return b.launchElementwise3("tanh_grad_f32", tanhOut, grad, out, n)
}
func (b *PuregoBackend) SoftmaxF64(a, c DeviceBuffer, rows, cols int) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cuda.SoftmaxF64: rows/cols must be > 0")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := check(cuCtxSetCurrent(b.primaryCtx), "cuCtxSetCurrent"); err != nil {
		return err
	}
	return b.launchSoftmax("softmax_f64", a, c, rows, cols)
}
func (b *PuregoBackend) SoftmaxF32(a, c DeviceBuffer, rows, cols int) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cuda.SoftmaxF32: rows/cols must be > 0")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := check(cuCtxSetCurrent(b.primaryCtx), "cuCtxSetCurrent"); err != nil {
		return err
	}
	return b.launchSoftmax("softmax_f32", a, c, rows, cols)
}

// ──────────────────────────────────────────────────────────
// Reduce F64/F32 — составные операции (Этап 5)
//
// Обоснование bind-стратегии: outer LockOSThread + defer UnlockOSThread
// на весь Sum/Mean/Softmax-метод. Все внутренние вызовы (Alloc, launch,
// CopyD2H, Free) сами делают LockOSThread + defer Unlock — это NESTED
// LockOSThread, Go docs явно разрешают (counter-based). Внешний Lock
// удерживает goroutine на одном OS thread до конца составной операции,
// пока nested Locks увеличивают счётчик; после return из внутренних
// методов defer'ы уменьшают счётчик, но outer defer держит его >= 1
// до конца композита. Race миграции между sub-операциями невозможен.
// ──────────────────────────────────────────────────────────

// launchReduce1 — 1-block launch с block=256 для kernel вида (a, out_scalar, n).
func (b *PuregoBackend) launchReduce1(fnName string, a DeviceBuffer, outPtr uintptr, n int) error {
	fn, err := b.getKernel(fnName)
	if err != nil {
		return err
	}
	va := a.deviceBuffer()
	args := struct {
		a   uintptr
		out uintptr
		n   int32
		_   int32
	}{va.ptr, outPtr, int32(n), 0}
	params := [3]unsafe.Pointer{
		unsafe.Pointer(&args.a),
		unsafe.Pointer(&args.out),
		unsafe.Pointer(&args.n),
	}
	if r := cuLaunchKernel(fn,
		1, 1, 1,
		256, 1, 1,
		0, 0,
		unsafe.Pointer(&params[0]),
		nil,
	); r != CUDA_SUCCESS {
		return fmt.Errorf("cuLaunchKernel(%s): %s", fnName, r.Error())
	}
	return nil
}

// launchSoftmax — 1-block-per-row launch для softmax.
func (b *PuregoBackend) launchSoftmax(fnName string, a, c DeviceBuffer, rows, cols int) error {
	fn, err := b.getKernel(fnName)
	if err != nil {
		return err
	}
	va, vc := a.deviceBuffer(), c.deviceBuffer()
	args := struct {
		a    uintptr
		c    uintptr
		rows int32
		cols int32
	}{va.ptr, vc.ptr, int32(rows), int32(cols)}
	params := [4]unsafe.Pointer{
		unsafe.Pointer(&args.a),
		unsafe.Pointer(&args.c),
		unsafe.Pointer(&args.rows),
		unsafe.Pointer(&args.cols),
	}
	if r := cuLaunchKernel(fn,
		uint32(rows), 1, 1,
		256, 1, 1,
		0, 0,
		unsafe.Pointer(&params[0]),
		nil,
	); r != CUDA_SUCCESS {
		return fmt.Errorf("cuLaunchKernel(%s): %s", fnName, r.Error())
	}
	return nil
}


func (b *PuregoBackend) SumF64(a DeviceBuffer, n int) (float64, error) {
	if n <= 0 {
		return 0, fmt.Errorf("cuda.SumF64: n must be > 0, got %d", n)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := check(cuCtxSetCurrent(b.primaryCtx), "cuCtxSetCurrent"); err != nil {
		return 0, err
	}
	tmp, err := b.Alloc(8)
	if err != nil {
		return 0, err
	}
	defer b.Free(tmp)
	if err := b.launchReduce1("sum_f64", a, tmp.ptr, n); err != nil {
		return 0, err
	}
	if err := check(cuCtxSynchronize(), "cuCtxSynchronize"); err != nil {
		return 0, err
	}
	buf := make([]byte, 8)
	if err := b.CopyD2H(buf, tmp); err != nil {
		return 0, err
	}
	return math.Float64frombits(binary.LittleEndian.Uint64(buf)), nil
}

func (b *PuregoBackend) SumF32(a DeviceBuffer, n int) (float32, error) {
	if n <= 0 {
		return 0, fmt.Errorf("cuda.SumF32: n must be > 0, got %d", n)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := check(cuCtxSetCurrent(b.primaryCtx), "cuCtxSetCurrent"); err != nil {
		return 0, err
	}
	tmp, err := b.Alloc(4)
	if err != nil {
		return 0, err
	}
	defer b.Free(tmp)
	if err := b.launchReduce1("sum_f32", a, tmp.ptr, n); err != nil {
		return 0, err
	}
	if err := check(cuCtxSynchronize(), "cuCtxSynchronize"); err != nil {
		return 0, err
	}
	buf := make([]byte, 4)
	if err := b.CopyD2H(buf, tmp); err != nil {
		return 0, err
	}
	return math.Float32frombits(binary.LittleEndian.Uint32(buf)), nil
}

func (b *PuregoBackend) MeanF64(a DeviceBuffer, n int) (float64, error) {
	s, err := b.SumF64(a, n)
	if err != nil {
		return 0, err
	}
	return s / float64(n), nil
}

func (b *PuregoBackend) MeanF32(a DeviceBuffer, n int) (float32, error) {
	s, err := b.SumF32(a, n)
	if err != nil {
		return 0, err
	}
	return s / float32(n), nil
}

// ──────────────────────────────────────────────────────────
// Linalg — cuBLAS DGEMM/SGEMM с row-major трюком (Этап 2)
// ──────────────────────────────────────────────────────────
//
// Наш контракт: C[MxN] = A[MxK] × B[KxN], хранение row-major.
// cuBLAS работает в col-major. Прямой вызов cublasDgemm с (M,N,K, A, B, C)
// в row-major-указателях интерпретировал бы данные как A^T[K×M], B^T[N×K],
// что даёт (A^T @ B^T)^T = B @ A — не то что нужно.
//
// Стандартный трюк — вычислить C^T = B^T @ A^T через col-major cuBLAS,
// СВАПНУВ операнды. В col-major-интерпретации:
//   * row-major A[M×K] выглядит как col-major A^T[K×M] размера (rows=K, cols=M);
//   * row-major B[K×N] → col-major B^T[N×K] размера (rows=N, cols=K);
//   * row-major C[M×N] → col-major C^T[N×M] размера (rows=N, cols=M).
//
// GEMM в col-major: C = op(A) @ op(B), размеры m×k, k×n → m×n.
// Хотим C^T[N×M] = B^T[N×K] @ A^T[K×M] → m=N, n=M, k=K.
// Оба op = OP_N (транспоны уже сидят в самом факте row/col-major сдвига).
// Первый операнд cuBLAS — row-major B (view B^T[N×K]), lda = N.
// Второй операнд cuBLAS — row-major A (view A^T[K×M]), ldb = K.
// Результат cuBLAS — row-major C (view C^T[N×M]), ldc = N.
//
// Сверено с goml/backend/cuda/cublas.go:MatMulF32 (тот же свап + N,M,K).

func (b *PuregoBackend) MatMulF64(a, bb, c DeviceBuffer, m, n, k int) error {
	if m <= 0 || n <= 0 || k <= 0 {
		return fmt.Errorf("cuda.MatMulF64: m/n/k must be > 0, got %d/%d/%d", m, n, k)
	}
	if err := b.bind(); err != nil {
		return err
	}
	defer runtime.UnlockOSThread()
	alpha, beta := 1.0, 0.0
	va, vb, vc := a.deviceBuffer(), bb.deviceBuffer(), c.deviceBuffer()
	s := cublasDgemm_v2(
		b.cublas,
		CUBLAS_OP_N, CUBLAS_OP_N,
		int32(n), int32(m), int32(k),
		unsafe.Pointer(&alpha),
		vb.ptr, int32(n), // B_row [K×N] as col-major B^T [N×K], lda=N
		va.ptr, int32(k), // A_row [M×K] as col-major A^T [K×M], ldb=K
		unsafe.Pointer(&beta),
		vc.ptr, int32(n), // C_row [M×N] as col-major C^T [N×M], ldc=N
	)
	if s != CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasDgemm_v2: %s", s.Error())
	}
	return nil
}

func (b *PuregoBackend) MatMulF32(a, bb, c DeviceBuffer, m, n, k int) error {
	if m <= 0 || n <= 0 || k <= 0 {
		return fmt.Errorf("cuda.MatMulF32: m/n/k must be > 0, got %d/%d/%d", m, n, k)
	}
	if err := b.bind(); err != nil {
		return err
	}
	defer runtime.UnlockOSThread()
	alpha, beta := float32(1.0), float32(0.0)
	va, vb, vc := a.deviceBuffer(), bb.deviceBuffer(), c.deviceBuffer()
	s := cublasSgemm_v2(
		b.cublas,
		CUBLAS_OP_N, CUBLAS_OP_N,
		int32(n), int32(m), int32(k),
		unsafe.Pointer(&alpha),
		vb.ptr, int32(n),
		va.ptr, int32(k),
		unsafe.Pointer(&beta),
		vc.ptr, int32(n),
	)
	if s != CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasSgemm_v2: %s", s.Error())
	}
	return nil
}
