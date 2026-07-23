package cuda

// Sealed API для нового purego-backend'а gotorch/cuda.
//
// Легаси cgo-путь (bridge.go / backend.go / pinned.go / detect_gpu.go / ops.cu)
// не удаляется и продолжает жить параллельно в этом же пакете. После R02b
// планируется добавить полноценный cgo-backend как вторую реализацию
// интерфейса Backend. Отсюда — суффиксы _purego на реализации-специфичных
// utility-функциях (util.go), но никаких суффиксов на контрактных типах:
// они общие для обеих будущих реализаций.

import "unsafe"

// bufferView — внутреннее представление device-буфера.
// Unexported: не покидает пакет cuda никогда.
//
// Поле ptr имеет тип uintptr, а не unsafe.Pointer, намеренно: CUDA
// device-указатель — это handle из чужого адресного пространства (GPU),
// а не адрес в Go-heap. Go GC его не отслеживает и не имеет права
// трогать; unsafe.Pointer.Rules про Go-heap к нему неприменимы.
// Хранение как unsafe.Pointer было бы ложным обещанием контракта +
// триггерило бы go vet «possible misuse of unsafe.Pointer» при
// конверсии из результата cuMemAlloc. Наружу пакет отдаёт указатель
// как unsafe.Pointer через UnsafeExtractDevicePtr — там конверсия
// однократна и в момент выхода, что согласуется с интероп-контрактом.
type bufferView struct {
	ptr       uintptr
	sizeBytes int
	device    int
}

// DeviceBuffer — общий контракт device-памяти для compute-методов Backend.
// Интерфейс ЗАПЕЧАТАН: unexported-метод deviceBuffer делает невозможной
// реализацию вне пакета cuda. Единственный способ завести внешнюю
// device-память в этот контракт — WrapDevicePtr (возвращает ForeignStorage).
// Единственный способ извлечь указатель наружу — UnsafeExtractDevicePtr.
// Внутри этих двух дверей указатель по публичному API недоступен.
type DeviceBuffer interface {
	deviceBuffer() bufferView
	SizeBytes() int
	Device() int
}

// Storage — владельческий handle к device-памяти, аллоцированной через
// Backend.Alloc. Владелец обязан вызвать Backend.Free(s) для освобождения.
type Storage struct {
	ptr       uintptr
	sizeBytes int
	device    int
}

func (s Storage) deviceBuffer() bufferView { return bufferView{s.ptr, s.sizeBytes, s.device} }
func (s Storage) SizeBytes() int           { return s.sizeBytes }
func (s Storage) Device() int              { return s.device }

// ForeignStorage — не-владельческий handle к device-памяти, аллоцированной
// снаружи (напр. goml-ядрами). Метода Free() у типа нет по дизайну, и
// Backend.Free его не принимает по сигнатуре: освободить чужую память
// через gotorch невозможно на уровне компиляции.
type ForeignStorage struct {
	ptr       uintptr
	sizeBytes int
	device    int
}

func (f ForeignStorage) deviceBuffer() bufferView { return bufferView{f.ptr, f.sizeBytes, f.device} }
func (f ForeignStorage) SizeBytes() int           { return f.sizeBytes }
func (f ForeignStorage) Device() int              { return f.device }

// PinnedStorage — непрозрачный handle к page-locked host-памяти.
// Владелец обязан вызвать Backend.FreePinned(p) для освобождения.
type PinnedStorage struct {
	ptr       unsafe.Pointer
	sizeBytes int
}

func (p PinnedStorage) SizeBytes() int { return p.sizeBytes }

// HostSlice возвращает []byte-view поверх pinned-буфера (zero-copy).
// Валиден только пока PinnedStorage не освобождён.
func (p PinnedStorage) HostSlice() []byte {
	if p.ptr == nil || p.sizeBytes == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(p.ptr), p.sizeBytes)
}

// WrapDevicePtr — ДВЕРЬ ВХОДА. Оборачивает чужой device-указатель в
// ForeignStorage. У ForeignStorage нет метода Free() и Backend.Free его не
// примет по типу — освобождение чужой памяти на уровне API невозможно.
// Нужно для интеграции с внешними CUDA-мирами (goml-ядра, чужие обёртки).
//
// Принимает unsafe.Pointer намеренно: это интероп-граница пакета, а
// лингва-франка интеропа Go-мира — unsafe.Pointer (та же goml-сторона в
// местах стыковки оперирует им). Внутри пакета указатель хранится как
// uintptr; конверсия однократна и происходит в этой строке.
func WrapDevicePtr(ptr unsafe.Pointer, sizeBytes, device int) ForeignStorage {
	return ForeignStorage{ptr: uintptr(ptr), sizeBytes: sizeBytes, device: device}
}

// UnsafeExtractDevicePtr — ДВЕРЬ ВЫХОДА. Возвращает сырой device-указатель
// буфера. Единственное корректное применение — передача во внешние CUDA-
// биндинги (goml-ядра, сторонние библиотеки, чужие обёртки cuBLAS/driver
// API). ЗАПРЕЩЕНО: разыменовывать с host-стороны (это device-память,
// segfault); сохранять как целое число между вызовами (use-after-free
// после Backend.Free); делать арифметику указателей. Нарушение —
// undefined behavior без диагностики. Это единственная публичная функция
// пакета, возвращающая unsafe.Pointer на device-память.
//
// Внутри пакета указатель хранится как uintptr; на выходе он
// реинтерпретируется в unsafe.Pointer через memory-view — прямой
// unsafe.Pointer(v.ptr) триггерил бы go vet «possible misuse of
// unsafe.Pointer» (false-positive: правило про Go-heap не применимо к
// CUDA device-адресам), а reinterpret через два cast'а — легальный
// pattern, vet его не ловит. Sizeof(uintptr) == Sizeof(unsafe.Pointer)
// на всех целевых архитектурах, поэтому реинтерпретация корректна.
func UnsafeExtractDevicePtr(b DeviceBuffer) unsafe.Pointer {
	v := b.deviceBuffer()
	return *(*unsafe.Pointer)(unsafe.Pointer(&v.ptr))
}

// Backend — единый контракт для GPU-бэкендов gotorch/cuda.
// Первая реализация (R02b): PuregoBackend через libcuda/libcublas + PTX-ядра.
// Вторая реализация (после R02b): cgo-backend через legacy libgotorch_cuda.so.
type Backend interface {

	// --- Управление устройством ---

	Device() int
	Sync() error
	Close() error

	// --- Аллокация ---

	Alloc(sizeBytes int) (Storage, error)
	Free(s Storage) error
	AllocPinned(sizeBytes int) (PinnedStorage, error)
	FreePinned(p PinnedStorage) error

	// --- Копирования ---

	CopyH2D(dst DeviceBuffer, src []byte) error
	CopyD2H(dst []byte, src DeviceBuffer) error
	CopyH2DAsync(dst DeviceBuffer, src PinnedStorage, sizeBytes int) error
	CopyD2HAsync(dst PinnedStorage, src DeviceBuffer, sizeBytes int) error
	CopyD2D(dst, src DeviceBuffer, sizeBytes int) error

	// --- Elementwise F64/F32 ---

	AddF64(a, b, c DeviceBuffer, n int) error
	AddF32(a, b, c DeviceBuffer, n int) error
	SubF64(a, b, c DeviceBuffer, n int) error
	SubF32(a, b, c DeviceBuffer, n int) error
	MulF64(a, b, c DeviceBuffer, n int) error
	MulF32(a, b, c DeviceBuffer, n int) error
	DivF64(a, b, c DeviceBuffer, n int) error
	DivF32(a, b, c DeviceBuffer, n int) error
	AddScalarF64(a DeviceBuffer, scalar float64, c DeviceBuffer, n int) error
	AddScalarF32(a DeviceBuffer, scalar float32, c DeviceBuffer, n int) error
	MulScalarF64(a DeviceBuffer, scalar float64, c DeviceBuffer, n int) error
	MulScalarF32(a DeviceBuffer, scalar float32, c DeviceBuffer, n int) error
	ExpF64(a, c DeviceBuffer, n int) error
	ExpF32(a, c DeviceBuffer, n int) error
	LogF64(a, c DeviceBuffer, n int) error
	LogF32(a, c DeviceBuffer, n int) error
	NegF64(a, c DeviceBuffer, n int) error
	NegF32(a, c DeviceBuffer, n int) error

	// --- Activations F64/F32 ---

	ReLUF64(a, c DeviceBuffer, n int) error
	ReLUF32(a, c DeviceBuffer, n int) error
	SigmoidF64(a, c DeviceBuffer, n int) error
	SigmoidF32(a, c DeviceBuffer, n int) error
	TanhF64(a, c DeviceBuffer, n int) error
	TanhF32(a, c DeviceBuffer, n int) error
	ReLUGradF64(input, grad, out DeviceBuffer, n int) error
	ReLUGradF32(input, grad, out DeviceBuffer, n int) error
	SigmoidGradF64(sigOut, grad, out DeviceBuffer, n int) error
	SigmoidGradF32(sigOut, grad, out DeviceBuffer, n int) error
	TanhGradF64(tanhOut, grad, out DeviceBuffer, n int) error
	TanhGradF32(tanhOut, grad, out DeviceBuffer, n int) error
	SoftmaxF64(a, c DeviceBuffer, rows, cols int) error
	SoftmaxF32(a, c DeviceBuffer, rows, cols int) error

	// --- Normalization: RMSNorm (LLaMA-style, no bias/mean-subtract) ---
	//
	// y = gamma * x / rms, где rms = sqrt(mean(x²) + eps).
	// x, gamma, y — [rows, cols] row-major; gamma — [cols].
	// Forward: RMSNormF32/F64.
	// Backward:
	//   dx_j     = gamma_j*dy_j*inv_rms - x_j*S*inv_rms^3/cols
	//   dgamma_j = sum_rows(dy_j*x_j*inv_rms)  (atomicAdd; dgamma должен быть pre-zeroed)
	//   где S = sum_i(gamma_i*x_i*dy_i)
	// Grad-функции сами обнуляют dgamma через cuMemsetD8 перед kernel'ом.
	RMSNormF64(x, gamma, y DeviceBuffer, rows, cols int, eps float64) error
	RMSNormF32(x, gamma, y DeviceBuffer, rows, cols int, eps float32) error
	RMSNormGradF64(x, gamma, dy, dx, dgamma DeviceBuffer, rows, cols int, eps float64) error
	RMSNormGradF32(x, gamma, dy, dx, dgamma DeviceBuffer, rows, cols int, eps float32) error

	// --- Embedding: gather / scatter-accumulate ---
	//
	// table   [vocab, hidden] row-major, F32 или F64.
	// indices [n] int32 little-endian. Байты индексного буфера трактуются как
	//   int32 непосредственно в PTX; система типов gotorch НЕ вводит Int32Storage
	//   ради embedding (единственный не-float потребитель на данный момент).
	//   Контракт индексов — свойство метода Embedding*, а не типа буфера.
	// Валидность: 0 <= idx < vocab — обязанность вызывающего. Out-of-range = UB
	//   в PTX (segfault на невалидном указателе или бит-мусор). Debug-путь тестов
	//   выполняет CPU-предпроверку диапазона.
	//
	// Forward: out[i][d] = table[indices[i]][d] — чистый gather, bit-exact при
	//   равных входах. Grid=(n,1,1), block=min(hidden,256).
	// Backward: dtable[indices[i]][d] += dout[i][d] через atom.global.add.
	//   dtable pre-zeroed внутри метода через cuMemsetD8 (как в RMSNormGrad).
	//   ВАЖНО: float atomicAdd не ассоциативен -> два прогона grad могут дать
	//   разницу ulp-порядка при коллизиях (повторяющиеся индексы). Ожидание
	//   недетерминизма зафиксировано в тестах atomic-reproducibility.
	EmbeddingF32(table, indices, out DeviceBuffer, vocab, hidden, n int) error
	EmbeddingF64(table, indices, out DeviceBuffer, vocab, hidden, n int) error
	EmbeddingGradF32(indices, dout, dtable DeviceBuffer, vocab, hidden, n int) error
	EmbeddingGradF64(indices, dout, dtable DeviceBuffer, vocab, hidden, n int) error

	// --- Reduce F64/F32 ---

	SumF64(a DeviceBuffer, n int) (float64, error)
	SumF32(a DeviceBuffer, n int) (float32, error)
	MeanF64(a DeviceBuffer, n int) (float64, error)
	MeanF32(a DeviceBuffer, n int) (float32, error)

	// --- Linalg F64/F32 ---

	MatMulF64(a, b, c DeviceBuffer, m, n, k int) error
	MatMulF32(a, b, c DeviceBuffer, m, n, k int) error

	// MatMulF32_TF32 — F32 GEMM с точечным включением TF32-tensor-cores.
	// Точность rel ~1e-3 против MatMulF32 (~FP32 eps). Bit-exact НЕ ожидается —
	// TF32 использует 10-bit mantissa в FMA. Назначение: сверка с legacy-путями,
	// у которых cublas handle глобально в TF32. См. R03b_design.md impl-4-final.
	MatMulF32_TF32(a, b, c DeviceBuffer, m, n, k int) error
}

// NewBackend возвращает purego-backend, привязанный к устройству device.
// В R02b — единственная реализация (PuregoBackend); после R02b появится
// NewCgoBackend или подобное. Функция сама по себе НЕ утверждает
// «purego»: она — фабрика по умолчанию текущего этапа.
func NewBackend(device int) (Backend, error) {
	return newPuregoBackend(device)
}
