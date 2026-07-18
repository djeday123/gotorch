package cuda

// Тесты Ворот 1 R02b — sealed-контракт + память + копирования.
// Требуют реальный CUDA GPU: если newPuregoBackend вернул ошибку про
// libcuda / no device — t.Skip, не считать провалом.

import (
	"encoding/binary"
	"math"
	"os"
	"strings"
	"testing"
	"unsafe"
)

// tryBackend создаёт PuregoBackend или пропускает тест с внятным сообщением.
func tryBackend(t *testing.T) *PuregoBackend {
	t.Helper()
	b, err := newPuregoBackend(0)
	if err != nil {
		t.Skipf("cuda unavailable, skipping: %v", err)
	}
	return b
}

// f64Bytes сериализует []float64 в little-endian байтовый буфер.
func f64Bytes(v []float64) []byte {
	buf := make([]byte, 8*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(x))
	}
	return buf
}

// bytesF64 десериализует байтовый буфер обратно в []float64.
func bytesF64(buf []byte) []float64 {
	v := make([]float64, len(buf)/8)
	for i := range v {
		v[i] = math.Float64frombits(binary.LittleEndian.Uint64(buf[i*8:]))
	}
	return v
}

// f32Bytes / bytesF32 — аналогично для float32.
func f32Bytes(v []float32) []byte {
	buf := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(x))
	}
	return buf
}
func bytesF32(buf []byte) []float32 {
	v := make([]float32, len(buf)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return v
}

// TestMemoryRoundtrip — Alloc → H2D → D2H → сравнение байт-в-байт.
// Паттерн — арифметическая прогрессия (не нули, чтобы поймать «забыли
// скопировать → буфер уже такой»).
func TestMemoryRoundtrip(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	// F64 пачка
	nF64 := 1024
	srcF64 := make([]float64, nF64)
	for i := range srcF64 {
		srcF64[i] = float64(i)*0.5 - 3.14
	}
	srcBytesF64 := f64Bytes(srcF64)

	s64, err := b.Alloc(len(srcBytesF64))
	if err != nil {
		t.Fatalf("Alloc F64: %v", err)
	}
	defer b.Free(s64)

	if err := b.CopyH2D(s64, srcBytesF64); err != nil {
		t.Fatalf("CopyH2D F64: %v", err)
	}

	dstBytesF64 := make([]byte, len(srcBytesF64))
	if err := b.CopyD2H(dstBytesF64, s64); err != nil {
		t.Fatalf("CopyD2H F64: %v", err)
	}

	for i := range srcBytesF64 {
		if srcBytesF64[i] != dstBytesF64[i] {
			t.Fatalf("F64 byte %d: src=%d dst=%d", i, srcBytesF64[i], dstBytesF64[i])
		}
	}
	got64 := bytesF64(dstBytesF64)
	for i, v := range got64 {
		if v != srcF64[i] {
			t.Fatalf("F64 elem %d: src=%g got=%g", i, srcF64[i], v)
		}
	}

	// F32 пачка
	nF32 := 2048
	srcF32 := make([]float32, nF32)
	for i := range srcF32 {
		srcF32[i] = float32(i)*0.25 + 1.0
	}
	srcBytesF32 := f32Bytes(srcF32)

	s32, err := b.Alloc(len(srcBytesF32))
	if err != nil {
		t.Fatalf("Alloc F32: %v", err)
	}
	defer b.Free(s32)

	if err := b.CopyH2D(s32, srcBytesF32); err != nil {
		t.Fatalf("CopyH2D F32: %v", err)
	}
	dstBytesF32 := make([]byte, len(srcBytesF32))
	if err := b.CopyD2H(dstBytesF32, s32); err != nil {
		t.Fatalf("CopyD2H F32: %v", err)
	}
	got32 := bytesF32(dstBytesF32)
	for i, v := range got32 {
		if v != srcF32[i] {
			t.Fatalf("F32 elem %d: src=%g got=%g", i, srcF32[i], v)
		}
	}
}

// TestPinnedRoundtrip — то же через pinned + async + Sync.
func TestPinnedRoundtrip(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	n := 512
	src := make([]float64, n)
	for i := range src {
		src[i] = float64(i) * 2.0
	}
	srcBytes := f64Bytes(src)

	// Pinned host: заполняем через HostSlice().
	pin, err := b.AllocPinned(len(srcBytes))
	if err != nil {
		t.Fatalf("AllocPinned: %v", err)
	}
	defer b.FreePinned(pin)
	copy(pin.HostSlice(), srcBytes)

	// Device buffer.
	dev, err := b.Alloc(len(srcBytes))
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	defer b.Free(dev)

	// pinned → device (async), sync, device → pinned2 (async), sync.
	if err := b.CopyH2DAsync(dev, pin, len(srcBytes)); err != nil {
		t.Fatalf("CopyH2DAsync: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync after H2DAsync: %v", err)
	}

	pin2, err := b.AllocPinned(len(srcBytes))
	if err != nil {
		t.Fatalf("AllocPinned dst: %v", err)
	}
	defer b.FreePinned(pin2)

	if err := b.CopyD2HAsync(pin2, dev, len(srcBytes)); err != nil {
		t.Fatalf("CopyD2HAsync: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync after D2HAsync: %v", err)
	}

	got := bytesF64(pin2.HostSlice())
	for i, v := range got {
		if v != src[i] {
			t.Fatalf("pinned roundtrip elem %d: src=%g got=%g", i, src[i], v)
		}
	}
}

// TestSealedInterface — проверка запечатывания DeviceBuffer.
//
// Тип вне пакета cuda не может реализовать DeviceBuffer:
//
//	// в чужом пакете:
//	type fake struct{}
//	func (fake) deviceBuffer() bufferView { return bufferView{} } // НЕ КОМПИЛИРУЕТСЯ:
//	  bufferView — unexported, метод deviceBuffer нельзя объявить вне cuda
//	func (fake) SizeBytes() int { return 0 }
//	func (fake) Device() int    { return 0 }
//
// Печать проверена конструкцией языка, рантайм-теста для неё не нужен.
// Здесь — позитивный тест: Storage и ForeignStorage присваиваются в
// переменную DeviceBuffer; round-trip WrapDevicePtr(UnsafeExtractDevicePtr(s))
// даёт ForeignStorage, через который CopyD2H читает те же байты.
func TestSealedInterface(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	// Позитив: Storage → DeviceBuffer.
	src := make([]float64, 64)
	for i := range src {
		src[i] = float64(i) - 10.0
	}
	srcBytes := f64Bytes(src)

	s, err := b.Alloc(len(srcBytes))
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	defer b.Free(s)
	if err := b.CopyH2D(s, srcBytes); err != nil {
		t.Fatalf("CopyH2D: %v", err)
	}

	var db DeviceBuffer = s
	if db.SizeBytes() != len(srcBytes) {
		t.Fatalf("SizeBytes via DeviceBuffer: got %d want %d", db.SizeBytes(), len(srcBytes))
	}
	if db.Device() != 0 {
		t.Fatalf("Device via DeviceBuffer: got %d want 0", db.Device())
	}

	// Round-trip: extract ptr, wrap back as ForeignStorage,
	// прочитать D2H через foreign — данные те же.
	ptr := UnsafeExtractDevicePtr(s)
	if ptr == nil {
		t.Fatalf("UnsafeExtractDevicePtr returned nil for live Storage")
	}
	foreign := WrapDevicePtr(ptr, s.SizeBytes(), s.Device())

	var db2 DeviceBuffer = foreign
	if db2.SizeBytes() != len(srcBytes) {
		t.Fatalf("foreign SizeBytes: got %d want %d", db2.SizeBytes(), len(srcBytes))
	}

	dstBytes := make([]byte, len(srcBytes))
	if err := b.CopyD2H(dstBytes, foreign); err != nil {
		t.Fatalf("CopyD2H via foreign: %v", err)
	}
	got := bytesF64(dstBytes)
	for i, v := range got {
		if v != src[i] {
			t.Fatalf("foreign readback elem %d: src=%g got=%g", i, src[i], v)
		}
	}
}

// TestFreeForeignNotCompilable — символическая проверка, что Backend.Free
// принимает только Storage, не ForeignStorage. Строка ниже
// закомментирована; если её раскомментировать, сборка упадёт с
//
//	cannot use foreign (variable of type ForeignStorage) as Storage value
//	in argument to backend.Free
//
// — что и требуется по контракту двух дверей.
func TestFreeForeignNotCompilable(t *testing.T) {
	// var backend Backend
	// var foreign ForeignStorage
	// backend.Free(foreign) // <-- этот вызов НЕ компилируется по сигнатуре
	_ = unsafe.Sizeof(ForeignStorage{}) // suppress unused-package for tests
}

// TestNoUintptrInPublicAPI — уточнённый контракт: uintptr запрещён в
// СИГНАТУРАХ экспортируемых функций/методов api.go и util.go. В полях
// unexported-структур (bufferView.ptr / Storage.ptr / ForeignStorage.ptr)
// uintptr легален с R02b Stage 1.5 (device-handle, не Go pointer;
// unsafe.Pointer.Rules про Go-heap неприменимы к CUDA device-адресам —
// подробно в комментарии bufferView).
//
// Практическая проверка: построчно, паттерн «func ... uintptr ...» в
// одной строке считается сигнатурой с utility-типом наружу.
// Дополнительно: обе двери (WrapDevicePtr / UnsafeExtractDevicePtr)
// должны в своих сигнатурах использовать unsafe.Pointer.
func TestNoUintptrInPublicAPI(t *testing.T) {
	for _, path := range []string{"api.go", "util.go"} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile(%s): %v", path, err)
		}
		for i, line := range strings.Split(string(data), "\n") {
			if strings.Contains(line, "func ") && strings.Contains(line, "uintptr") {
				t.Fatalf("%s:%d — signature contains uintptr: %q", path, i+1, strings.TrimSpace(line))
			}
		}
	}
	apiData, err := os.ReadFile("api.go")
	if err != nil {
		t.Fatalf("ReadFile(api.go): %v", err)
	}
	src := string(apiData)
	if !strings.Contains(src, "func WrapDevicePtr(ptr unsafe.Pointer,") {
		t.Fatal("api.go: WrapDevicePtr must accept unsafe.Pointer as first arg")
	}
	if !strings.Contains(src, "func UnsafeExtractDevicePtr(b DeviceBuffer) unsafe.Pointer") {
		t.Fatal("api.go: UnsafeExtractDevicePtr must return unsafe.Pointer")
	}
}

// TestCopyD2D — прямая проверка device-to-device копии.
// Схема: H2D паттерн A → D2D A→B → D2H(B) → сравнение байт-в-байт.
// Оба dtype-паттерна (F64 арифметическая прогрессия + F32 та же).
func TestCopyD2D(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	// F64
	nF64 := 512
	srcF64 := make([]float64, nF64)
	for i := range srcF64 {
		srcF64[i] = float64(i)*0.1 - 7.5
	}
	srcBytesF64 := f64Bytes(srcF64)

	a64, err := b.Alloc(len(srcBytesF64))
	if err != nil {
		t.Fatalf("Alloc a64: %v", err)
	}
	defer b.Free(a64)
	b64, err := b.Alloc(len(srcBytesF64))
	if err != nil {
		t.Fatalf("Alloc b64: %v", err)
	}
	defer b.Free(b64)

	if err := b.CopyH2D(a64, srcBytesF64); err != nil {
		t.Fatalf("CopyH2D F64: %v", err)
	}
	if err := b.CopyD2D(b64, a64, len(srcBytesF64)); err != nil {
		t.Fatalf("CopyD2D F64: %v", err)
	}
	dstBytesF64 := make([]byte, len(srcBytesF64))
	if err := b.CopyD2H(dstBytesF64, b64); err != nil {
		t.Fatalf("CopyD2H F64: %v", err)
	}
	for i := range srcBytesF64 {
		if srcBytesF64[i] != dstBytesF64[i] {
			t.Fatalf("F64 D2D byte %d: src=%d dst=%d", i, srcBytesF64[i], dstBytesF64[i])
		}
	}
	got64 := bytesF64(dstBytesF64)
	for i, v := range got64 {
		if v != srcF64[i] {
			t.Fatalf("F64 D2D elem %d: src=%g got=%g", i, srcF64[i], v)
		}
	}

	// F32
	nF32 := 1024
	srcF32 := make([]float32, nF32)
	for i := range srcF32 {
		srcF32[i] = float32(i)*0.25 + 3.0
	}
	srcBytesF32 := f32Bytes(srcF32)

	a32, err := b.Alloc(len(srcBytesF32))
	if err != nil {
		t.Fatalf("Alloc a32: %v", err)
	}
	defer b.Free(a32)
	b32, err := b.Alloc(len(srcBytesF32))
	if err != nil {
		t.Fatalf("Alloc b32: %v", err)
	}
	defer b.Free(b32)

	if err := b.CopyH2D(a32, srcBytesF32); err != nil {
		t.Fatalf("CopyH2D F32: %v", err)
	}
	if err := b.CopyD2D(b32, a32, len(srcBytesF32)); err != nil {
		t.Fatalf("CopyD2D F32: %v", err)
	}
	dstBytesF32 := make([]byte, len(srcBytesF32))
	if err := b.CopyD2H(dstBytesF32, b32); err != nil {
		t.Fatalf("CopyD2H F32: %v", err)
	}
	got32 := bytesF32(dstBytesF32)
	for i, v := range got32 {
		if v != srcF32[i] {
			t.Fatalf("F32 D2D elem %d: src=%g got=%g", i, srcF32[i], v)
		}
	}
}
