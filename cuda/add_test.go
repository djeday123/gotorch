package cuda

// Ворота 3 R02b — PTX-инфраструктура + AddF64/AddF32.
// Проверка на размерах 1, 255, 256, 257, 100000 (границы grid при block=256).
// Плюс негативный TestKernelLoadFailure на битой PTX-строке.

import (
	"math/rand"
	"strings"
	"testing"
	"unsafe"
)

// runAddF64 — одна форма F64: fill A, B → H2D → AddF64 → D2H → сверка.
func runAddF64(t *testing.T, b *PuregoBackend, n int) {
	t.Helper()
	r := rand.New(rand.NewSource(int64(n) * 7))
	aH := make([]float64, n)
	bH := make([]float64, n)
	for i := 0; i < n; i++ {
		aH[i] = r.NormFloat64()
		bH[i] = r.NormFloat64()
	}
	aBytes := f64Bytes(aH)
	bBytes := f64Bytes(bH)

	aS, err := b.Alloc(len(aBytes))
	if err != nil {
		t.Fatalf("Alloc A (n=%d): %v", n, err)
	}
	defer b.Free(aS)
	bS, err := b.Alloc(len(bBytes))
	if err != nil {
		t.Fatalf("Alloc B: %v", err)
	}
	defer b.Free(bS)
	cS, err := b.Alloc(len(aBytes))
	if err != nil {
		t.Fatalf("Alloc C: %v", err)
	}
	defer b.Free(cS)

	if err := b.CopyH2D(aS, aBytes); err != nil {
		t.Fatalf("H2D A: %v", err)
	}
	if err := b.CopyH2D(bS, bBytes); err != nil {
		t.Fatalf("H2D B: %v", err)
	}
	if err := b.AddF64(aS, bS, cS, n); err != nil {
		t.Fatalf("AddF64: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	cBytes := make([]byte, len(aBytes))
	if err := b.CopyD2H(cBytes, cS); err != nil {
		t.Fatalf("D2H C: %v", err)
	}
	gotC := bytesF64(cBytes)
	for i := 0; i < n; i++ {
		want := aH[i] + bH[i]
		if gotC[i] != want {
			t.Fatalf("AddF64 n=%d idx=%d: got=%g want=%g (a=%g b=%g)",
				n, i, gotC[i], want, aH[i], bH[i])
		}
	}
}

func runAddF32(t *testing.T, b *PuregoBackend, n int) {
	t.Helper()
	r := rand.New(rand.NewSource(int64(n) * 11))
	aH := make([]float32, n)
	bH := make([]float32, n)
	for i := 0; i < n; i++ {
		aH[i] = float32(r.NormFloat64())
		bH[i] = float32(r.NormFloat64())
	}
	aBytes := f32Bytes(aH)
	bBytes := f32Bytes(bH)

	aS, err := b.Alloc(len(aBytes))
	if err != nil {
		t.Fatalf("Alloc A (n=%d): %v", n, err)
	}
	defer b.Free(aS)
	bS, err := b.Alloc(len(bBytes))
	if err != nil {
		t.Fatalf("Alloc B: %v", err)
	}
	defer b.Free(bS)
	cS, err := b.Alloc(len(aBytes))
	if err != nil {
		t.Fatalf("Alloc C: %v", err)
	}
	defer b.Free(cS)

	if err := b.CopyH2D(aS, aBytes); err != nil {
		t.Fatalf("H2D A: %v", err)
	}
	if err := b.CopyH2D(bS, bBytes); err != nil {
		t.Fatalf("H2D B: %v", err)
	}
	if err := b.AddF32(aS, bS, cS, n); err != nil {
		t.Fatalf("AddF32: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	cBytes := make([]byte, len(aBytes))
	if err := b.CopyD2H(cBytes, cS); err != nil {
		t.Fatalf("D2H C: %v", err)
	}
	gotC := bytesF32(cBytes)
	for i := 0; i < n; i++ {
		want := aH[i] + bH[i]
		if gotC[i] != want {
			t.Fatalf("AddF32 n=%d idx=%d: got=%g want=%g (a=%g b=%g)",
				n, i, gotC[i], want, aH[i], bH[i])
		}
	}
}

// addSizes — границы grid при block=256.
var addSizes = []int{1, 255, 256, 257, 100000}

func TestAddF64(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	for _, n := range addSizes {
		n := n
		t.Run(subName(n), func(t *testing.T) {
			runAddF64(t, b, n)
		})
	}
}

func TestAddF32(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()
	for _, n := range addSizes {
		n := n
		t.Run(subName(n), func(t *testing.T) {
			runAddF32(t, b, n)
		})
	}
}

func subName(n int) string {
	return "n=" + itoa(n)
}
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	neg := n < 0
	if neg {
		n = -n
	}
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// TestKernelLoadFailure — битая PTX-строка должна давать внятную ошибку,
// не панику. Проверяем через прямой вызов cuModuleLoadData на garbage.
// Не поднимаем backend — модуль загружается на текущий primary-context.
func TestKernelLoadFailure(t *testing.T) {
	b := tryBackend(t)
	defer b.Close()

	badPTX := append([]byte("this is not valid PTX\n"), 0)
	var module uintptr
	r := cuModuleLoadData(&module, unsafe.Pointer(&badPTX[0]))
	if r == CUDA_SUCCESS {
		cuModuleUnload(module)
		t.Fatal("expected cuModuleLoadData to fail on garbage input, got SUCCESS")
	}
	// Ожидаем INVALID_PTX / JIT_COMPILER_ERROR / INVALID_IMAGE — всё три
	// приемлемы, драйвер выбирает по конкретному повреждению. Главное —
	// не panic, а внятная ошибка через CUresult.Error().
	msg := r.Error()
	if !strings.Contains(msg, "CUDA_ERROR_") {
		t.Fatalf("expected CUDA_ERROR_ prefix in error, got: %s", msg)
	}
	t.Logf("cuModuleLoadData(garbage) failed as expected: %s", msg)
}
