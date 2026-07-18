//go:build !gpu

package cuda

// Заглушка для Ворот 6.2 R02b — активна при сборке БЕЗ тега gpu.
//
// Печатает точную причину пропуска: сравнение с legacy cgo-путём
// требует -tags gpu (тот путь использует cgo, libgotorch_cuda.so и
// заголовки cuda.h). Реальная реализация — parity_test.go под тегом gpu.
//
// Запуск полного парити:
//   go test -tags gpu -count=1 -run TestParityLegacy ./cuda/

import "testing"

func TestParityLegacyMatMulF64(t *testing.T) {
	t.Skip("parity vs legacy cgo backend requires -tags gpu " +
		"(legacy path uses cgo + libgotorch_cuda.so + cuda.h); " +
		"run: go test -tags gpu -run TestParityLegacy ./cuda/")
}

func TestParityLegacyAddF64(t *testing.T) {
	t.Skip("parity vs legacy cgo backend requires -tags gpu " +
		"(legacy path uses cgo + libgotorch_cuda.so + cuda.h); " +
		"run: go test -tags gpu -run TestParityLegacy ./cuda/")
}

func TestParityLegacyMulF64(t *testing.T) {
	t.Skip("parity vs legacy cgo backend requires -tags gpu " +
		"(legacy path uses cgo + libgotorch_cuda.so + cuda.h); " +
		"run: go test -tags gpu -run TestParityLegacy ./cuda/")
}
