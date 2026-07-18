package cuda

// Utility-функции purego-мира. Имена содержат суффикс _purego намеренно:
// в этом же пакете живёт legacy cgo-мир с симметричными функциями без
// суффикса (DetectGPU/DeviceInfo/Init/DeviceCount/DeviceName/MemoryInfo из
// detect_*.go и bridge.go). После введения второго cgo-backend'а после
// R02b symmetричная маркировка станет DetectGPU_cgo/... — сейчас legacy
// имена остаются без суффикса как исторический артефакт до отдельной
// правки-переименования.
//
// Подчёркивание в имени — не Go-конвенция и golint будет ворчать. Это
// осознанное решение: симметрия с будущим _cgo важнее косметики линтера.
//
// Инициализация целиком живёт внутри newPuregoBackend, поэтому публичного
// Init_purego нет: пользователь получает готовый Backend через NewBackend
// и не имеет причин звать голый Init самостоятельно.

import (
	"fmt"
)

// DetectGPU_purego — есть ли хоть одно CUDA-устройство. Дублирует смысл
// legacy DetectGPU из detect_gpu.go/detect_cpu.go, но реализовано через
// libcuda driver API вместо cgo-моста.
func DetectGPU_purego() bool {
	if err := initDriver(); err != nil {
		return false
	}
	if r := cuInit(0); r != CUDA_SUCCESS {
		return false
	}
	return DeviceCount_purego() > 0
}

// DeviceCount_purego — сколько CUDA-устройств доступно системе.
func DeviceCount_purego() int {
	if err := initDriver(); err != nil {
		return 0
	}
	if r := cuInit(0); r != CUDA_SUCCESS {
		return 0
	}
	var n int32
	if r := cuDeviceGetCount(&n); r != CUDA_SUCCESS {
		return 0
	}
	return int(n)
}

// DeviceName_purego — имя устройства с индексом device.
func DeviceName_purego(device int) string {
	if err := initDriver(); err != nil {
		return ""
	}
	if r := cuInit(0); r != CUDA_SUCCESS {
		return ""
	}
	var dev int32
	if r := cuDeviceGet(&dev, int32(device)); r != CUDA_SUCCESS {
		return ""
	}
	nameBuf := make([]byte, 256)
	if r := cuDeviceGetName(&nameBuf[0], 256, dev); r != CUDA_SUCCESS {
		return ""
	}
	for i, b := range nameBuf {
		if b == 0 {
			return string(nameBuf[:i])
		}
	}
	return string(nameBuf)
}

// MemoryInfo_purego — свободная и общая память привязанного контекста.
// Возвращает нули, если primary context не был retained (нет Backend'а).
func MemoryInfo_purego() (free, total uint64) {
	if err := initDriver(); err != nil {
		return 0, 0
	}
	var f, t uint64
	if r := cuMemGetInfo(&f, &t); r != CUDA_SUCCESS {
		return 0, 0
	}
	return f, t
}

// DeviceInfo_purego — человекочитаемая строка про device 0.
func DeviceInfo_purego() string {
	if !DetectGPU_purego() {
		return "no GPU (purego)"
	}
	name := DeviceName_purego(0)
	n := DeviceCount_purego()
	return fmt.Sprintf("%s (purego, %d device(s))", name, n)
}
