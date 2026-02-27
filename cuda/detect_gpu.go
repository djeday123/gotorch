//go:build gpu

package cuda

import "fmt"

// DetectGPU returns true if at least one CUDA GPU is available.
func DetectGPU() bool {
	return DeviceCount() > 0
}

// DeviceInfo returns name and compute capability of device 0,
// plus free/total memory. Example: "NVIDIA RTX 4090 (sm_89), 22.1/24.0 GB".
func DeviceInfo() string {
	n := DeviceCount()
	if n == 0 {
		return "no GPU"
	}
	name := DeviceName(0)
	free, total := MemoryInfo()
	freeGB := float64(free) / 1e9
	totalGB := float64(total) / 1e9
	return fmt.Sprintf("%s, %.1f/%.1f GB free (%d device(s))", name, freeGB, totalGB, n)
}
