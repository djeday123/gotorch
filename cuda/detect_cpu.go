//go:build !gpu

package cuda

// DetectGPU returns true if a CUDA GPU is available at runtime.
// In CPU-only builds this always returns false.
func DetectGPU() bool { return false }

// DeviceInfo returns a human-readable description of the GPU.
// In CPU-only builds returns "no GPU (CPU-only build)".
func DeviceInfo() string { return "no GPU (CPU-only build)" }
