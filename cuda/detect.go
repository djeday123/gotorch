// Package cuda provides GPU backend support for gotorch_v1.
//
// Build tags:
//   - (none):  CPU-only build. DetectGPU returns false, DeviceInfo returns "no GPU".
//   - gpu:     Full CUDA build. Requires CUDA Toolkit 12.x and driver >= 525.
//
// To build with GPU support:
//
//	make build-gpu
//
// or manually:
//
//	CGO_LDFLAGS="-L./cuda -lgotorch_cuda -lcublas -lcudart" go build -tags gpu ./...
package cuda
