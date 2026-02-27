# =============================================================================
# gotorch_v1 — Build System
# =============================================================================
#
# CPU (default, no CUDA needed):
#   make             — build all packages
#   make test        — run all CPU tests
#
# GPU (requires CUDA Toolkit 12.x, driver >= 525):
#   make build-gpu                          — compile for all default archs
#   make build-gpu CUDA_ARCHS="80 86 89"   — target specific archs only
#   make test-gpu                           — run GPU tests
#   make clean                              — remove build artifacts
#
# =============================================================================

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CUDA_DIR    := $(CURDIR)/cuda
CUDA_LIB    := $(CUDA_DIR)/libgotorch_cuda.so
NVCC        ?= nvcc
CUDA_TOOLKIT ?= /usr/local/cuda

# ---------------------------------------------------------------------------
# Multi-arch configuration
#
# CUDA_ARCHS: list of SM generations to embed in the binary.
#   sm_80  → A100, A30         (Ampere)
#   sm_86  → RTX 3090, A40     (Ampere)
#   sm_89  → RTX 4090, L40     (Ada Lovelace)  ← current dev machine
#   sm_90  → H100, H200        (Hopper)
#
# PTX fallback: the last arch is also compiled to PTX (compute_XX).
# This allows the CUDA JIT to compile kernels for future GPU generations
# at runtime without requiring a new build.
# ---------------------------------------------------------------------------
CUDA_ARCHS  ?= 80 86 89 90

# Build -gencode flags for each arch
GENCODE_FLAGS := $(foreach arch,$(CUDA_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))
# Add PTX for the highest arch (forward-compat JIT fallback)
CUDA_ARCHS_LIST := $(CUDA_ARCHS)
LATEST_ARCH     := $(lastword $(CUDA_ARCHS_LIST))
GENCODE_FLAGS   += -gencode arch=compute_$(LATEST_ARCH),code=compute_$(LATEST_ARCH)

NVCC_FLAGS := \
    -O3 \
    -std=c++14 \
    $(GENCODE_FLAGS) \
    --compiler-options -fPIC \
    -I$(CUDA_TOOLKIT)/include

CGO_CFLAGS  := -I$(CUDA_DIR)
CGO_LDFLAGS := -L$(CUDA_DIR) -lgotorch_cuda -L$(CUDA_TOOLKIT)/lib64 -lcublas -lcudart -Wl,-rpath,$(CUDA_DIR) -Wl,-rpath,$(CUDA_TOOLKIT)/lib64

# ---------------------------------------------------------------------------
# CPU targets (default)
# ---------------------------------------------------------------------------

.PHONY: all build test fmt vet

all: build

build:
	go build ./...

test:
	go test ./... -v

fmt:
	gofmt -w .

vet:
	go vet ./...

# ---------------------------------------------------------------------------
# GPU targets
# ---------------------------------------------------------------------------

.PHONY: build-gpu test-gpu cuda-lib info

## Compile CUDA kernels + cuBLAS wrapper into a shared library
cuda-lib: $(CUDA_LIB)

$(CUDA_LIB): $(CUDA_DIR)/ops.cu $(CUDA_DIR)/cuda.h
	@echo "==> Compiling CUDA kernels (archs: $(CUDA_ARCHS) + PTX sm_$(LATEST_ARCH))"
	$(NVCC) $(NVCC_FLAGS) \
	    -shared \
	    -o $@ \
	    $<
	@echo "==> Built: $@"

## Build all Go packages with GPU support
build-gpu: cuda-lib
	@echo "==> Building Go (gpu tag, CGo → libgotorch_cuda)"
	CGO_CFLAGS="$(CGO_CFLAGS)" \
	CGO_LDFLAGS="$(CGO_LDFLAGS)" \
	go build -tags gpu ./...
	@echo "==> GPU build complete"

## Run GPU tests (requires CUDA device)
test-gpu: cuda-lib
	@echo "==> Running GPU tests"
	CGO_CFLAGS="$(CGO_CFLAGS)" \
	CGO_LDFLAGS="$(CGO_LDFLAGS)" \
	LD_LIBRARY_PATH="$(CUDA_DIR):$(CUDA_TOOLKIT)/lib64:$$LD_LIBRARY_PATH" \
	go test -tags gpu -v ./cuda/...
	@echo "==> CPU tests (verify GPU tag doesn't break CPU packages)"
	CGO_CFLAGS="$(CGO_CFLAGS)" \
	CGO_LDFLAGS="$(CGO_LDFLAGS)" \
	LD_LIBRARY_PATH="$(CUDA_DIR):$(CUDA_TOOLKIT)/lib64:$$LD_LIBRARY_PATH" \
	go test -tags gpu ./tensor/... ./autograd/... ./nn/... ./optim/...

## Print GPU/CUDA environment info
info:
	@echo "NVCC:         $(shell which $(NVCC) 2>/dev/null || echo 'NOT FOUND')"
	@echo "CUDA_TOOLKIT: $(CUDA_TOOLKIT)"
	@echo "CUDA_ARCHS:   $(CUDA_ARCHS)"
	@echo "GENCODE:      $(GENCODE_FLAGS)"
	@nvcc --version 2>/dev/null || echo "nvcc not in PATH"
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

.PHONY: clean
clean:
	rm -f $(CUDA_LIB) $(CUDA_DIR)/*.o
	go clean ./...
	@echo "==> Cleaned"
