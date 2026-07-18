# RECON_R01a — CUDA runtime linkage

## 0. Версия

```
$ git -C /data/lib/podman-data/projects/gotorch/v6/ rev-parse HEAD
950d2c724d56973209d218c4ce2a9bb1b10092db

$ git -C /data/lib/podman-data/projects/gotorch/v6/ describe --tags
v6.0.0-1-g950d2c7
```

## 1. Инвентарь директории

```
$ ls -la cuda/
total 1148
drwxr-xr-x  2 root root    4096 Jul 13 11:09 .
drwxr-xr-x 12 root root    4096 Jul 13 12:28 ..
-rw-r--r--  1 root root    8943 Jul 13 11:09 backend.go
-rw-r--r--  1 root root    4324 Jul 13 11:09 bridge.go
-rw-r--r--  1 root root    2345 Jul 13 11:09 cuda.h
-rw-r--r--  1 root root     367 Jul 13 11:09 detect_cpu.go
-rw-r--r--  1 root root     417 Jul 13 11:09 detect.go
-rw-r--r--  1 root root     587 Jul 13 11:09 detect_gpu.go
-rwxr-xr-x  1 root root 1076120 Jul 13 11:09 libgotorch_cuda.so
-rw-r--r--  1 root root   13454 Jul 13 11:09 ops.cu
-rw-r--r--  1 root root    7331 Jul 13 11:09 ops_test.go
-rw-r--r--  1 root root    4360 Jul 13 11:09 pinned.go
-rw-r--r--  1 root root    4598 Jul 13 11:09 pinned_test.go
-rw-r--r--  1 root root    5143 Jul 13 11:09 README.md
-rw-r--r--  1 root root    1907 Jul 13 11:09 tensor_gpu.go
```

```
$ file cuda/*.go
cuda/backend.go:     ASCII text
cuda/bridge.go:      ASCII text
cuda/detect.go:      ASCII text
cuda/detect_cpu.go:  ASCII text
cuda/detect_gpu.go:  ASCII text
cuda/ops_test.go:    ASCII text
cuda/pinned.go:      ASCII text
cuda/pinned_test.go: ASCII text
cuda/tensor_gpu.go:  ASCII text
```

```
$ wc -l cuda/*.go
  330 cuda/backend.go
  132 cuda/bridge.go
   11 cuda/detect_cpu.go
   14 cuda/detect.go
   24 cuda/detect_gpu.go
  318 cuda/ops_test.go
  138 cuda/pinned.go
  245 cuda/pinned_test.go
   77 cuda/tensor_gpu.go
 1289 total
```

## 2. Grep

```
$ grep -rn 'import "C"' cuda/
cuda/pinned.go:9:import "C"
cuda/bridge.go:11:import "C"
```

```
$ grep -rn 'ebitengine/purego' cuda/
(пусто — exit code 1)
```

```
$ grep -rn 'cgo LDFLAGS' cuda/
cuda/bridge.go:7:#cgo LDFLAGS: -L. -lgotorch_cuda -lcublas -lcudart
```

## 3. Ответ

CGO

Обоснование. В директории `cuda/` два `.go`-файла содержат `import "C"` — `bridge.go` и `pinned.go`. Главный из них — `bridge.go`: он несёт `#cgo`-директиву и объявляет всю Go-обёртку над сишным API из `cuda.h`. Полный преамбул-блок:

```go
// cuda/bridge.go:1-15
//go:build gpu

package cuda

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -lgotorch_cuda -lcublas -lcudart
#include "cuda.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)
```

Все Go-обёртки прямо вызывают C-функции через `C.<name>`. Ниже — блок реального вызова (allocations + H2D/D2H + один из compute-путей, cuBLAS DGEMM идёт через `gpu_matmul_f64`, экспорт которой определён в `cuda.h`):

```go
// cuda/bridge.go:44-70
// Malloc allocates bytes on the GPU. Caller must call Free when done.
func Malloc(bytes int) unsafe.Pointer {
	return C.gpu_malloc(C.size_t(bytes))
}

// Free releases GPU memory previously allocated with Malloc.
func Free(ptr unsafe.Pointer) {
	C.gpu_free(ptr)
}

// H2D copies len(data)*8 bytes from a []float64 slice to GPU pointer dst.
func H2D(dst unsafe.Pointer, data []float64) {
	if len(data) == 0 {
		return
	}
	C.gpu_memcpy_h2d(dst, unsafe.Pointer(&data[0]), C.size_t(len(data)*8))
}

// D2H copies n float64 values from GPU pointer src into dst slice.
// dst must have cap >= n.
func D2H(dst []float64, src unsafe.Pointer, n int) {
	if n == 0 {
		return
	}
	C.gpu_memcpy_d2h(unsafe.Pointer(&dst[0]), src, C.size_t(n*8))
}
```

```go
// cuda/bridge.go:130-132
func GPUMatMul(A, B, C unsafe.Pointer, M, N, K int) {
	C.gpu_matmul_f64((*C.double)(A), (*C.double)(B), (*C.double)(C), C.int(M), C.int(N), C.int(K))
}
```

Второй файл с `import "C"` — `pinned.go` — не имеет собственных `#cgo`-директив, он использует символы из того же преамбула (стандартный cgo-многофайловый пакет):

```go
// cuda/pinned.go:1-14
//go:build gpu

package cuda

/*
#include "cuda.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"github.com/djeday123/gotorch/tensor"
	"unsafe"
)
```

Механизм подтверждён: **cgo → shared library `libgotorch_cuda.so` (лежит в `cuda/`) → внутрь линкуются `libcublas` и `libcudart`**. `libgotorch_cuda.so` собирается из `ops.cu` (CUDA C++ ядра) и экспортирует C-ABI, объявленный в `cuda.h` (`extern "C"`).

## 4. Побочные наблюдения

- **CUDA-библиотеки**: `-lcudart` (Runtime API — `cudaMalloc/cudaMemcpy` через `gpu_*`-обёртки в `ops.cu`) + `-lcublas` (используется как минимум для `gpu_matmul_f64`, судя по объявлению в `cuda.h:50-52`).
- **Dtypes в CUDA-вызовах**: только `double` (float64) — все экспортируемые функции суффиксованы `_f64` (`gpu_add_f64`, `gpu_matmul_f64`, …). Ни `float`, ни `half`, ни `__nv_bfloat16`, ни `int8` в `cuda.h` не объявлены; в `bridge.go` типизация только `*C.double`. Значит cuBLAS вызывается как **DGEMM**, не SGEMM.
- **Внешний device-указатель**: **нет**. `GPUTensor` в `cuda/tensor_gpu.go:13-17` имеет поля `ptr unsafe.Pointer, shape []int, size int` — все **unexported**. Публичные конструкторы: `NewGPUTensor(t *tensor.Tensor)` (H2D upload) и `NewGPUTensorEmpty(shape ...int)` (`Malloc`). Публичного `NewGPUTensorFromPtr` / `WrapDevicePtr` / `RegisterExternalPointer` в пакете нет; `grep -rn 'External|Wrap|Register|Foreign|DevicePtr|FromDevice|WithDevice' cuda/` — пусто. Наружу отдаётся только геттер `func (g *GPUTensor) Ptr() unsafe.Pointer` (`tensor_gpu.go:77`), но обратной операции — принять чужой device-указатель и обернуть его в `*GPUTensor` — нет.
