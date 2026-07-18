# RECON_R02a — инвентарь gotorch/cuda и проектирование Backend-интерфейса

## 0. Версия

```
$ git -C /data/lib/podman-data/projects/gotorch/v6 rev-parse HEAD
950d2c724d56973209d218c4ce2a9bb1b10092db

$ git -C /data/lib/podman-data/projects/gotorch/v6 describe --tags
v6.0.0-1-g950d2c7
```

Module path — `github.com/djeday123/gotorch` (см. `go.mod`, без `/v6` в пути; директория v6 — просто локальная папка).

## 1. Полный экспорт пакета cuda/

### Сырой grep-вывод

```
$ grep -n '^type ' cuda/*.go
cuda/tensor_gpu.go:13:type GPUTensor struct {
cuda/pinned.go:24:type PinnedTensor struct {
cuda/backend.go:25:type GPUBackend struct {

$ grep -n '^var ' cuda/*.go
(пусто, exit 1)

$ grep -n '^const ' cuda/*.go
cuda/ops_test.go:12:const gpuTol = 1e-9

$ grep -n '^func ' cuda/*.go
(см. полный вывод ниже)
```

`const gpuTol` — из `ops_test.go` (unexported, тестовый). Экспортируемых `var` и `const` в пакете нет.

Все `.go`-файлы кроме `detect.go` и `detect_cpu.go` защищены `//go:build gpu`. В CPU-only-сборке экспортируются только `DetectGPU()` и `DeviceInfo()` (стабы из `detect_cpu.go`).

### Таблица Types

| Символ | Файл:строка | Заголовок и поля |
|---|---|---|
| `GPUTensor` | `cuda/tensor_gpu.go:13-17` | `struct { ptr unsafe.Pointer; shape []int; size int }` — все поля unexported |
| `PinnedTensor` | `cuda/pinned.go:24-28` | `struct { ptr unsafe.Pointer; shape []int; size int }` — все поля unexported |
| `GPUBackend` | `cuda/backend.go:25-27` | `struct { device int }` — поле unexported |

Interfaces в пакете нет.

### Таблица Functions

Экспортируемые (первая буква — заглавная), включая методы на экспортируемых receiver'ах. Отсортировано по файлу и строке.

| # | Символ | Файл:строка | Сигнатура |
|---|---|---|---|
| 1 | `Init` | `bridge.go:19` | `func Init(device int) (int, error)` |
| 2 | `DeviceCount` | `bridge.go:28` | `func DeviceCount() int` |
| 3 | `MemoryInfo` | `bridge.go:33` | `func MemoryInfo() (free, total uint64)` |
| 4 | `DeviceName` | `bridge.go:40` | `func DeviceName(device int) string` |
| 5 | `Malloc` | `bridge.go:45` | `func Malloc(bytes int) unsafe.Pointer` |
| 6 | `Free` | `bridge.go:50` | `func Free(ptr unsafe.Pointer)` |
| 7 | `H2D` | `bridge.go:55` | `func H2D(dst unsafe.Pointer, data []float64)` |
| 8 | `D2H` | `bridge.go:64` | `func D2H(dst []float64, src unsafe.Pointer, n int)` |
| 9 | `GPUAdd` | `bridge.go:76` | `func GPUAdd(a, b, c unsafe.Pointer, n int)` |
| 10 | `GPUSub` | `bridge.go:79` | `func GPUSub(a, b, c unsafe.Pointer, n int)` |
| 11 | `GPUMul` | `bridge.go:82` | `func GPUMul(a, b, c unsafe.Pointer, n int)` |
| 12 | `GPUDiv` | `bridge.go:85` | `func GPUDiv(a, b, c unsafe.Pointer, n int)` |
| 13 | `GPUAddScalar` | `bridge.go:88` | `func GPUAddScalar(a unsafe.Pointer, scalar float64, c unsafe.Pointer, n int)` |
| 14 | `GPUMulScalar` | `bridge.go:91` | `func GPUMulScalar(a unsafe.Pointer, scalar float64, c unsafe.Pointer, n int)` |
| 15 | `GPUReLU` | `bridge.go:94` | `func GPUReLU(a, c unsafe.Pointer, n int)` |
| 16 | `GPUSigmoid` | `bridge.go:97` | `func GPUSigmoid(a, c unsafe.Pointer, n int)` |
| 17 | `GPUTanh` | `bridge.go:100` | `func GPUTanh(a, c unsafe.Pointer, n int)` |
| 18 | `GPUExp` | `bridge.go:103` | `func GPUExp(a, c unsafe.Pointer, n int)` |
| 19 | `GPULog` | `bridge.go:106` | `func GPULog(a, c unsafe.Pointer, n int)` |
| 20 | `GPUNeg` | `bridge.go:109` | `func GPUNeg(a, c unsafe.Pointer, n int)` |
| 21 | `GPUReLUGrad` | `bridge.go:112` | `func GPUReLUGrad(a, grad, out unsafe.Pointer, n int)` |
| 22 | `GPUSigmoidGrad` | `bridge.go:115` | `func GPUSigmoidGrad(sig, grad, out unsafe.Pointer, n int)` |
| 23 | `GPUTanhGrad` | `bridge.go:118` | `func GPUTanhGrad(tanhOut, grad, out unsafe.Pointer, n int)` |
| 24 | `GPUSum` | `bridge.go:121` | `func GPUSum(a unsafe.Pointer, n int) float64` |
| 25 | `GPUMean` | `bridge.go:124` | `func GPUMean(a unsafe.Pointer, n int) float64` |
| 26 | `GPUSoftmax` | `bridge.go:127` | `func GPUSoftmax(a, c unsafe.Pointer, rows, cols int)` |
| 27 | `GPUMatMul` | `bridge.go:130` | `func GPUMatMul(A, B, C unsafe.Pointer, M, N, K int)` |
| 28 | `NewGPUTensor` | `tensor_gpu.go:20` | `func NewGPUTensor(t *tensor.Tensor) (*GPUTensor, error)` |
| 29 | `NewGPUTensorEmpty` | `tensor_gpu.go:37` | `func NewGPUTensorEmpty(shape ...int) (*GPUTensor, error)` |
| 30 | `GPUTensor.ToCPU` | `tensor_gpu.go:52` | `func (g *GPUTensor) ToCPU() *tensor.Tensor` |
| 31 | `GPUTensor.Free` | `tensor_gpu.go:59` | `func (g *GPUTensor) Free()` |
| 32 | `GPUTensor.Shape` | `tensor_gpu.go:67` | `func (g *GPUTensor) Shape() []int` |
| 33 | `GPUTensor.Size` | `tensor_gpu.go:74` | `func (g *GPUTensor) Size() int` |
| 34 | `GPUTensor.Ptr` | `tensor_gpu.go:77` | `func (g *GPUTensor) Ptr() unsafe.Pointer` |
| 35 | `DetectGPU` | `detect_cpu.go:7` / `detect_gpu.go:8` | `func DetectGPU() bool` |
| 36 | `DeviceInfo` | `detect_cpu.go:11` / `detect_gpu.go:14` | `func DeviceInfo() string` |
| 37 | `NewPinnedTensor` | `pinned.go:32` | `func NewPinnedTensor(shape ...int) (*PinnedTensor, error)` |
| 38 | `PinnedTensor.Free` | `pinned.go:60` | `func (p *PinnedTensor) Free()` |
| 39 | `PinnedTensor.Slice` | `pinned.go:71` | `func (p *PinnedTensor) Slice() []float64` |
| 40 | `PinnedTensor.Shape` | `pinned.go:76` | `func (p *PinnedTensor) Shape() []int` |
| 41 | `PinnedTensor.Size` | `pinned.go:83` | `func (p *PinnedTensor) Size() int` |
| 42 | `PinnedTensor.ToGPU` | `pinned.go:88` | `func (p *PinnedTensor) ToGPU() (*GPUTensor, error)` |
| 43 | `PinnedTensor.FromGPU` | `pinned.go:100` | `func (p *PinnedTensor) FromGPU(g *GPUTensor) error` |
| 44 | `StreamSync` | `pinned.go:110` | `func StreamSync()` |
| 45 | `H2DAsync` | `pinned.go:116` | `func H2DAsync(dst *GPUTensor, src *PinnedTensor) error` |
| 46 | `D2HAsync` | `pinned.go:127` | `func D2HAsync(dst *PinnedTensor, src *GPUTensor) error` |
| 47 | `PinnedTensor.ToCPUTensor` | `pinned.go:134` | `func (p *PinnedTensor) ToCPUTensor() *tensor.Tensor` |
| 48 | `NewGPUBackend` | `backend.go:30` | `func NewGPUBackend(device int) (*GPUBackend, error)` |
| 49 | `GPUBackend.Close` | `backend.go:40` | `func (b *GPUBackend) Close()` |
| 50 | `GPUBackend.Device` | `backend.go:46` | `func (b *GPUBackend) Device() int` |
| 51 | `GPUBackend.Upload` | `backend.go:49` | `func (b *GPUBackend) Upload(t *tensor.Tensor) (*GPUTensor, error)` |
| 52 | `GPUBackend.Download` | `backend.go:54` | `func (b *GPUBackend) Download(g *GPUTensor) *tensor.Tensor` |
| 53 | `GPUBackend.Add` | `backend.go:62` | `func (b *GPUBackend) Add(a, x *GPUTensor) (*GPUTensor, error)` |
| 54 | `GPUBackend.Sub` | `backend.go:74` | `func (b *GPUBackend) Sub(a, x *GPUTensor) (*GPUTensor, error)` |
| 55 | `GPUBackend.Mul` | `backend.go:86` | `func (b *GPUBackend) Mul(a, x *GPUTensor) (*GPUTensor, error)` |
| 56 | `GPUBackend.Div` | `backend.go:98` | `func (b *GPUBackend) Div(a, x *GPUTensor) (*GPUTensor, error)` |
| 57 | `GPUBackend.AddScalar` | `backend.go:114` | `func (b *GPUBackend) AddScalar(a *GPUTensor, s float64) (*GPUTensor, error)` |
| 58 | `GPUBackend.MulScalar` | `backend.go:123` | `func (b *GPUBackend) MulScalar(a *GPUTensor, s float64) (*GPUTensor, error)` |
| 59 | `GPUBackend.ReLU` | `backend.go:136` | `func (b *GPUBackend) ReLU(a *GPUTensor) (*GPUTensor, error)` |
| 60 | `GPUBackend.Sigmoid` | `backend.go:145` | `func (b *GPUBackend) Sigmoid(a *GPUTensor) (*GPUTensor, error)` |
| 61 | `GPUBackend.Tanh` | `backend.go:154` | `func (b *GPUBackend) Tanh(a *GPUTensor) (*GPUTensor, error)` |
| 62 | `GPUBackend.Exp` | `backend.go:163` | `func (b *GPUBackend) Exp(a *GPUTensor) (*GPUTensor, error)` |
| 63 | `GPUBackend.Log` | `backend.go:172` | `func (b *GPUBackend) Log(a *GPUTensor) (*GPUTensor, error)` |
| 64 | `GPUBackend.Neg` | `backend.go:181` | `func (b *GPUBackend) Neg(a *GPUTensor) (*GPUTensor, error)` |
| 65 | `GPUBackend.ReLUGrad` | `backend.go:194` | `func (b *GPUBackend) ReLUGrad(input, grad *GPUTensor) (*GPUTensor, error)` |
| 66 | `GPUBackend.SigmoidGrad` | `backend.go:206` | `func (b *GPUBackend) SigmoidGrad(sigOutput, grad *GPUTensor) (*GPUTensor, error)` |
| 67 | `GPUBackend.TanhGrad` | `backend.go:218` | `func (b *GPUBackend) TanhGrad(tanhOutput, grad *GPUTensor) (*GPUTensor, error)` |
| 68 | `GPUBackend.Sum` | `backend.go:234` | `func (b *GPUBackend) Sum(a *GPUTensor) float64` |
| 69 | `GPUBackend.Mean` | `backend.go:238` | `func (b *GPUBackend) Mean(a *GPUTensor) float64` |
| 70 | `GPUBackend.Softmax` | `backend.go:248` | `func (b *GPUBackend) Softmax(a *GPUTensor) (*GPUTensor, error)` |
| 71 | `GPUBackend.MatMul` | `backend.go:266` | `func (b *GPUBackend) MatMul(a, x *GPUTensor) (*GPUTensor, error)` |
| 72 | `GPUBackend.RunUnary` | `backend.go:288` | `func (b *GPUBackend) RunUnary(t *tensor.Tensor, op func(*GPUTensor) (*GPUTensor, error)) (*tensor.Tensor, error)` |
| 73 | `GPUBackend.RunBinary` | `backend.go:302` | `func (b *GPUBackend) RunBinary(a, x *tensor.Tensor, op func(*GPUTensor, *GPUTensor) (*GPUTensor, error)) (*tensor.Tensor, error)` |

**Итого экспортируемых функций/методов: 73.** (Тестовые `Test*`/`Benchmark*` и `almostEqualGPU`, `allCloseSlices` из `ops_test.go`/`pinned_test.go` — не включены, они unexported/тестовые. `checkSameSize` из `backend.go:325` — unexported.)

### Таблица Vars

Экспортируемых `var` в пакете **нет**.

### Таблица Consts

Экспортируемых `const` в пакете **нет** (`gpuTol` — unexported тестовый).

## 2. Кто вне cuda/ этим пользуется

Модуль-путь по `go.mod` — `github.com/djeday123/gotorch`. Ожидаемая форма импорта — `"github.com/djeday123/gotorch/cuda"`.

Выполнены обе релевантные команды из v6-корня (в моём случае — `/data/lib/podman-data/projects/gotorch/v6/`):

```
$ grep -rn 'djeday123/gotorch/cuda\|djeday123/gotorch/v6/cuda' --include='*.go' .
(пусто, exit 1)

$ grep -rn '"cuda"' --include='*.go' .
(пусто, exit 1)

$ grep -rn 'gotorch/v6/cuda' --include='*.go' .
(пусто, exit 1)

$ grep -rn 'gotorch/cuda' --include='*.go' .
(пусто, exit 1)
```

**Никто вне пакета `cuda/` его не импортирует.** Всё, что видит внешние символы, — это только сам пакет `cuda/` (плюс тесты внутри него: `ops_test.go`, `pinned_test.go`). Т.е. `nn/`, `autograd/`, `tensor/`, `optim/`, `data/`, `amp/`, `export/`, `examples/` — на CUDA-путь не завязаны, работают через CPU-tensor.

Резюме секции:
- Использующих мест вне `cuda/`: **0**.
- Топ-5 наиболее «дёргаемых» символов среди внешних пользователей: **не применимо** (нет пользователей).
- Практическое следствие: R02c-миграция «удалить legacy cgo» **не ломает** ни один call-site за пределами `cuda/`. Единственные сломанные будут внутренние тесты `cuda/ops_test.go` и `cuda/pinned_test.go`, которые в R02b естественно перегоним на новый интерфейс.

## 3. Инвентарь операций по классам

### Alloc/Free (7 функций)

| Функция | Файл:строка |
|---|---|
| `func Malloc(bytes int) unsafe.Pointer` | `bridge.go:45` |
| `func Free(ptr unsafe.Pointer)` | `bridge.go:50` |
| `func NewGPUTensor(t *tensor.Tensor) (*GPUTensor, error)` | `tensor_gpu.go:20` |
| `func NewGPUTensorEmpty(shape ...int) (*GPUTensor, error)` | `tensor_gpu.go:37` |
| `func (g *GPUTensor) Free()` | `tensor_gpu.go:59` |
| `func NewPinnedTensor(shape ...int) (*PinnedTensor, error)` | `pinned.go:32` |
| `func (p *PinnedTensor) Free()` | `pinned.go:60` |

### Copy (H2D/D2H/D2D — 11 функций)

| Функция | Файл:строка |
|---|---|
| `func H2D(dst unsafe.Pointer, data []float64)` | `bridge.go:55` |
| `func D2H(dst []float64, src unsafe.Pointer, n int)` | `bridge.go:64` |
| `func (p *PinnedTensor) Slice() []float64` | `pinned.go:71` |
| `func (p *PinnedTensor) ToGPU() (*GPUTensor, error)` | `pinned.go:88` |
| `func (p *PinnedTensor) FromGPU(g *GPUTensor) error` | `pinned.go:100` |
| `func H2DAsync(dst *GPUTensor, src *PinnedTensor) error` | `pinned.go:116` |
| `func D2HAsync(dst *PinnedTensor, src *GPUTensor) error` | `pinned.go:127` |
| `func (p *PinnedTensor) ToCPUTensor() *tensor.Tensor` | `pinned.go:134` |
| `func (g *GPUTensor) ToCPU() *tensor.Tensor` | `tensor_gpu.go:52` |
| `func (b *GPUBackend) Upload(t *tensor.Tensor) (*GPUTensor, error)` | `backend.go:49` |
| `func (b *GPUBackend) Download(g *GPUTensor) *tensor.Tensor` | `backend.go:54` |

D2D (`cudaMemcpy DeviceToDevice`) как явной функции **нет** — не реализована.

### Fill (0 функций)

Нет отдельных `Fill`/`Zero`/`Ones`/`Constant`. `Malloc` возвращает неинициализированную память; `NewPinnedTensor` зануляет через Go-slice-view (`pinned.go:47-51`) — не через GPU-kernel.

### Elementwise (18 функций)

| Функция | Файл:строка |
|---|---|
| `func GPUAdd(a, b, c unsafe.Pointer, n int)` | `bridge.go:76` |
| `func GPUSub(a, b, c unsafe.Pointer, n int)` | `bridge.go:79` |
| `func GPUMul(a, b, c unsafe.Pointer, n int)` | `bridge.go:82` |
| `func GPUDiv(a, b, c unsafe.Pointer, n int)` | `bridge.go:85` |
| `func GPUAddScalar(a unsafe.Pointer, scalar float64, c unsafe.Pointer, n int)` | `bridge.go:88` |
| `func GPUMulScalar(a unsafe.Pointer, scalar float64, c unsafe.Pointer, n int)` | `bridge.go:91` |
| `func GPUExp(a, c unsafe.Pointer, n int)` | `bridge.go:103` |
| `func GPULog(a, c unsafe.Pointer, n int)` | `bridge.go:106` |
| `func GPUNeg(a, c unsafe.Pointer, n int)` | `bridge.go:109` |
| `func (b *GPUBackend) Add(a, x *GPUTensor) (*GPUTensor, error)` | `backend.go:62` |
| `func (b *GPUBackend) Sub(a, x *GPUTensor) (*GPUTensor, error)` | `backend.go:74` |
| `func (b *GPUBackend) Mul(a, x *GPUTensor) (*GPUTensor, error)` | `backend.go:86` |
| `func (b *GPUBackend) Div(a, x *GPUTensor) (*GPUTensor, error)` | `backend.go:98` |
| `func (b *GPUBackend) AddScalar(a *GPUTensor, s float64) (*GPUTensor, error)` | `backend.go:114` |
| `func (b *GPUBackend) MulScalar(a *GPUTensor, s float64) (*GPUTensor, error)` | `backend.go:123` |
| `func (b *GPUBackend) Exp(a *GPUTensor) (*GPUTensor, error)` | `backend.go:163` |
| `func (b *GPUBackend) Log(a *GPUTensor) (*GPUTensor, error)` | `backend.go:172` |
| `func (b *GPUBackend) Neg(a *GPUTensor) (*GPUTensor, error)` | `backend.go:181` |

### Activations (14 функций)

| Функция | Файл:строка |
|---|---|
| `func GPUReLU(a, c unsafe.Pointer, n int)` | `bridge.go:94` |
| `func GPUSigmoid(a, c unsafe.Pointer, n int)` | `bridge.go:97` |
| `func GPUTanh(a, c unsafe.Pointer, n int)` | `bridge.go:100` |
| `func GPUReLUGrad(a, grad, out unsafe.Pointer, n int)` | `bridge.go:112` |
| `func GPUSigmoidGrad(sig, grad, out unsafe.Pointer, n int)` | `bridge.go:115` |
| `func GPUTanhGrad(tanhOut, grad, out unsafe.Pointer, n int)` | `bridge.go:118` |
| `func GPUSoftmax(a, c unsafe.Pointer, rows, cols int)` | `bridge.go:127` |
| `func (b *GPUBackend) ReLU(a *GPUTensor) (*GPUTensor, error)` | `backend.go:136` |
| `func (b *GPUBackend) Sigmoid(a *GPUTensor) (*GPUTensor, error)` | `backend.go:145` |
| `func (b *GPUBackend) Tanh(a *GPUTensor) (*GPUTensor, error)` | `backend.go:154` |
| `func (b *GPUBackend) ReLUGrad(input, grad *GPUTensor) (*GPUTensor, error)` | `backend.go:194` |
| `func (b *GPUBackend) SigmoidGrad(sigOutput, grad *GPUTensor) (*GPUTensor, error)` | `backend.go:206` |
| `func (b *GPUBackend) TanhGrad(tanhOutput, grad *GPUTensor) (*GPUTensor, error)` | `backend.go:218` |
| `func (b *GPUBackend) Softmax(a *GPUTensor) (*GPUTensor, error)` | `backend.go:248` |

### Reduce (4 функции)

| Функция | Файл:строка |
|---|---|
| `func GPUSum(a unsafe.Pointer, n int) float64` | `bridge.go:121` |
| `func GPUMean(a unsafe.Pointer, n int) float64` | `bridge.go:124` |
| `func (b *GPUBackend) Sum(a *GPUTensor) float64` | `backend.go:234` |
| `func (b *GPUBackend) Mean(a *GPUTensor) float64` | `backend.go:238` |

Max/Min/ArgMax — **не реализованы**.

### Linalg (2 функции)

| Функция | Файл:строка |
|---|---|
| `func GPUMatMul(A, B, C unsafe.Pointer, M, N, K int)` | `bridge.go:130` |
| `func (b *GPUBackend) MatMul(a, x *GPUTensor) (*GPUTensor, error)` | `backend.go:266` |

BatchMatMul/Dot — **не реализованы** на CUDA-стороне.

### Query (13 функций/методов)

| Функция | Файл:строка |
|---|---|
| `func Init(device int) (int, error)` | `bridge.go:19` |
| `func DeviceCount() int` | `bridge.go:28` |
| `func MemoryInfo() (free, total uint64)` | `bridge.go:33` |
| `func DeviceName(device int) string` | `bridge.go:40` |
| `func StreamSync()` | `pinned.go:110` |
| `func DetectGPU() bool` | `detect_cpu.go:7` / `detect_gpu.go:8` |
| `func DeviceInfo() string` | `detect_cpu.go:11` / `detect_gpu.go:14` |
| `func (g *GPUTensor) Shape() []int` | `tensor_gpu.go:67` |
| `func (g *GPUTensor) Size() int` | `tensor_gpu.go:74` |
| `func (g *GPUTensor) Ptr() unsafe.Pointer` | `tensor_gpu.go:77` |
| `func (p *PinnedTensor) Shape() []int` | `pinned.go:76` |
| `func (p *PinnedTensor) Size() int` | `pinned.go:83` |
| `func (b *GPUBackend) Device() int` | `backend.go:46` |

### Util (не операции — 4 функции)

| Функция | Файл:строка |
|---|---|
| `func NewGPUBackend(device int) (*GPUBackend, error)` | `backend.go:30` |
| `func (b *GPUBackend) Close()` | `backend.go:40` |
| `func (b *GPUBackend) RunUnary(...) (*tensor.Tensor, error)` | `backend.go:288` |
| `func (b *GPUBackend) RunBinary(...) (*tensor.Tensor, error)` | `backend.go:302` |

### Итого

`7 + 11 + 0 + 18 + 14 + 4 + 2 + 13 + 4 = 73` ✓ — сходится с общим числом экспортируемых из Секции 1.

## 4. Предлагаемый интерфейс Backend (правка R02a-fix + R02b-fix sealed)

Проектная идея: единый Go-интерфейс `Backend`, за которым живёт runtime-реализация. В R02b будет ровно одна реализация — `purego`-based. Cgo-путь v6 остаётся в файле рядом (`backend.go`/`bridge.go`), но НЕ подкладывается под интерфейс до этапа R02c, где старые `GPUTensor`/`GPUBackend` уходят.

Дизайн-принципы:

1. **Указатель device-памяти не имеет публичной точки выхода из пакета кроме одной именованной двери.** Интерфейс `DeviceBuffer` **запечатан** (`sealed`) unexported-методом `deviceBuffer() bufferView`: тип из-за границы пакета `cuda` физически не может его реализовать, потому что не может объявить unexported-метод чужого пакета. Compute-методы `Backend` принимают этот интерфейс, читают view через `b.deviceBuffer()` и разыменовывают `.ptr` — легально, потому что реализация лежит в том же пакете. Единственный способ **вывести** указатель наружу — функция `UnsafeExtractDevicePtr` в `util.go` (см. дверь-выхода ниже). Единственный способ **завести** чужой указатель внутрь — `WrapDevicePtr` (дверь-входа, возвращает `ForeignStorage`).
2. **Владение — типом.** Разделены два вида device-памяти: `Storage` — наша аллокация (её можно и нужно `Backend.Free`), `ForeignStorage` — чужая (метода `Free` у типа нет — «нельзя освободить» проверяется компилятором, а не рантаймом). Оба реализуют `DeviceBuffer`.
3. **Dtype в имени метода.** `MatMulF64` / `MatMulF32`, а не `MatMul(a, b DeviceBuffer, dtype DType)`. Для будущих FP16/BF16/FP8 добавятся отдельные методы, старые сигнатуры не сломаются, ошибка «не тот dtype» — compile-time.
4. **Backend не знает про `tensor.Tensor`.** Прикладные обёртки типа `Upload/Download` — уровнем выше (в `nn/` или в `cuda/util.go`), чтобы не создавать циклическую зависимость `tensor` ↔ `cuda`.
5. **Ошибки — везде.** Ни один backend-метод не молчит: `error` в возврате обязателен для всех, что могут упасть в runtime (Alloc, Copy, Softmax etc.). Query-геттеры (Device) без ошибки.
6. **Stream — implicit.** Как сейчас (default stream 0). `Sync()` — единственный синхронизатор; cuBLAS/cuStream_t не торчат.

### Код интерфейса (предлагается положить в `cuda/api.go`)

```go
package cuda

import "unsafe"

// bufferView — внутреннее представление device-буфера.
// Unexported: не покидает пакет cuda никогда.
type bufferView struct {
	ptr       unsafe.Pointer
	sizeBytes int
	device    int
}

// DeviceBuffer — общий контракт device-памяти для compute-методов Backend.
// Интерфейс ЗАПЕЧАТАН: unexported-метод deviceBuffer делает невозможной
// реализацию вне пакета cuda. Единственный способ завести внешнюю
// device-память в этот контракт — WrapDevicePtr (возвращает ForeignStorage).
// Единственный способ извлечь указатель наружу — UnsafeExtractDevicePtr
// в util.go. Внутри этих двух дверей указатель по публичному API недоступен.
type DeviceBuffer interface {
	// deviceBuffer возвращает внутренний view. Unexported — печать интерфейса.
	deviceBuffer() bufferView
	// SizeBytes возвращает размер буфера в байтах.
	SizeBytes() int
	// Device возвращает индекс устройства, на котором лежит буфер.
	Device() int
}

// Storage — владельческий handle к device-памяти, аллоцированной через
// Backend.Alloc. Владелец обязан вызвать Backend.Free(s) для освобождения.
type Storage struct {
	ptr       unsafe.Pointer
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
	ptr       unsafe.Pointer
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

// SizeBytes возвращает размер pinned-аллокации в байтах.
func (p PinnedStorage) SizeBytes() int { return p.sizeBytes }

// HostSlice возвращает []byte-view поверх pinned-буфера (zero-copy).
// Валиден только пока PinnedStorage не освобождён.
func (p PinnedStorage) HostSlice() []byte {
	return unsafe.Slice((*byte)(p.ptr), p.sizeBytes)
}

// Backend — единый контракт для GPU-бэкендов.
// В R02b реализуется purego; в дальнейшем может быть добавлен cgo-backend.
type Backend interface {

	// --- Управление устройством ---

	// Device возвращает индекс устройства, на которое привязан backend.
	Device() int
	// Sync блокирует до завершения всех ранее запущенных операций на default stream.
	Sync() error
	// Close освобождает ресурсы backend'а (cuBLAS-handle и т.д.).
	Close() error

	// --- Аллокация device- и pinned host-памяти ---

	// Alloc выделяет sizeBytes байт на device и возвращает владельческий handle.
	Alloc(sizeBytes int) (Storage, error)
	// Free освобождает device-аллокацию. Принимает только Storage — ForeignStorage
	// по типу передать сюда невозможно, что и требуется по дизайну.
	Free(s Storage) error
	// AllocPinned выделяет sizeBytes байт page-locked host-памяти.
	AllocPinned(sizeBytes int) (PinnedStorage, error)
	// FreePinned освобождает pinned-аллокацию.
	FreePinned(p PinnedStorage) error

	// --- Копирования (dtype-нейтральные, байтовые) ---

	// CopyH2D копирует src (host) → dst (device), синхронно.
	CopyH2D(dst DeviceBuffer, src []byte) error
	// CopyD2H копирует src (device) → dst (host), синхронно.
	CopyD2H(dst []byte, src DeviceBuffer) error
	// CopyH2DAsync копирует sizeBytes из pinned host → device асинхронно.
	CopyH2DAsync(dst DeviceBuffer, src PinnedStorage, sizeBytes int) error
	// CopyD2HAsync копирует sizeBytes из device → pinned host асинхронно.
	CopyD2HAsync(dst PinnedStorage, src DeviceBuffer, sizeBytes int) error
	// CopyD2D копирует device → device внутри одного устройства.
	CopyD2D(dst, src DeviceBuffer, sizeBytes int) error

	// --- Elementwise F64/F32 (c[i] = a[i] op b[i]) ---

	// AddF64 поэлементно складывает a и b, результат в c (n элементов).
	AddF64(a, b, c DeviceBuffer, n int) error
	// AddF32 — F32-версия AddF64.
	AddF32(a, b, c DeviceBuffer, n int) error
	// SubF64 поэлементно вычитает: c = a - b.
	SubF64(a, b, c DeviceBuffer, n int) error
	// SubF32 — F32-версия SubF64.
	SubF32(a, b, c DeviceBuffer, n int) error
	// MulF64 поэлементно умножает: c = a * b.
	MulF64(a, b, c DeviceBuffer, n int) error
	// MulF32 — F32-версия MulF64.
	MulF32(a, b, c DeviceBuffer, n int) error
	// DivF64 поэлементно делит: c = a / b.
	DivF64(a, b, c DeviceBuffer, n int) error
	// DivF32 — F32-версия DivF64.
	DivF32(a, b, c DeviceBuffer, n int) error
	// AddScalarF64 добавляет скаляр к каждому элементу: c = a + scalar.
	AddScalarF64(a DeviceBuffer, scalar float64, c DeviceBuffer, n int) error
	// AddScalarF32 — F32-версия AddScalarF64.
	AddScalarF32(a DeviceBuffer, scalar float32, c DeviceBuffer, n int) error
	// MulScalarF64 умножает каждый элемент на скаляр: c = a * scalar.
	MulScalarF64(a DeviceBuffer, scalar float64, c DeviceBuffer, n int) error
	// MulScalarF32 — F32-версия MulScalarF64.
	MulScalarF32(a DeviceBuffer, scalar float32, c DeviceBuffer, n int) error
	// ExpF64 поэлементно: c = exp(a).
	ExpF64(a, c DeviceBuffer, n int) error
	// ExpF32 — F32-версия ExpF64.
	ExpF32(a, c DeviceBuffer, n int) error
	// LogF64 поэлементно: c = log(a).
	LogF64(a, c DeviceBuffer, n int) error
	// LogF32 — F32-версия LogF64.
	LogF32(a, c DeviceBuffer, n int) error
	// NegF64 поэлементно: c = -a.
	NegF64(a, c DeviceBuffer, n int) error
	// NegF32 — F32-версия NegF64.
	NegF32(a, c DeviceBuffer, n int) error

	// --- Activations F64/F32 ---

	// ReLUF64 forward: c = max(0, a).
	ReLUF64(a, c DeviceBuffer, n int) error
	// ReLUF32 — F32-версия ReLUF64.
	ReLUF32(a, c DeviceBuffer, n int) error
	// SigmoidF64 forward: c = 1 / (1 + exp(-a)).
	SigmoidF64(a, c DeviceBuffer, n int) error
	// SigmoidF32 — F32-версия SigmoidF64.
	SigmoidF32(a, c DeviceBuffer, n int) error
	// TanhF64 forward: c = tanh(a).
	TanhF64(a, c DeviceBuffer, n int) error
	// TanhF32 — F32-версия TanhF64.
	TanhF32(a, c DeviceBuffer, n int) error
	// ReLUGradF64 backward: out = grad * (input > 0).
	ReLUGradF64(input, grad, out DeviceBuffer, n int) error
	// ReLUGradF32 — F32-версия ReLUGradF64.
	ReLUGradF32(input, grad, out DeviceBuffer, n int) error
	// SigmoidGradF64 backward: out = grad * sigOut * (1 - sigOut).
	SigmoidGradF64(sigOut, grad, out DeviceBuffer, n int) error
	// SigmoidGradF32 — F32-версия SigmoidGradF64.
	SigmoidGradF32(sigOut, grad, out DeviceBuffer, n int) error
	// TanhGradF64 backward: out = grad * (1 - tanhOut^2).
	TanhGradF64(tanhOut, grad, out DeviceBuffer, n int) error
	// TanhGradF32 — F32-версия TanhGradF64.
	TanhGradF32(tanhOut, grad, out DeviceBuffer, n int) error
	// SoftmaxF64 row-wise softmax на 2D-тензоре [rows x cols].
	SoftmaxF64(a, c DeviceBuffer, rows, cols int) error
	// SoftmaxF32 — F32-версия SoftmaxF64.
	SoftmaxF32(a, c DeviceBuffer, rows, cols int) error

	// --- Reduce F64/F32 ---

	// SumF64 возвращает сумму всех n элементов.
	SumF64(a DeviceBuffer, n int) (float64, error)
	// SumF32 — F32-версия SumF64.
	SumF32(a DeviceBuffer, n int) (float32, error)
	// MeanF64 возвращает среднее по всем n элементам.
	MeanF64(a DeviceBuffer, n int) (float64, error)
	// MeanF32 — F32-версия MeanF64.
	MeanF32(a DeviceBuffer, n int) (float32, error)

	// --- Linalg F64/F32 ---

	// MatMulF64 вычисляет C = A @ B; A[MxK], B[KxN], C[MxN], row-major.
	MatMulF64(a, b, c DeviceBuffer, m, n, k int) error
	// MatMulF32 — F32-версия MatMulF64 (SGEMM через cuBLAS).
	MatMulF32(a, b, c DeviceBuffer, m, n, k int) error
}

// NewBackend — фабрика: возвращает конкретную реализацию Backend
// (в R02b — purego-based; в будущем может выбирать между реализациями).
// Возвращаемое значение — интерфейс, а не конкретный тип, чтобы вызывающий код
// не зависел от реализации.
func NewBackend(device int) (Backend, error) { panic("wire in R02b") }
```

**Считаем методы `Backend`**:
- `Device` + `Sync` + `Close` = 3
- `Alloc` + `Free` + `AllocPinned` + `FreePinned` = 4
- `CopyH2D` + `CopyD2H` + `CopyH2DAsync` + `CopyD2HAsync` + `CopyD2D` = 5
- Elementwise F64/F32 (9 уникальных ops × 2 dtypes) = 18
- Activations F64/F32 (7 уникальных ops × 2 dtypes) = 14
- Reduce F64/F32 (2 уникальных ops × 2 dtypes) = 4
- Linalg F64/F32 (1 op × 2 dtypes) = 2

**Итого: 3 + 4 + 5 + 18 + 14 + 4 + 2 = 50 методов.**

### Utility-слой (предлагается `cuda/util.go`, файл в R02a НЕ создаём)

Функции, которые НЕ входят в интерфейс, потому что общие для всех реализаций (или платформенно-нейтральные):

```go
// cuda/util.go

package cuda

// DetectGPU — есть ли хоть одно CUDA-устройство. Не в интерфейсе,
// потому что вызывается до создания Backend'а.
func DetectGPU() bool

// DeviceCount — сколько CUDA-устройств доступно системе.
func DeviceCount() int

// DeviceName — имя устройства с индексом device.
func DeviceName(device int) string

// DeviceInfo — человекочитаемая строка "имя, свободно/всего GB, N устройств".
func DeviceInfo() string

// MemoryInfo — свободная и общая память device 0 в байтах.
func MemoryInfo() (free, total uint64)

// WrapDevicePtr оборачивает чужой device-указатель в ForeignStorage.
// У ForeignStorage нет метода Free() и Backend.Free его не примет по типу —
// освобождение чужой памяти на уровне API невозможно. Нужно для goml-интеграции.
// ЕДИНСТВЕННАЯ дверь входа device-указателя в запечатанный контракт DeviceBuffer.
func WrapDevicePtr(ptr unsafe.Pointer, sizeBytes, device int) ForeignStorage

// UnsafeExtractDevicePtr возвращает сырой device-указатель буфера.
// ЕДИНСТВЕННОЕ корректное применение — передача во внешние CUDA-биндинги
// (goml-ядра, сторонние библиотеки, чужие обёртки cuBLAS/driver API).
// ЗАПРЕЩЕНО: разыменовывать с host-стороны (это device-память, segfault);
// сохранять как uintptr между вызовами (use-after-free после Backend.Free);
// делать арифметику указателей. Нарушение — undefined behavior без диагностики.
// ЕДИНСТВЕННАЯ дверь выхода device-указателя из пакета cuda.
func UnsafeExtractDevicePtr(b DeviceBuffer) unsafe.Pointer
```

**Файл предлагается положить в `cuda/api.go`, все четыре типа (`bufferView`, `DeviceBuffer`, `Storage`, `ForeignStorage`) — там же**, потому что они — публичный контракт входа/выхода интерфейса `Backend` (плюс один unexported view). Держать интерфейс отдельно от типов его аргументов усложняет чтение без выигрыша: любая реализация будет читать поля `Storage`/`ForeignStorage` через `.deviceBuffer().ptr` в рамках одной пакетной границы. `bufferView` unexported — не покидает пакет.

### Функции, не вошедшие в интерфейс

Инвентарь Секции 3 показывает **36** dtype-специфичных compute-функций (18 Elementwise + 14 Activations + 4 Reduce). Из них в интерфейсе Backend представлено 44 (= 9 + 7 + 2 elementwise/activation/reduce уникальных операций × 2 dtypes). Расхождение объясняется тем, что «36 функций» — это **двойной счёт**: каждая уникальная операция в текущем `cuda/` присутствует и как низкоуровневая обёртка на `unsafe.Pointer` в `bridge.go`, и как высокоуровневый метод на `*GPUBackend` в `backend.go`. Оба вызова кончаются в одном C-символе (`gpu_add_f64` и т.п.).

Ниже — постатейно, что покрыто интерфейсом и что оставлено вне.

**Elementwise — 18 функций Секции 3 = 9 пар (bridge + backend-метод) → 9 методов интерфейса × 2 dtypes = 18 в интерфейсе.**

| Уникальная операция | Bridge-функция | Backend-метод | В интерфейсе |
|---|---|---|---|
| Add | `GPUAdd` | `GPUBackend.Add` | `AddF64`, `AddF32` |
| Sub | `GPUSub` | `GPUBackend.Sub` | `SubF64`, `SubF32` |
| Mul | `GPUMul` | `GPUBackend.Mul` | `MulF64`, `MulF32` |
| Div | `GPUDiv` | `GPUBackend.Div` | `DivF64`, `DivF32` |
| AddScalar | `GPUAddScalar` | `GPUBackend.AddScalar` | `AddScalarF64`, `AddScalarF32` |
| MulScalar | `GPUMulScalar` | `GPUBackend.MulScalar` | `MulScalarF64`, `MulScalarF32` |
| Exp | `GPUExp` | `GPUBackend.Exp` | `ExpF64`, `ExpF32` |
| Log | `GPULog` | `GPUBackend.Log` | `LogF64`, `LogF32` |
| Neg | `GPUNeg` | `GPUBackend.Neg` | `NegF64`, `NegF32` |

Пропущенных уникальных операций нет.

**Activations — 14 функций Секции 3 = 7 пар → 7 методов интерфейса × 2 dtypes = 14 в интерфейсе.**

| Уникальная операция | Bridge | Backend | В интерфейсе |
|---|---|---|---|
| ReLU | `GPUReLU` | `GPUBackend.ReLU` | `ReLUF64`, `ReLUF32` |
| Sigmoid | `GPUSigmoid` | `GPUBackend.Sigmoid` | `SigmoidF64`, `SigmoidF32` |
| Tanh | `GPUTanh` | `GPUBackend.Tanh` | `TanhF64`, `TanhF32` |
| ReLUGrad | `GPUReLUGrad` | `GPUBackend.ReLUGrad` | `ReLUGradF64`, `ReLUGradF32` |
| SigmoidGrad | `GPUSigmoidGrad` | `GPUBackend.SigmoidGrad` | `SigmoidGradF64`, `SigmoidGradF32` |
| TanhGrad | `GPUTanhGrad` | `GPUBackend.TanhGrad` | `TanhGradF64`, `TanhGradF32` |
| Softmax | `GPUSoftmax` | `GPUBackend.Softmax` | `SoftmaxF64`, `SoftmaxF32` |

Пропущенных уникальных операций нет.

**Reduce — 4 функции Секции 3 = 2 пары → 2 методов интерфейса × 2 dtypes = 4 в интерфейсе.**

| Уникальная операция | Bridge | Backend | В интерфейсе |
|---|---|---|---|
| Sum | `GPUSum` | `GPUBackend.Sum` | `SumF64`, `SumF32` |
| Mean | `GPUMean` | `GPUBackend.Mean` | `MeanF64`, `MeanF32` |

Пропущенных уникальных операций нет.

**Ожидаемые ТЗ-кандидаты, которых в инвентаре `cuda/` v6 нет и в интерфейс не добавляются:**

`grep -nE 'GPU(Pow|Sqrt|Abs|Floor|Ceil|Round|Sign|GELU|SiLU|LeakyReLU|ELU|Softplus|Max|Min|ArgMax)' cuda/*.go` — пусто. `grep -nE 'gpu_(pow|sqrt|abs|floor|ceil|round|sign|gelu|silu|leaky|elu|softplus|max|min|argmax)' cuda/*.{go,h,cu}` — пусто.

- **PowScalar, Sqrt, Abs, Floor, Ceil, Round, Sign** — в `cuda/` не реализованы, ни в `bridge.go`, ни в `ops.cu`, ни в `cuda.h`. В интерфейс R02a не добавляются: интерфейс отражает то, что покрыто в текущем коде. Появятся — добавим одной парой методов на операцию.
- **GELU, SiLU, LeakyReLU, ELU, Softplus** — не реализованы в `cuda/`. Есть только в CPU-пути `nn/` (Go-код). Добавлять пустые слоты в интерфейс не будем, чтобы R02b/purego-backend не отдавал `ErrNotImplemented` на неполовину-планированные вещи.
- **Max, Min, ArgMax** — не реализованы. Не добавляются по той же причине.

Прикладные обёртки, оставленные вне интерфейса намеренно (они принадлежат уровню поверх Backend):

- `GPUBackend.Upload` / `GPUBackend.Download` — работают с `*tensor.Tensor`, тянут импорт `github.com/djeday123/gotorch/tensor` в `cuda/`; в интерфейсе не место. Переедут в `cuda/util.go` (или в `tensor/` рядом с `Tensor`) после R02c.
- `GPUBackend.RunUnary` / `GPUBackend.RunBinary` — сахар над Upload+op+Download; тоже уровень поверх, не в контракте backend'а.
- `GPUTensor.ToCPU` / `PinnedTensor.ToCPUTensor` / `PinnedTensor.Slice` — прикладные конверсии тензора, не операция backend'а.
- `NewGPUTensor` / `NewGPUTensorEmpty` / `NewPinnedTensor` / `GPUTensor.Free` / `PinnedTensor.Free` / `PinnedTensor.ToGPU` / `PinnedTensor.FromGPU` / `H2DAsync` / `D2HAsync` — реализуются в терминах `Backend.Alloc` / `Backend.AllocPinned` / `Backend.CopyH2D*` / `Backend.CopyD2H*`. В R02c старые типы удалятся, эти конкретные функции — тоже.
- Read-only геттеры (`GPUTensor.Shape`, `GPUTensor.Size`, `GPUTensor.Ptr`, `PinnedTensor.Shape`, `PinnedTensor.Size`) — свойства прикладных типов, не backend'а. Аналогичное свойство `SizeBytes`/`Device` на новом `Storage`/`ForeignStorage`/`PinnedStorage` уже есть.
- `Init` / `NewGPUBackend` / `GPUBackend.Close` / `GPUBackend.Device` — заменяются фабрикой `NewBackend(device)` и методами `Backend.Close()` / `Backend.Device()`.

## 5. Особые случаи и открытые вопросы

- **Streams и cuBLAS-handle.** В текущем коде cuBLAS-handle статически глобален внутри `.cu`-стороны (`ops.cu:43`: `static cublasHandle_t g_cublas = NULL;`). Stream тоже implicit (default stream 0), а `StreamSync()` из `pinned.go:110-112` — вызов `cudaDeviceSynchronize` без stream-аргумента. В новом интерфейсе это отражено методом `Backend.Sync()` без параметра-stream; сам `cublasHandle_t`/`cudaStream_t` живут внутри purego-реализации и **наружу не торчат** (ни как `Handle`-тип, ни как поле `Storage`). Если в будущем потребуется multi-stream — добавится отдельный `type Stream struct{ ... }` рядом с `Storage`, но это уже нарушение текущего контракта, планировать не сейчас.
- **C-callbacks.** Поиск: `grep -rn 'callback\|C.CFunc\|purego.NewCallback\|extern.*_Cfunc\|GoBridge' cuda/*.go` — **пусто** (exit 1). Go-функции в C для вызова из ядра нигде не передаются. R02b/purego-реализация не столкнётся с самой сложной покерной картой purego (callback-thunks c правильным ABI).
- **Внешний device-указатель и sealed-контракт.** В текущем коде — **нет** конструктора вида `WrapDevicePtr` / `RegisterExternal` / `FromDevicePtr`. Поиск: `grep -rn -E 'External|Wrap|Register|FromPtr|FromDevice|WithDevice|Foreign' cuda/` — **пусто**. Наружу торчит только read-only геттер `func (g *GPUTensor) Ptr() unsafe.Pointer` (`tensor_gpu.go:77`), но обратной обёртки нет. В новом API это исправлено дизайном с двумя именованными дверями и запечатанным (sealed) контрактом:
  - **Дверь входа — `WrapDevicePtr(ptr, sizeBytes, device) ForeignStorage`** в `cuda/util.go`. Возвращает `ForeignStorage`. Тип не имеет метода `Free()`, `Backend.Free` его не принимает по сигнатуре — освобождение чужой памяти невозможно на уровне компилятора, а не рантайма.
  - **Дверь выхода — `UnsafeExtractDevicePtr(b DeviceBuffer) unsafe.Pointer`** в `cuda/util.go`. Единственная публичная функция пакета, возвращающая device-указатель наружу. Применение — передача во внешние CUDA-биндинги (goml-ядра, чужие обёртки cuBLAS/driver API).
  - **Запечатанный интерфейс `DeviceBuffer`** с unexported-методом `deviceBuffer() bufferView`. Тип из-за пределов пакета `cuda` физически не может его реализовать (unexported-методы чужого пакета не декларируются), значит все compute-аргументы гарантированно приходят из `Storage` или `ForeignStorage`. Публичного `Ptr()` в интерфейсе нет — указатель просто не проходит через публичный API.
  
  Это критично для будущей goml-интеграции, где goml-ядра сами аллоцируют device-память и передают её в gotorch для forward pass через слои `nn/`, а gotorch отдаёт свои `Storage` наружу через `UnsafeExtractDevicePtr` при вызове чужих ядер. Обе двери именованные, каждый вызов виден в grep, ни одна не оправдана «случайным разыменованием».

## 6. Резюме

- **Экспортируемых символов всего: 76.** (73 функции/метода + 3 типа: `GPUTensor`, `PinnedTensor`, `GPUBackend`.)
- **Методов в предложенном интерфейсе `Backend`: 50.**
- **Реализуется ли полностью через purego:** да, полностью. R02b/purego-backend делает `dlopen` **трёх** библиотек:
  - `libcudart.so.12` — для `cudaMalloc`/`cudaFree`/`cudaMemcpy`/`cudaMemcpyAsync`/`cudaDeviceSynchronize`/`cudaMallocHost`/`cudaFreeHost`/`cudaGetDeviceCount`/`cudaSetDevice`/`cudaMemGetInfo`/`cudaGetDeviceProperties`.
  - `libcublas.so.12` — для `cublasCreate`/`cublasDestroy`/`cublasDgemm`/`cublasSgemm`.
  - `libcuda.so.1` — driver API, опционально (для `cuModuleLoadData` + `cuLaunchKernel`, см. ниже).
  
  Callback'ов нет (см. Секцию 5), значит purego-thunks с ABI-сложностями не нужны.

**libgotorch_cuda.so остаётся в директории `cuda/` как задел на будущее.** В purego-backend этапа R02b он **не подгружается**. Elementwise/activation/reduce-операции реализуются в purego-backend как PTX-ядра, скомпилированные из Go-строк и загруженные через `cuModuleLoadData` + `cuLaunchKernel` из CUDA driver API (по образцу `/data/lib/podman-data/projects/goml/backend/cuda/kernels.go`). Если в будущем возникнет потребность подключить готовые ядра из `libgotorch_cuda.so` через `purego.Dlopen` — это делается без изменения интерфейса `Backend`, добавлением альтернативной реализации в отдельном файле.

Обходов через cgo в интерфейсе **не предусмотрено**. Если в R02b выяснится, что какой-то cudart/cublas/driver-символ не берётся через `dlsym` в конкретной версии драйвера — это будет отдельная запись сюда и в open questions R02b, не в архитектурный дефект интерфейса.
