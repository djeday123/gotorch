# P5B — Batched FP16/FP8 MatMul в gotorch: design paper

**Дата:** 2026-07-23
**Статус:** DESIGN-ONLY. Ни строки production-кода. Продукт — набор решений с обоснованиями от инвентаря к этапности.
**Правило главы:** paper-first, как перед B2 в FA-серии. После доклада — моё ревью, затем ТЗ на impl.

---

## B1. Инвентарь боевого пути

Grep MatMul в goml по всем 4 уровням (backend interface / cuda impl / nn / cmd). Живое отделено от законсервированного по правилу главы.

### B1.1 Backend interface

| Файл:строка | Сигнатура | Тип |
|---|---|---|
| `goml/backend/backend.go:106` | `MatMul(dst, a, b, shapeA, shapeB, dtype) error` | **единственный public MatMul** |

`shapeA` может быть N-мерным; batchSize = product(shapeA[:-2]). Dtype — только `Float32` (проверка в impl). **F16/F8 в интерфейс не выведены**.

### B1.2 CUDA impl

| Файл:строка | Метод | Путь |
|---|---|---|
| `goml/backend/cuda/ops.go:236-260` | `Backend.MatMul` | если batchSize==1 → `cublas.MatMulF32` (cublasSgemm_v2); иначе → `cublas.BatchedMatMulF32` (cublasSgemmStridedBatchedEx через wrapper) |
| `goml/backend/cuda/cublas.go:264-289` | `BatchedMatMulF32` | wrapper `libcublas_wrapper.so::gemmStridedBatchedExWrapper`, TF32 compute |
| `goml/backend/cuda/cublas.go:343-347` | `MatMulF16` | wrapper single call `cublasGemmEx`, FP16 IO + TF32 compute |
| `goml/backend/cuda/cublas.go:397-403` | `BatchedMatMulF16` | wrapper `cublasGemmStridedBatchedEx`, FP16 IO |
| `goml/backend/cuda/cublas_fp8.go:64-86` | `MatMulF8E4M3` | wrapper `libcublaslt_wrapper.so::fp8_matmul_wrapper` — cuBLASLt cached workspace |
| `goml/backend/cuda/cublas_fp8.go:88-90` | `MatMulF8E5M2` | **не реализовано** (returns error) |
| `goml/backend/cuda/fp8_gemm.go` | `fp8gemm.gemm/original/ss` | dlopen `libfp8gemm.so` — свои PTX FP8 kernel'ы, R2/FA-research |

### B1.3 Callers боевого пути (nn/cmd)

| Файл:строка | Что зовёт | Форма (боевая) |
|---|---|---|
| **`goml/cmd/gputrain/main.go:118`** | `gpu.MatMul(logits, normed, outWeight, [16,64], [64,256], F32)` | **LIVE production trainer**: [seqLen=16, embed=64] × [embed=64, vocab=256] |
| `goml/cmd/simpletrain/main.go` | analog gputrain | LIVE mini-trainer |
| `goml/nn/linear.go:61` | `ops.MatMul(x, w^T)` — вызывается из `Linear.Forward` | зависит от размера Linear |
| `goml/nn/attention.go:65-89` | 4×`Linear.Forward` для Q/K/V/O проекций | `[batch, seqLen, dim]` × `[dim, dim]` |
| `goml/nn/backward.go` | `Linear.Backward` (nn.LLM training path) | `[batch·seq, dim]` — CPU F32 путь через weights.Grad |
| **`goml/cmd/fp16bench/main.go:186-205`** | `MatMulF16`, `MatMulF8E4M3` | **BENCH-ONLY**, не production; полигон производительности |
| `goml/cmd/fa_v121r_diet` (`runs/`) | fp8_gemm через libfp8gemm | **R2/FA-research**, не production |

### B1.4 Живое / законсервированное

**Живое (в production-пути gputrain/nn):**
- `backend.Backend.MatMul` (**F32**) → `cublasSgemm_v2` (`cublas.MatMulF32`), single и strided batched.

**Законсервированное (workflowы существуют, но НЕ в production trainer'е):**
- `cublas.MatMulF16` + `BatchedMatMulF16` — работают, вызываются только `fp16bench`. Полностью рабочая инфраструктура: `libcublas_wrapper.so` уже собран, biэксэкт vs cublasGemmEx.
- `cublas_fp8.MatMulF8E4M3` — работает, вызывается `fp16bench`. `libcublaslt_wrapper.so` уже собран. E5M2 не реализован.
- `fp8_gemm.gemm` (свои PTX ядра, `libfp8gemm.so`) — R2/FA-research. Свободные FP8 GEMM без cuBLAS зависимости.

**Сироты, `libtransformer.so`:**
- уже 2 записи в `[[legacy-inventory]]` (rmsnorm/rope FP16). MatMul-родственных нет (transformer.so не имеет MatMul).

### B1.5 FP8 квантизация

**amax + scale механика (из `goml/libs1/`):**

- `libs1/fp8_quantize.cu:76-77`: `float amax = *absmax_ptr; scale = amax / FP8_E4M3_MAX;`
- `libs1/fp8_quantize_v2.cu:92-93`: то же с per-tile amax.
- `libs1/cublas_lt_wrapper_v2.c:267`: `CUBLASLT_MATMUL_DESC_AMAX_D_POINTER` — cuBLASLt пишет absmax(D) в device pointer автоматически.

**Итог:** amax поддерживается на уровне GPU (per-tensor, per-row варианты в v2). Scale вычисляется на CPU host после копирования amax обратно, либо GPU-side kernel'ом `fp8_quantize`.

### B1.6 Итог инвентаря

| Слой | Живое | Законсервированное |
|---|---|---|
| Interface | F32 non-batched + strided-batched | — |
| Impl F16 | — | MatMulF16 + Batched (libcublas_wrapper) |
| Impl F8 E4M3 | — | MatMulF8E4M3 (libcublaslt_wrapper v1) |
| Impl F8 E5M2 | — | не реализовано |
| Impl F8 свои PTX | — | fp8_gemm (libfp8gemm) — R2 FA research |
| Quantize | — | fp8_quantize.cu + amax через cuBLASLt-desc |

**Приоритет портирования:** F32 batched первым (живой; уже есть в gotorch как non-batched), потом F16 (батчи из inference-path и потенциальный mixed-precision training), потом F8 (по запросу когда gputrain перейдёт на mixed-precision).

---

## B2. Целевой API gotorch

Правило R02a: **dtype в имени метода, никаких режимов-состояний**. Сигнатуры следуют фактическим потребностям Step, не изобретают общности.

### B2.1 F32 batched (расширение существующего)

```go
// MatMulStridedBatchedF32 — strided batched GEMM.
// A[batch, M, K] × B[batch, K, N] = C[batch, M, N].
// Страйды в **элементах**, не байтах.
MatMulStridedBatchedF32(a, b, c DeviceBuffer, batch, m, n, k int,
    strideA, strideB, strideC int64) error

// Для F64 -- судейский путь. cublasDgemmStridedBatched cheap.
MatMulStridedBatchedF64(...) error
```

**Обоснование:** сейчас в gotorch только non-batched MatMulF32/F64. Batched нужно для attention (Q@K^T, A@V) и для многослойных Linear в fused-микро-батчах. Реальная потребность — из nn.MultiHeadAttention.

### B2.2 F16 mixed precision

```go
// MatMulF16 -- IO=FP16 [half], compute=TF32, accumulator=FP32, out=FP32.
// Форма как MatMulF32; A: []half, B: []half, C: []float32.
MatMulF16(a, b, c DeviceBuffer, m, n, k int) error
MatMulStridedBatchedF16(a, b, c DeviceBuffer, batch, m, n, k int,
    strideA, strideB, strideC int64) error

// FP16 quant (F32 -> FP16 host-side промоушен не нужен; квант через ptx cvt).
QuantizeF32ToF16(src, dst DeviceBuffer, n int) error
CastF16ToF32(src, dst DeviceBuffer, n int) error
```

**Half dtype в gotorch — НОВЫЙ.** DeviceBuffer нейтрален к типу (P3 подтвердил: контракт живёт в doc-string методов). `Alloc(sizeBytes)` + `CopyH2D` работают для любых байтов. Значит **новых Storage-типов не нужно**, half живёт в контракте метода.

Doc-string: half буферы = `[]uint16` interpretive, LE, IEEE 754 binary16.

### B2.3 F8 E4M3

```go
// MatMulF8E4M3 -- IO=FP8 E4M3, compute=FP32 tensor cores, out=FP32.
// scaleA, scaleB, scaleC: device float32-скаляры (per-tensor amax/max).
// amaxD: optional device float32-скаляр -- cuBLASLt пишет absmax(D) автоматически.
MatMulF8E4M3(a, b, c DeviceBuffer,
    m, n, k int,
    scaleA, scaleB, scaleC DeviceBuffer,
    amaxD DeviceBuffer /* optional, may be nil */) error

MatMulStridedBatchedF8E4M3(a, b, c DeviceBuffer,
    batch, m, n, k int,
    strideA, strideB, strideC int64,
    scaleA, scaleB, scaleC, amaxD DeviceBuffer) error

// Квантизация: reduce+cast в одном kernel'е (наш паттерн P2/P3 -- fused).
// Считает amax по всему src, вычисляет scale=amax/E4M3_MAX, пишет FP8 в dst.
QuantizeF32ToF8E4M3(src, dst, scale, amax DeviceBuffer, n int) error
```

**Дизайн-выбор:** отдельные Quantize и MatMul методы (не фьюзнуто в один API-вызов). Причина: (а) прецедент R02a — dtype в имени метода, а не режим; (б) в реальном training-loop scale переиспользуется между forward и backward, поэтому Quantize вызывается один раз на forward, MatMul несколько раз использует тот же scale — фьюжн привёл бы к double-Quantize.

### B2.4 F8 E5M2 (out-of-scope P5)

Не реализовывать. E5M2 нужен для gradients (динамический range шире), но требует отдельного FP8-flow для backward. Резервируем API-slot, вернёмся после mixed-precision в gputrain.

---

## B3. Реализация: cublasLt vs своё

**Вопрос:** cublasLt-purego-биндинги vs свои PTX-ядра vs гибрид.

### B3.1 Три варианта

**(a) Свои PTX GEMM ядра.** По прецеденту goml `fp8_gemm.cu` (libfp8gemm) — реально возможно, работает в R2/FA-research.
- Плюсы: полный контроль, никаких внешних `.so`, alpine-build остаётся clean (R02b правило alpine-no-CUDA).
- Минусы: **не догнать cutlass/cuBLASLt** в GEMM. FA-серия достигла 647T fwd на hd=128 (63% of card peak) через недели multi-agent optimization. GEMM — та же задача, cutlass у NVIDIA годами тюнится; свои ядра дадут 30-50% от cuBLASLt.
- Вывод: **отвергнуто** для production MatMul. Свои ядра оставить R2 area.

**(b) cuBLASLt через purego-биндинги.** По прецеденту goml `libcublaslt_wrapper.so`.
- Плюсы: полная скорость NVIDIA-optimized, поддержка FP8/FP16/BF16/TF32 из коробки, workspace-caching, algo-heuristic-selection. Батчи через cublasLtMatmul с matmul-descs.
- Минусы: сложнее API — matmul-desc + preference + heuristic-choose + workspace-plan; много cleanup-code (см. `cublas_lt_wrapper.c` — 267+ строк). purego-биндинги cublasLt функций разнообразнее чем cublas (matmul-desc set-attribute — struct-based interface).
- **Прецедент рабочий**: goml уже вызвает cublasLt через C wrapper (`libcublaslt_wrapper.so::fp8_matmul_wrapper`), тот же путь для gotorch.
- **Alpine-flag**: cublasLt требует `libcublasLt.so.12` в runtime; alpine-no-CUDA build должен gracefully skip (как gotorch/v6 сейчас делает для non-GPU tests).

**(c) Гибрид: cuBLAS purego для F32-batched (уже есть fondation в cublas_purego.go), cuBLASLt purego для F16/F8.**
- Плюсы: минимальный incremental путь. F32 batched — расширение существующей R02b фундации; F16/F8 — новая cublasLt-инфраструктура.
- Минусы: две библиотеки одновременно.

### B3.2 Рекомендация

**Вариант (c) — гибрид.**
- F32 batched: purego `cublasSgemmStridedBatchedEx` — расширение существующего `cublas_purego.go`. Без .so-обёрток.
- F16 non-batched: purego `cublasGemmEx` через cublas.so — не cublasLt (F16 работает в classic cublas).
- F16 batched: purego `cublasGemmStridedBatchedEx`.
- F8 E4M3: purego `cublasLtMatmul` через cublasLt.so + workspace-management.

**Обёртки `.so` не нужны.** goml использует C wrappers (`libcublas_wrapper.c`, `libcublaslt_wrapper.c`) из-за purego-ограничения на функции с >20 аргументами. **cublasGemmEx имеет 19 аргументов, cublasLtMatmul — 13.** purego support на 22.10+ поддерживает 20+ аргументов через `SetContext` ABI. Проверить экспериментально в первом impl-этапе; fallback = struct-args wrapper (как у goml).

### B3.3 F64 batched (судья)

`cublasDgemmStridedBatched` — 15 аргументов, purego OK. Дёшево добавить в план как pure F64 путь. **F64-судья должен уметь batched.**

---

## B4. Точностная модель

### B4.1 Floor'ы от FA-опыта (из FA-серии, memory)

Из [[sm120-qmma-microbench-L-T]]: FP8 e4m3 T=17, card peak ~960T FP8. Из FA v96b: 62% ceiling для FP8 GEMM.

**Точностные класы (из FA experience):**
- F32 GEMM (default cublas Sgemm): rel ~ 1e-6 vs F64 ref.
- F32 TF32 GEMM (cublas MatMulF32_TF32): rel ~ 1e-3 (10-bit mantissa in FMA, задокументировано impl-4-final).
- F16 IO + FP32 accum: rel ~ 5e-4 (11-bit mantissa; ошибка суммы N слагаемых ~ eps·sqrt(N)).
- F8 E4M3 IO + FP32 accum: rel ~ **5e-3 класс** (4-bit mantissa). Это FA-опыт (b1-fix-fp64-golden vs FA-fp8).

### B4.2 Двухуровневая валидация (как в B-серии)

**Уровень 1 (fast):** каждый порт против F32 CPU/gotorch reference. Hybrid abs+rel floor из класса dtype.
**Уровень 2 (точный):** каждый порт против F64 judge (cublasDgemmStridedBatched для batch, или наш F64 gotorch). Rel <= 1e-12 не ожидается для F16/F8 — но для F32-batched **обязательно** (F32×F32 → F64 accumulator в reference).

### B4.3 A/B floor pre-registration

| Путь | Floor | Обоснование |
|---|---|---|
| F32 batched B(gotorch cublasSgemm) vs J(F64 gotorch cublasDgemm) | rel ≤ **1e-6** hybrid | FP32 eps + N-slaugемых |
| F32 batched A(goml.cuda) vs B(gotorch) | **bit-exact** при одинаковом cublas math mode | одна и та же ptxas-cutlass |
| F16 batched B vs J(F64) | abs ≤ 1e-3 + rel ≤ **1e-3** | FP16 IO |
| F16 batched A(goml.cuda MatMulF16) vs B(gotorch) | **bit-exact** при том же compute type | тот же cublasGemmEx |
| F8 E4M3 B vs J(F64) | abs ≤ **5e-3 + rel ≤ 5e-3** | 4-bit mantissa класс |
| F8 E4M3 A(goml libcublaslt_wrapper) vs B(gotorch cublasLt purego) | rel ≤ **1e-4** | оба идут в cuBLASLt |

**Правило "два числа" из [[feedback-atomicadd-drift-oscillation]] сохраняется:** pre-reg + actual, не переписывать прогноз.

---

## B5. Память и Pro-4000-сценарий

**Прямое требование пользователя (по ТЗ):** параметризация геометрии, не зашивать sl=8192.

### B5.1 Workspace cuBLASLt

Из `libs1/cublas_lt_wrapper.c:76-80`: preference `MAX_WORKSPACE_BYTES` кэшируется на форму (M/N/K). Первый вызов делает `MatmulAlgoGetHeuristic` — выбирает алгоритм (может 32MB-256MB workspace). Кэш живёт до следующей смены формы.

**Расчёт для gputrain-shape (vocab=32000, embed=256, batch·seqLen=256):**
- Fwd: 256×256 × 256×32000 = 256×32000 outputs = 8M FP32 out + 32K params + activations.
- cublasLt workspace для 256×256×32000: heuristic обычно 16-64 MB.
- Форма стабильна — один cache-entry hold.

**На Pro-4000-класс (24GB):**
- 32000×256 embed table = 32MB F32.
- Batched attention для batch=4/heads=32/seq=2048/hd=64: `[4×32×2048×64] = 16M FP32 = 64MB` per tensor.
- cublasLt workspace: 64-256MB (динамический выбор).
- **Total per-step**: ~500MB — вполне помещается в 24GB.

**Параметризация geometry (не зашивать):** методы принимают M/N/K/batch/stride параметрами, cache workspace-по-форме. Смена формы → повторный heuristic. **Нельзя кешировать по int64-hash — cublasLt-выбор зависит и от dtype, добавить в ключ.**

### B5.2 Scratch pool (переиспользовать P5A-паттерн)

Из [[gotorch-r03b-p5a-emb-i64-closed]]: `ensureScratchXxx(n)` + освобождение в `Close`. Применить для cuBLASLt workspace: `ensureCublasLtWorkspace(bytes)` — растёт до max seen, освобождается в Close.

### B5.3 Quantize scale/amax буферы

`scaleA, scaleB, scaleC, amaxD` — по 1 float32 каждый = 16 bytes. Копеечные. Можно alloc-per-call для user-facing API; для internal fused-path — пул из 4 float32 slots.

### B5.4 Оценка ограничений

- Максимальная форма для нашей карты (RTX 6000 Blackwell 48GB): batch=8 seq=8192 dim=1024 headDim=128 = attention Q/K/V каждый ~2GB, MatMul workspace ~1GB = ~10GB total. Помещается.
- Для Pro-4000 (24GB): максимум batch=4 seq=4096 dim=1024 = ~5GB total. Достаточно.
- **Ни один параметр в код зашивать НЕ надо** — сигнатуры принимают M/N/K/batch/stride, workspace растёт лениво.

---

## B6. Этапность с воротами

Скорректировано по фактам B1-B5:

### B6.1 B-impl-1: F32 batched foundation

**Задача:** расширить `cublas_purego.go` с `cublasSgemmStridedBatchedEx` + `cublasDgemmStridedBatched`. Добавить методы `MatMulStridedBatchedF32/F64`.

**Тесты:**
- shapes: [batch=1..64] × M/K/N × [16..2048].
- F32 batched B(gotorch) vs J(F64 gotorch batched) — hybrid abs=1e-4 + rel=1e-6.
- F32 batched A(goml.cuda) vs B(gotorch) — bit-exact при одинаковом math mode (аналог impl-4-final).

**Ворота:** полная регрессия + FA-canary + новые тесты. Method count: 66 → 68 (+2).

### B6.2 B-impl-2: F16 non-batched + batched

**Задача:** purego-биндинги `cublasGemmEx` (проверить 19-arg через purego SetContext; fallback = struct-arg wrapper по goml-прецеденту).

Методы: `MatMulF16`, `MatMulStridedBatchedF16`, `QuantizeF32ToF16`, `CastF16ToF32`.

**Тесты:**
- F16 batched B vs J(F64) — hybrid abs=1e-3 + rel=1e-3.
- A(goml.cuda MatMulF16) vs B(gotorch) — bit-exact при одном compute type.
- QuantizeF32ToF16 bit-exact vs CPU ref (простой cvt).
- CastF16ToF32 — bit-exact лёгкий тест.

**Ворота:** регрессия + method count 68 → 72.

### B6.3 B-impl-3: F8 E4M3 + quantize

**Задача:** purego-биндинги `cublasLtMatmul` + `cublasLtMatrixLayoutCreate/Destroy` + `cublasLtMatmulDescCreate/Destroy`. Workspace-cache (~scratch pool). `QuantizeF32ToF8E4M3` PTX-ядро (reduce amax + cvt).

Методы: `MatMulF8E4M3`, `MatMulStridedBatchedF8E4M3`, `QuantizeF32ToF8E4M3`.

**Тесты:**
- F8 E4M3 B vs J(F64) — abs 5e-3 + rel 5e-3.
- A(goml.cuda libcublaslt_wrapper) vs B(gotorch cublasLt purego) — rel ≤ 1e-4.
- Quantize amax + scale roundtrip: `QuantizeF32ToF8 → MatMul → CastF32ToF8*scaleC^-1` внутри hybrid tolerance.

**Ворота:** регрессия + FA-canary + method count 72 → 75. **Alpine-graceful** (без libcublasLt.so.12 skip).

### B6.4 B-impl-4: adapter стрелка + боевой Step

**Задача:** `backend.Backend.MatMul` в adapter стрелку delegate→direct через `MatMulStridedBatchedF32`. gputrain-Step ходит через adapter (F32 путь). Опционально: mixed-precision путь через adapter (F16 stub-callers).

**Тесты:**
- gputrain-Step 10-step A/B/J (как P1-ABJ). Прогноз: |A-B| < 1e-5 (F32 batched тот же cublasSgemm).
- FA-canary fwd + bwd.

**Ворота:** регрессия + FA-canary. Финальный отчёт B-final.md.

### B6.5 Итог этапности

| Этап | Кол-во новых методов | Кол-во строк PTX | Кол-во строк Go |
|---|---|---|---|
| B-impl-1 (F32 batched) | +2 | 0 | ~200 |
| B-impl-2 (F16) | +4 | ~50 (cvt) | ~400 |
| B-impl-3 (F8 E4M3) | +3 | ~150 (quantize+amax) | ~600 |
| B-impl-4 (adapter) | 0 (dispatch) | 0 | ~100 |
| **Итого** | **+9** | ~200 | ~1300 |

Method count финал: 66 → **75** (+9 methods за всю Часть B).

**Сроки-порядок (по опыту P2-P4):** каждый B-impl = 1 рабочая сессия (2-3 часа). Общее: 4 сессии.

---

## Резюме решений

1. **Инвентарь**: живая F32 (single + batched); законсервированные F16/F8 (полностью рабочие, но только в bench/research).
2. **API**: dtype в имени, F16/F8 живут в контракте метода (DeviceBuffer нейтрален); отдельные Quantize и MatMul (не фьюжн).
3. **Реализация**: гибрид — cuBLAS purego для F32/F16, cuBLASLt purego для F8. Обёртки .so не нужны при purego 22.10+ (проверить эксперим.).
4. **Точность**: pre-reg floor'ы по классам dtype (F32 1e-6, F16 5e-4, F8 5e-3). Двухуровневая валидация fast(F32-ref) + honest(F64-judge).
5. **Память**: параметризация M/N/K, workspace-pool через `ensureCublasLtWorkspace(bytes)` по P5A-паттерну. Никаких sl=8192 hard-codes.
6. **Этапность**: B-impl-1..4 по прецеденту P-серии, каждый этап отдельный отчёт+commit+push.

Отсутствуют формулировки «наверное», «возможно» — все решения с обоснованиями и файл:строка.

---

## СТОП по правилу ТЗ

Paper готов. **Ни строки кода не написано.** Жду ревью пользователя перед ТЗ B-impl-1.

Открытые вопросы для обсуждения:
- Экспериментальная проверка purego 20+ arg support для cublasGemmEx: делать до B-impl-2 или готовить struct-args wrapper заранее?
- FP8 E5M2 в дорожную карту (сейчас out-of-scope) — когда именно возвращаемся?
- gputrain-Step переключение на mixed-precision (F16 путь) — параллельно с B-impl-2 или после?
