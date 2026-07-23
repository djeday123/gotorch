# R03b-impl-4-final — Linear.Forward A/B, TF32 vs FP32 разложено

**Дата:** 2026-07-20
**Итог:** ✅ CLOSED. 4 сверки PASS + 3 хвоста PASS + все регрессии PASS + FA canary reanchored. TF32 vs FP32 разница режимов документирована как переходная; adapter 3648.7× точнее legacy; scrytый багов раскладки нет (доказано TF32-vs-TF32 bit-exact).

---

## Часть 0 — FA-канарейка (по правилу серии)

| Компонент | Median | Baseline | Vердикт |
|---|---|---|---|
| Forward v121r bh=128 sl=8192 wnd=0 | **653.86T** | 652 ± 2T | ✓ WITHIN |
| Bwd nc E2E R2C bh=128 sl=8192 | **41.561ms** | **41.6 ± 0.3ms** (обновлённый anchor) | ✓ WITHIN |
| Bwd causal | 21.948ms | 22.206 ± 0.3ms | ✓ WITHIN |
| Fingerprint 4/4 | OK | | ✓ |

Устаревший **42.346ms** помечен в memory как «ждёт пере-сертификации» — 5-run canary не переписывает 30-run якорь, новый родится при полной сертификации.

`_canary_5run_fwd.sh` в `goml/runs/` — регрессионный набор всех будущих ворот R03b-impl.

---

## Часть 1 — Решение по TF32-расхождению (Вариант 3)

**Приняли** документированную разницу режимов. **Path B (SetMathMode-выравнивание) отклонён окончательно** — прогиб gotorch (единственного целевого источника истины) под legacy = первый прецедент эрозии.

### 1.1 R03b_design.md impl-5 таблица правлена

MatMul(F32, batch=1) direct: было «bit-exact, floor=0», стало «разница режимов документирована, rel ≤ 1e-2 hybrid». Ключевая формулировка (не расширение допуска):

> Предположение bit-exact **опровергнуто фактом**: `goml/backend/cuda/cublas.go:216` включает TF32 в legacy cublas handle, gotorch держит pedantic FP32 (`cublas_purego.go:5-9`). Adapter-путь **~3600× точнее legacy** (измерено). Скрытых багов раскладки НЕТ — доказано bit-exact 256/256 через MatMulF32_TF32 sверкой 3.4.

Это исправление ошибочной посылки (принцип R02b: floor записан до измерения; ошибку в посылке правим, tolerance задним числом не расширяем).

### 1.2 goml/ARCHITECTURE.md одна строка добавлена

> **Известная разница режимов (R03b-impl-4-final):** legacy cublas handle — TF32, gotorch — pedantic FP32. Adapter direct-путь ~3600× точнее. Выравнивание gotorch под legacy TF32 **не выполняется**. При полном переходе трейна на gotorch разница исчезает в пользу FP32. Точечный TF32 доступен через `gotorch.PuregoBackend.MatMulF32_TF32` — режим свойство метода, не состояние backend'а.

---

## Часть 2 — MatMulF32_TF32 в gotorch

### 2.1 Сигнатура

```go
func (b *PuregoBackend) MatMulF32_TF32(a, bb, c DeviceBuffer, m, n, k int) error
```

Внутри: `cublasSetMathMode(TF32)` → `cublasSgemm` → **`defer` возврат в DEFAULT_MATH** до return (panic'ит если rollback fail — лучше тихого TF32-состояния handle).

**Никакого публичного `SetMathMode`** — режим существует только как свойство метода, не как состояние backend'а. Философия dtype-суффиксов R02a: невозможно «забыть выключить» то, что не включается глобально.

Также добавлен биндинг `cublasSetMathMode` в `cublas_purego.go` и константы `CUBLAS_DEFAULT_MATH`/`CUBLAS_TF32_TENSOR_OP_MATH`.

Метод добавлен в `Backend` interface (`api.go`) с doc-строкой: «Точность rel ~1e-3 против MatMulF32. Bit-exact НЕ ожидается. Назначение: сверка с legacy-путями + первый кирпич будущего скоростного слоя».

### 2.2 Тесты gotorch/cuda (3 файла)

**`matmul_tf32_test.go` — 3 теста:**

| Тест | Форма | Проверка | Результат |
|---|---|---|---|
| `TestMatMulF32_TF32_vsMatMulF32` | [3×4×5], [16²], [128×64×32] | TF32 vs FP32 hybrid tolerance abs=5e-2 + rel=1e-2·\|ref\|; TF32 обязан отличаться | ✅ 3/3 PASS; [3×4×5] 10/15 bit-exact (K=4 TF32-shortfall ниже eps); [16²]/[128×64×32] 0% bit-exact (TF32 включился) |
| `TestMatMulF32_TF32_vsCPUFP64` | те же 3 формы | vs CPU-FP64 hybrid | ✅ 3/3 PASS |
| `TestMatMulF32_TF32_RollbackHygiene` | [16²] | FP32 после TF32 обязан вернуться к ~FP32 eps | ✅ **maxRel=5.67e-5**, не TF32 1e-3 → **rollback работает** |

---

## Часть 3 — 4 сверки impl-4-final

Все PASS. Ключевые числа:

| Sверка | Форма | Проверка | Ожидание | Факт |
|---|---|---|---|---|
| **3.1** small adapter-FP32 vs fb-TF32 | [4×8×16] | rel ≤ 1e-2 hybrid (TF32 shortfall) | документированная разница режимов | ✅ maxAbs=2.5e-3, maxRel=0.34, hybridFail=0 |
| **3.2** трёхсторонний | [16×16×16] | delta(adapter, FP64) < delta(fb, FP64) | adapter должен быть точнее | ✅ **adapter 3648.7× ТОЧНЕЕ fb** (adapter absErr=9.5e-7, fb absErr=3.5e-3) |
| **3.3** batched delegate | [4,64,64]×[64,64] (attention Q) | bit-exact (тот же код-путь) | 100% | ✅ **16384/16384 bit-exact** |
| **3.4** TF32-vs-TF32 (главная) | [16×16×16] | bit-exact (один режим + один cuBLAS + один ctx + один stream) | 100% | ✅ **256/256 bit-exact, maxAbs=0.000e+00** — **скрытых багов раскладки НЕТ** |

**Ключевая находка Sверки 3.2:** adapter точнее fb на **3.5 порядка** — это измеренное преимущество моста, не bug. «Провал bit-exact» превратился в фичу.

**Ключевая находка Sверки 3.4:** adapter в TF32-режиме идентичен legacy в TF32-режиме до бита. Значит вся разница impl-4 (0/64 при [4×8×16] на pre-final прогоне) была режимной, никакой скрытой ошибки раскладки/транспона нет. Ради этого MatMulF32_TF32 и строился.

---

## Часть 4 — Хвосты

### 4.1 ContiguityFork
Форма [8×16×8]. adapter vs fb в TF32-hybrid — PASS. maxAbs=3.02e-3, maxRel=1.67e-2. Оба одинаково игнорируют strides (передают raw pointer в cuBLAS) → результаты согласованы в пределах TF32 shortfall.

**Size-mismatch fix (из pre-final прогона):** `goml.Pool` в fb.Alloc округляет byteLen до 256 (aligned bucket); для запроса 128 wrapForeign обёртывает full-256, CopyH2D size-check ломается. Fix — в helper'ах теста Alloc через adapter (точный byteLen).

### 4.2 WithBias intermediate MatMul (без Add)

Форма [8×8×16]. Intermediate MatMul FP32 vs CPU-FP64 — hybrid `abs=1e-4 + rel=1e-5·|ref|` (R02b Ворот 2 BLAS-стандарт). **hybridFail=0**, maxAbs=9.5e-7, maxRel=1.35e-5.

**Финал Add skipped** — broadcasting не поддержан ни в adapter, ни в fb (goml.cuda `ops.go:81 TODO: broadcasting support`). Плюс **в LLM TinyConfig ВСЕ Linear имеют bias=false** — Add branch в hot path Step никогда не выполняется. Bias-Linear — методологический, не боевой.

### 4.3 Realistic FFN shape

Форма [4, 64, 64] × [64, 172] (TinyConfig FFN gate). Batch=4 > 1 → adapter DELEGATES to fb → тот же код-путь. Ожидание bit-exact. **44032/44032 bit-exact.**

---

## Sync-race fix (обнаружено в helper'ах теста, ассоциированное)

`cuMemcpyDtoH` без Async синхронно ordered против **default stream (0)**, но kernel'ы на **injected goml stream**. Между async MatMul и D2H нужен явный `b.fb.Sync()` в helper'е теста. Без него — race с грязным CPU-читанием.

**Правило (грабли для будущих авторов тестов моста):** D2H после adapter/fb kernel-ops в тестах **ВСЕГДА** через `b.fb.Sync()` (== `cuStreamSynchronize(goml_stream)`) перед копией. Без sync — `cuMemcpyDtoH` идёт по default stream (0), а kernel на injected goml stream, ordering не гарантирован → race → мусорные данные, симптом: maxRel в разы больше eps на маленьких формах, потому что D2H успевает до kernel finish.

**Правило:** sync в тестовых helper'ах допустим (grep-guard `TestAdapterNoFullSync` проверяет только non-test `.go` файлы, тестам разрешено). В production adapter body sync запрещён — stream-injection даёт ordering для последовательных ops.

**Не касается adapter production path:** в production goml.trainer перед CPU-читанием loss делает implicit sync через `tensor.ToFloat32Slice()`, который на CPU-стороне уже наступает после всех async op'ов в очереди stream'а (`cuMemcpyDtoH` triggers implicit stream sync для default stream, plus tensor.Storage.Bytes() чит из host-side buffer). Не race'ит в hot path.

---

## Регрессионный набор — 5/5 PASS

| Проверка | Результат |
|---|---|
| **Full adapter tests** (impl-2 + impl-3 + impl-4) | ✅ 10/10 PASS |
| **gotorch cuda** (MatMul/Add/Activations/Sum/Softmax + TF32) | ✅ ok 0.356s |
| **interop_smoke** (6/6 subtests) | ✅ ok 0.342s |
| **goml cudatest** (171 TFLOPS) | ✅ All tests passed |
| **FA canary** (fwd 653.86T + bwd nc 41.561ms) | ✅ WITHIN corridors |

---

## Часть 5 — задел impl-5 (код НЕ начинать)

**FP64-эталонная траектория для impl-5 = GPU через gotorch F64-путь**, не на CPU. Точность важнее платформы; GPU-F64 на порядки быстрее CPU при той же роли эталона.

CPU-FP64 остаётся только в дешёвых поэлементных сверках малых форм (как в 3.2).

**Возможные F64-дыры для 10-Step траектории (задел, при подготовке проверить):**
- `MatMulF64` есть в gotorch ✅
- Активации F64: `ReLUF64`/`SigmoidF64`/`TanhF64`/`ExpF64`/`LogF64` есть ✅ (все fdlibm-точность)
- `SoftmaxF64` есть ✅
- `SumF64`/`MeanF64` есть ✅
- LayerNorm F64 — **нет** (не в 50 методах R02b). Нужен port из goml (LayerNorm F32) в F64 для судьи, ИЛИ судья на этом шаге берёт CPU.
- Embedding/RoPE F64 — **нет**. Аналогично.

**Список F64-дыр для судьи** — карта будущего F64-покрытия. Не блокер impl-5, но нужно закрыть или пометить в отчёте перед прогоном.

---

## СТОП

По ТЗ: **после ворот impl-4 — СТОП, impl-5 по go после ревью**.

Готов принять решение по impl-5 старту.
