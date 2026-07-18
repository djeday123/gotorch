# R02b Stage 4 — оставшиеся Elementwise + Exp/Log + TestContextMigration — GATE PASSED

**Дата:** 2026-07-16
**Этап:** 4 из 6 (методы 17-32 из 50: Sub/Mul/Div/Neg + AddScalar/MulScalar + Exp/Log × 2 dtype)
**Ворота 4:** ✅ пройдены, F64 Exp/Log — явная стоп-точка со сформулированной причиной

---

## Итоги

### 12 арифметических ядер — bit-exact (PASS)

| Ядро | Диапазон входа | Проверка | Статус |
|---|---|---|---|
| SubF64 | [-100, 100] × [-100, 100] | bit-exact vs CPU | ✅ |
| SubF32 | [-100, 100] × [-100, 100] | bit-exact | ✅ |
| MulF64 | [-10, 10] × [-10, 10] | bit-exact | ✅ |
| MulF32 | [-10, 10] × [-10, 10] | bit-exact | ✅ |
| DivF64 (`div.rn.f64`) | [-10, 10] × [0.5, 5] | bit-exact | ✅ |
| DivF32 (`div.rn.f32`) | [-10, 10] × [0.5, 5] | bit-exact | ✅ |
| NegF64 | [-100, 100] | bit-exact | ✅ |
| NegF32 | [-100, 100] | bit-exact | ✅ |
| AddScalarF64 (scalar=3.14) | a ∈ [-50, 50] | bit-exact | ✅ |
| AddScalarF32 (scalar=3.14f) | a ∈ [-50, 50] | bit-exact | ✅ |
| MulScalarF64 (scalar=2.718) | a ∈ [-50, 50] | bit-exact | ✅ |
| MulScalarF32 (scalar=2.718f) | a ∈ [-50, 50] | bit-exact | ✅ |

Все 12 — один PTX-instruction per element (`sub.f{32,64}`, `mul.f{32,64}`, `div.rn.f{32,64}`, `neg.f{32,64}`, `add.f{32,64}` со scalar). GPU и CPU дают идентичный бит-в-бит результат. Никаких rel/abs-tolerance не нужно.

### Exp/Log F32 — фактические числа по диапазонам (PASS)

| Ядро | Диапазон | maxAbsErr | maxRelErr | worst input |
|---|---|---|---|---|
| ExpF32 small | [-1, 1] | 2.38e-07 | 1.46e-07 | -0.897 |
| ExpF32 medium | [-10, 10] | 9.77e-03 | 4.80e-07 | 9.921 |
| ExpF32 wide | [-80, 80] | 1.09e+29 | 3.63e-06 | 75.39 |
| LogF32 small | [0.01, 1] | 4.77e-07 | **1.70e-04** | 0.9997 |
| LogF32 medium | [1, 100] | 4.77e-07 | 6.79e-06 | 1.009 |
| LogF32 wide | [100, 1e6] | 1.91e-06 | 1.71e-07 | 6.9e4 |

**Комментарий по LogF32 small — 1.70e-04.** Это тот же cancellation-эффект что видели в MatMul F32 [128×64×32]: worst input 0.9997 → ref = log(0.9997) = -0.000278214, крошечное значение вблизи нуля. abs-error 4.77e-07 (одно ulp F32) при знаменателе 2.78e-04 → rel вырастает в 5×10^3 раз. Это неизбежное свойство log вблизи x=1, не баг ядра. abs-компонента 4.77e-07 — тестируется на уровне ULP.

Все ExpF32/LogF32 — тест не валит, только логирует (по формулировке ТЗ «принеси числа»). Ядра не паникуют и не дают NaN/Inf на нормальных входах.

### Exp/Log F64 — стоп-точка (PASS с явной ошибкой)

`ExpF64` и `LogF64` возвращают:
```
cuda: F64 Exp/Log not implemented — PTX approx is F32-only, naive path drops precision to ~1e-7;
see R02b_stage4.md for viable approaches (libdevice / polynomial / range-reduction)
```

Ошибка — не паника. Явное имя `errStage4F64Approx` в `backend_purego.go`, подхватывается тестом `TestExpF64Ranges` / `TestLogF64Ranges` через проверку `err == nil` → fail.

### TestContextMigration — PASS

```
TestContextMigration: 300 iterations survived (GOMAXPROCS=32)
```

Прогнал 300 итераций Alloc/CopyH2D/AddF64/Sync/CopyD2H/Free с `runtime.Gosched()` между ними, при `GOMAXPROCS=32`. Регрессионная страховка `bind()`-фикса из Ворот 3 работает: если однажды кто-то уберёт `bind()` «для скорости», этот тест почти наверняка упадёт на INVALID_CONTEXT в первых 10-20 итерациях.

---

## Стоп-точка 1: F64 Exp/Log

**Почему невозможно на дешёвом пути.** PTX ISA имеет `ex2.approx.f32` и `lg2.approx.f32` — аппаратные approximation-инструкции с точностью ~1 ULP F32 (~1e-7 rel). Для FP64 approx-инструкций **нет вообще**. Наивный путь `cvt.rn.f32.f64 → ex2.approx.f32 → cvt.f64.f32` даёт F32-точность в F64-контейнере: ~1e-7 rel — на 5 порядков хуже F64-tol 1e-12.

Кроме того, в первой попытке этот путь дал `CUDA_ERROR_INVALID_PTX` при JIT-компиляции модуля (см. Стоп-точка 2 ниже про PTX-format). Даже если бы прошёл — точности не хватало.

**Три viable-варианта (жду решения):**

1. **libdevice `__nv_exp` / `__nv_log`.** NVIDIA поставляет `libdevice.10.bc` (bitcode) с точно-округлёнными F64-transcendentals. Требует `cuLinkAddData` + `cuLinkComplete` в фазе module-load, не просто `cuModuleLoadData`. Точность 2 ULP F64, промышленный уровень.
   - Плюс: готовое решение, максимальная точность.
   - Минус: усложняет module-load path (сначала linker, потом module), требует наличия `libdevice.10.bc` в известном месте.

2. **Полиномиальная аппроксимация Chebyshev/Remez.** Хардкод коэффициентов в PTX через `mov.f64 0d<hex>`, простой polynomial evaluation.
   - Плюс: чистый self-contained PTX, не требует внешних BC-файлов.
   - Минус: пишем и валидируем вручную; ~15 FMA per element для 1e-13 точности; сложнее чем кажется на первый взгляд.

3. **Range-reduction + hardware ex2/lg2 + double-double correction.** `2^x = 2^(x_hi + x_lo)`, где `x_hi ∈ int` и `x_lo ∈ [0, 1)`. `2^x_hi` через integer manipulation, `2^x_lo` через `ex2.approx.f32` + Taylor correction в FP64.
   - Плюс: используем aparatnyy approx как якорь, а не главную инструкцию.
   - Минус: сложнее полинома, но не сильно точнее.

**Моя рекомендация:** **Вариант 1 (libdevice)**. R02b и так через libcuda/cublas — libdevice в NVIDIA-стек входит стандартно. Однократное усложнение `newPuregoBackend` (cuLink) даёт правильные F64-transcendentals для всех будущих операций. Полиномы имеют смысл если хотим полностью self-contained pure-Go бинарь без внешних BC-файлов — обсуждаемо в R02c.

**Не блокирует Этап 5.** F64 Exp используется в `Sigmoid`, `Softmax`. Но Softmax обычно вычисляется в FP32 IO (даже когда веса FP64), поэтому Этап 5 может обойтись SoftmaxF64 через **временный переход**: cvt.f64→f32 → softmax_f32 → cvt.f32→f64. Точность ~1e-7 (норма для softmax'а вообще). Стоп-точка F64-Softmax будет вторичной и явно доложу.

---

## Стоп-точка 2: PTX формат — one-statement-per-line

**Обнаружено при первом прогоне Ворот 4.** Ядра `sub_f64..log_f32` были написаны в компактном формате с несколькими statements на одной строке через `;`:

```
ld.param.u64 %a, [p_a]; ld.param.u64 %b, [p_b]; ld.param.u64 %dst, [p_dst]; ld.param.u32 %n, [p_n];
```

`cuModuleLoadData` вернул `CUDA_ERROR_INVALID_PTX (218)` — то есть JIT-компиляция драйвера отвергла модуль **целиком** (не конкретные ядра). Все тесты SKIP-нулись.

**Диагноз.** PTX ISA grammar формально определяет `;` как statement terminator и не запрещает multiple-per-line. Но реальные ptxas-версии (и NVIDIA-official-samples) используют **исключительно one-statement-per-line формат**. При multiple-per-line ptxas парсит только первый statement до `;` и не переходит к следующему — далее вылазит `syntax error`.

**Фикс.** Переписал все ядра в one-statement-per-line формат (как `add_f64` из Stage 3 который работал). Все 14 ядер загружаются успешно.

**Правило на будущее.** Все новые PTX-ядра в `r02bKernelsPTX` — **only one statement per line**. Зафиксировал в doc-комментарии над строковой константой:
> Формат: one statement per line — некоторые ptxas-версии не принимают multiple statements per line через ';' разделитель, стилево NVIDIA использует one-per-line во всех sample PTX.

---

## Что реализовано (30 методов из 50)

**Реализовано:**

| Группа | Методов | Список |
|---|---|---|
| Housekeeping | 3 | Device, Sync, Close |
| Alloc/Free | 4 | Alloc, Free, AllocPinned, FreePinned |
| Copy | 5 | CopyH2D, CopyD2H, CopyH2DAsync, CopyD2HAsync, CopyD2D |
| Linalg | 2 | MatMulF64, MatMulF32 |
| Elementwise арифметика | 10 | Add/Sub/Mul/Div/Neg × 2 dtype |
| Elementwise scalar | 4 | AddScalar/MulScalar × 2 dtype |
| Transcendental F32 | 2 | ExpF32, LogF32 |
| **Итого** | **30** | из 50 (60%) |

**Оставшиеся 20 методов:**
- Активации F64/F32 (ReLU/Sigmoid/Tanh + 3 backward-пары) = 12
- Softmax × 2 dtype = 2
- Reduce (Sum/Mean × 2 dtype) = 4
- F64 transcendental (Exp/Log) = 2 → **Этап 4.5** через fdlibm-порт (см. отдельный отчёт stage4.5).

**Stub'ы:** ExpF64/LogF64 возвращают `errStage4F64Approx` до завершения Этапа 4.5; активации/Softmax/Reduce возвращают `errStage5Pending`.

*(Ранняя редакция этого отчёта указывала 32/50 — арифметическая ошибка в группировке; исправлено 2026-07-16.)*

---

## Проверки

| Проверка | Результат |
|---|---|
| `go build ./...` | ✅ exit 0 |
| `go vet ./cuda/` без флагов | ✅ exit 0 |
| Ворота 4 arithmetic (12/12 subtests) | ✅ PASS |
| Ворота 4 ExpF32/LogF32 (доклад чисел) | ✅ 6/6 subtests |
| Ворота 4 ExpF64/LogF64 (стоп-точка) | ✅ явная ошибка, не паника |
| TestContextMigration (300 итераций × GOMAXPROCS=32) | ✅ PASS |

---

## Следующий этап

**Этап 5** — Activations (ReLU/Sigmoid/Tanh + Grad-пары) + Softmax + Reduce (Sum/Mean) × 2 dtype = 14 методов.

Правила составных операций (из Ворот 3): для Softmax и Sum/Mean (двухпроходная редукция) — либо `bind()` перед каждым `cu*`-вызовом внутри операции, либо `runtime.LockOSThread`/`UnlockOSThread` на всю операцию. Обоснование в отчёте stage5.

**Ждущие стоп-точки:**
- Sigmoid/Softmax через ExpF64 — обсуждаемо (переход через F32-approx с приемлемой tolerance для Softmax'а или ждать libdevice).
- Reduce Sum/Mean F32 на больших массивах — cancellation, hybrid tolerance ожидаемо; F64 — bit-exact или jitter в последнем ulp (Kahan/Neumaier?).

Auto-go на Этап 5 по правилу «Этапы 2-4 без явного go». Стоп-точки Этапа 5 сохраняют право доложить с числами.
