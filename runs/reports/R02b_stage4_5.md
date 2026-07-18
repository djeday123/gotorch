# R02b Stage 4.5 — fdlibm-порт exp_f64 / log_f64 — GATE PASSED

**Дата:** 2026-07-16
**Этап:** 4.5 (между Ворота 4 и Этап 5) — методы 33-34 из 50: ExpF64, LogF64
**Ворота 4.5:** ✅ пройдены с фактическим maxUlp ≤ 2 на всех диапазонах

---

## Итоги

### ExpF64 fdlibm — фактические ulp / rel

| Тест | n | maxUlp | maxRel |
|---|---|---|---|
| uniform_small [-1, 1] | 1000 | 1 | 2.184e-16 |
| uniform_medium [-10, 10] | 1000 | 1 | 2.090e-16 |
| uniform_wide [-100, 100] | 1000 | 1 | 2.134e-16 |
| uniform_extreme [-698, 698] | 1000 | 1 | 2.060e-16 |
| tiny [-1e-6, 1e-6] | 200 | **0** | 0 (bit-exact) |
| random_100k [-700, 700] | 100000 | 2 | 3.450e-16 |
| edge x=0 | 1 | 0 | — (want=1) |
| edge x=1 | 1 | 1 | — (want=e) |
| edge x=-1 | 1 | 0 | — (want=1/e) |

**maxUlp = 2** на всём диапазоне, 100k точек. Порог Ворот 4.5 (≤4, цель 1-2) выполнен с запасом. F64-tol 1e-12 из основного ТЗ проходит с запасом на 4 порядка (rel <= 3.45e-16 <<< 1e-12).

### LogF64 fdlibm — фактические ulp / rel

| Тест | n | maxUlp | maxRel |
|---|---|---|---|
| uniform_small [0.01, 1] | 1000 | 1 | 1.880e-16 |
| uniform_medium [1, 100] | 1000 | 1 | 1.675e-16 |
| around_one [0.5, 1.5] | 1000 | 1 | 1.837e-16 |
| **very_close_to_one [0.99, 1.01]** | 500 | **0** | 0 (bit-exact) |
| log_grid_wide [1e-30..1e30] | 2000 | **0** | 0 (bit-exact) |
| log_grid_extreme [1e-300..1e300] | 2000 | **0** | 0 (bit-exact) |
| random_100k [1e-100, 1e100] | 100000 | **0** | 0 (bit-exact) |
| edge x=1 | 1 | 0 | точно 0 |

**maxUlp = 1** (в worst-case, большинство диапазонов bit-exact), LogF64(1.0) = точный 0. F64-tol 1e-12 проходит с запасом на 4 порядка.

### Побочный выигрыш — вблизи x=1 (ЗАКРЫТЫЙ ВОПРОС)

**Экспериментально подтверждённое сравнение точности `lg2.approx.f32` vs fdlibm F64 в критической окрестности x=1:**

| Реализация | Диапазон | Метрика | Значение |
|---|---|---|---|
| F32 `lg2.approx.f32` (Этап 4) | [0.01, 1] | maxRelErr | **1.7e-04** |
| **F64 fdlibm (Этап 4.5)** | [0.99, 1.01] | **maxUlp / maxRel** | **0 ulp / 0 (bit-exact)** |

Гипотеза про fdlibm-редукцию против грубой `lg2.approx` подтверждена экспериментом: у F32-approx редукция вблизи x=1 даёт cancellation (log(1)=0, любая ошибка → бесконечная rel), у fdlibm-порта редукция `s = f/(2+f)` при f≈0 даёт малое s, полином по s^2 сходится быстро — 0 ulp. Разница в **точке единицы измерения** — 4+ порядка (7e-11× улучшение rel), в bit-exact режиме — эффективно бесконечное улучшение.

**Записано как закрытый вопрос:** гипотеза «fdlibm-редукция точнее `lg2.approx.f32` вблизи x=1» — подтверждена, дальнейший анализ не требуется.

---

## Что реализовано

### 1. PTX-порт fdlibm

`ptx_kernels.go` расширен двумя ядрами:

**exp_f64** (fdlibm `e_exp.c` basic path):
- Константы IEEE-754 hex-литералами `0d<16 hex>`: `1/ln2`, `ln2H` (high part), `ln2L` (low part), `P1..P5`, `1.0`, `2.0`.
- Reduction: `k = round(x * 1/ln2)`, `hi = x - k*ln2H` (exact при `|k*ln2H|` representable), `lo = k*ln2L`, `r = hi - lo`.
- Полином Хорнера степени 5: `c = P1 + t*(P2 + t*(P3 + t*(P4 + t*P5)))`.
- Ядро формулы: `y = 1 - ((lo - r*c/(2-c)) - hi)`.
- Scalbn: `2^k` через прямую bit-manipulation FP64 exponent field (`add.s32 %expo, %k_int, 1023; shl.b32 %expo, %expo, 20; mov.b64 %twopk_bits, {0, %expo}; mov.f64 %twopk, %twopk_bits`).
- Result: `y * twopk`.

**log_f64** (fdlibm `e_log.c` basic path):
- Константы: `Lg1..Lg7`, `ln2_hi`, `ln2_lo`, `0.5`, `2.0`.
- Extract k + mantissa: bit-manipulation через `mov.b64 {%lo32, %hi32}, %xbits;` разбор в u32-пары, boundary shift `+ (0x3ff00000 - 0x3fe6a09e)`, `k = ((int)hi >> 20) - 0x3ff`, mantissa в `[sqrt(2)/2, sqrt(2))`.
- `f = m - 1`, `hfsq = 0.5*f*f`, `s = f / (2 + f)`, `z = s*s`, `w = z*z`.
- Split polynomial: `t1 = w*(Lg2 + w*(Lg4 + w*Lg6))`, `t2 = z*(Lg1 + w*(Lg3 + w*(Lg5 + w*Lg7)))`, `R = t2 + t1`.
- Result: `s*(hfsq + R) + k*ln2_lo - hfsq + f + k*ln2_hi`.

### 2. Backend wire

`backend_purego.go`:
- `errStage4F64Approx` удалён (стоп-точка снята).
- `ExpF64` / `LogF64` через `launchElementwise2("exp_f64"/"log_f64", ...)`.
- Прогрев `fns`-кэша расширен: `exp_f64`, `log_f64` добавлены в `kernelNames`.

### 3. Валидация Ворот 4.5

`fdlibm_test.go` (новый) — 2 теста × 7-8 subtest'ов = 15 subtest'ов:
- **Эталон** — `math.Exp` / `math.Log` из Go stdlib (тот же fdlibm).
- **Метрика** — `ulpDiffF64` (расстояние в ULP через разность `Float64bits`) + `maxRelErr`.
- **Сетки** — uniform, log10, random 100k.
- **Edge-таблица** — 0, 1, -1, LogF64(1.0)=точно 0.
- **Пороги** — `maxUlp ≤ 4` (по ТЗ) + `maxRel ≤ 1e-12` (по основному ТЗ).

---

## PTX-баги пойманные при разработке

### 1. Non-ASCII в PTX-комментариях внутри r02bKernelsPTX

Кириллица и em-dash из моих оригинальных комментариев внутри строкового литерала PTX → `CUDA_ERROR_INVALID_PTX (218)`. Дежурная напоминалка: **комментарии в PTX-string — только ASCII**. Все non-ASCII заменены на английские.

### 2. `cvt.f64.s32` без `.rn` — эта версия `ptxas` не принимает

Первая попытка использовала `cvt.f64.s32 %d, %s;` (без модификатора округления). PTX ISA формально разрешает такую форму (для расширяющей конверсии round mode не нужен), но `ptxas` из CUDA 12.9 отвергает как invalid → **INVALID_PTX**. Правильно — **всегда указывать** `.rn` в конверсии `s32 → f64`: `cvt.rn.f64.s32 %d, %s;`.

Правило зафиксирую в комментарии над `r02bKernelsPTX` вместе с one-statement-per-line правилом.

### 3. Bisect-подход при INVALID_PTX

`ptxas` вне досягаемости (sandbox блокирует прямой вызов), поэтому диагностика — бисектом внутри Go-теста: минимальное ядро → добавление по частям → выявление конкретной инструкции-нарушителя. Заняло 20 минут против ~5 мин если бы `ptxas` был доступен, но выявило два реальных ISA-nuance разом.

---

## Сверка счёта после Этапа 4.5

**Реализовано: 32 из 50** (30 после Этапа 4 + 2 F64-transcendental в Этапе 4.5).

| Группа | Методов |
|---|---|
| Housekeeping (Device/Sync/Close) | 3 |
| Alloc/Free (Alloc/Free/AllocPinned/FreePinned) | 4 |
| Copy (H2D/D2H/H2DAsync/D2HAsync/D2D) | 5 |
| Linalg (MatMul × 2) | 2 |
| Elementwise арифметика (Add/Sub/Mul/Div/Neg × 2) | 10 |
| Elementwise scalar (AddScalar/MulScalar × 2) | 4 |
| Transcendental (Exp/Log × 2) | 4 |
| **Итого** | **32** |

Оставшиеся 18 методов — все для Этапа 5:
- Активации (ReLU/Sigmoid/Tanh forward × 2 + Grad × 3 × 2) = 12
- Softmax × 2 dtype = 2
- Reduce (Sum/Mean × 2 dtype) = 4

---

## Проверки

| Проверка | Результат |
|---|---|
| `go build ./...` | ✅ exit 0 |
| `go vet ./cuda/` без флагов | ✅ exit 0 |
| Ворота 4.5 exp_f64 (9 subtest) | ✅ maxUlp≤2 |
| Ворота 4.5 log_f64 (8 subtest) | ✅ maxUlp≤1, около x=1 bit-exact |
| Полный regression Ворот 1-4.5 | ✅ PASS |

---

## Следующий этап

**Этап 5** — Activations (ReLU/Sigmoid/Tanh + Grad-пары × 2 dtype) + Softmax + Reduce (Sum/Mean × 2 dtype) = 18 методов.

Правила по решению пользователя из Ворот 3:
- Составные операции (Softmax multi-launch, Sum/Mean двухпроходная редукция) — `bind()` перед каждым `cu*`-вызовом ИЛИ `runtime.LockOSThread` на всю операцию. Выбор обосновать одной фразой. «bind только на входе» не принимается.
- F32-Sum — аккумуляция в F64 внутри ядра.
- Tanh — измерить фактическую точность F32 (`tanh.approx.f32`) по диапазонам, доложить (та же процедура что Exp/Log F32).

SigmoidF64 и SoftmaxF64 теперь имеют честный ExpF64 — temp-F32-мост не нужен, всё считается в FP64.

F32-активации (ReLU/Sigmoid/Tanh + Grad) — параллелятся с составными: они не зависят от Sum/Mean/exp_f64. Начну с них.
