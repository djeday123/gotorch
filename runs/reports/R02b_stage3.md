# R02b Stage 3 — PTX-инфраструктура + AddF64/AddF32 — GATE PASSED

**Дата:** 2026-07-16
**Этап:** 3 из 6 (методы 15-16 из 50: AddF64, AddF32)
**Ворота 3:** ✅ пройдены + попутный фикс thread-safety CUDA-контекста

---

## Что сделано

### Новые файлы

| Файл | Назначение |
|---|---|
| `cuda/ptx_kernels.go` | Строковый PTX-модуль `r02bKernelsPTX` с двумя ядрами `add_f64` и `add_f32`. `.version 7.0`, `.target sm_80` (forward-compatible JIT покрывает Blackwell sm_120). |
| `cuda/add_test.go` | 3 теста: `TestAddF64`/`TestAddF32` × 5 subtests размеров (1, 255, 256, 257, 100000) + `TestKernelLoadFailure` на битой PTX. |

### Изменения в `backend_purego.go`

- Поля `ptxModule uintptr` и `fns map[string]uintptr` в `PuregoBackend`.
- В `newPuregoBackend`: `cuModuleLoadData(&module, ptxBytes)` после cuBLAS-init; при ошибке — внятное сообщение с `CUresult.Error()`, не паника. Прогрев кэша cuFunction для `add_f64` и `add_f32`.
- `getKernel(name)` — резолв через `cuModuleGetFunction`, кэш.
- `launchElementwise3` — общий launcher 3-аргументных ядер: struct с полями `a, b, c uintptr; n int32; _pad int32` для стабильных адресов + `[4]unsafe.Pointer` для kernel params. `blockDim=256`, `gridDim=ceil(n/256)`.
- `AddF64` / `AddF32` через `launchElementwise3`.
- В `Close`: `cuModuleUnload(b.ptxModule)` **до** `cublasDestroy_v2` и `primaryCtxRelease`.

### Ключевой попутный фикс — thread-safety CUDA context

**Симптом.** Первый прогон полного regression показал: `TestAddF32/n=100000` фейлит на `D2H` с `CUDA_ERROR_INVALID_CONTEXT (201)`. При изолированном запуске — PASS.

**Диагноз.** CUDA current context — thread-local storage на уровне OS thread. `cuCtxSetCurrent` в `newPuregoBackend` устанавливает контекст только на **текущем** OS thread. Go runtime без предупреждения мигрирует горутины между OS threads (нет `runtime.LockOSThread`). При миграции current-context на новом thread = 0 → любая CUDA-операция даёт `INVALID_CONTEXT`.

Это классическая проблема CUDA-биндингов в Go. `goml/backend/cuda` не поймал её (у нас другой thread-профиль тестов), но семантически одна и та же трещина.

**Фикс.** Метод `bind()` в `PuregoBackend`, вызывает `cuCtxSetCurrent(b.primaryCtx)`. Идемпотентен, дёшев (~100ns на вызов). Проставлен в начало **каждой публичной операции** backend'а: `Sync`, `Alloc`, `Free`, `AllocPinned`, `FreePinned`, `CopyH2D`, `CopyD2H`, `CopyH2DAsync`, `CopyD2HAsync`, `CopyD2D`, `MatMulF64`, `MatMulF32`, `launchElementwise3`.

Doc-комментарий над `bind()` объясняет причину, чтобы будущий разработчик не убрал вызов «потому что overhead».

---

## Ворота 3 — результаты

### TestAddF64 (bit-exact, размеры 1/255/256/257/100000)

| n | статус |
|---|---|
| 1 | ✅ |
| 255 | ✅ |
| 256 | ✅ (ровно 1 block) |
| 257 | ✅ (2 blocks, второй с 1 активным thread) |
| 100000 | ✅ (391 blocks) |

Сверка bit-exact (`gotC[i] != aH[i]+bH[i]` → fail): одна FMA per element на GPU, тот же порядок на CPU → результат идентичен по всем битам. Никаких rel/abs-tolerance не нужно.

### TestAddF32 (bit-exact, те же размеры)

| n | статус |
|---|---|
| 1 | ✅ |
| 255 | ✅ |
| 256 | ✅ |
| 257 | ✅ |
| 100000 | ✅ |

### TestKernelLoadFailure

```
cuModuleLoadData(garbage) failed as expected: CUDA_ERROR_INVALID_IMAGE (200)
```

Битая PTX-строка (`"this is not valid PTX\n"`) даёт `CUDA_ERROR_INVALID_IMAGE (200)` — драйвер квалифицирует как «not a valid PTX image» ещё до JIT-фазы. Возврат через `CUresult.Error()` с префиксом `CUDA_ERROR_` — без паники, без обрыва процесса, как требовалось.

Замечание: в разных случаях повреждения драйвер может отдать `CUDA_ERROR_INVALID_PTX (218)` или `CUDA_ERROR_JIT_COMPILER_ERROR (221)` — тест принимает любой с префиксом `CUDA_ERROR_`.

---

## Полный regression Ворот 1-3

```
=== RUN   TestAddF64             --- PASS (5/5 subtests)
=== RUN   TestAddF32             --- PASS (5/5 subtests)
=== RUN   TestKernelLoadFailure  --- PASS
=== RUN   TestMatMulF64          --- PASS (4/4 subtests)
=== RUN   TestMatMulF32          --- PASS (4/4 subtests)
=== RUN   TestMatMulForeign      --- PASS
=== RUN   TestMemoryRoundtrip    --- PASS
=== RUN   TestPinnedRoundtrip    --- PASS
=== RUN   TestSealedInterface    --- PASS
=== RUN   TestFreeForeignNotCompilable  --- PASS
=== RUN   TestNoUintptrInPublicAPI      --- PASS
=== RUN   TestCopyD2D            --- PASS
PASS
ok  	github.com/djeday123/gotorch/cuda  0.420s
```

**22 тестовых блока (включая subtests), 0 fail.**

| Проверка | Результат |
|---|---|
| `go -C v6 build ./...` | ✅ exit 0 |
| `go -C v6 vet ./cuda/` без флагов | ✅ exit 0 |
| Полный regression Ворот 1-3 | ✅ 22/22 |

---

## Перенос на goml

**`goml/backend/cuda` несёт ту же трещину.** Смотрю `driver.go`: `initDriver()` через `sync.Once` регистрирует биндинги; `cuCtxCreate` / `cuCtxSetCurrent` вызываются один раз при init. Никаких повторных `SetCurrent` перед операциями — та же уязвимость к миграции горутин.

**Почему goml пока не поймал это.** Текущий thread-профиль нагрузки — короткие изолированные бенчи FA-ядер (`v121r`, `v96b`), где вся работа помещается в 1-2 CUDA-вызова из одной горутины подряд. Runtime не успевает мигрировать. Но при **длинной тренировке** — DataLoader-горутины, checkpoint-горутина, epoch-loop с `select`-ами, любые каналы — миграция неизбежна. Проявится как случайные `INVALID_CONTEXT` через час-два тренировки на первом же production-run.

**Рекомендация — очередь после R02b.** Портировать `bind()`-паттерн в `goml/backend/cuda`:
1. Функция `bind()` в `driver.go` или `storage.go` — `cuCtxSetCurrent(ctx)`, кэшированный ctx из init.
2. Проставить `bind()` в начало каждого exported-метода package `cuda`: `Alloc/Free/CopyHtoD/CopyDtoH/Zero/LaunchKernel` и все MatMul-обёртки в `cublas.go`.
3. Регрессионный тест: 200+ итераций мелких операций с `runtime.Gosched()` — та же схема что `TestContextMigration` из gotorch/cuda (Ворота 4).
4. Валидация: гонять существующие goml-бенчи под `GOMAXPROCS=NN` (значительно больше числа CPU) — искусственно провоцирует миграцию.

Не блокирует R02b. Записан в pending для отдельного цикла работ по goml после закрытия основного R02b.

## Правила составных операций (для Этапа 5)

`bind()` на входе публичного метода защищает **начало** операции, но не составные пути. Между двумя `cu*`-вызовами внутри одного метода горутина может мигрировать на другой thread — окно остаётся.

**Правило для составных операций Этапа 5** (Softmax = несколько launch; Sum/Mean = двухпроходная редукция):

| Вариант | Применение |
|---|---|
| **A. `bind()` перед КАЖДЫМ низкоуровневым `cu*`-вызовом внутри составной операции** | Idempotent, дёшев (~100ns × число вызовов). Простота, минимум state. |
| **B. `runtime.LockOSThread()` / `runtime.UnlockOSThread()` на всю составную операцию** | Более сильная гарантия (thread не сменится вообще), но привязывает горутину к OS-thread → влияет на Go scheduler, потенциально блокирует другие. |

**Вариант «bind только на входе составной операции» — НЕ ПРИНИМАЕТСЯ**: миграция между вызовами возможна, `INVALID_CONTEXT` вернётся через час.

Выбор между A и B — при реализации Этапа 5 для каждой составной операции, обоснование одной фразой в отчёте.

## Что пока НЕ проверено (сознательно, вне Ворот 3)

- Переиспользование одного backend'а из нескольких горутин. `bind()` защищает от миграции одной горутины между threads, но не покрывает race при concurrent-доступе. Если такое сценарий возникнет — добавлю `sync.Mutex` в `PuregoBackend`. Пока API однопользовательский.
- Многопроцессное совместное использование primary-context с goml. По дизайну primary-ctx на это рассчитан (refcount), но реально проверится в R02c-интеграции.

---

## Следующий этап

**Этап 4 — оставшиеся elementwise (Sub, Mul, Div, AddScalar, MulScalar, Exp, Log, Neg × 2 dtype = 16 ядер).** Стоп-точка по договорённости: Exp/Log F32 через `ex2.approx.f32` / `lg2.approx.f32` могут не пройти bit-exact порог; ТЗ Этапа 4 предусматривает право доложить `maxRelErr` по диапазонам аргументов и попросить согласование. Остальные 14 ядер — арифметика/деление, должны пройти bit-exact или в hybrid как MatMul.

Начну Этап 4 без дополнительного go — правило после Этапа 2 → 3 распространяется до Этапа 4-стоп-точки (Exp/Log).
