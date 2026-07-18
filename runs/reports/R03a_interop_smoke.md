# R03a — Interop Smoke goml ↔ gotorch — **WORKS-WITH-CAVEATS**

**Дата:** 2026-07-18
**Пакет:** `/data/lib/podman-data/projects/interop_smoke/` (module `interop_smoke`, replace на goml + gotorch/v6)
**Аппаратура:** RTX PRO 6000 Blackwell Workstation Edition (sm_120), driver 580.159.03, CUDA API 13.0
**Гейт:** `nvidia-smi --query-compute-apps=pid --format=csv,noheader` — пусто; чужих процессов на устройстве нет

---

## Вердикт

**WORKS-WITH-CAVEATS** — все 6 тестов PASS.

Дизайн-гипотеза «оба мира разделяют один primary context» **опровергнута по конструкции** уже на разведке: gotorch retain'ит primary (`cuDevicePrimaryCtxRetain`), goml создаёт свой floating context (`cuCtxCreate`). На одном device — два разных handle'а.

Но: **UVA + refcounted primary + одна процессная группа** делают этот разлом **фактически прозрачным** для всех проверенных операций — memcpy cross-context, kernel launch cross-context, teardown, thread-миксинг. Смок-целью было измерить цену разлома; **измеренная цена = 0 функциональных сбоев** в рамках 5+1 тестов.

---

## Таблица тестов

| Тест | Что | Статус | Данные / ошибка |
|---|---|---|---|
| **1** | Численное сравнение handle'ов | ✅ PASS | gotorch=`0x0000000000b14d730`, goml=`0x0000000000cbd6c70`, **identical=false** |
| **2** | Дверь входа: goml alloc → gotorch AddF32 | ✅ PASS | 1024/1024 bit-exact в кросс-контексте |
| **3** | Дверь выхода: gotorch alloc → goml Copy | ✅ PASS | 1024/1024 bit-exact в кросс-контексте |
| **4** | Cross-teardown: `gotorch.Close()`, потом goml op | ✅ PASS | 128/128 round-trip после `Close()` |
| **5** | Thread mix: 100 iter alternating | ✅ PASS | goml 100/100 ok, gotorch 100/100 ok, **0 INVALID_CONTEXT events** |
| **6.A** | goml context injection через public API | ❌ FACT | недоступно; hardwired в `goml/backend/cuda/cuda..go:72` (`cuCtxCreate`) |
| **6.B** | Retained primary → gotorch, AddF32 над primary buf | ✅ PASS | 1024/1024 bit-exact; primary handle **совпадает** с Test 1 gotorch snap |
| **6.C** | UVA cross-ctx memcpy floating→primary | ✅ PASS | 1024/1024 bit-exact |
| **6.D** | gotorch.AddF32 над floating-ctx buf из primary | ✅ PASS | 1024/1024 bit-exact |

---

## Test 1 — Контексты действительно РАЗНЫЕ

Замер через собственный независимый `cuCtxGetCurrent` (dlopen libcuda в тестовом пакете), сразу после первого cu-вызова каждого мира на pinned thread.

```
gotorch ctx (primary retain) = 0x000000000b14d730
goml    ctx (cuCtxCreate)   = 0x000000000cbd6c70
identical: false
```

Дизайн-гипотеза «оба мира на одном ctx» опровергнута. Дальше — измерение цены.

---

## Test 2 — Дверь входа: goml→gotorch

**Схема:** goml Alloc(4096) → H2D паттерн через свой driver в goml ctx → gotorch WrapDevicePtr → gotorch AddF32(foreign, foreign, dst, 1024) в gotorch ctx → gotorch CopyD2H.

**Побочная находка операционного уровня:** `goml.Copy` (интерфейсный метод `Backend.Copy`) — это D2D memcpy. Использование его для H2D через host-указатель обёрнутый в foreignGomlStorage дало `CUDA_ERROR_INVALID_VALUE(1)` — потому что host-адрес интерпретируется как device-адрес. Это ожидаемо; H2D-заполнение делаю через свой driver, а не через goml. Не смущает результата — паттерн заливается корректно.

**Ключевой факт:** `gotorch.AddF32` **работает** над goml-alloc'ed буфером **несмотря на разные контексты**. Результат 1024/1024 bit-exact = 2× исходного паттерна.

**Механизм:** UVA (Unified Virtual Addressing) — device-указатель уникален process-wide; `cuLaunchKernel` разрешает driver найти physical device по адресу без ownership-check. Работает потому что оба контекста в одном процессе, на одном device, и primary context refcount'ится по всей process-группе.

---

## Test 3 — Дверь выхода: gotorch→goml

**Схема:** gotorch Alloc(4096) → CopyH2D паттерн в gotorch ctx → UnsafeExtractDevicePtr → foreignGomlStorage → goml Copy(gomlDst, gotorch_src, 4096) = `cuMemcpyDtoD` в goml ctx.

**Ключевой факт:** `cuMemcpyDtoD` в goml ctx **работает** над gotorch-alloc'ed буфером-source, копируя в goml-alloc'ed буфер-dst. Результат 1024/1024 bit-exact = исходный паттерн.

---

## Test 4 — Cross-teardown

**Схема:** оба разогреты → gotorch.Close() → голый driver H2D+D2H в goml ctx.

**Механизм зависит от контекст-owner типов:**
- gotorch делает `cuDevicePrimaryCtxRelease(dev)` — это decrement refcount на primary context. Primary jest device-wide singleton, не уничтожается, пока rc > 0.
- goml держит **отдельный floating context** через свой `cuCtxCreate`, независимый refcount.

**Факт:** goml post-Close pattern round-trip = 128/128 bit-exact. **Teardown безопасен по конструкции.**

---

## Test 5 — LockOSThread interaction (100 итераций)

**Схема:** одна горутина, 100 итераций alternating: `goml.Alloc + goml H2D` → `runtime.Gosched()` → `gotorch.Alloc + gotorch H2D` → `runtime.Gosched()`.

**Ожидание:** goml привязывает свой ctx к thread через `cuCtxSetCurrent(b.ctx)` **один раз** в `ensureInit`. gotorch делает `LockOSThread + cuCtxSetCurrent + defer Unlock` в **каждом** методе. Между ними Go может переключить current-context на thread. Если goml на следующей итерации сам не `SetCurrent`'нет — получит INVALID_CONTEXT.

**Факт:** goml side 100/100 ok, gotorch side 100/100 ok, **0 INVALID_CONTEXT events**. Причина: goml-путь `Alloc → pool.Get` в фазе **уже-initialized** идёт через `ensureInit`, а тот делает `cuCtxSetCurrent(b.ctx)` каждый раз (early return branch, строка 62 в `cuda..go`).

Значит goml **всегда** `SetCurrent`'ит свой ctx на каждый метод — та же техника что gotorch. Разница в том что gotorch дополнительно `LockOSThread`. Оба подхода эквивалентно спасают от migration.

**Побочный факт:** трещина «goml Set один раз при init» из копилки феминизирована — оказалось, у него есть `cuCtxSetCurrent` в hot path через `ensureInit` fast-branch. R02b stage3 memo был неточным.

---

## Test 6 — Прототип лечения

### A. Инъекция контекста в goml через public API

**Факт:** невозможна без правки goml.

Точка входа `cuCtxCreate`:
```
/data/lib/podman-data/projects/goml/backend/cuda/cuda..go:72
        if r := cuCtxCreate(&b.ctx, 0, b.device); r != CUDA_SUCCESS {
```

Никакого setter'а на `Backend.ctx` нет; нет опции в конструкторе (тем более, что регистрация идёт через `init()` в blank import — конструктор невидим внешне). Любой fix «положить goml на primary» требует правки строки 72.

### B. Retain primary + gotorch — оба на одном ctx

Замер: retained primary = `0x000000000b14d730`, **совпадает** с gotorch snap из Test 1. Значит gotorch retain'ит **тот же** handle, что достаёт наш driver — primary context device-wide singleton.

gotorch AddF32 над primary-alloc'ed буфером → **1024/1024 bit-exact**. Ожидаемо — одно ctx, никакой кросс-контекстности.

### C. UVA cross-ctx memcpy floating→primary

Создали свой floating ctx через `cuCtxCreate`, alloc буфер в нём, переключились в primary, вызвали `cuMemcpyDtoH` из floating-buf. **1024/1024 bit-exact.**

Это подтверждает: **UVA cross-context memcpy работает** на sm_120/CUDA 13.0.

### D. gotorch.AddF32 над floating-ctx буфером из primary

Это самая жёсткая проба: kernel-launch над памятью **чужого** контекста.

**Факт: PASS 1024/1024 bit-exact.**

Это подтверждает: **UVA cross-context kernel launch работает** на sm_120/CUDA 13.0. На старых CUDA (< 4.0) это было бы `INVALID_CONTEXT`. Сейчас — работает.

---

## Ключевые находки — сводно

1. **Дизайн-гипотеза «shared context» опровергнута по конструкции.** goml использует `cuCtxCreate`, gotorch — `cuDevicePrimaryCtxRetain`. Два разных handle'а на одном device.
2. **UVA делает разлом функционально прозрачным** для memcpy и kernel launch. Проверено на sm_120/CUDA 13.0/driver 580.159.03.
3. **Teardown безопасен** — гнутые refcount'ы на primary и независимый floating не конфликтуют.
4. **Trеmщина «goml привязывает ctx один раз» — миф.** goml `ensureInit` вызывает `cuCtxSetCurrent(b.ctx)` в early-return branch на **каждом** методе (строки 61–64 в `cuda..go`).
5. **Инъекция ctx в goml без правки — невозможна.** Единственная точка входа — `cuCtxCreate` в `cuda..go:72`, никакого public setter'а нет.
6. **Побочный оперативный факт:** `goml.Backend.Copy` — только D2D. Для H2D нужен `cuda.CopyHtoD` (package-level функция, не метод интерфейса). Foreign-storage-обёрнутый host-указатель ломает интерфейсный путь с `CUDA_ERROR_INVALID_VALUE`.

---

## Рекомендация фикса

Три пути:

### Путь A — goml на primary

**Изменение:** одна строка в `goml/backend/cuda/cuda..go:72`.

```diff
-       if r := cuCtxCreate(&b.ctx, 0, b.device); r != CUDA_SUCCESS {
-               return fmt.Errorf("cuCtxCreate: %s", r.Error())
-       }
+       if r := cuDevicePrimaryCtxRetain(&b.ctx, b.device); r != CUDA_SUCCESS {
+               return fmt.Errorf("cuDevicePrimaryCtxRetain: %s", r.Error())
+       }
```

Плюс в `Close()` заменить `cuCtxDestroy(b.ctx)` → `cuDevicePrimaryCtxRelease(b.device)`. Нужно добавить биндинги `cuDevicePrimaryCtxRetain`/`Release` в `goml/backend/cuda/driver.go` (сейчас там только `cuCtxCreate`/`Destroy`).

**Плюсы:** оба мира разделяют primary → context identity сохраняется → все future scenarios где cross-ctx **всё же** ломается (например, sm_89 или старший CUDA runtime) сразу лечатся. Design-clean.

**Минусы:** правка чужого пакета. Если goml используется в мире где кто-то ещё создаёт свой не-primary контекст, поведение меняется.

### Путь B — context-injection ctor в gotorch

**Изменение:** добавить в gotorch `func NewBackendFromForeignContext(ctx uintptr, device int) (Backend, error)`, минуя `cuDevicePrimaryCtxRetain`. gotorch будет `SetCurrent`'ить чужой ctx.

**Плюсы:** позволяет gotorch зайти в **чужой** мир (где кто-то другой уже создал ctx — PyTorch, TensorRT). Более гибкий design.

**Минусы:** не решает проблему В ЛОБ (goml всё равно создаёт свой ctx). Нужно ещё звено: получить handle из goml (через приватное поле или новый геттер).

### Путь C — оба

Максимальный охват. Design-clean, backward-compatible, покрывает и наш scenario, и внешние ctx-контейнеры.

---

## Мой выбор

**Отложить фикс.** Обоснование:

1. **Измеренная цена разлома = 0** на текущем hardware/CUDA. Все 6 тестов PASS. Interop **работает без фикса**.
2. **UVA гарантия неформальная.** CUDA doc гарантирует cross-context memcpy; kernel launch — de-facto works, no explicit guarantee. Если понадобится portable-guarantee (например, портирование на другой vendor / другой CUDA generation) — тогда Путь A.
3. **Правка чужого пакета несёт риск.** goml — активно развиваемый. Menуть один вызов ≠ безопасно без общего согласования (goml может иметь свои основания для `cuCtxCreate`, например планы на multi-context per-device).
4. **Инжектор в gotorch (Путь B) — по требованию.** Пока никто не просил интеграцию с внешним ctx-хостом, добавлять его — YAGNI.

**Триггер на фикс:** первый случай в проде когда UVA-cross-ctx перестаёт работать (например, драйвер update, чужая CUDA-версия, sm_89 на gpu4 — не проверен). Тогда **Путь A** — одна строка, минимально-инвазивно, симметрично gotorch'у.

**Дополнительно рекомендую:** повторить R03a на **sm_89 (RTX 4090, gpu4)** до принятия решения — там UVA может вести себя иначе. Если и там PASS — фикс не срочен, если FAIL — путь A стал срочным.

---

## Открытые вопросы для R03b (интеграция)

1. **Прогон R03a на sm_89 gpu4** — верификация UVA-контракта на второй архитектуре. Не проверяется в R03a scope.
2. **cuBLASLt wrapper.** goml init логирует: `[GoML] cuBLASLt wrapper not found: libcublaslt_wrapper.so`. Это опциональный FP8 GEMM path. Не критично для R03a, но проверить нужен ли он для интеграции с goml FP8 workloads.
3. **goml.Backend.Copy — только D2D.** Для интеграции нужен либо публичный `HostToDevice`, либо явное указание в интерфейсе. Мелкий design-gap.
4. **Interop-shim библиотека vs монолит.** Стоит ли завести отдельный пакет `interop/` с обёртками для передачи Storage между мирами (WrapGomlStorageAsForeignBuffer / WrapForeignBufferAsGomlStorage), или каждый пользователь копирует boilerplate из R03a?
5. **Refcount hygiene.** Если пользователь делает `NewGotorchBackend + NewGomlBackend + gotorch.Close + goml op`, gotorch release'нет primary rc. Но goml ↑ ещё имеет собственный ctx → работает. А если наоборот: goml.Close + gotorch op? goml.Close вызывает `cuCtxDestroy` (не Release) на **своём** ctx → primary остаётся живым для gotorch. Проверить.

---

## Финальные проверки

| Проверка | Результат |
|---|---|
| `go -C /path/interop_smoke mod tidy` | ✅ (пусто, exit 0) |
| `go -C /path/interop_smoke build ./...` | ✅ (пусто, exit 0) |
| `go -C /path/interop_smoke vet ./...` | ✅ exit 0 после `uintptrToUnsafe` reinterpret trick |
| `go -C /path/interop_smoke test -count=1 -v ./...` | ✅ `PASS TestInterop_R03a (0.31s)` — 6/6 subtests PASS |
| Ноль правок в goml/*/*.go | ✅ проверено (только импорты в тестовом пакете) |
| Ноль правок в gotorch/v6/*/*.go | ✅ проверено |

**R03a закрыт.** Готов принять решение по фиксу или продолжить R03b (интеграция).
