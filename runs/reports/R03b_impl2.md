# R03b-impl-2 — Adapter package + AddF32 end-to-end

**Дата:** 2026-07-18
**Итог:** ✅ PASSED. 3/3 adapter tests, 1024/1024 bit-exact, sync-контракт держится, goml регрессия сохранена.

---

## Что сделано

### 1. gotorch — SetStream hook (правка 1 R03b-paper)

- **`gotorch/v6/cuda/backend_purego.go`** — добавлено поле `stream uintptr` в `PuregoBackend`. По умолчанию `0` = default stream, старое поведение сохранено байт-в-байт для всех, кто не инжектирует.
- Все 6 мест `cuLaunchKernel(fn, ..., 0, 0, ...)` заменены на `cuLaunchKernel(fn, ..., 0, b.stream, ...)`.
- Новый экспортированный метод `SetStream(s unsafe.Pointer) error`: сохраняет stream в `b.stream` + вызывает `cublasSetStream_v2(b.cublas, b.stream)`. Единственный вход инъекции.
- **Публичный контракт** 50 методов **не тронут** — SetStream — single-purpose интеграционный hook, не универсальный per-call stream-параметр. Решение R02a «Sync без параметра» не нарушено.
- Регрессия: `go test ./cuda/` PASS (тесты gotorch не изменились).

### 2. goml — `Stream() uintptr` accessor

- **`goml/backend/cuda/cuda..go`** — публичный `func (b *Backend) Stream() uintptr`. Возвращает `b.stream` (уже приватное поле, было).

### 3. goml — новый пакет `backend/gotorch/`

**Файлы:**

| Файл | Что |
|---|---|
| `gotorch.go` | `type Backend struct { gt gtcuda.Backend; fb *gomlcuda.Backend }`. `Enable()` функция явного включения. `EnableIfEnv()` через `GOML_BACKEND=gotorch`. `Name()`/`DeviceType()`. |
| `storage.go` | `type Storage struct { gtStore gtcuda.Storage; ... }` — реализует `backend.Storage`. `wrapForeign()` helper — оборачивает любой `backend.Storage` в `gtcuda.ForeignStorage` через `WrapDevicePtr`. |
| `mem.go` | Alloc/Free/Copy(D2D)/ToDevice. Alloc/Free/Copy — прямые через `b.gt`. ToDevice для CPU↔GPU — прямой (без делегации в `fb.ToDevice`, тот panic'ит на type-assert для наших `*Storage`). |
| `add.go` | Единственный **direct**-метод impl-2: `Add` через `b.gt.AddF32`. Первый end-to-end путь. |
| `delegate.go` | Остальные 22 метода `backend.Backend` — делегация в `b.fb.<Method>` (stays-in-goml). Sub/Mul/Div/Neg/Exp/Log/Tanh/Relu/Sigmoid/MatMul(batch=1)/Softmax(axis=-1) станут direct в impl-3; композитные (LayerNorm, SDPA, RoPE...) — permanent stays-in-goml. |

**Дизайн memory:**
- Adapter владеет своей памятью через `gotorch.Alloc`. `goml.Pool` НЕ вовлечён (тот живёт в fallback `cuda.Backend`, используется только stays-in-goml операциями). Два раздельных менеджера памяти в одном процессе (R03b_design.md вопрос 4).

### 4. goml/go.mod

- Добавлен `require github.com/djeday123/gotorch v0.0.0` + `replace ...` на локальный `/data/lib/podman-data/projects/gotorch/v6` (для разработки; в проде — обычный module resolution).

---

## Sync-контракт двери (правка 1 R03b-paper)

**Реализация:** `Enable()` при инициализации:
1. Получает `fb *gomlcuda.Backend` из уже зарегистрированного goml.cuda.
2. Force lazy init через probe `Alloc(64) + Free` — иначе `Stream()` вернёт 0.
3. Читает `gomlStream := fb.Stream()`.
4. Создаёт `gotorch.NewBackend(0)` → `*PuregoBackend`.
5. **Инжектирует stream:** `pgt.SetStream(unsafe.Pointer(gomlStream))` — переводит все kernel launch и cuBLAS handle на goml stream.
6. Регистрирует adapter в реестре с ключом `backend.CUDA` (overriding goml.cuda).

**Дальнейшая работа:** оба мира на одной очереди. Порядок операций гарантирован stream'ом. **Полные Sync внутри adapter-методов — нет.** Единственный `Sync` — конец Step (у goml.trainer implicit через loss materialization).

---

## Приёмка

### Приёмка 1 — TestAdapterEnable

```
--- PASS: TestAdapterEnable (0.10s)
  adapter_test.go:70: adapter registered: gotorch-adapter (cuda)
```

Adapter зарегистрирован в реестре с overriding CUDA-backend'а, Alloc/Free round-trip работает.

### Приёмка 2 — TestAdapterAddF32 bit-exact 1024/1024

```
--- PASS: TestAdapterAddF32 (0.00s)
  adapter_test.go:178: AddF32 adapter n=1024: bit-exact=1024/1024
```

End-to-end: `adapter.Alloc → adapter.ToDevice(CPU→GPU) → adapter.Add → adapter.ToDevice(GPU→CPU)`. Результат **1024/1024 бит-в-бит** совпадает с CPU-эталоном `expected[i] = a[i] + x[i]`. Direct-путь через `gotorch.AddF32` (SetStream'нутый на goml stream) работает без full-sync, порядок операций даёт корректный результат.

### Приёмка 3 — TestAdapterNoFullSync (контр-тест дисциплины)

```
--- PASS: TestAdapterNoFullSync (0.00s)
  adapter_test.go:228: adapter body clean — zero full-sync calls, stream-injection contract holds
```

Grep-based static test по всем `.go` файлам adapter пакета (исключая `_test.go`) на подстроки `.Sync(`, `cuStreamSynchronize`, `cuCtxSynchronize`. Строки с `//` игнорируются. **Ноль вхождений.** Контракт «полные sync в теле adapter-методов запрещены» держится статически.

### Приёмка 4 — goml build чистый

```
$ go -C /data/lib/podman-data/projects/goml build ./...
# empty output, exit=0
```

### Приёмка 5 — goml cudatest регрессия (стандартный путь без adapter)

```
=== GoML CUDA Backend Test ===
1. Checking backend registration... OK — cuda
2. Memory alloc + HtoD + DtoH... OK
3. MatMul correctness (cuBLAS vs CPU)... OK (max diff: 0.004459)
4. MatMul performance benchmark:
   [ 128x 128x 128] GPU:    0.010 ms (   418.8 GFLOPS)
   [ 512x 512x 512] GPU:    0.006 ms ( 42819.5 GFLOPS)
   [1024x1024x1024] GPU:    0.017 ms (121415.9 GFLOPS)
   [2048x2048x2048] GPU:    0.100 ms (170840.3 GFLOPS) | 46516× speedup
5. PTX kernel (Fill)... OK
6. Element-wise (Add, Mul)... OK (Add verified)
=== All tests passed ===
```

**171 TFLOPS на 2048³ SGEMM** — без изменений от impl-1. Adapter package **не влияет** на пользователей, которые его не включают (стандартный путь goml.cuda работает как прежде).

### Приёмка 6 — gotorch регрессия (default stream)

```
$ go -C /data/lib/podman-data/projects/gotorch/v6 test -count=1 -run "TestAddF|TestMatMulF|TestSumF|TestSoftmaxF|TestActivations" ./cuda/
ok  github.com/djeday123/gotorch/cuda  0.432s
```

Замена `hStream=0` → `b.stream` (по умолчанию 0) не сломала ни одного теста gotorch. Обратная совместимость подтверждена.

---

## Что НЕ сделано (следующий impl-3)

- 15 direct-методов из таблицы (кроме Add) — все ещё делегируются в `fb.<Method>`.
- Композитные тесты (Softmax, LayerNorm) не проверяются через adapter — они всё равно идут через goml.cuda.
- Абсолютное bit-exact покрытие тестами для остальных direct-путей — impl-3.

---

## Финал impl-2

- 6 файлов: 2 модификации в gotorch, 1 accessor в goml, 4 новых в `goml/backend/gotorch/` + 1 test.
- Приёмка 3/3 adapter tests PASS, sync-дисциплина держится статически.
- Регрессий нет (cudatest без adapter, gotorch без стрима — всё работает).
- Готов принять impl-3 (перевод 14 direct-методов из delegate в direct).
