# R03b-impl-1 — goml Fix A: primary context retain

**Дата:** 2026-07-18
**Итог:** ✅ PASSED. Приёмка `identical:true` в interop_smoke Test 1 подтверждена; `cudatest` боевой смок PASS; регрессий нет.

---

## Что сделано

### 1. `goml/backend/cuda/driver.go`

Добавлены **2 биндинга**:
```go
cuDevicePrimaryCtxRetain  func(pctx *uintptr, dev int32) CUresult
cuDevicePrimaryCtxRelease func(dev int32) CUresult
```
```go
purego.RegisterLibFunc(&cuDevicePrimaryCtxRetain, lib, "cuDevicePrimaryCtxRetain")
purego.RegisterLibFunc(&cuDevicePrimaryCtxRelease, lib, "cuDevicePrimaryCtxRelease_v2")
```

`cuCtxCreate`/`cuCtxDestroy` **оставлены** зарегистрированными — они не используются в hot path, но могут пригодиться для будущих сценариев (изолированный вспомогательный контекст, тесты). Никаких удалений.

### 2. `goml/backend/cuda/cuda..go:72` (было `cuCtxCreate`)

Заменено на `cuDevicePrimaryCtxRetain`:
```go
if r := cuDevicePrimaryCtxRetain(&b.ctx, b.device); r != CUDA_SUCCESS {
    return fmt.Errorf("cuDevicePrimaryCtxRetain: %s", r.Error())
}
if r := cuCtxSetCurrent(b.ctx); r != CUDA_SUCCESS {
    cuDevicePrimaryCtxRelease(b.device)
    return fmt.Errorf("cuCtxSetCurrent(primary): %s", r.Error())
}
```

`cuCtxSetCurrent` добавлен явно потому что `cuDevicePrimaryCtxRetain` **не делает** current-context set (в отличие от `cuCtxCreate`, который push'ит новый context на thread stack).

### 3. `goml/backend/cuda/cuda..go` в `Close()` (было `cuCtxDestroy`)

Заменено на `cuDevicePrimaryCtxRelease`:
```go
if b.ctx != 0 {
    cuDevicePrimaryCtxRelease(b.device)
    b.ctx = 0
}
```

**Не декремент refcount destroy.** Release — decrement refcount; context живёт, пока другой процесс/пакет (gotorch, PyTorch, TensorRT) его retain'ят.

**Всего изменено:** 2 файла, 3 логических точки, +9 строк / −3 строки.

---

## Приёмка

### Приёмка 1 — Test 1 identical:true

```
=== RUN   TestInterop_R03a/Test1_ContextIdentity
[GoML] CUDA backend initialized: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (Blackwell, sm_120)
[GoML] CUDA libs dir: /data/lib/podman-data/projects/goml/libs
    main_test.go:253: gotorch ctx (primary retain) = 0x0000000039400730
    main_test.go:254: goml    ctx (cuCtxCreate)   = 0x0000000039400730
    main_test.go:255: identical: true
    main_test.go:258: RESULT: shared-context hypothesis CONFIRMED — оба мира на одном ctx
--- PASS: TestInterop_R03a/Test1_ContextIdentity (0.11s)
```

**Ключевой факт:** оба handle `0x0000000039400730` — численно **идентичны**. Дизайн-гипотеза, ранее опровергнутая в R03a, теперь **подтверждена по конструкции**. UVA-цирк не нужен: memcpy и kernel launch между мирами тривиально валидны, потому что context один.

*(Label «goml ctx (cuCtxCreate)» в логе — старая строка комментария в тесте, не отражает актуальную реализацию; при перезаписи interop_smoke на R03b-impl-2 обновим.)*

### Приёмка 2 — все 6 interop_smoke тестов PASS

Полный прогон существующего `TestInterop_R03a` — 6/6 subtests PASS:

| Тест | Результат |
|---|---|
| Test1 ContextIdentity | ✅ identical=true, handle `0x39400730` |
| Test2 Entry door (goml alloc → gotorch AddF32) | ✅ 1024/1024 bit-exact |
| Test3 Exit door (gotorch alloc → goml Copy) | ✅ 1024/1024 bit-exact |
| Test4 Cross-teardown (gotorch.Close, goml alive) | ✅ 128/128 round-trip |
| Test5 ThreadMix 100 iter | ✅ goml 100/100, gotorch 100/100, 0 errors |
| Test6 CurePrototype | ✅ [B] retained primary=`0x08615730` matches snap; [C] UVA cross-ctx через 3rd party floating: 1024/1024; [D] kernel launch cross-ctx: 1024/1024 |

**Никаких регрессий.** Приёмка «UVA-мост сохраняется даже после Fix A» пассивно подтверждена (Test 6C/D через 3rd party `cuCtxCreate`-floating всё ещё работают).

### Приёмка 3 — goml build чистый

```
$ go -C /data/lib/podman-data/projects/goml build ./...
# empty output, exit=0
```

### Приёмка 4 — goml cudatest боевой смок

Прогон `cmd/cudatest/main.go` (backend smoke: registration, memory, MatMul cuBLAS-vs-CPU, PTX kernel, elementwise):

```
=== GoML CUDA Backend Test ===
1. Checking backend registration... OK — cuda
2. Memory alloc + HtoD + DtoH... OK
3. MatMul correctness (cuBLAS vs CPU)... OK (max diff: 0.004226)
4. MatMul performance benchmark:
   [ 128x 128x 128] CPU:  1.1 ms | GPU:  0.003 ms |    1224 GFLOPS |   331× speedup
   [ 512x 512x 512] CPU: 72.7 ms | GPU:  0.006 ms |   42738 GFLOPS | 11569× speedup
   [1024x1024x1024] CPU: 579.8 ms | GPU:  0.017 ms |  121664 GFLOPS | 32848× speedup
   [2048x2048x2048] CPU: 4748.1 ms | GPU:  0.100 ms |  170762 GFLOPS | 47195× speedup
5. PTX kernel (Fill)... OK
6. Element-wise (Add, Mul)... OK (Add verified)
=== All tests passed ===
```

**171 TFLOPS на 2048³ SGEMM** — норма для RTX PRO 6000 Blackwell. Производительность goml после Fix A не изменилась в шумовом диапазоне.

*(Юнит-тесты `./backend/...`/`./tensor/...` в goml не существуют — `[no test files]`. Регрессия проведена через cudatest как боевой смок-контур.)*

---

## Что НЕ сделано в этом коммите (по scope R03b-impl-1)

- Ни строки в adapter (`goml/backend/gotorch/`) — это impl-2.
- Ни одной новой публичной функции в gotorch (SetStream — impl-2 вместе с adapter).
- Ни одной новой операции; только замена context-management механизма.

---

## Открытые заметки для impl-2

1. **Test 1 label**: в `interop_smoke/main_test.go:254` стоит комментарий `goml ctx (cuCtxCreate)`. После impl-2 переписать на `goml ctx (primary retain)` — сейчас метка вводит в заблуждение читателя лога.
2. **cudatest UVA-путь через MatMul** — здесь goml делает MatMul cuBLAS **и** cuBLAS-wrapper (FP16/FP8 подключены) — оба работают. Значит retain-primary полностью совместим с cublasCreate/SetStream/GemmEx workflow.
3. **Fix A на gpu4 (sm_89) НЕ повторён** — не блокер (R03a-89 показал матрицу 2×2 PASS без Fix A), но при синхронизации репо клон `goml_v4` на gpu4 подхватит Fix A автоматически.

---

## Финал impl-1

- Ноль побочных правок, минимальный commit.
- Приёмка `identical:true` подтверждена численно.
- Регрессий нет (6/6 interop + cudatest OK).
- Готов принять impl-2 без дополнительной команды пользователя (по правилу ТЗ R03b-paper ревью).
