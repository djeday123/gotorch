# B-impl-1 доработка — Native strided-batched через C-wrapper

**Дата:** 2026-07-23
**Итог:** ✅ ПРОЙДЕН. Native `cublas[S/D]gemmStridedBatched` теперь основной путь через тонкий C-wrapper (обходит purego <18 args лимит). Loop-fallback остаётся с warning при отсутствии `.so`. Раскрыта старая broadcast-UB в `fb.MatMul` -- adapter теперь исправлен, старые delegate-парity тесты корректно скипнуты (обёснение в коде).

---

## Мотивация

По ревью B-impl-1: loop-fallback стал боевой стрелкой в adapter, что противоречит духу R03b правки 2 («loop — тест-заглушка, не для production»). Цель главы -- индустриальный фундамент, не паритет с legacy.

Решение: тонкий C-wrapper по прецеденту goml (`libcublas_wrapper.c`, `libcublaslt_wrapper.c`) с packed struct-args entry points. 2-arg сигнатура (`(stream, args*)`) обходит purego лимит.

---

## Файлы

**Новые:**
- `gotorch/v6/libs/blas_wrapper.c` (194 lines) — 4 entry points через struct-args:
  - `gt_sgemm_strided_batched(args)` — F32 (SgemmStridedBatched).
  - `gt_dgemm_strided_batched(args)` — F64.
  - `gt_gemm_ex(args)` — universal typed GEMM (для B-impl-2 F16).
  - `gt_gemm_strided_batched_ex(args)` — universal typed strided-batched (для B-impl-2 F16 batched).
- `gotorch/v6/libs/Makefile.blas_wrapper` — сборка `libgotorch_blas_wrapper.so`. Resolves cublas_v2.h + libcublas.
- `gotorch/v6/cuda/cublas_wrapper_purego.go` (172 lines) — Go биндинги через purego. Struct layouts точно зеркалят C natural alignment. `HasBlasWrapper()` предикат для тестов.
- `gotorch/v6/cuda/matmul_batched_wrapper_test.go` (150 lines) — A/B wrapper vs loop:
  - F32: bit-exact 608/2048, maxRel 1.54e-4, hybrid PASS.
  - F64: **bit-exact 2048/2048** (cublasDgemmStridedBatched vs cublasDgemm loop).

**Изменённые:**
- `gotorch/v6/cuda/backend_purego.go`: `MatMulStridedBatchedF32/F64` — 2 пути. Wrapper (`HasBlasWrapper()`) → `gt[SD]gemm_strided_batched`; иначе `matMulBatchedF32/F64Loop` (fallback) с warning в stderr.
- `goml/backend/gotorch/matmul_softmax.go`: **strideB=0 при broadcast** (shapeB имеет меньше batch-размерностей чем shapeA). Раскрытая старая UB в loop-путях.
- `goml/backend/gotorch/linear_test.go`: 3.3/4.3 **корректно скипнуты** с диагностическим комментарием о broadcast-UB в fb.

---

## Ключевой урок доработки: **cuBLAS handles не переносятся через dlopen**

Диагностический прогон показал:
```
[wrapper-diag] cublasCreate inside wrapper: 0, local_h=0x15893cb0, caller_h=0x133cd120
```

Handle из purego-loaded libcublas.so **НЕ работает** внутри wrapper.so-linked libcublas.so, даже когда оба dlopen резолвят один on-disk `libcublas.so.13`. Возвращает `CUBLAS_STATUS_NOT_INITIALIZED (1)`. Cублас держит per-load state.

**Правильный дизайн:** wrapper owns свой local `cublasHandle_t`, Go caller передаёт **CUstream** (не handle). Wrapper вызывает `cublasSetStream_v2(local_h, stream)` перед каждой GEMM. Стоимость -- один state mutation setStream, не sync -- дёшево.

Отдельная запись в feedback-memory.

---

## Broadcast-UB раскрыта

Форма `[batch=4, M, K] × [K, N]` — B non-batched. Правильная semantics: одна B для всех batches → `strideB=0`.

**Что делали ранее loop-пути (adapter+fb):**
```go
strideB := uintptr(K * N * 4)
for i := 0; i < batch; i++ {
    ... bPtr + strideB*i ...
}
```
Для B non-batched это **чтение за границы буфера** — UB. adapter loop и fb loop оба страдали, читали тот же мусор из GPU памяти (аллокатор кладёт соседние блоки), давали одинаковый output → тесты 3.3/4.3 «bit-exact».

**После B-impl-1 доработки:** adapter fix (strideB=0), fb остаётся buggy. Расхождение = 0.24 maxAbs. Тесты 3.3/4.3 больше не работают как paritу; **правильная semantics** через B/J F64 CPU reference (существующий `TestAdapterMatMul_Batched_AvsB`).

Скип с комментарием (объяснение в коде) вместо удаления — сохраняет историю решения.

---

## Тесты (все PASS)

**gotorch/v6/cuda:**
| Тест | Форма | Результат |
|---|---|---|
| MatMulStridedBatchedF32_Batch1EqNonBatched (wrapper) | b=1 m=16 n=32 k=24 | maxRel 1.5e-4, hybrid PASS |
| MatMulStridedBatchedF32_Batch1EqNonBatched (loop) | как выше, без wrapper | **bit-exact 512/512** |
| MatMulStridedBatchedF32_Shapes | [1..16, ...] | hybrid all PASS |
| MatMulStridedBatchedF64_Shapes | [1..8, ...] | rel <= 6e-12 (actual floor 1e-11) |
| MatMulStridedBatchedF32_BJudge | b=4 m=32 n=64 k=48 | maxRel 5.6e-4 (hybrid PASS) |
| **WrapperVsLoop_F32** (NEW) | b=4 m=16 n=32 k=24 | maxRel 1.5e-4, hybrid PASS |
| **WrapperVsLoop_F64** (NEW) | b=4 m=16 n=32 k=24 | **bit-exact 2048/2048** |

**goml/backend/gotorch:**
- adapter regression: PASS (3.3/4.3 skipped с диагностикой).
- B/J F32 batched: maxRel 5.4e-5, 0 fails.

---

## Регрессия

| Гейт | Результат |
|---|---|
| gotorch/v6/cuda full | ✅ ok 0.789s |
| adapter regression | ✅ ok 0.427s |
| interop_smoke 6/6 | ✅ ok 0.348s |
| f64ref 9/9 | ✅ ok 0.374s |
| P1-ABJ 10 шагов | ✅ PASS |
| FA-canary fwd v121r | mean **654.00T** (+0.31% thermal, WITHIN) |
| NoFullSync grep guard | ✅ clean |

---

## Резолюция .so (для CI и dev workflow)

Loader (`resolveBlasWrapperPath` в `cublas_wrapper_purego.go`):
1. `$GOTORCH_LIBS_DIR/libgotorch_blas_wrapper.so`
2. cwd (`./libgotorch_blas_wrapper.so`)
3. exec-dir/libs
4. ldconfig fallback (through purego)

**Missing .so → soft degradation:** stderr warning с инструкцией сборки + loop-fallback. Adapter/пользователь не падает, просто медленнее.

---

## Побочные находки

1. **cuBLAS handle НЕ переносится через dlopen** — принципиально важно для любых новых wrappers через purego. Правило: **wrapper owns local handle, caller передаёт stream**. Записано в feedback-memory.
2. **Broadcast-UB в loop-batched** — старая тихая ошибка в adapter + fb, скрывалась «paritу» тестами. Native cublas обнажил. Правильный дизайн: strideB=0 при broadcast.
3. **F64 wrapper vs loop bit-exact** — cублас DgemmStridedBatched и DgemmLoop дают идентичный результат. F32 версии отличаются на F32 rounding-level. Это уточнение к paper B4 -- F64 внутри cublas детерминировано между вариантами.

---

## Метрический учёт

Методы: **68** (без изменений). Прирост -- инфраструктурный (wrapper + fallback + tests).

---

## СТОП по ТЗ

B-impl-1 доработка закрыта. Далее по плану: **B-impl-2 F16** через уже готовый `gt_gemm_ex` + `gt_gemm_strided_batched_ex` в wrapper.
