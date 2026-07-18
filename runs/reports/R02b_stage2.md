# R02b Stage 2 — cuBLAS DGEMM + SGEMM MatMul — GATE PASSED

**Дата:** 2026-07-16
**Этап:** 2 из 6 (Linalg F64/F32, методы 13-14 из 50)
**Ворота 2:** ✅ пройдены с FP64-эталоном + hybrid F32-tolerance

---

## Что сделано

### Новые/изменённые файлы

| Файл | Изменение |
|---|---|
| `cuda/cublas_purego.go` | **Новый.** 5 биндингов: `cublasCreate_v2`/`cublasDestroy_v2`/`cublasSetStream_v2` + `cublasDgemm_v2` + `cublasSgemm_v2`. Без GemmEx, batched, TF32 math-mode — как в ТЗ. |
| `cuda/backend_purego.go` | Добавлено поле `cublas uintptr` в `PuregoBackend`. `newPuregoBackend`: после `cuDevicePrimaryCtxRetain` вызывает `cublasCreate_v2` + `cublasSetStream_v2(0)`. `Close`: `cublasDestroy_v2` до `cuDevicePrimaryCtxRelease`. `MatMulF64`/`MatMulF32` реализованы. |
| `cuda/matmul_test.go` | **Новый.** `TestMatMulF64`/`TestMatMulF32` через subtests по 4 формам + `TestMatMulForeign` через `WrapDevicePtr`. |

### Row-major трюк — обоснование в коде

23-строчный комментарий в `backend_purego.go` над `MatMulF64` объясняет вывод:
- row-major C[MxN] = A[MxK] × B[KxN] эквивалентно col-major C^T[N×M] = B^T[N×K] × A^T[K×M].
- Оба op = `OP_N` (транспоны уже сидят в самом факте row/col-major сдвига).
- Первый операнд cuBLAS — row-major B (view B^T[N×K]), `lda = N`.
- Второй — row-major A (view A^T[K×M]), `ldb = K`.
- Результат — row-major C (view C^T[N×M]), `ldc = N`.
- Порядок аргументов: `(N, M, K)` вместо `(M, N, K)`.

Сверено с `goml/backend/cuda/cublas.go:244-261` — идентичный свап.

### TF32 умышленно НЕ включён

`cublasSetMathMode` не вызывается — default math mode = FULL FP32 accumulate/multiply. TF32 срезал бы точность до ~1e-3 rel, требуя ослабления hybrid tolerance. Прописано в `cublas_purego.go` doc-комментарии.

---

## Финальные maxRelErr — Ворота 2

### F64 (element-wise rel < 1e-12)

| Форма (M×K×N) | maxRelErr | worst idx | статус |
|---|---|---|---|
| 3×4×5 | 3.66e-15 | — | ✅ |
| 16×16×16 | 5.38e-15 | — | ✅ |
| 128×64×32 | 1.73e-13 | — | ✅ |
| 1×1×1 | 0 | — | ✅ |
| foreign 32×16×24 | 3.29e-14 | — | ✅ |

**Комментарий.** После перевода эталона на канонический FP64-путь (у F64 он и был FP64, но проверил на регрессию) — цифры **идентичны прежним**: эталон не сдвинулся, ошибка cuBLAS DGEMM ~ K × eps_f64 ≈ 64 × 2.2e-16 ≈ 1.4e-14 на 128×64×32, что согласуется с наблюдаемым 1.73e-13 (небольшой запас за счёт cancellation).

### F32 — hybrid `|got - ref| < 1e-4 + 1e-5 × |ref|`

| Форма (M×K×N) | maxAbsErr | maxRelErr | hybrid PASS? |
|---|---|---|---|
| 3×4×5 | 8.94e-08 | 3.05e-07 | ✅ |
| 16×16×16 | 9.54e-07 | 2.89e-06 | ✅ |
| 128×64×32 | 3.82e-06 | **1.19e-04** | ✅ (abs ≪ 1e-4) |
| 1×1×1 | 0 | 0 | ✅ |

**Ключевой случай — 128×64×32.** maxRelErr = 1.19e-04 действительно превышает 1e-5 rel-порог, ровно как предсказано анализом cancellation. Но maxAbsErr = 3.82e-06 — в 26× меньше 1e-4 abs-порога, поэтому hybrid bound `1e-4 + 1e-5 × |ref|` всегда покрывает: **абсолютная компонента поглощает cancellation-случаи, относительная ловит систематические ошибки** (перепутанные операнды, транспоны, битый lda/ldb — на них ошибка была бы на порядки больше 1e-4 abs).

Предыдущий прогон (F32-эталон) давал rel 7.6e-05 на этой же форме. После перехода на FP64-эталон rel вырос до 1.19e-4 — потому что эталон стал точнее, и разница с cuBLAS теперь чистая, без шума собственной проверялки. Это ровно то, что ожидалось.

---

## FP64-эталон — прямой урок B1

CPU-эталон **для обоих dtype** считает в FP64:
```go
func cpuMatMulF32(a, b []float32, m, n, k int) []float32 {
    c := make([]float32, m*n)
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            var s float64 // ← FP64 accumulator
            for l := 0; l < k; l++ {
                s += float64(a[i*k+l]) * float64(b[l*n+j])
            }
            c[i*n+j] = float32(s) // ← округление в самом конце
        }
    }
    return c
}
```

Это прямая калька с решения из B1 (сага 93.3% → 100%): эталон обязан быть точнее проверяемого, иначе `measured_err = ref_noise + subject_err`, и различить невозможно. Двухуровневый стандарт «FP64-golden как истина» распространён на все тесты gotorch.

Приложил doc-комментарий над `cpuMatMulF32` и в шапке файла `matmul_test.go` — 15 строк, объясняющих почему аккумулятор в FP64 и почему допуск на F32 hybrid, а не element-wise rel. Явно предупреждает будущего читателя: **не ужесточай допуск обратно из благих намерений — упрёшься в тот же cancellation.**

---

## Комментарии в коде теста — предотвращение регрессии

В шапке `matmul_test.go`:
> Element-wise rel=1e-5 для FP32 GEMM с K≥64 НЕДОСТИЖИМ из-за cancellation: eps × K × max_partial ≈ 1.19e-7 × 64 × 20 ≈ 1e-4 абсолютной ошибки в worst-case-сумме. При малом |ref| относительная ошибка на элементе взлетает; это не баг реализации, это неизбежное свойство FP32-суммирования. Абсолютная компонента 1e-4 в hybrid покрывает cancellation, относительная 1e-5 ловит систематические ошибки (перепутанные операнды, транспоны, битый lda/ldb).
>
> **НЕ ужесточай допуск F32 обратно из благих намерений — упрёшься в тот же cancellation.**

---

## Проверки

| Проверка | Команда | Результат |
|---|---|---|
| Build | `go -C v6 build ./...` | ✅ exit 0 |
| Vet | `go -C v6 vet ./cuda/` | ✅ exit 0 (без флагов) |
| Ворота 1 тесты | `go test ./cuda/ -run 'TestMemoryRoundtrip\|TestPinned\|TestSealed\|TestFree\|TestNoUintptr\|TestCopyD2D'` | ✅ 6/6 |
| Ворота 2 тесты | `go test ./cuda/ -run 'TestMatMulF64\|TestMatMulF32\|TestMatMulForeign'` | ✅ 9/9 (4 F64 subtests + 4 F32 subtests + foreign) |

---

## Флаг эскалации — отработан

Реальность разошлась с ТЗ на F32-tolerance (1e-5 rel был неоптимистичен для FP32 SGEMM с cancellation). Не ослабил молча — принёс числа и три варианта, пользователь выбрал hybrid + FP64-эталон. Прецедент из B1 подтвердил правильность подхода.

**Пометка для будущих этапов:** если Exp/Log/Softmax F32 (Этапы 4-5) покажут аналогичный разрыв — повторю ту же схему (принести числа и варианты). ТЗ Этапа 4 уже предусматривает этот путь для approx-инструкций.

---

## Следующий этап

**Auto-go на Этап 3** по решению пользователя. Этап 3:
- `ptx_kernels.go` — PTX-строки для AddF64 и AddF32 (`.target sm_80`, `.version 7.0`).
- `backend_purego.go` — поля `ptxModule uintptr` + `fns map[string]uintptr`, инициализация в `newPuregoBackend` после cuBLAS, реализация `AddF64`/`AddF32` через `cuLaunchKernel`.
- Тесты: `TestAddF64`/`TestAddF32` на размерах 1, 255, 256, 257, 100000 (границы grid); `TestKernelLoadFailure` с битой PTX-строкой (внятная ошибка вместо паники).
- Grid: block=256, grid=ceil(n/256), одномерная схема. Оптимизации не цель — корректность.
- Стоп-точка на Этапе 4 (Exp/Log approx-точность — там ТЗ предусматривает возврат с вопросом).
