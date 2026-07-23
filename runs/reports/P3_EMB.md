# P3-EMB — Embedding port в gotorch: второй порт главы

**Дата:** 2026-07-23
**Итог:** ✅ ПРОЙДЕН. Forward bit-exact на всех тестовых формах (прогноз подтверждён), backward hybrid PASS с запасом 500-1000× vs floor. Первое живое (production-path) legacy-ядро в главе — попадает в подсписок «Живые» инвентаря.
**Побочная находка:** первый прогнозный промах — атомическая репродуцируемость F32 недооценена sqrt(N)·eps оценкой; реальный drift 44× выше прогноза. Записано честно как факт, floor'ы обновлены с запасом относительно измерения.

---

## Этап 1 — Recon (paper)

**goml эталон (backend.Backend interface + PTX):**

| Слой | Файл:строка | Свойство |
|---|---|---|
| Interface | `goml/backend/backend.go:115` | `Embedding(dst, weight, indices, vocab, embed, seqLen, dtype)` |
| CUDA impl | `goml/backend/cuda/ops.go:341` | вызывает `embedding_f32` через `launch(seqLen, 1, 1, blockDim=min(embed,256))` |
| PTX kernel | `goml/backend/cuda/kernels.go:523` | `.visible .entry embedding_f32(p_dst, p_weight, p_indices, p_embed_dim)` |
| Adapter delegate | `goml/backend/gotorch/delegate.go:50` | stays-in-goml (`b.fb.Embedding(...)`) |

**Ключевые характеристики эталона:**
- **Только forward**, только F32.
- **Индексы: int64** (`ld.global.s64 %idx64` + `cvt.u32.s64` внутри kernel).
- Раскладка weight: `[vocab, embed_dim]` row-major (`weight[idx][d]` = `weight + idx*dim + d`).
- Grid: `(seqLen, 1, 1)`, block: `min(embed_dim, 256)` — 1 CTA на выходную строку.
- **Callers (живой путь):**
  - `goml/cmd/gputrain/main.go:109` — **LIVE** (production-путь; F32, int64 tokens).
  - `goml/cmd/simpletrain/main.go` — LIVE.
  - `goml/nn/embedding.go:33` — LIVE (nn.Embedding.Forward).
  - `goml/nn/model.go:113` — LIVE (nn.LLM).

**Backward механика эталона (`goml/nn/backward.go:230`):**
- **CPU F32**, **детерминированный** (последовательный цикл).
- Код: `dW[idx*embedDim + d] += dout[s*embedDim + d]` — collision handling через накопление в порядке появления индекса.
- Использует `accumulateGrad` (CPU-цикл).
- **НЕТ GPU backward kernel'а** — nn.Embedding.Backward работает только через CPU-путь.
- В live gputrain-пути **backward не вызывается** (обновляется только outWeight); nn.LLM.Trainer вызывает CPU backward.

**Живость:** goml.cuda.Embedding — **живая**, попадает в подсписок «Живые» инвентаря. Порт-приоритет высокий.

**Решение по fusion:** Порт **чистый** (без fusion). Backward выводим сами через atomic-scatter (эталон CPU не масштабируется на GPU; собственная реализация — единственный вариант).

---

## Прогнозы, записанные ДО измерения

| # | Метрика | Прогноз | Обоснование |
|---|---|---|---|
| P1 | Forward B/J | **bit-exact** | gather = memcpy строки, никакой арифметики |
| P2 | A/B fwd (goml int64 vs adapter int32) | **bit-exact при равных значениях** | dtype индексов не влияет на output при равных вокабуляр-значениях |
| P3 | Backward B/J (без коллизий) | **bit-exact** | нет race, atomic == простой store |
| P4 | Backward B/J (с коллизиями) | hybrid abs=1e-4+rel=1e-5 | float atomicAdd не ассоциативен |
| P5 | Atomic repro F32 (5 runs) | maxRel ≤ **1e-6** | оценка sqrt(N)·eps_f32 = √32·6e-8 ≈ 3e-7 |
| P6 | Atomic repro F64 (5 runs) | maxRel ≤ **1e-14** | оценка sqrt(N)·eps_f64 ≈ 1e-16·6 |
| P7 | Numerical grad F64 h=1e-6 | rel ≤ **1e-8** | стандарт главы |

---

## Этап 2 — int32 буферы в gotorch

**Решение (записано в `cuda/api.go` doc-string):**
Не заводить полный Int32-мир. Использовать существующие `Alloc(nBytes)` + `CopyH2D/D2H` — они типонезависимы (передают байты). int32 индексы = 4 bytes/elem, little-endian. Контракт индексного буфера — свойство **метода** (`Embedding*`), а не типа storage.

Валидность `0 <= idx < vocab` — обязанность вызывающего; out-of-range = UB (в PTX = чтение неверного адреса → segfault/бит-мусор). В debug-пути тесты выполняют CPU-предпроверку диапазона перед upload.

Файл `probe.go` (uintptr, cuMemcpyDtoH и т.д. уже байтовые) не требует правок. Никаких новых типов, никаких новых копирований — контракт живёт в doc-string.

---

## Этап 3 — Порт ядер

**4 PTX kernels** в `gotorch/v6/cuda/ptx_kernels.go`:
```
embedding_f32(table, indices, out, hidden, n)       — gather, 1 CTA/row, block=min(hidden,256)
embedding_f64(table, indices, out, hidden, n)       — F64 gather
embedding_grad_f32(indices, dout, dtable, hidden, n) — scatter atom.global.add.f32
embedding_grad_f64(indices, dout, dtable, hidden, n) — atom.global.add.f64
```

**Соответствие 6 правилам PTX** (safety-net R02b): ASCII-only, `%tidx` конвенция, SMEM не используется (нет reduction), `cvt.rn` где нужно, one-stmt-per-line, JIT-log включён. Первый прогон **прошёл** без ASCII/register-name catch'ей.

**Go wrappers** (`backend_purego.go`) + Backend interface (`api.go`): 4 методов, kernelNames registration. Grad-обёртки zeroят `dtable` через `cuMemsetD8` перед kernel launch.

**Тесты** (`gotorch/v6/cuda/embedding_test.go`) — 9 тестов:

| Тест | Форма/условие | Результат |
|---|---|---|
| Forward F32 shapes | [7,3,5], [1,1,1], [256,64,16], [32000,256,256], **[50000,512,256]** battle | bit-exact **131072/131072** worst-case |
| Forward F64 shapes | [7,3,5], [1,1,1], [256,64,16], [8000,128,64] | bit-exact **8192/8192** worst-case |
| Edge equal indices | [16,8,32] всё нули | bit-exact, `y=table[0]` для всех |
| Edge boundary indices | [100,8,4] `[0, 99, 0, 99]` | bit-exact |
| Grad F32 shapes | [7,3,5], [256,64,16], [32000,256,256] noCol; [100,32,512] withCol | maxRel 2.2e-5 with coll; 0 без коллизий |
| Grad F64 shapes | как выше | maxRel 6.4e-14 with coll |
| Atomic repro F32 | 5 runs, 32 идексов в [0..8) | **maxRel 4.4e-5** (прогноз промахнулся 44×) |
| Atomic repro F64 | 5 runs, 32 идексов в [0..8) | **maxRel 2.6e-14** (прогноз 2.6×) |
| Numerical F64 h=1e-6 | [5,4,7], vocab=5 | worstRel 7.9e-9 (запас 1.3× vs floor 1e-8) |

**Прогноз P1 (fwd bit-exact) — подтверждён.** Прогноз P3 (grad bit-exact без коллизий) — подтверждён (`maxRel=0`). Прогноз P4 (hybrid grad с коллизиями) — подтверждён с большим запасом.

**Прогнозы P5/P6 (atomic reproducibility) — НЕ УДЕРЖАЛИ.** Причина: при 32 коллизиях/строку GPU thread scheduling существенно варьирует **порядок** сложений, drift накапливается ближе к линейному `O(eps·|sum|·N)`, а не sqrt-корневому. Измеренные значения:
- F32: **4.4e-5** vs прогноз 1e-6 (промах 44×)
- F64: **2.6e-14** vs прогноз 1e-14 (промах 2.6×)

Actual floor'ы (с запасом относительно измеренного): F32 = **1e-4**, F64 = **1e-13**. Оба прогноза И actual-floor'ы явно логируются тестом (пометка `pre-reg X, actual Y`).

**Урок для главы:** Оценка `sqrt(N)·eps` для atomicAdd-drift занижена, когда N коллизий на строку большое (десятки) и величина суммы того же порядка что и слагаемые. Использовать `O(eps·N)` как гарантирующий верхний прогноз при таких условиях.

---

## Этап 4 — Adapter direct + A/B + J-судья

**Adapter extension methods** (`goml/backend/gotorch/embedding.go`):

4 метода на конкретном типе `*gotorch.Backend` (extension API через type-assertion, как в P2-RMS). Через backend.Backend interface **не идут** — там сигнатура goml.Embedding с int64 индексами, drop-in замена невозможна без изменения gputrain (правило read-only). Порт ценен как library method и как подготовка nn.LLM/Trainer, где мы контролируем dtype индексов.

**Форма LLM-tiny [vocab=32000, hidden=256, n=64]** — default nn.Config.

**Три теста B vs J + один тест A vs B** (`goml/backend/gotorch/embedding_test.go`):

| Тест | Результат | Прогноз |
|---|---|---|
| B(adapter F32) vs J(F64) fwd [32000,256,64] | bit-exact **16384/16384** | ✅ подтверждён (P1) |
| A(goml int64) vs B(adapter int32) fwd [256,64,16] | bit-exact **1024/1024** | ✅ подтверждён (P2) |
| B(adapter F32) vs J(F64) bwd [256,64,16] compression=8 | maxAbs 1.19e-7, maxRel 5.84e-8, 0 fails | запас **>500×** vs floor abs=1e-4+rel=1e-5 |

**Стрелка delegate→direct:** отложена. gputrain использует `backend.Backend.Embedding(int64)`, наш kernel — int32. Замена = breaking change по dtype индексов. Записано в отчёт; порт всё равно ценен как library method (extension на *gotorch.Backend).

**Sync-контракт:** без внутреннего Sync (injected stream). `TestAdapterNoFullSync` grep-guard остаётся зелёным.

---

## Этап 5 — Регрессия ворот

| Гейт | Результат |
|---|---|
| **P3-EMB gotorch/v6/cuda** (9 тестов) | ✅ PASS |
| **P3-EMB adapter B vs J + A vs B** (3 теста) | ✅ PASS |
| adapter regression (все R03b/P1/P2 + P3) | ✅ ok 0.433s |
| gotorch cuda tests (R02b + P2 + P3) | ✅ ok 0.673s |
| interop_smoke 6/6 | ✅ ok 0.344s |
| goml cudatest `go run ./cmd/cudatest/` | ✅ All tests passed |
| f64ref grad checks 9/9 | ✅ ok 0.423s |
| P1-ABJ 10 шагов (изоляция) | ✅ PASS |
| **FA-canary fwd v121r** | mean **654.12T** (baseline 652±2T, +0.32%) — thermal drift (та же природа что и в P2, 37→41C) |
| NoFullSync grep guard | ✅ clean |

**Про FA-canary fwd:** +2.12T = +0.32% над baseline. Диапазон `[653.47, 654.37]` (spread 0.9T), стабильно WITHIN honest tolerance. Не регрессия. Сравнимо с P2 закрытием (mean 653.97T).

---

## Метрический учёт

**gotorch/v6/cuda Backend methods**: 54 → **58** (+4 EmbeddingF32/F64/GradF32/GradF64).

**Файлы:**

| Файл | Изменение |
|---|---|
| `gotorch/v6/cuda/ptx_kernels.go` | +4 PTX kernels |
| `gotorch/v6/cuda/backend_purego.go` | +4 kernel registrations + 4 Go wrappers + 2 launcher helpers |
| `gotorch/v6/cuda/api.go` | +4 methods в Backend interface + doc-string по int32 контракту |
| `gotorch/v6/cuda/embedding_test.go` | **NEW** — 9 тестов |
| `goml/backend/gotorch/embedding.go` | **NEW** — 4 adapter extension methods |
| `goml/backend/gotorch/embedding_test.go` | **NEW** — 3 B/J теста + A/B forward тест |

---

## Побочные находки

1. **Legacy inventory обновление**: **`goml/backend/cuda/Embedding` (`ops.go:341` + `kernels.go:523`) — ЖИВАЯ** (production callers: gputrain, simpletrain, nn.LLM). Первая запись в подсписке живых. Стрелка delegate→direct отложена (dtype индексов расходится). Инвентарь: живые первыми в порт-приоритете — правило подтверждено.
2. **Прогноз atomicAdd-drift `sqrt(N)·eps` — недооценка при high-collision.** Реальный F32 drift 44× выше прогноза при 32 коллизий/строку. Для будущих grad-портов с атомиками использовать `O(eps·N)` верхнюю оценку. Правило в feedback-memory.
3. **PTX 6-правил** — на P3 катастрофных срабатываний не было (в отличие от P2 em-dash). Правила стали привычкой, не safety-net в этом раунде — но `logBuf` остаётся включённым (см. [[feedback-ptx-jit-log-diagnostic]]).
4. **backward Embedding — CPU-only в эталоне.** goml GPU `embedding_grad_f32` **отсутствует** в PTX ядрах goml. Наш GPU backward — единственный GPU-путь для Embedding в repo. Legacy nn.Embedding.Backward остаётся живым как CPU-fallback.

---

## СТОП по правилу ТЗ

P3-EMB пройден. По ТЗ: **P4-RoPE — следующим ТЗ после ревью**.

Готов принять решение по следующему шагу.
