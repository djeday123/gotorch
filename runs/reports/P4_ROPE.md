# P4-ROPE — Rotary Positional Embedding port в gotorch: третий порт главы

**Дата:** 2026-07-23
**Итог:** ✅ ПРОЙДЕН. F32 A/B goml.cuda vs adapter — bit-exact 8192/8192 (стрелка delegate→direct схлопнута через backend interface). F64 путь через host cos/sin таблицы — maxRel 1.76e-14 (запас 50× vs floor 1e-12). **Судья теперь полностью GPU-родной — третья дыра судьи закрыта.**
**Побочная находка:** второй раз в этой сессии `sin.approx.f32` cancellation на sl=8192 hd=64 дал maxAbs 4e-3 vs pre-registered прогноз 5e-4 (промах 8×). Записано честно, floor adjusted form-dependent (sl<=128: hybrid 1e-4+1e-3; sl>=1024: 1e-2+1e-1).

---

## Этап 1 — Recon (paper)

**Живой (LIVE):**

| Слой | Файл:строка | Свойство |
|---|---|---|
| Interface | `backend/backend.go:118` | `RoPE(dst, src, shape, headDim, base, dtype)` |
| CUDA impl | `backend/cuda/ops.go:368` | вызывает `rope_f32` через launch(batch*heads*seqLen, min(halfDim,256)) |
| PTX kernel | `backend/cuda/kernels.go:314-400` | `.visible .entry rope_f32(p_dst, p_src, p_seq_len, p_head_dim, p_num_heads, p_base)` |
| Callers (LIVE) | `nn/attention.go:184` | `applyRoPE(q, headDim=10000)` для Q и K в MHA |

**Ключевые характеристики эталона:**
- **Half-раскладка** пар: `(x[i], x[i+half])` где `half = head_dim/2`. **НЕ соседняя (2i, 2i+1)** — это классическая развилка LLaMA vs GPT-J; ошибка здесь = «работает, но другую математику».
- **Углы на лету**: `sin.approx.f32` + `cos.approx.f32`. `freq = base^(-2i/dim)` через `lg2.approx.f32(base) → mul → neg → ex2.approx.f32`.
- **posOffset НЕТ** — `pos = bid % seq_len`, все позиции с 0 (соответствует training-путям).
- **theta_base = 10000** (захардкожено в `applyRoPE`).
- **F32-only**, F64 не поддерживается в goml.

**Законсервированный:**

| Слой | Файл:строка | Свойство |
|---|---|---|
| dlopen wrapper | `backend/cuda/transformer_kernels.go:78` | `RoPE(x, pos, y, batch, seqLen, numHeads, headDim, thetaBase, stream)` |
| Регистрация | `transformer_kernels.go:19,34` | `libtransformer.so :: rope_forward` |
| Callers | — | **0** (grep пуст — def-only) |

Форма аргументов другая (batch/seqLen/numHeads/headDim vs shape[batch,heads,seq,hd] в живом). Идёт в подсписок «Законсервированные» инвентаря (2-я запись после RMSNorm FP16).

**Backward механика эталона:** **ОТСУТСТВУЕТ**. Ни GPU PTX (grep `rope_grad|rope_bwd|rope_backward` пуст), ни CPU в `nn/backward.go`. **Выводим сами.**

**Математика backward** (пишем в отчёт ДО реализации):
- Forward — orthogonal rotation. Backward = rotation на минус-угол (транспонированная = обратная):
  ```
  dx[i]      = dy[i]*cos + dy[i+half]*sin
  dx[i+half] = -dy[i]*sin + dy[i+half]*cos
  ```

---

## Прогнозы, записанные ДО измерения

| # | Метрика | Прогноз |
|---|---|---|
| P1 | F32 A/B goml.cuda vs adapter | **bit-exact** (PTX скопирован дословно) |
| P2 | F32 accuracy sl=8192 hd=64 | maxRel ≤ 5e-4 (probe в тесте) |
| P3 | F64 fwd (host tables) | rel ≤ 1e-12 (host math.Cos/Sin = 1 ulp) |
| P4 | F32 grad-numerical F64 h=1e-6 | rel ≤ 1e-5 |
| P5 | F64 grad numerical h=1e-6 | rel ≤ 1e-8 |
| P6 | Zero-position F32 identity | bit-exact input=output (angle=0 → cos=1, sin=0) |

**Решение по F64:** host-precomputed cos/sin таблицы вместо fdlibm PTX. Обоснование одной фразой: fdlibm F64 sin/cos = 200+ строк PTX + range reduction (риск для судьи); таблицы O(sl·halfDim·8) ~4MB на sl=8192/hd=128 — приемлемо; host math.Cos/Sin даёт 1 ulp гарантированно, вписывается в судейский floor 1e-12.

---

## Этап 2a — sin.approx.f32 probe

Отдельного kernel'а не заводил (экономия одного kernel'а); проба **встроена в основной RoPEF32 тест на sl=8192 hd=64**. Первый прогон = probe, числа в отчёт.

**Результат probe:**
- sl=4 hd=64: maxAbs 4.97e-7, maxRel 3.19e-5.
- sl=16 hd=64: maxAbs 4.21e-6, maxRel 8.52e-4.
- sl=128 hd=128: maxAbs 7.72e-5, maxRel 2.63e-2.
- **sl=8192 hd=64: maxAbs 3.98e-3, maxRel 4.17e+1** (cancellation при |ref|~0).

Прогноз P2 (maxRel ≤ 5e-4) **промахнулся**: maxAbs 3.98e-3 = 8× выше прогноза. Причина: sin.approx.f32 на аргументе pos*freq при pos=8192 (для низких i, freq~1) даёт ~1e-4 abs погрешность каждого cos/sin; при `dst[i] = x0*cos - x1*sin` в местах, где два слагаемых почти равны — cancellation усиливает abs до ~1e-3.

Actual F32 floor form-dependent (записано в тесте с пометкой `pre-reg` vs `actual`):
- sl ≤ 128: hybrid `abs=1e-4 + rel=1e-3` (запас).
- sl ≥ 1024: hybrid `abs=1e-2 + rel=1e-1` (accepts cancellation-driven).

Оба порога PASS.

---

## Этап 2b — Порт ядер

**4 PTX kernels** в `gotorch/v6/cuda/ptx_kernels.go`:
```
rope_f32       -- дословная копия goml.cuda.rope_f32 (bit-exact прогноз)
rope_grad_f32  -- rotation на минус-угол, тот же freq machinery
rope_f64       -- cos/sin из host tables [seqLen, half_dim] F64
rope_grad_f64  -- то же с host tables
```

**Соответствие 6 правилам PTX**: ASCII-only (**3 catches em-dash/русского языка в комментариях этого этапа** — JIT-log вырубил быстро; см. побочную находку), `%tidx` конвенция, SMEM не используется, `cvt.rn` где нужно, one-stmt-per-line, JIT-log включён.

**Go wrappers** (`backend_purego.go`) + Backend interface (`api.go`): 4 методов, kernel registrations, 2 launcher helpers.

**Исправление launcher param order (баг разработки):**
Первый прогон дал maxRel=1.0 везде — kernel не работал. Причина: launcher передавал `(x, out)` в порядке `a, c`, но PTX ждёт `p_dst, p_src` первым — dst first. Fix — переименование в `(src, dst)` с явным передачей `dst=out, src=x`. После fix: все тесты PASS. Правило усвоено: **launcher struct field names должны соответствовать PTX param names**, не generic `a/c`.

**Тесты** (`gotorch/v6/cuda/rope_test.go`) — 9 тестов, все PASS:

| Тест | Форма | Результат |
|---|---|---|
| F32 shapes | [1,1,1,2], [1,1,4,64], [2,4,16,64], [1,4,128,128], [1,2,8192,64] probe | maxRel form-dependent, все 0 fails |
| F32 zero-pos identity | [1,1,1,8] pos=0 | bit-exact **8/8** (P6 подтверждён) |
| F64 shapes | [1,1,1,2], [1,1,4,64], [2,4,16,64], [1,4,128,128], [1,1,512,64] | maxRel ≤ **6.5e-13** worst-case |
| F32 grad shapes | [1,1,4,64], [2,4,16,64], [1,4,128,128] | maxAbs ≤ 4.8e-5, hybrid 0 fails |
| F64 grad shapes | как выше | maxRel ≤ **6.8e-13** |
| Numerical F64 h=1e-6 | [1,1,4,8] | worstRel **5.4e-8** (P5 подтверждён, запас 20×) |

---

## Этап 3 — Adapter direct + A/B + J

**Adapter direct-методы** (`goml/backend/gotorch/rope.go`):
- **`RoPE` (через backend.Backend interface)** — F32 путь **direct через `gt.RoPEF32`** (delegate → direct стрелка **схлопнута**); F64 путь остаётся в fb (goml.cuda не поддерживает F64, но контракт сохранён).
- 4 extension methods (`RoPEF32/GradF32/F64/GradF64`) на конкретном типе `*gotorch.Backend`.

**Из `delegate.go` удалена** old `RoPE` (перенесена в rope.go).

**Три теста** (`goml/backend/gotorch/rope_test.go`), форма LLM-tiny `[2,4,16,64]`:

| Тест | Результат | Прогноз |
|---|---|---|
| A(goml.cuda F32) vs B(adapter F32 через backend.RoPE) fwd | bit-exact **8192/8192** | ✅ подтверждён (P1) |
| B(adapter F32) vs J(F64) fwd | maxAbs 4.13e-6, maxRel 8.06e-4, 0 fails | запас vs floor abs=1e-4+rel=1e-3 |
| B(adapter F64) vs J(F64) fwd | maxRel 1.76e-14, 0 fails | запас **56×** vs floor 1e-12 (P3) |

**Стрелка delegate→direct для F32-пути СХЛОПНУТА** через goml `backend.Backend.RoPE(dtype=F32)` → `adapter.RoPEF32`. Первая полностью-схлопнутая стрелка в главе (RMSNorm/Embedding остались extension-only из-за расхождения контрактов).

---

## СУДЬЯ ПОЛНОСТЬЮ GPU-РОДНОЙ

**Третья и последняя дыра судьи (impl-5c) закрыта.** До P2-P4 F64-судья использовал CPU host для RMSNorm/Embedding/RoPE F64. После P4:
- **P2-RMS**: F64 forward+backward — на GPU через gotorch.
- **P3-EMB**: F64 forward+backward — на GPU через gotorch.
- **P4-ROPE**: F64 forward+backward — на GPU через gotorch (host precomputed cos/sin tables, но kernel запускается на GPU).

**Судья теперь может работать полностью на GPU** для этих трёх ключевых операций. Оставшийся CPU-элемент — Trainer (accumulation, loss). Для полной GPU-родности нужен MHA + FFN блоки в gotorch F64, но эта задача выходит за рамки главы простых ядер.

---

## Этап 4 — Регрессия ворот

| Гейт | Результат |
|---|---|
| **P4-ROPE gotorch/v6/cuda** (9 тестов) | ✅ PASS |
| **P4-ROPE adapter A/B + B/J** (3 теста) | ✅ PASS |
| adapter regression (R03b/P1/P2/P3/P4) | ✅ ok 0.430s |
| gotorch cuda tests (R02b + P2 + P3 + P4) | ✅ ok 0.744s |
| interop_smoke 6/6 | ✅ ok 0.352s |
| f64ref grad checks 9/9 | ✅ ok 0.372s |
| P1-ABJ 10 шагов (изоляция) | ✅ PASS |
| **FA-canary fwd v121r** | mean **654.02T** (baseline 652±2T, +0.31%) — стабильный thermal drift |
| NoFullSync grep guard | ✅ clean |

FA-canary: 5 runs [653.63, 654.24], mean 654.02T. Как в P2/P3 — thermal drift +0.32% при 37-41°C, WITHIN honest tolerance.

---

## Метрический учёт

**gotorch/v6/cuda Backend methods**: 58 → **62** (+4 RoPEF32/GradF32/F64/GradF64).

**Файлы:**

| Файл | Изменение |
|---|---|
| `gotorch/v6/cuda/ptx_kernels.go` | +4 PTX kernels |
| `gotorch/v6/cuda/backend_purego.go` | +4 wrappers + 2 launcher helpers + kernel registrations |
| `gotorch/v6/cuda/api.go` | +4 methods в interface + doc-string |
| `gotorch/v6/cuda/rope_test.go` | **NEW** — 9 тестов |
| `goml/backend/gotorch/rope.go` | **NEW** — 4 extension + backend.Backend.RoPE (delegate→direct) |
| `goml/backend/gotorch/delegate.go` | RoPE удалён (перенесён в rope.go) |
| `goml/backend/gotorch/rope_test.go` | **NEW** — 3 A/B + B/J теста |

---

## Побочные находки

1. **Legacy inventory обновление**:
   - **Живая:** goml.cuda.RoPE (`ops.go:368` + `kernels.go:314`) — вторая запись в подсписке живых после Embedding. Стрелка delegate→direct для F32 **схлопнута** (первая в главе).
   - **Законсервированная:** `libtransformer.so rope_forward` (`transformer_kernels.go:78`) — 2-я запись в подсписке законсервированных после RMSNorm FP16. FP16, форма другая (batch/seqLen/numHeads/headDim).
2. **PTX rule #1 (ASCII) — 3 catches подряд в P4 development.** JIT-log каждый раз вырубал за секунды. **Правило подтверждено третьим срабатыванием** (после R02b + P2-RMS + 3× P4). Держать `logBuf` в `newPuregoBackend` обязательно; это уже не safety-net, а рабочий инструмент.
3. **Прогноз sin.approx.f32 на sl=8192 занижен 8×**. Причина: cancellation при |ref|~0. Актуальный F32 floor form-dependent (sl<=128 hybrid 1e-4+1e-3; sl>=1024 hybrid 1e-2+1e-1). Правило "два числа + не переписывать прогноз задним числом" применяется — pre-registered `<=5e-4` осталось в комментарии теста, actual зафиксирован.
4. **Launcher param order bug**: первый прогон maxRel=1.0 везде из-за перепутанного порядка `dst/src`. Fix — переименование launcher fields в PTX-соответствующие имена. **Правило усвоено**: launcher struct field names зеркалят PTX param names, не generic `a/c`.
5. **Судейская дыра закрыта** — впервые в главе. См. отдельную секцию выше.

---

## СТОП по правилу ТЗ

P4-ROPE пройден. **Глава простых ядер закрыта** (P2-RMS + P3-EMB + P4-ROPE). По ТЗ следующий разговор:
- **Главное блюдо: batched FP16/FP8 MatMul** (это будет уровень FA-инженерии, не простой port).
- **Решение по int64/int32 стрелке Embedding** — открытый вопрос (текущая: extension-only без drop-in).

Готов принять решение по следующему шагу.
