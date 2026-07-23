# R03b — Финальная сводка серии goml ↔ gotorch интеграция

**Дата:** 2026-07-20
**Итог:** ✅ Серия R03b закрыта. Мост построен, валидирован, инфраструктура для будущих портов готова. **Открытая находка impl-5c** передаёт задачу «goml.LLM ↔ GPU совместимость» в главу портирования.

---

## Хроника — 8 этапов

| Этап | Что | Ворота |
|---|---|---|
| **R03b-paper** | Проектный документ 6 решений + таблица покрытия 32 методов (47% direct / 44% stays-in-goml / 6% gap) + план impl-1..5 | Doc |
| **R03b-impl-1** | Fix A: goml → primary CUDA context. `cuCtxCreate → cuDevicePrimaryCtxRetain` в `cuda..go:72`. Приёмка `identical:true handle 0x39400730` | ✅ |
| **R03b-impl-2** | Adapter package `goml/backend/gotorch/`. `SetStream` в gotorch. AddF32 первый end-to-end direct. TestAdapterNoFullSync grep-guard | ✅ 3/3 PASS |
| **R03b-impl-3** | 14 direct-методов adapter (Neg/Exp/Log/Tanh/Relu/Sigmoid/Sub/Mul/Div/MatMul batch=1/Softmax axis=-1) | ✅ 10/10 PASS, все floor выдержаны с запасом |
| **R03b-impl-4** | Linear.Forward A/B — обнаружено TF32 vs FP32 расхождение legacy | STOP → правки |
| **R03b-impl-4-final** | 4 сверки: [3.1] hybrid, [3.2] adapter **3648× точнее** legacy, [3.3] delegate bit-exact, **[3.4] TF32-vs-TF32 bit-exact — доказано скрытых багов раскладки НЕТ**. MatMulF32_TF32 в gotorch. FA canary новый anchor 41.6ms | ✅ 7/7 PASS |
| **R03b-impl-5a** | F64 foundation: LinearF64/LayerNormF64/EmbeddingF64/AdamWF64 + numerical grad + AdamW manual formula | ✅ 4/4 PASS с запасом 10-100× |
| **R03b-impl-5b** | F64 transformer: RoPEF64/AttentionF64/FFNF64/TransformerBlockF64 + dQ/dK/dV раздельно | ✅ 4/4 PASS |
| **R03b-impl-5c** | LLMF64 sanity ✅. **Полный ABJ blocked gap `goml.LLM ↔ GPU`** — веса на CPU, training на CPU в текущей версии goml | ✅ partial + finding |

---

## Ключевые артефакты

### Код

- **gotorch:** `SetStream` hook + `MatMulF32_TF32` (impl-2, impl-4-final).
- **goml:** Fix A (impl-1) + `Stream()` accessor + пакет `backend/gotorch/` adapter (impl-2..4-final).
- **f64ref:** `goml/internal/f64ref/` — полный F64 stack (~1300 строк), 18 grad checks PASS.

### Документы

- `R03b_design.md` — проектный (6 решений, таблица покрытия, impl-план)
- `R03b_impl1.md` — Fix A
- `R03b_impl2.md` — adapter + SetStream
- `R03b_impl3.md` — 14 direct + FA canary
- `R03b_impl4.md` — 4 сверки TF32-vs-FP32 разложено
- `R03b_impl5a.md` — F64 фундамент
- `R03b_impl5b.md` — F64 трансформер
- `R03b_impl5c.md` — LLMF64 sanity + критическая находка
- **R03b_final.md** (этот) — сводка
- `ARCHITECTURE.md` (gotorch + goml) — целевая картина стека + правила переходного периода

---

## Регрессионный набор (5/5 PASS)

- adapter tests 10/10
- gotorch cuda (impl-3 + TF32 + rollback hygiene)
- interop_smoke 6/6
- goml cudatest 171 TFLOPS
- FA canary fwd 653.86T + bwd nc 41.561ms (новый anchor 41.6±0.3ms)
- f64ref grad checks 9/9

---

## Ключевые находки серии

1. **UVA cross-context прозрачен** на sm_120/sm_89, MPS on/off (R03a матрица 4/4).
2. **goml legacy cublas handle = TF32**, gotorch = pedantic FP32 (impl-4-final). Adapter direct-путь **3648.7× точнее** legacy на MatMul.
3. **Скрытых багов раскладки нет** — доказано bit-exact 256/256 через MatMulF32_TF32 сверкой 3.4.
4. **`ARCHITECTURE.md` правило**: gotorch = единственный целевой источник истины; goml/backend/cuda = read-only эталон переходного периода.
5. **Sync-race правило для тестов**: D2H после kernel'а через `b.fb.Sync()` (goml stream); default stream не гарантирует ordering.
6. **FA-canary в регрессионном наборе** каждых ворот (принцип «каждая правка контекста/stream проверяется на самом дорогом активе»).
7. **[Критично] goml.LLM работает на CPU в текущей версии** — `NewLinear`/`NewEmbedding` игнорируют device параметр, FromSlice всегда даёт CPU-tensor. Мост через GPU не задействован в training loop.

---

## Незакрытая задача (для главы портирования)

**Полный ABJ 10-step trainer** требует goml.LLM GPU-совместимости — минимум `NewLinear`/`NewEmbedding` должны кладить weights на device через `FromSlice → ToDevice(device)` или через прямой Zeros/Alloc в device backend. Это **~200 строк правок в goml.nn** ИЛИ **правильное решение — портирование `nn.Linear`/`nn.Embedding` в gotorch** (по плану ARCHITECTURE.md: F32 покрытие в gotorch). После порта ABJ становится тривиальным (F32 на gotorch, F64 через f64ref).

**Плюс:** судья F64 (f64ref) полностью готов и ждёт этого порта. Он универсально применим ко всем будущим port'ам ядер goml→gotorch.

---

## После серии — следующий этап

**Глава портирования** (по ARCHITECTURE.md):
- Порт `nn.Linear`, `nn.Embedding` в gotorch как GPU-first слои (fix ABJ blocker одновременно).
- Порт LayerNorm, RoPE, SDPA в gotorch с A/B через adapter vs goml.cuda vs f64ref.
- Порт fused-AdamW в gotorch (currently host-loop CPU).
- Каждый порт: (1) port goml-эталона в gotorch, (2) numerical vs f64ref, (3) A/B через мост, (4) переключение goml.trainer.

Приоритеты — обсуждение с пользователем перед стартом.

---

## R03b объявляется закрытым.

- 8 этапов, ~2500 строк нового кода, 9 отчётов, 3 репо push'ей (goml + gotorch + interop_smoke).
- Мост работает, документирован, покрыт тестами.
- Судья F64 готов к применению в главе портирования.
- Финдинг про goml.LLM ↔ GPU gap — ценный вход в следующую фазу.
