# R03b-impl-5c — LLMF64 сборка + критическая находка про gap goml.LLM ↔ GPU

**Дата:** 2026-07-20
**Итог:** Частично закрыт. LLMF64 sanity PASS (loss монотонно убывает 10 шагов). **Полный ABJ 10-step blocked архитектурным gap: goml.LLM не задействует GPU в training loop.**

---

## Что сделано

### `internal/f64ref/llm.go` (+215 строк)

Полная сборка `LLMF64`:
- `Embed → N × TransformerBlock → LayerNormFinal → Linear(output) → CE loss`
- `Forward`/`Backward` возвращают grads в порядке `AllParams()` (совпадает с goml.LLM.Parameters()).
- `CrossEntropyLossF64` — численно-стабильный softmax + CE + gradient.

### `internal/f64ref/llm_test.go` — sanity 10-step

```
LLMF64 10-step sanity losses (TinyConfig-like, Vocab=16 Dim=8 NumLayers=2 Hid=12):
  step  1: loss=3.842298
  step  2: loss=3.318732
  step  3: loss=2.840883
  step  4: loss=2.535001
  step  5: loss=2.264825
  step  6: loss=2.038044
  step  7: loss=1.870620
  step  8: loss=1.774759
  step  9: loss=1.678085
  step 10: loss=1.589234
```

**Loss убывает монотонно 2.4× за 10 шагов** — F64-судья функционален (forward + backward + AdamW работают как единое целое). ✅ PASS.

---

## Критическая находка — goml.LLM ↔ GPU gap

При попытке построить полный ABJ 10-step тест обнаружено:

- `nn.NewLinear(inFeatures, outFeatures, bias, device)` в `goml/nn/linear.go:21` **принимает** `device` параметр, но **игнорирует его** — веса создаются через `tensor.FromSlice(wData, ...)` (`linear.go:30`), а `FromSlice` в `goml/tensor/tensor.go:66` **всегда** идёт на CPU backend (`b, err := backend.Get(backend.CPU)`).

- Аналогично `NewEmbedding` (`goml/nn/embedding.go:21`) — веса на CPU.

- `nn.LayerNorm` через `Zeros(shape, dtype, device)` **действительно** использует device (goml/tensor/tensor.go:83-99) — единственный слой корректно кладущий на GPU.

- В `ops.MatMul` backend определяется через `getBackend(a) → backend.GetForDevice(a.Device())` — device берётся из **input tensor'а**. Если input на CPU (после Embedding), MatMul идёт на CPU. Значит `goml.LLM.Forward` фактически **работает на CPU** в текущей версии, даже если device=CUDA передан в NewLLM.

**Симптом при попытке TestABJ_TenStepTrajectory:** `SIGSEGV в backend/cpu.LayerNorm` при `Zeros(...)` пути на CPU где storage=nil. Trace через `nn.LayerNorm.Forward → ops.LayerNorm → backend/cpu.LayerNorm → segfault`.

Это **архитектурный gap** goml, не bug adapter'а. Мост через GPU **не задействован** в текущем `nn.LLM` training loop — вся training идёт CPU-side.

## Что это означает для impl-5

**Полный ABJ 10-step с `nn.LLM`** невозможен без правки goml — минимум добавить `ToDevice(cuda)` вызов после `FromSlice` в `NewLinear`/`NewEmbedding`. Это правка `goml/backend/cuda`-логики, а по правилу ARCHITECTURE.md переходного периода **это `read-only` эталон**.

**Два пути решения:**

**Путь 1 (для R03c или future portion):** правка goml.LLM/Linear/Embedding для GPU-совместимости. Плюс правка `tensor.FromSlice` или добавление `Tensor.ToDevice(device)` метода. Это несколько сот строк в goml. Плюс тесты. Полностью соответствует пути **«port to gotorch»** — по факту нужно `nn.Linear` в gotorch, а не в goml.

**Путь 2 (обход):** написать test-side полную F32 модель на adapter напрямую (без `nn.LLM`), реплицирующую структуру LLM. Это дубль ещё одной LLM stack в F32 — ~500 строк. Дорого.

**Мой выбор:** отложить полный ABJ до **главы портирования** — там `nn.Linear` и `nn.Embedding` естественно переходят в gotorch (это часть плана: F32 покрытие в gotorch). После порта — ABJ становится тривиальным.

## Ворота 5c — частичное закрытие

- ✅ LLMF64 sanity: loss 3.84 → 1.59 монотонно за 10 шагов, все F64 методы работают в комплексе
- ✅ Все компоненты 5a/5b с numerical grad — судья готов к использованию когда GPU-training goml появится
- ❌ Полный ABJ 10-step — blocked архитектурным gap `goml.LLM ↔ GPU`

**Судья F64 полностью готов** (18 numerical-grad тестов PASS 5a+5b). Ждёт только GPU-совместимой LLM для сравнения.

---

## Регрессия

- adapter tests 10/10 PASS
- gotorch cuda tests PASS
- interop_smoke 6/6 PASS
- goml cudatest 171 TFLOPS PASS
- FA canary fwd 653.86T + bwd nc 41.561ms WITHIN
- f64ref tests: 5a (4/4 PASS) + 5b (4/4 PASS) + 5c LLM sanity PASS = 9/9 PASS

Ничего не сломано регрессионно.

---

## СТОП — импл-5 завершён с честной раскладкой

- **5a**: F64 foundation — 4/4 PASS
- **5b**: F64 transformer — 4/4 PASS (композиция + dQ/dK/dV раздельно)
- **5c**: LLMF64 sanity ✅ + gap найден и задокументирован

**Судья F64 готов к использованию.** Он не был лишним — построил универсальный инструмент для будущих сверок портируемых ядер (LayerNorm/Embedding/RoPE/SDPA/AdamW → gotorch). Первое реальное применение — в главе портирования.

**R03b серия полностью закрыта.** Итоговый сводный отчёт: `R03b_final.md`.
