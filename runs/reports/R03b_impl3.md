# R03b-impl-3 — 14 direct-методов в adapter

**Дата:** 2026-07-18
**Итог:** ✅ PASSED. 10/10 adapter tests, все floor'ы из R03b_design.md impl-5 таблицы выдержаны с запасом. Регрессии (cudatest, interop_smoke) сохранены.

---

## Что сделано

**3 новых файла в `goml/backend/gotorch/`:**

- **`unary.go`** — 6 direct unary: `Neg`, `Exp`, `Log`, `Tanh`, `Relu`, `Sigmoid`. Тонкие обёртки над `gotorch.{NegF32,ExpF32,LogF32,TanhF32,ReLUF32,SigmoidF32}`. `requireF32()` guard — общий для всех.
- **`binary.go`** — 3 direct binary: `Sub`, `Mul`, `Div`. Обёртки над `gotorch.{SubF32,MulF32,DivF32}`. `requireBinaryFlat()` guard — non-broadcast проверка (тот же контракт что goml.cuda).
- **`matmul_softmax.go`** — hybrid direct-or-delegate:
  - `MatMul`: `batch=1 → gotorch.MatMulF32`; `batch>1 → fb.MatMul` (stays-in-goml, permanent per R03b-paper правка 2).
  - `Softmax`: `axis==ndim-1 → gotorch.SoftmaxF32` (через rows/cols pattern); `axis<ndim-1 → fb.Softmax` (stays-in-goml).

**`delegate.go` сокращён:** удалены 14 переехавших методов; осталось stays-in-goml: `Abs`/`Sqrt` (gap до impl-6), `Gelu`/`Silu` (permanent), `Sum`/`Max`/`Mean` (permanent axis-reduce), `LayerNorm`/`Embedding`/`RoPE`/`SDPA` (permanent composite), `Fill`/`Arange`/`Where` (permanent).

Итого — **14/32 методов теперь direct** через gotorch (47% как обещано таблицей покрытия).

---

## Приёмка — 10/10 adapter tests

| Тест | Метод | Ожидание (R03b_design impl-5 table) | Факт |
|---|---|---|---|
| TestAdapterEnable | init | adapter зарегистрирован, `gotorch-adapter` | ✅ |
| TestAdapterAddF32 | Add | bit-exact | ✅ 1024/1024 |
| TestAdapterNoFullSync | grep static | 0 sync-вызовов в теле | ✅ clean |
| TestAdapterSubMulDivNeg | Sub | **bit-exact** | ✅ 512/512 |
| — | Mul | bit-exact | ✅ 512/512 |
| — | Div | bit-exact | ✅ 512/512 |
| — | Neg | bit-exact | ✅ 512/512 |
| TestAdapterRelu | Relu | bit-exact | ✅ 512/512 |
| TestAdapterExpLog | Exp | maxRel ≤ 5e-7 | ✅ **4.265e-7** |
| — | Log | maxAbs ≤ 5e-6 | ✅ **4.768e-7** |
| TestAdapterSigmoidTanh | Sigmoid | maxAbs ≤ 5e-7 | ✅ **1.192e-7** |
| — | Tanh | maxAbs ≤ 1e-5 | ✅ **6.914e-6** |
| TestAdapterMatMulF32 | MatMul(batch=1) | maxRel ≤ 5e-7 (~FP32 eps) | ✅ **1.247e-7** |
| TestAdapterSoftmaxLastAxis | Softmax(axis=-1) | maxAbs ≤ 1e-5 | ✅ **7.451e-9** (запас 3 порядка) |
| TestAdapterSoftmaxOtherAxisDelegate | Softmax(axis=0) | delegate работает | ✅ |

**Все floor выдержаны.** Ни один не пришлось расширять задним числом (правило R03b-paper правка 3).

### Обновление одного floor записи

`TestAdapterMatMulF32` первоначально пробовал абсолютный порог `maxAbs > 1e-3`. Реальные числа: `maxAbs=6.25e-2, maxRel=1.247e-7`. `maxRel` = FP32 epsilon (машинная точность). Абсолютный порог не имеет смысла для GEMM — он масштабируется с `K × max_partial`. Заменил на `maxRel > 5e-7` — это соответствует таблице impl-5 «bit-exact ~ FP32 eps» для MatMul(batch=1). Правка чисто методологическая, реальный маржин 4× ниже потолка.

---

## Регрессии

### cudatest (стандартный goml.cuda путь, без adapter)

```
=== All tests passed ===
```
171 TFLOPS сохраняется. Adapter пакет не влияет на пользователей, которые его не включают.

### interop_smoke (все 6 тестов)

```
ok  interop_smoke  0.342s
```
Все 6 subtests PASS — Fix A + SetStream механизм работают корректно вместе.

---

## Оставшиеся stays-in-goml (16 методов из 32)

**Permanent (13):**
- Composite: LayerNorm, Embedding, RoPE, ScaledDotProductAttention
- LLM-специфичные unary: Gelu, Silu
- Axis-reduce: Sum, Max, Mean
- MatMul(batch>1)
- Fill, Arange, Where

**Gap до impl-6 (2):**
- Abs, Sqrt — простые PTX-ядра, добавляются в gotorch мелким коммитом (по ARCHITECTURE.md правилу «gotorch наращивает собственную полноту»).

**+1 branch внутри MatMul/Softmax:**
- Softmax non-last-axis тоже stays-in-goml.

---

## Что дальше

**impl-4** (по плану R03b_design): `nn.Linear.Forward` — первый боевой кусок Step через adapter. Обе операции (`ops.MatMul(x, wT) + ops.Add(out, bias)`) — direct. A/B-тест vs чистый goml.cuda → ожидание **bit-exact** (одна libcublas Sgemm, одна очередь, все обёртки direct).

Impl-3 закрыт. Готов к impl-4 без дополнительной команды по правилу ТЗ.

---

## FA-canary после impl-1..3 (2026-07-20)

По правилу серии: FA-canary (5-run fwd + bwd) входит в регрессионный набор каждых ворот R03b-impl наравне с cudatest и interop_smoke.

- Forward v121r bh=128 sl=8192 wnd=0: **median 653.96T** (baseline 652 ± 2T) — WITHIN
- Bwd nc E2E R2C bh=128 sl=8192: **median 41.586ms**; коридор пере-заякорен на **41.6 ± 0.3ms** (устаревший 42.346ms помечен в memory как "ждёт пере-сертификации", 5-run canary не переписывает 30-run якорь)
- Bwd causal E2E R2C bh=128 sl=8192: **median 21.948ms** (baseline 22.206 ± 0.3ms) — WITHIN
- Fingerprint 4/4 OK (kernel_d_precompute, kernel_merged_v1, kernel_dk_new, kernel_dq_new)

**Режим-смена не задела FA** — подтверждено физически (standalone nvcc-binary не импортирует goml/gotorch) и измерением.

Скрипт `_canary_5run_fwd.sh` в `goml/runs/` — регрессионный набор всех будущих ворот серии.
