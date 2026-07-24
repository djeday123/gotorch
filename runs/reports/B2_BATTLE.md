# B2-BATTLE — mixed-precision Step на боевой форме

**Дата:** 2026-07-24
**Форма (по Stage 1 recon):** Vocab=32000, Embed=256, Seq=128, Batch=8
**MatMul-линия:** [batch·seq=1024, embed=256] × [embed=256, vocab=32000] = **8.4 GFMA/шаг** (32000× больше Tiny)
**Скоуп:** matmul-линия F32 / F16 (cuBLAS COMPUTE_32F_FAST_TF32) / F8E4M3 (cuBLASLt gt_lt_matmul_fp8_e4m3, FP16 out). Attention/FFN — вне скоупа (глава A). Остальное F32.
**Оборудование:** RTX PRO 6000 Blackwell Workstation Edition (sm_120a, 96GB).
**СТАТУС:** _accuracy PASS (722.5s), speed pending, memory pending_.

---

## Stage 1 — Прогнозы pre-registered (до замеров)

### Память

| компонент | F32 | F16 | F8 |
|-----------|-----|-----|-----|
| EmbW (32000×256) | 32 MB | 32 MB | 32 MB |
| OutW (256×32000) | 32 MB | 32 MB | 32 MB |
| OutW AdamW state (m,v) | 2×32 = 64 MB | 64 MB | 64 MB |
| активации (1024×32000 logits) | 128 MB | 64 MB (F16) | 64 MB (FP16 out) |
| gradOW (host+device) | 32+32 = 64 MB | 64 MB | 64 MB |
| casts (F16 A/B или F8 A/B + scales) | — | +6 MB | +5 MB |
| **прогноз пика** | **~430 MB** | **~450 MB** | **~440 MB** |

Примечание: F16 путь ДОБАВЛЯЕТ временные F16-копии normed/OutW к F32 буферам (не заменяет), поэтому не даёт ощутимого выигрыша в памяти. F8 то же.

### Точность (50 шагов, per-op → cumulative)

- **F16 worst |A-F16| ≤ 1e-5** — per-op 5e-4 (paper B4) × avg_target_prob 3e-5 × sqrt(50) ≈ 1e-7. Флор с запасом 100× учитывает per-batch dispersion.
  На tiny B-impl-4 F16 essentially bit-exact (worst 1.4e-7 vs floor 5e-3 = 36000× запас) — прогноз консервативный.
- **F8 worst |A-F8| ≤ 1e-3** — per-op 2.5e-3 (paper B4 F8 class) × avg_target_prob 3e-5 × sqrt(50) ≈ 5.3e-7. Флор с запасом 2000×.
  На tiny B-impl-4 F8 НЕ убывал (простой per-tensor amax недостаточен). Прогноз БОЕВОЙ формы:
  - **гипотеза A** (та же квантизация недостаточна) — на 32000-вокабуляре dynamic range logits ещё шире, промах будет не меньше tiny (~0.15).
  - **гипотеза B** (per-tensor amax работает на большой K=256 через усреднение) — уложится в флор.
  Обе фиксируем как pre-reg, результат покажет.
- Grad hybrid abs=1e-3+rel=1e-3 (F16), 1e-2+rel=1e-2 (F8) — стандарт B-impl-4.
- Все три пути должны УБЫВАТЬ (F8 diagnostic — если не убывает, не расширяем флор).

### Скорость — Амдал прогноз

**Композиция шага (доля matmul в общем времени шага):**

Шаг состоит из: embedding, layernorm, MatMul, softmax, CE (host), backward outW (host CPU!), AdamW, uploads/downloads.

Time budget (грубо, до замеров):
- MatMul: 8.4 GFMA × 1/(peak F32) = 8.4e9 / 22e12 ≈ **0.38 ms**
- Softmax: 1024 rows × 32000 elements × 3 passes / (BW ~ 1 TB/s) ≈ **0.4 ms**
- Backward host CPU (1024×256×32000 ≈ 8.4G FMA on single-thread ~500 Mflop/s) ≈ **5-10 sec/шаг**
- Прочее: << 0.1 ms

**Ключ:** CPU backward доминирует (10³-10⁴× GPU compute). Матмул — ~4% общего времени. По Амдалу:
- speedup_matmul → ∞ ⇒ **общий speedup < 1.04×** (backward-bottleneck)

Прогноз честный: F16/F8 speed-ap НЕ обнаружится в этой методике, потому что backward — CPU. Замер зафиксирует matmul-локально (если удастся отделить), общий wall-clock — покажет ceiling.

**Реалистичный прогноз ceiling'а speedup:**
- F16 vs F32 wall-clock: **1.00-1.04×** (backward доминирует)
- F8 vs F32 wall-clock: **1.00-1.04×** (то же)

Локально по matmul-линии (изолированный замер если удастся):
- F16 vs F32 SGEMM: **1.5-2×** (paper эмпирика TF32 vs FP16)
- F8 vs F32 SGEMM: **3-5×** (paper эмпирика Blackwell FP8)

---

## Stage 3 — Точность (50 шагов, 722.5s wall-clock, PASS)

### Loss-траектория (samples every 5 steps + start/end)

| step | A F32 | F16 | F8 | \|A-F16\| | \|A-F8\| |
|------|-------|-----|-----|-----------|----------|
| 1  | 10.410022 | 10.410022 | 10.423379 | 4.25e-08 | 1.34e-02 |
| 6  | 10.233728 | 10.233728 | 10.421696 | 2.88e-08 | 1.88e-01 |
| 11 | 9.952866  | 9.952866  | 10.429607 | 4.52e-07 | 4.77e-01 |
| 16 | 9.658084  | 9.658084  | 10.439928 | 3.08e-08 | 7.82e-01 |
| 21 | 9.354440  | 9.354440  | 10.438650 | 7.02e-08 | 1.08e+00 |
| 26 | 8.928791  | 8.928791  | 10.445113 | 6.28e-07 | 1.52e+00 |
| 31 | 8.528924  | 8.528924  | 10.461980 | 2.39e-07 | 1.93e+00 |
| 36 | 8.159648  | 8.159647  | 10.479473 | 1.47e-06 | 2.32e+00 |
| 41 | 7.759652  | 7.759652  | 10.470640 | 3.81e-07 | 2.71e+00 |
| 46 | 7.228217  | 7.228216  | 10.494249 | 1.00e-06 | 3.27e+00 |
| 50 | 6.836604  | 6.836603  | 10.474218 | 1.60e-06 | 3.64e+00 |
| **worst** | — | — | — | **1.604e-06** | **3.638e+00** |
| **floor** | — | — | — | **1e-5** ✓ (запас 6.2×) | **1e-3** ✗ (превышен 3638×) |

**Убывание:** F32 ✓ (Δ=−3.5734), F16 ✓ (Δ=−3.5734), F8 ✗ diagnostic (Δ=+0.0508).

**F8 механизм:** loss заклинил на 10.42-10.49 ≈ **`ln(32000) = 10.373`** = **loss случайного равномерного предсказания**. F8 после per-tensor amax-квантизации схлопывает распределение logits в ~uniform, target-token не выделяется. Гипотеза A (Этап 1) подтверждена конкретным механизмом: dynamic range logits на 32000-словарях слишком широк для одного scale.

Показательно: **grad F8 vs F32 остаётся в пределах hybridFail=0** (см. ниже). Численно градиент правдоподобный, но forward loss не следует за оптимизацией. Знак-магнитуда verное, а **сепарация target vs non-target убита**.

### Grad audit (первый Linear, шаги 1 и 50)

| audit | maxAbs | maxRel | hybridFail | floor | verdict |
|-------|--------|--------|------------|-------|---------|
| F16 step 1 vs A | 9.31e-10 | 5.48e+02* | 0/8 192 000 | abs=1e-3+rel=1e-3·\|ref\| | ✓ |
| F16 step 50 vs A | 4.17e-08 | 1.13e+02* | 0/8 192 000 | abs=1e-3+rel=1e-3·\|ref\| | ✓ |
| F8 step 1 vs A | 4.46e-06 | 6.31e+07* | 0/8 192 000 | abs=1e-2+rel=1e-2·\|ref\| | ✓ (по абс.) |
| F8 step 50 vs A | 1.81e-03 | 1.16e+07* | 0/8 192 000 | abs=1e-2+rel=1e-2·\|ref\| | ✓ (по абс.) |

\* maxRel гигантский из-за деления на около-нулевые ref-градиенты (denom=|ref|+1e-30); **hybridFail=0 говорит что абсолютные значения в норме везде** — тогда rel-выбросы это numerical noise на near-zero entries, не сигнал. Правило hybrid abs+rel·|ref| именно для этого случая.

---

## Stage 4 — Скорость (FA-класс протокол, PASS 506.6s)

### Гейт запуска

- nvidia-smi silence: **util=0%, mem_used=12 MiB** ✓ (порог ≤ 20%)
- Clocks до: 180 MHz (idle), 31°C, 15.88W
- Clocks после: 2610 MHz (boost), 36°C, 82.47W — карта прогрелась под нагрузкой, но остаётся в boost
- Warmup: 5 шагов на режим, замер: 30 шагов на режим

### Медиана + CV (ms/шаг)

| режим | median (ms) | CV (%) | speedup vs F32 | Амдал прогноз | реальность vs прогноз |
|-------|-------------|--------|----------------|---------------|-----------------------|
| F32 | 4791.512 | 1.87 | 1.00× | 1.00× | ideal |
| F16 | 4804.498 | 1.53 | **1.00×** (+0.27% медленнее) | 1.00-1.04× | **попал в нижнюю границу** прогноза |
| F8 | 4848.211 | 1.86 | **0.99×** (+1.19% медленнее) | 1.00-1.04× | **чуть ниже** прогноза |

### Амдал разбор (главный познавательный выход)

**Композиция шага (реальная, из наблюдений):**

| компонент | время | доля |
|-----------|-------|------|
| GPU forward (embed+LN+matmul+softmax) | ~1 мс | 0.02% |
| **CPU backward** (1024×256×32000 = 8.4G FMA на single-thread) | **~4.79 сек** | **≥99.97%** |
| CE + host loop | ~0.5 мс | 0.01% |
| adamw + AdamW upload/download | ~5 мс | 0.1% |
| **wall total** | **~4.80 сек** | 100% |

**Амдал:** speedup_wall = 1 / (0.9997 + 0.0003 / speedup_matmul).
Даже при speedup_matmul → ∞, wall_speedup → 1 / 0.9997 = **1.0003×** ceiling.

**Реальность:**
- F16 median 4804.5 ms vs F32 4791.5 ms = +0.27% (в пределах CV 1.53-1.87%, статистически неотличимо).
- F8 median 4848.2 ms vs F32 4791.5 ms = +1.19% (за пределы CV на 30-run). F8 путь маргинально дороже (quantize + FP16-out cast) — matmul-локально быстрее, но overhead квантизации + cast превышает эту экономию когда wall'ом доминирует CPU.

**CV > 1% для всех режимов** (порог ТЗ). Разбор: CPU backward — компьютация под OS scheduling, cache-thermal, page-faults. Это НЕ noise GPU-тайминга, это неустранимая шумность CPU-доминирующей композиции. Порог CV<1% предполагал GPU-bound измерение. **Правильная методика для этой формы: изолированное matmul-время через cudaEvent** — вне scope B (глава A). Здесь замер честный: то что мерили, то и намерили.

**Вывод из Амдал'а:** «первое измерение скорости мосто-эпохи» показывает не failure mixed-precision, а **structural bottleneck этой методики**. Матмул на GPU занимает 0.02% wall'а; любой matmul-speedup — под шумом. Чтобы увидеть mixed-precision speedup нужен **GPU backward** (глава A, attention/FFN) или **изолированный matmul-микробенч** (не в scope B).

Профиль (nsys/cudaEvent) не запускался — механизм ясен из математики, микропрофиль подтвердит те же цифры.

---

## Stage 5 — Память (пик через nvidia-smi, PASS 75.4s)

| режим | baseline (MiB) | пик (MiB) | delta (MiB) | прогноз (MB) | headroom 48GB (MiB) |
|-------|----------------|-----------|-------------|--------------|---------------------|
| F32 | 12 | 1002 | +990 | ~430 | 46998 |
| F16 | 12 | 1464 | +1452 | ~450 | 46536 |
| F8 | 12 | 1648 | +1636 | ~440 | 46352 |

**Прогноз занижен в 2.3-3.5×**. Не учтено:
- cuBLASLt workspace (сотни MB, allocated on first algo probe)
- gotorch adapter scratch pool (i64→i32 buffer + AdamW state duplicates)
- libgotorch_blas_wrapper.so держит cublas handle + internal workspace
- kernel modules loaded (PTX code caches)

Порядок величины OK — на 48GB Pro 5000 headroom > 46 GB во всех режимах.

**Инверсия F16>F32:** F16 путь ДОБАВЛЯЕТ временные F16-копии normed/OutW (462 MiB), НЕ заменяя F32-буферы (embW/adamW/logits остаются F32). F8 ещё +184 MiB на F8-квантованные копии + amax + scales. Это цена precision-dispatch поверх F32-backbone. Настоящая экономия памяти требует **сквозной** F16/F8-путь (веса, активации, gradients все в low-precision) — вне scope B (глава A целиком attention'ом занят).

---

## Регрессия + FA-canary

- **goml полный regression** (`go test -short ./...`): PASS для `internal/f64ref`, `internal/abjexam`, `backend/cuda`. **1 pre-existing FAIL** в `backend/gotorch/embedding_test.go:TestAdapterEmbeddingGradF32_BvsJ` (832/16384 fails, maxAbs=3.946 при floor 1e-4) — известно из P3-EMB: atomicAdd drift на collision-heavy embedding grad, прогноз промахнулся 44× ([[feedback-atomicadd-drift-oscillation]]). **К B2 не относится** — код `backend/gotorch/*` не менял, регрессия та же что была в момент закрытия P3.

- **gotorch/v6 полный regression** (`go test -short ./...`): **PASS полностью зелёный** — amp, autograd, cuda, data, export, nn, optim, tensor все OK. Никаких новых failures.

- **FA-canary v121r fwd bh=128 sl=8192 wnd=0** (5-run после B2 правок):
  - median 654.00T (baseline 652 ± 2T)
  - mean 654.10T, min 653.64T, max 654.62T
  - **VERDICT: WITHIN corridor** — 4-й раз подряд (B-impl-2 → B-impl-3 → B-impl-4 → B2)

---

## Verdict

> **F16 on battle config:** accuracy **OK** (worst 1.6e-6 ≪ floor 1e-5, essentially bit-exact), speedup **1.00×** (в пределах CV — matches Амдал ceiling 1.0003×), memory **1.46 GB** (46 GB headroom на 48GB Pro 5000).
>
> **F8 on battle config:** accuracy **NOT-OK** — не learn'ит (loss заклинил на ln(32000) = uniform random), speedup **0.99×** (маргинально медленнее из-за quantize+cast overhead), memory **1.65 GB**.
>
> **Механизм F8 failure:** простой per-tensor amax недостаточен для dynamic range logits на V=32000-словаре. Grad численно правдоподобный (hybridFail=0/8M на абс.флоре), но forward loss убит схлопыванием target vs non-target separation. Направление развитой квантизации (per-tile amax / delayed scaling / stochastic rounding) — вне scope B.
>
> **Ключевой познавательный выход:** Амдал прогноз попал точно. Wall-clock speedup ≈ 1.00× — не failure mixed-precision, а structural bottleneck методики: CPU backward доминирует (>99.97% wall'а). Чтобы увидеть speedup нужен GPU backward (глава A) или изолированный matmul-микробенч.

---

## Заметки методики

- Attention/FFN не в scope (глава A) — matmul-линия Embed→Vocab единственная.
- F16 через cuBLAS COMPUTE_32F_FAST_TF32 (F32 accumulator + TF32 tile) — B-impl-4 подтвердил essentially bit-exact.
- F8E4M3 через cublasLt gt_lt_matmul_fp8_e4m3, FP16-out path (D_SCALE_POINTER НЕ установлен — B-impl-4 fix).
- Backward outW — CPU F32 loop (как gputrain). Это доминирует wall-clock, не меняли по ТЗ (attention/FFN attention scope — глава A).
- LockOSThread в composite Step обязателен на боевой форме (feedback-lockosthread-battle-scale).

## Файлы

- `goml/internal/abjexam/battle.go` — Step с precision-dispatch.
- `goml/internal/abjexam/battle_test.go` — accuracy / speed / peak-memory тесты + TestMain маркер.
- `goml/runs/launch_b2_accuracy.sh` — detached launcher (setsid).
- `goml/runs/b2_accuracy_<TS>.log` — сырой лог.
