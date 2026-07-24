# A-BRIDGE — Дизайн-бумага A1 (attention через мост goml↔gotorch)

**Дата:** 2026-07-24
**Статус:** paper, СТОП после — ревью до кода.
**Основание:** ТЗ A-BRIDGE (после закрытия B2).
**Разведка:** Explore-агент, 98 tool calls, 422s.

---

## Резюме — где ТЗ разошлось с фактами

ТЗ предполагает: «Attention сейчас на CPU и жирнейшая часть CPU-backward глыбы. Провести существующие FA-ядра через мост, заменить CPU-путь в боевом Step».

Разведка выявила **три критических разрыва** с этой посылкой:

**Р1. Боевого GPU-Step с attention НЕ существует.**
- `goml/main.go` — единственный call-site nn.LLM (полная attention-модель); девайс **CPU**, форма TinyConfig (Batch=1, Seq=32, NumHeads=4, HeadDim=16, NumLayers=2).
- `goml/train/trainer.go` — тоже CPU-путь; вызывает `model.ForwardWithCache/Backward` на CPU-backend.
- `goml/cmd/gputrain` — GPU smoke test, без attention (только embed+LN+linear+softmax).
- `goml/internal/abjexam/battle.go` (B2-BATTLE Step) — matmul-only, комментарий *«без attention/FFN»*.
- `cmd/simpletrain` использует `SimpleLM` без attention.

**⇒ Нет production-Step где attention работает на GPU — есть только два раздельных мира.** Задача A — построить (или адаптировать) такой Step, а не заменить существующий.

**Р2. B2's «CPU-backward 99.97% wall'а» — это НЕ attention.**
Это единственный host-цикл `gradOW = normed^T @ gradLogits` в `battle.go:238-247` (output-projection matmul, 1024×256×32000). Attention'ной глыбы в B2 нет — attention там отсутствует по scope. Значит **Амдал для главы A надо считать заново** на настоящей LLM.Backward-композиции, не переносить B2's число.

**Р3. FA-ядра готовы, но не end-to-end.**
- Shipped `libfa_sm120.so` (v121r fwd, 652T): **hd ∈ {64,128}**, **без LSE-выхода** в C-ABI. LSE-эмиттер существует в исходнике (`_v121r_train_kernel.cu`, `fa96b_train_kernel`), но не подключён к сборке (`libs/fa_sm120/Makefile` не включает train_kernel в SRCS_CU).
- Backward (v0.2.0 260T fused, `release_v0.2.0/src/`): **никакого `.so` вообще нет**, `extern "C"` тоже нет. Только namespaced C++ launcher'ы (`fa_bwd_merged_v1::launch_merged` и т.п.). **`hd=128` only.**
- **hd=32 (SmallConfig в nn/model.go)** не поддерживается ни fwd, ни bwd.

⇒ ТЗ спрашивает архитектурное решение, но перед ним стоят **инфраструктурные gate'ы**: (i) собрать LSE-fwd, (ii) собрать backward `.so`, (iii) выбрать целевую форму под hd∈{64,128}.

---

## 1. Карта attention-линии (факты)

### 1a. Единственная реальная точка вызова — CPU-путь main.go

**Forward** `nn/attention.go:60-160` — `MultiHeadAttention.Forward`:
- Layout: `[B,S,D]` после Wq/Wk/Wv → reshape `[B,S,H,hd]` → transpose `[0,2,1,3]` → **`[B,H,S,hd]` (BHSD)**, contiguous.
- RoPE (line 108-115) → `backend.RoPE`.
- Attention: `ops.ScaledDotProductAttention(q, k, v, numHeads, causal)` (line 132) → `backend.ScaledDotProductAttention`.
- Dispatch: CPU (`goml/backend/cpu/cpu.go:394`) или CUDA composed (`backend/cuda/ops.go:411-548`, cublasSgemm×2 + softmax kernel, F32, causal via -1e9-fill).
- Adapter в gotorch — delegate в fb (`backend/gotorch/delegate.go:56-58`), т.е. в живом коде attention даже через adapter уходит в наивную CPU/CUDA composed реализацию, никаких FA-ядер.

**Backward** `nn/backward_attn.go:19-246` — единственный кешированный backward-путь, вызывается из `LLM.Backward`. **Полностью Go host loops на `[]float32`:**
- Recompute (lines 131-146): `4·S²·hd` FMAs per head.
- dV loop (169-177): `S²·hd` FMAs.
- dScores = dO·V^T (181-189): `S²·hd`.
- Softmax-grad (192-200): `S²`.
- dQ = dPre·K (203-211): `S²·hd`.
- dK = dPre^T·Q (214-222): `S²·hd`.
- **Итого per attention block per step**: `≈ 5·B·H·S²·hd` FMAs.
- В том же файле `LLM.Backward` делает CPU embedding backward через `.ToFloat32Slice()` (lines 386-398).

### 1b. Формы

- **main.go TinyConfig**: `B=1, S=32, H=4, hd=16, L=2`. Attention per step: `5·1·4·32²·16 = 328 K FMAs × 2 = 655 K` — ничтожно.
- **SmallConfig** (`nn/model.go:22-34`): `H=8, hd=32, L=6, D=256, MaxSeq=512`. **`hd=32` FA-ядрами НЕ поддерживается.**
- **B2-battle** (уже поднят): `V=32000, D=256, S=128, B=8` — но без attention.
- **FA канонические**: `bh=128, sl=8192, hd=128` для v121r fwd (652T) и bwd fused (260T).

**⇒ Существующие формы (Tiny/Small) НЕ бьются с поддерживаемыми формами FA.** Нужен новый config под A-путь.

---

## 2. Контракт FA-ядер (факты из .cu/.h файлов)

### 2a. Forward — libfa_sm120 (v0.1.0, `goml/libs/fa_sm120/`)

C-ABI из `include/fa_sm120.h`:
```c
fa_status_t fa_forward(fa_ctx_t* ctx,
    const void* q, const void* k, const void* v,  // FP8 e4m3 (uint8_t*), [BH, S, HD]
    void* o,                                       // FP16 out,           [BH, S, HD]
    int batch_heads, int seq_len, int head_dim,
    int causal, int window,
    float scale, fa_stream_t stream);
```

**Ограничения** (кодовые проверки):
- `hd ∈ {64, 128}` (`fa_sm120.h:14, 62`, `fa_ctx.cu:120-135`)
- `bh > 0, sl > 0, window ∈ [0, sl]`
- Layout **`[BH, S, HD]`** row-major, contiguous. Batch и heads collapsed в `BH = B*H`.
- Dispatcher (`fa_ctx.cu:143-155`): только `FA_KERNEL_V121R` wired; остальные IDs возвращают `FA_ERR_INTERNAL "not yet linked (SHIP-2)"`.
- **LSE НЕ отдаётся текущим C-ABI.** Train-kernel с LSE (`_v121r_train_kernel.cu`, `fa96b_train_kernel`, `float* L_out [bh, sl]`) существует, но не в SRCS_CU (`Makefile:20`).

### 2b. Backward — release_v0.2.0/src (260T fused)

**`extern "C"` НЕ существует**, все entry points C++ namespaced:

```cpp
launch_d_precompute(const __half* O, const __half* dO, float* D,
                    int bh, int sl, int hd, cudaStream_t s);   // fa_bwd_dk.cu:522

launch_merged(const uint8_t* Q, K, V, const __half* dO,
              const float* L, const float* D,                    // ← L из fwd, D из d_precompute
              uint8_t* dS_nat, uint8_t* dS_T,                    // dual-write, stride_ds padded
              float* dV,                                          // FP32 [bh, sl, hd]
              int bh, int sl, int hd,
              int causal, int window, float scale, cudaStream_t s);   // fa_bwd_merged_v1.cu:513

launch_dk_new(const uint8_t* Q, const uint8_t* dS_T, float* dK,
              int bh, int sl, int hd, ...);                       // fa_bwd_dk_new.cu:306

launch_dq_new(const uint8_t* K, const uint8_t* dS_nat, float* dQ,
              int bh, int sl, int hd, ...);                       // fa_bwd_dq_new.cu:338
```

**Каноническая R2C-цепочка** (`release_v0.2.0/README.md`, `docs/cert/cert_summary.md`):
```
d_precompute → merged_v1 (ds_gen+dV) → dk_new → dq_new
  0.34 ms      25.12 ms                 8.42 ms   8.46 ms    (total ≈ 42.3 ms bwd nc, bh=128 sl=8192 hd=128)
```

**Ограничения**:
- **hd=128 only** (`fprintf/exit(1)` в четырёх launcher'ах, `fa_bwd_dk_new.cu:306-326` и т.д.).
- sm_120a only.
- Dtypes: Q/K/V = FP8 e4m3 (uint8_t*); dO = FP16 (`__half*`); dQ/dK/dV = FP32; L, D = FP32 `[bh, sl]`; dS_nat/dS_T = FP8 stride_ds-padded.
- **`stride_ds = (sl + 15) & ~15`** — обязательная padded stride для dS буферов (multiple файлов подтверждают).
- **Dual-write dS**: merged пишет `dS_nat` для dq_new и `dS_T` для dk_new одновременно.

**Caller обязан:**
- Alloc & zero-init `dQ, dK, dV` — FP32, `bh*sl*hd*4` each.
- Alloc `dS_nat, dS_T` — FP8, `bh*sl*stride_ds` each.
- Alloc `D` — FP32, `bh*sl*4`.
- Обеспечить `L` из fwd (`bh*sl*4`).

### 2c. Legacy wired-но-не-используемое

- `goml/backend/cuda/flash_attention_backward.go` — purego binding для `libflash_attention_v58_backward.so` (FP16, v58 127T, correctness <0.0014 vs FP32 CPU ref). **Ноль вызовов** во всём проекте.
- `goml/libs/fa_sm120/go/gofa/gofa.go` — cgo binding для `libfa_sm120.so` fwd. **Ноль вызовов**. Свой Go-модуль (`github.com/djeday123/fa-blackwell-fp8/go/gofa`), потребует `replace` в go.mod.

---

## 3. Где живут ядра и как прокидываются

- **Shipped `.so` в `goml/libs/`**: `libfa_sm120.so` (fwd 652T, v121r only), + legacy `libflash_attention_v54..v58_backward.so`.
- **v0.2.0 bwd source** — не собран, только tests/benchmarks (`release_v0.2.0/Makefile:14-38`).
- **gotorch/v6/ARCHITECTURE.md:13, 67**: *«fa-blackwell-fp8 — отдельная библиотека… НЕ часть gotorch, специфична по архитектуре и dtype-миксу; подключается опционально когда нужен FA-путь.»* Grep по `Attention|attn|flash|FA_` в gotorch/v6/ подтверждает: attention там наивный CPU MHA (`nn/attention.go:11-34`), никаких FA-биндингов.

**⇒ FA-ядра сейчас — «арсенал в подвале»: ready, но никакой продовый Go-код их не зовёт.**

---

## 4. Новая Амдал'а (без замеров, порядки величин)

B2's 4.79s/step CPU-backward = только gradOW (Embed→Vocab matmul). Для LLM.Backward нужен другой раскладной.

**Гипотетический BattleAConfig (hd=128, чтобы попасть в FA-регион):**
- `B=4, S=128, D=256, H=8, hd=32` — ПРОБЛЕМА hd=32 (не FA).
- Альтернатива: `B=4, S=128, D=1024, H=8, hd=128` — попадаем в FA-регион, но модель раздувается.
- Альтернатива: `B=4, S=512, D=512, H=8, hd=64` — на границе FA (hd=64 поддерживается только fwd v121r, backward hd=128 only).

**Только `hd=128` даёт полную (fwd+bwd) FA-цепочку.** ⇒ конфигу для A нужен `hd=128`.

**Для BattleAConfig `B=4, S=128, D=1024, H=8, hd=128, L=6`:**

Per step FMAs (single-thread ~500 Mflop/s CPU):
- Attention grad: `5·4·8·128²·128 = 168 M × 6 = 1.0 G` (`≈ 2 sec` CPU)
- Wq/Wk/Wv/Wo linear grads: `5·4·128·1024² = 2.7 G × 6 = 16 G` (`≈ 32 sec` CPU — ой)
- FFN grad (D=1024, FFN=4·D=4096): `~2·4·128·1024·4096 = 4.3 G × 6 = 26 G` (`≈ 52 sec` CPU)
- Output projection (grad_Wout): `~4·128·1024·32000 = 17 G` (`≈ 34 sec`)
- Итого ≈ 120 sec/step CPU

**Amdahl если только attention уезжает на GPU:**
- До A: attention 2 s / total 120 s = 1.6%
- Wall speedup ceiling = 1/(1 - 0.016) = **1.016× (то же самое что F16 speedup в B2 — сотые доли процента)**

**⇒ Attention НЕ доминирует в CPU-backward'е на этих формах.** Матмулы линеек и FFN значительно жирнее. **Правильный порядок работы после A может быть: сначала linear-backward-GPU, потом attention.**

**Уточнение критической важности:** *если* `S` увеличивается сильно (S²·hd в attention vs S·D² в linear), attention начинает доминировать. Критическая точка: `S²·hd > S·D²` ⇒ `S > D²/hd`. Для `D=1024, hd=128`: `S > 8192`. То есть attention доминирует только на **длинных контекстах**.

**⇒ Для короткого-контекста трена ( `S < ~1K` ) выгоднее сначала linear-backward.**

**Замер обязателен для честной приоритизации.** Recon-агент не смог верифицировать raw ratio без реального замера LLM.Backward (грепом не нашёл per-op timing'ов).

---

## 5. Три архитектурных варианта слоя стыковки

### Вариант (a) — extension methods на `*gotorch.Backend`

Паттерн повторяет B-impl-4 (MatMulF16/F8 через `libgotorch_blas_wrapper.so`).

Плюсы:
- Знакомая архитектура (B-impl-2/3/4 доказали работоспособность).
- Unified с mixed-precision infra.

Минусы:
- **Ломает `gotorch/v6/ARCHITECTURE.md:13, 67`** — правило «fa-blackwell-fp8 opt-in, не часть gotorch».
- gotorch/v6/cuda.Backend получает Blackwell+FP8-специфичный attention API, который не переносится на другие backends.

Работы:
1. Собрать LSE-fwd + backward `.so` (см. Q2, Q3 ниже — инфраструктурные gate'ы).
2. Написать `libgotorch_fa_wrapper.so` (или dlopen'ить libfa_sm120.so прямо).
3. Добавить `AttentionFwdF8`, `AttentionBwdF8` в gotorch/v6/cuda API.
4. Adapter extension в goml/backend/gotorch/attention_fa.go.
5. Wiring через nn/attention.go MultiHeadAttention.Forward/Backward.

Сложность: **high** (3 репо: goml/libs/fa_sm120, gotorch/v6/cuda, goml/backend/gotorch).

### Вариант (b) — прямой goml purego binding + бридж-дисциплина

Минимальный путь: purego bindings в `goml/backend/cuda/fa_forward.go` + `fa_backward.go` поверх собранных `.so`, вызов из `nn/attention.go` через backend-агностичный интерфейс.

Плюсы:
- **Не ломает ARCHITECTURE gotorch/v6.**
- Быстрее к работающему end-to-end пути.
- FA живёт в goml (свой мир), gotorch/v6 остаётся FA-agnostic.

Минусы:
- Не unified — при следующем FA-родственнике (H100 sm_90, sm_100) та же работа повторится.
- Ручная бридж-дисциплина (LockOSThread, injected stream) руками — рядом с существующим MP-путём через adapter асимметрично.

Работы:
1. Собрать LSE-fwd + backward `.so` (те же gate'ы).
2. Purego bindings ~200 строк.
3. `backend.ScaledDotProductAttention` overload или новая интерфейс-функция `backend.AttentionFA_FP8`.
4. Wiring через nn/attention.go.

Сложность: **medium** (1 репо: goml).

### Вариант (c) — Гибрид: gotorch с явным FA-shim

gotorch/v6 получает `Attention*` API в интерфейсе, но реализация выделена в отдельный shim который dlopen'ит `libfa_sm120.so` opt-in. Fallback CPU для случая когда `.so` отсутствует.

Плюсы:
- **Соблюдает ARCHITECTURE через runtime opt-in.**
- Unified для будущих FA-backends.

Минусы:
- Больше кода (два уровня dispatch).
- Runtime-check лишний слой.

Сложность: **high** (архитектурно чище, но не быстрее).

### Решение пользователя — FA integration ladder (2026-07-24)

Не одиночный вариант, а **лестница из трёх ступеней со сверкой на каждой** — доказательство архитектуры двух книг тремя побитовыми сверками одного вычисления.

**Ступень 1 (= вся глава А целиком) — FA через существующие goml-биндинги в бридж-дисциплине.**
- Purego binding для `libfa_sm120.so` (+ будущий `libfa_bwd_sm120.so`) в `goml/backend/cuda/`.
- Мостовая дисциплина: один stream (fb.Stream()), LockOSThread, Sync-границы, canary до/после.
- **Здесь живёт вся тяжёлая сверка**: F64-судья, A/B/J-трасса, correctness gate 11 форм, 50-шаг траектория. Основная научная работа главы.
- Соответствует варианту (b) выше по локации кода (goml/backend/cuda), но с явным акцентом что это **не финальная архитектура** — стартовая ступень.

**Ступень 2 (после А, отдельный короткий этап) — переезд биндингов в fa-blackwell-fp8/gofa.**
- Train-вариант вызова: `forward-with-L` + `backward chain` — в `fa-blackwell-fp8/go/gofa/` (их архитектурное место по [gotorch v6 ARCHITECTURE](../../../gotorch/v6/ARCHITECTURE.md:13,67)).
- goml переключается на `import fa-blackwell-fp8/gofa` через `replace` в `go.mod`.
- **Сверка ступени 2**: тот же `.so` под капотом → прогон обязан быть **bit-exact со ступенью 1**. Ошибки при этой сверке = баги упаковки (Go-обёртки, cgo/purego дисциплина), не вычислений. Ступень короткая по сути — это переезд контракта.

**Ступень 3 (после 2) — gotorch объявляет gofa опциональной зависимостью, attention-слой nn-уровня.**
- В gotorch появляется тонкий `nn.FlashAttention` слой (композиция `gotorch.Tensor` + `gofa`-вызов, паттерн как `torch + flash-attn` в Python).
- **НЕ копирование обёрток в gotorch** — зависимость, границы библиотек сохранены. Именно это «границы двух книг» о котором ARCHITECTURE говорит.
- **Сверка ступени 3**: bit-exact со ступенью 2 (та же цепочка вызовов, другой upper layer).

**Выход лестницы:** «человек завтра» получает **два независимых входа** — (i) `gofa` напрямую для тех кто уже работает с FA-контрактом, (ii) `gotorch.nn.FlashAttention` для composed-путей. Оба доказаны быть эквивалентными через две побитовые сверки одного вычисления.

**Расписание работ переиначивается:**
- Глава A = ступень 1: всё что ниже в §7-§9.
- Пост-A ступени 2 и 3 — отдельные короткие главы (по разговору), не в scope A1-A9.

Отдельно: этап **A-0** (перед A-2 = LSE-forward) — GPU matmul-backward для gradOW и связанные, реальный первый ход по backward-ускорению. См. §12.

---

## 6. Прогнозы ДО кода (pre-registered)

### 6a. Accuracy floor

FA v0.2.0 сертифицированный:
- fwd F32-ref vs FP8 в v121r: certified корректность на бенче (`docs/cert/cert_summary.md`).
- bwd fused: 5e-3 class non-causal, до 3e-2 causal-stress (paper анализ).

**Прогноз для 50-step траектории BattleA:**
- Per-step attention output error: 5e-3 relative.
- 50-шаг cumulative через softmax + Wo + FFN усиление: floor **1e-2 vs F32-эталон**.
- Grad hybrid: abs=5e-3, rel=5e-3.

Требуемый экзамен: A/B/J 50 шагов на BattleA-форме, F32-путь vs FA-путь, F64-судья на 10-step коротком отрезке.

### 6b. Speed — новая Амдал'а с уточнением

Прогноз до замера: attention CPU-backward на BattleA (`B=4, S=128, D=1024, hd=128, L=6`) — **не доминирующий кусок** (~1.6% wall'а). Ожидаемый wall-speedup 1.01-1.02× — в шуме CV.

**Ключевая рекомендация: перед этапом A2 замерить LLM.Backward по компонентам** (nsys profile или per-op timing на BattleA-конфиге). Это скажет: (α) сколько реально ест attention, (β) стоит ли идти в атаку на attention первым или на linear.

**Alt-прогноз для длинных контекстов:** если BattleA будет иметь `S=8192`, тогда attention доминирует (S²·hd term). На FA-канонической форме `bh=128 sl=8192 hd=128` attention fwd+bwd fused ≈ 50ms, тогда как CPU-эквивалент ≈ минуты — speedup ×1000+ на этом куске. Полный wall-speedup зависит от того что делает embedding+FFN на этой форме — тоже надо замерить.

### 6c. Memory

Per attention block на BattleA (`bh=B·H=32, sl=128, hd=128`):
- Q/K/V FP8: `3 · 32·128·128 = 1.5 MB`
- L, D FP32: `2 · 32·128·4 = 32 KB`
- dS_nat/dS_T FP8 stride_ds-padded: `2 · 32·128·144 = 1.1 MB`
- dQ/dK/dV FP32: `3 · 32·128·128·4 = 6 MB`
- O FP16: `32·128·128·2 = 1 MB`
- Итого ≈ **10 MB/attention block × 6 layers = 60 MB**.

Порядок negligible на 48GB Pro 5000.

---

## 7. Инфраструктурные gate'ы (обязательно перед любым вариантом слоя)

Эти работы — префикс любого A-этапа. Оценки времени грубые.

**G1. LSE-forward.** Добавить `_v121r_train_kernel.cu` в `libs/fa_sm120/Makefile:20 SRCS_CU`, экспозировать `fa_forward_train(...)` с параметром `float* L_out [bh, sl]` в `fa_sm120.h/fa_ctx.cu`. Пересобрать `libfa_sm120.so`. **Риск**: reg count v121r_train_kernel уже близок к 255 (v121r fwd = 244); ptxas отчёт обязателен. ~1-3 часа.

**G2. Backward `.so`.** Написать `libs/fa_bwd_sm120/wrapper.cu` с `extern "C"` обёртками для `launch_d_precompute`, `launch_merged`, `launch_dk_new`, `launch_dq_new`. Makefile + сборка `libfa_bwd_sm120.so`. **Риск**: buffer ownership правила (stride_ds, dual dS, D-precompute каждый шаг) требуют аккуратности. ~3-6 часов.

**G3. Целевой config BattleA.** Определить `hd=128` конфиг (полный fwd+bwd) с параметрами `(B, H, S, D, L, FFN)` под задачи A: (i) реалистичный размер для трена (25-100M параметров), (ii) попадает в FA-регион, (iii) не разрывает существующий боевой B2-скоуп. Кандидат: `B=4, H=8, S=128, D=1024, hd=128, L=6, FFN=4096` (≈50M params). ~1 час обсуждения.

**G4. Замер LLM.Backward компонент** (nsys profile или per-op timing на BattleA). Даст честный Амдал-прогноз для оценки value главы A. ~2-4 часа.

Только после G1-G4 имеет смысл принимать финальное архитектурное решение (a/b/c).

---

## 8. Разбивка предполагаемых этапов A

Условно, при рекомендованном варианте (b):

- **A2 — G1**: LSE-forward сборка + smoke test.
- **A3 — G2**: Backward `.so` сборка + correctness gate 11 форм.
- **A4 — G4**: Профиль LLM.Backward компонент на BattleA — **если attention <5% wall'а, ПЕРЕОСМЫСЛИТЬ приоритеты** (может, linear-backward-GPU важнее).
- **A5**: Purego bindings + backend-агностичный интерфейс.
- **A6**: Wiring через `nn/attention.go`, dispatcher F32-fallback / FA-path.
- **A7**: A/B/J 50-шаг экзамен на BattleA.
- **A8**: FA-класс скоростной замер + Amdahl-проверка.
- **A9**: Gates + отчёт + commit.

---

## 9. Дисциплина главы A

По ТЗ:
- **FA-canary** до и после каждого этапа A2..A9 (fwd v121r 5-run, baseline 652±2T; bwd fused 5-run, baseline 41.6±0.3ms). Первое movement из corridor → разбор.
- **Correctness gate 11 форм** (комбинация {bh∈{4,32,128}} × {sl∈{128, 1024, 8192}} × {wnd∈{0, 1024}} × {causal∈{0, 1}}) — при любом изменении FA-вызовов.
- Скоростные замеры FA-класс: silence gate + clocks до/после + median + CV 30-run.
- Setsid-детач протокол для всех прогонов > 5 мин (`feedback-detached-long-runs`).

---

## 10. Что не найдено — требует уточнения от пользователя

1. **Целевая форма BattleA-config** — не описана в ТЗ, нет A-handoff документа. Grep A-BRIDGE/A_BRIDGE/A_HANDOFF пуст. Требует решения пользователя.
2. **Реальный вклад attention в LLM.Backward wall'а** — не замерен ни в одном отчёте. Recon-агент подтвердил.
3. **`nn/feedforward.go`, `nn/linear.go`** — не прочитаны разведкой, точный CPU/GPU разбор linear backward неизвестен. Может помочь ещё больше сжать G4.
4. **Планы fa-blackwell-fp8 v0.2 shipping** — будет ли собран `libfa_bwd_sm120.so` в апстриме FA-репо, или мы делаем сборку в скоупе А. Требует уточнения.

---

## 12. Этап A-0 — matmul-backward на GPU (пре-attention ход)

**Решение пользователя (2026-07-24, после ревью бумаги):** перед погружением в LSE/backward-`.so`/attention начать с **A-0** — переезд matmul-backward'а на GPU через существующий wrapper.

### Мотивация

- B2 показал: **CPU host-loop backward (`gradOW = normed^T @ gradLogits`, `battle.go:229-237`) = 99.97% wall'а** = 4.79 сек из 4.80 сек шага.
- Это не attention (attention в B2 нет), а **output-projection matmul** — тот же класс `A^T @ B` что и большинство linear-backward'ов в реальной LLM.
- Существующий `libgotorch_blas_wrapper.so`'s `gt_gemm_ex` **уже принимает transa/transb** флаги (`blas_wrapper.c:78, 118, 155-172`, struct `GemmExArgs` с полями `transa, transb`). Инфраструктура готова.
- A-0 — **fastest wall-clock win** в главе A: один matmul переезжает с CPU на GPU, wall падает с ~4.8с до ~10-15мс = **~300× speedup**.
- Заодно даёт **честный Амдал'а с картой остаточных CPU-кусков** — станет видно куда бить дальше (CE-on-GPU? gradLogits-scatter-on-GPU? attention?).

### Что делаем

1. Добавить wrapper-based F32 matmul с trans-флагами:
   - `gotorch/v6/cuda`: метод `MatMulF32Ex(a, b, c, m, n, k, transA, transB, alpha, beta)` через `gt_gemm_ex` (F32 IO + F32 compute).
   - `goml/backend/gotorch`: extension `MatMulF32Ex` на `*gotorch.Backend` (паттерн B-impl-4).

2. Написать `trainStepBattleA0` в `internal/abjexam/battle.go`:
   - Всё как в `trainStepBattle` (F32 baseline), кроме backward-строки:
     - вместо CPU host-loop `for i,e,v: gradOW[e,v] += normed[i,e] * gradLogits[i,v]`
     - используем: upload gradLogits на GPU (~5мс), `MatMulF32Ex(normed, gradLogits, gradOW, K=Embed, N=Vocab, M=m, transA=T, transB=N)` (~0.5мс), skip normed D2H.
   - `runtime.LockOSThread` обязателен (feedback-lockosthread-battle-scale).

3. Тесты в `battle_test.go`:
   - **TestA0_Battle_Accuracy**: A/B — A=F32 CPU-bwd (B2 baseline), B=F32 GPU-bwd. 50 шагов, идентичные seed. Loss diff floor 1e-5 (см. floor'ы ниже). Grad hybrid abs=1e-3 + rel=1e-3.
   - **TestA0_Battle_Speed**: 30-run FA-класс на F32-GPU-bwd. Ожидаемо median ~10-15мс vs B2's 4791.5мс.
   - **TestA0_Battle_CPUMap**: профильный запуск — измерить wall'ы отдельных host-компонент (probs D2H, CE host loop, gradLogits H2D) → карта остаточных CPU-кусков для §12.6 нового Амдал'а.

### Floor'ы pre-registered (до измерения)

**Accuracy:**
- cublasSgemm F32 (row-major через swap trick) accumulates в F32, host loop тоже F32, но порядок суммирования разный. Ожидание per-op diff ≈ `K · eps = 256 · 6e-8 = 1.5e-5` absolute.
- **worst |gradOW_gpu - gradOW_cpu_bwd| floor**: abs=1e-3 + rel=1e-3·|ref| (тот же hybrid что в B-impl-4).
- **worst |loss_gpu - loss_cpu| floor** (50 шагов): 1e-5 (accumulated через softmax).
- **Убывание**: обязательно, оба пути.
- STOP-правило: разбор при превышении floor'а, не расширение.

**Скорость:**
- Прогноз median wall/step: **10-15 мс** (доминирует probs D2H 128MB @ 25GB/s ≈ 5мс + gradLogits H2D 128MB ≈ 5мс).
- Прогноз speedup: **~300×** vs F32 CPU-bwd (4791.5мс → ~15мс).
- CV прогноз: возможно всё ещё >1% из-за host↔GPU copies (не полностью GPU-bound).
- Если speedup <100× — разбор (микропрофиль cudaEvent на компоненты).

**Память:**
- +gradLogits GPU буфер: `m*Vocab*4 = 128 MB`.
- +gradOW GPU буфер: `Embed*Vocab*4 = 32 MB` (уже был через upload после CPU-bwd — не растёт).
- Итого прогноз пика F32-GPU-bwd: ~1130 MB (vs B2 F32 1002 MB). Заголовок 46-47 GB на 48GB Pro 5000.

**Новый Амдал (пост-A-0, для приоритизации следующих ходов):**
- Прогноз шага после A-0:
  ```
  GPU forward (embed+LN+matmul+softmax)      ~1 мс   6-10%
  probs D2H (128 MB)                          ~5 мс  35-50%
  CE + gradLogits build on host               ~0.5 мс  3-5%
  gradLogits H2D (128 MB)                     ~5 мс  35-50%
  MatMul-bwd GPU (gradOW = normed^T @ gL)     ~0.5 мс  3-5%
  adamw GPU + Sync                            ~0.3 мс  2-3%
  ---
  Wall total                                  ~12-15 мс   100%
  ```
- **Следующий кандидат**: CE + gradLogits scatter на GPU (сохранит 10мс = 2/3 wall'а). Это pre-attention ход.
- attention всё ещё на не в scope B2 (не запущен). Для главы A (attention) — стартовать после того как CE-scatter уйдёт на GPU (иначе даже с attention на GPU останется 5мс probs D2H bottleneck).

### Этапы A-0

- **A-0.1**: добавить `MatMulF32Ex` в gotorch/v6/cuda + adapter extension (goml/backend/gotorch).
- **A-0.2**: `trainStepBattleA0` в battle.go.
- **A-0.3**: A/B accuracy 50 шагов (детач).
- **A-0.4**: 30-run скорость (детач).
- **A-0.5**: CPU-map profile — измерить остаточные CPU-компоненты.
- **A-0.6**: gates + `runs/reports/A_0.md` + bundle-commit.
- **СТОП** для обсуждения пользователем перед началом A-1 (следующий ход по карте).

---

## 13. Что предлагается пользователю решить перед A-1

Ко времени когда A-0 завершится, будут числа для решения:

**Р1.** Согласовать: A-BRIDGE **не заменяет** существующий attention (его на GPU нет), а **создаёт** GPU attention-путь на новом BattleA-конфиге, параллельно оставляя CPU-путь main.go как есть.

**Р2.** Согласовать: FA-integration ladder из трёх ступеней (§5.2), глава A = ступень 1.

**Р3.** BattleA-конфиг (§7 G3) — `B, H, S, D, hd=128, L, FFN`.

**Р4.** Порядок работ после A-0 — что бить дальше по новой Амдал-карте (§12.6):
- CE + gradLogits scatter на GPU?
- LSE-forward сборка (G1)?
- Backward `.so` сборка (G2)?
- Attention forward + backward через gofa/purego?

Решение по Р4 после чисел A-0.

---

**СТОП. Ждём ревью и решения пользователя перед кодом.**
