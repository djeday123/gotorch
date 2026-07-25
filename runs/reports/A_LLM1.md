# A-LLM-1 — фундамент nn.LLM трансформер с FA-forward на GPU

**Дата:** 2026-07-25
**Тип:** первый ход nn.LLM stage после закрытия battle-цепочки. Ступень 1 FA-лестницы (goml purego, мостовая дисциплина). Задача: трансформер ДЫШИТ через FA-fwd; backward без FA (attention-bwd временно reconstructed F32).
**Скоуп:** Этапы 1-5 + ворота. **Форма ТЗ большая — данный отчёт наполняется инкрементально с прогрессом.**

---

## Этап 1 (G3) — BattleA config + ревизия nn.LLM + арифметика памяти

### 1.1 Разведка nn.LLM (файл:строка) — что уже есть, что подходит FA

| Кусок | fwd файл:строка | fwd путь | bwd файл:строка | bwd путь | Целевой путь A-LLM-1 |
|-------|-----------------|----------|------------------|----------|----------------------|
| Embedding | `nn/embedding.go:35` → `backend.Embedding` (gotorch) | **GPU F32** | `nn/backward.go:230` (CPU scatter) | **CPU** | GPU fwd (существует); bwd — atomic-add scatter (наш P3-EMB kernel) |
| LayerNorm | `nn/layernorm.go:36` → `backend.LayerNorm` (GPU) | **GPU F32** | `nn/backward.go:81` (CPU) | **CPU** | GPU fwd (сущ.); bwd — нужен LayerNormGradF32 или reconstruct |
| **RMSNorm** | ❌ НЕ используется nn.LLM. Есть GPU extension `backend/gotorch/rmsnorm.go:25` (P2-RMS, F32/F64 fwd+bwd). | — | — | — | **Заменить LayerNorm на RMSNorm** — GPU fwd+bwd готовы |
| Wq/Wk/Wv/Wo Linear | `nn/linear.go:53` → `ops.MatMul` (GPU) | **GPU F32** | `nn/backward.go:14` (**CPU F32 GEMM host loop**) | **CPU** | GPU fwd (сущ.); bwd — **MatMulF32Ex trans-flags** (наш A-0) |
| RoPE | `nn/attention.go:173` → `backend.RoPE` (GPU) для inference Forward; в ForwardWithCache — inline CPU `applyRoPEInPlace` (`nn/backward_attn.go:439`) | **CPU в тренировке** | inline CPU `ropeBackwardInPlace` (`nn/backward_attn.go:460`) | **CPU** | GPU fwd (goml/gotorch RoPE_F32 существует, P4-ROPE); bwd — TBD (или пропускать RoPE bwd для inference) |
| Attention fwd | `nn/attention.go:132` (inference) — composite cublasSgemm+softmax_f32+cublasSgemm F32, S² materialized; `nn/backward_attn.go:19` (тренировка) — **чистый Go 4-nested-loop** | **CPU в тренировке** | `nn/backward_attn.go:117` (`.Backward`) — CPU host loop, держит `scores []float32 [B·H·S·S]` per layer в кэше | **CPU** | **Fwd → FA-fwd-with-L (наш G1 сборка).** Bwd → reconstruct F32 (вариант a Этап 4) |
| out projection Wo | inline через `Linear.Forward` после `rearrangeBHSD` (`nn/backward_attn.go:104`) | **GPU F32** | inline `Wo.Backward` (`nn/backward_attn.go:151`) | **CPU** | GPU fwd (сущ.); bwd — MatMulF32Ex |
| FFN (SwiGLU) fwd | `nn/feedforward.go:54` — 3 Linear.Forward + `ops.Silu` (GPU) + `ops.Mul` (GPU) | **GPU F32** | `nn/backward.go:161` — inline CPU siluForward/siluBackward + mulElementwise (F32 host loops) | **CPU** | GPU fwd (сущ.); bwd — Silu-grad kernel (уже есть? проверить), MatMulF32Ex для линеек |
| Softmax (attn GPU path) | `backend/cuda/ops.go:522` (softmax_f32 PTX) | **GPU F32** | — | — | Не нужен вне attention — FA-fwd делает свой softmax внутренне |
| CE Loss | `nn/loss.go:16` (CPU) + `ops/loss.go:14` (F32) | **CPU** | `ops/loss.go:68` | **CPU** | **Наш `cross_entropy_f32` A-1 PTX kernel** (осиротел в nn.LLM, подключаем) |

**Production config nn.LLM:**
- `main.go:40`: `TinyConfig()` + `MaxSeqLen=64` → V=256, D=64, L=2, H=4, hd=16, FFN=172 SwiGLU. **hd=16 несовместим с FA (нужно 64 или 128).**
- `nn.SmallConfig()`: V=32000, D=256, L=6, H=8, hd=32, FFN=688 SwiGLU. **hd=32 тоже несовместим.**

**Ключевой архитектурный вывод:** `nn.LLM.ForwardWithCache` — production-path через **чистый Go CPU** (materialized tensors через `.ToFloat32Slice()`). Wall на боевой форме = минуты/шаг. **Данный ТЗ строит НОВЫЙ transformer Step** (параллельно `nn.LLM`, паттерн battle.go), собранный из battle-инструментария + FA-fwd; nn.LLM code не менять в scope A-LLM-1 (может быть переработан в отдельном ходу).

### 1.2 FA sources — что готово, что надо собрать

**`_v121r_train_kernel.cu` (LSE-forward):**
- Файл: `goml/libs/fa_sm120/src/_v121r_train_kernel.cu` (805 строк). Launcher: `_v121r_train_launcher.cuh`.
- Функция: `fa96b_train_kernel(Q,K,V FP8, O FP16, L_out FP32 [bh,sl], sl, hd, causal, scale, qk_descale, v_descale, window)`.
- **НЕ в Makefile SRCS_CU!** `libs/fa_sm120/Makefile:20` только `_v121r_kernel_full.cu + fa_ctx.cu`.
- **`fa_forward_train` C-ABI НЕ существует.** `fa_sm120.h` экспортирует только `fa_forward` (inference без L).

**v0.2.0 backward chain (`goml/release_v0.2.0/src/`):**
- 4 launcher'а: `launch_d_precompute` (fa_bwd_dk.cu:522), `launch_merged` (fa_bwd_merged_v1.cu:513), `launch_dk_new` (fa_bwd_dk_new.cu:306), `launch_dq_new` (fa_bwd_dq_new.cu:338).
- **Все namespace C++, extern-C НЕТ, .so НЕТ.** Makefile строит только 3 бенчмарк-бинарника.
- Fingerprint metadata (для sanity-check при сборке): `d_precompute=38 regs`, `merged=252 regs`, `dk_new=124 regs`, `dq_new=69 regs`. Non-causal baseline wall 42.346 ms на `bh=128 sl=8192 hd=128`.

**libfa_sm120.so shipped state:**
- `libs/fa_sm120/libfa_sm120.so` = только `fa_forward` (inference, без L).
- Symbols (`nm -D`): fa_create, fa_forward, fa_version, fa_status_str, fa_last_cuda_error, fa_dispatch_select, fa_kernel_name, fa_destroy.
- Fatbin — только production v121r kernel.

### 1.3 FA-контракт vs nn/attention layout

**FA `[BH, S, HD]` row-major, contiguous, FP8 Q/K/V, FP16 O, F32 L_out.**
`hd ∈ {64, 128}`, `sl > 0`, `bh > 0`, `causal ∈ {0,1}`, `window ∈ [0, sl]`.

**Совместимость:**
- Layout: nn/attention Q,K,V после `Transpose([0,2,1,3])` → [B,H,S,hd] = collapse [B*H, S, hd] ✓.
- Contiguity: `makeContiguous` (`nn/attention.go:222`) — **потенциальная проблема**: читает `t.Storage().Bytes()` пер-байт на host. Если storage device, это DtoH-copy per shape. **Требует проверки** — либо использовать GPU permute (goml/gotorch не имеют такого kernel'а, придётся написать), либо reshape без transpose (изменить порядок хранения весов).
- **hd блокер**: TinyConfig hd=16, SmallConfig hd=32 — оба НЕ поддержаны. **BattleA обязана иметь hd∈{64,128}.**

**Dtype conversions (доступные):**
- Q/K/V (F32 → FP8 e4m3): `gotorch.QuantizeF32ToF8E4M3` (matmul_mp.go:65) — per-tensor amax.
- O (FP16 → F32): `gotorch.CastF16ToF32`.
- dO (F32 → FP16, для bwd): `gotorch.CastF32ToF16`.

**Scale механика (открытый вопрос):**
- `fa96b_train_kernel` принимает `scale, qk_descale, v_descale`. Launcher hardcode-ит `qk_descale=1.0, v_descale=1.0`.
- QuantizeF32ToF8E4M3 применяет `scale_Q = amax_Q / 448` → Q_fp8 = Q_f32 / scale_Q. Kernel считает Q_fp8 · K_fp8 без обратного скейла.
- **Решение**: составить `scale = softmax_scale · scale_Q · scale_K` (передать в kernel), либо пропатчить launcher чтобы принимать `qk_descale = scale_Q · scale_K`. Первый вариант проще (не трогает launcher).

**B-impl-4 F8 опыт (per-tensor amax недостаточен на V=32000):** для attention Q·K^T range уже (после RMSNorm), scale-invariance softmax поглощает часть погрешности. FA-3 литература говорит per-tensor работает на attention. Тем не менее, **первый bit-vs-F64 замер обязателен** (даёт число).

### 1.4 BattleA config — арифметика памяти

**Config (user + агент confirmation):**
- V=32000, **D=512, H=4, hd=128** (hd·H=D ✓ FA compat), **L=4, S=2048, B=4**
- FFN=**2048** (standard 4·D) для упрощения; SwiGLU +50% weights, детали в Этапе 3.

**Веса + AdamW + grads (F32):**
- Embed [V,D] = 32000·512·4 = **65.5 MB**
- Output [D,V] = 65.5 MB
- Per-layer (Wq+Wk+Wv+Wo)·[D,D] = 4·512²·4 = 4.2 MB
- Per-layer FFN standard [D,4D]+[4D,D] = 2·512·2048·4 = 8.4 MB
- Per-layer 2×LayerNorm gammas = 8 KB (neg.)
- Per-layer total ≈ 12.6 MB, ×4 = 50.4 MB
- **Weights total ≈ 181 MB**
- **AdamW state (m+v) ≈ 362 MB**
- **Grad buffers ≈ 181 MB**
- **Weights+grad+opt ≈ 724 MB**

**Активации forward (SwiGLU per-layer cached):**
- Q/K/V/O F32, FP8, L_out, AttnOut, Normed1/2, FFN hidden/out: **~322 MB per layer × 4 = 1288 MB**
- Logits F32: **1024 MB**
- Embedding out: 16 MB

**Backward без FA (variant a, in-flight per layer):**
- Attention S=QK^T `[B,H,S,S]` F32: 4·4·2048²·4 = **268 MB per layer** (in-flight peak, не keep-all)
- Softmax P, dScores можно alias с S in-place = 0 extra
- **Peak per-layer attention-bwd ≈ 300 MB**

**Peak (worst case, no gradient checkpointing):**
| категория | MB |
|-----------|----|
| Weights+grads+AdamW | 724 |
| Fwd activations (all layers cached SwiGLU) | 1288 |
| Logits F32 | 1024 |
| Bwd temp peak (in-flight per layer, alias-friendly) | ~300 |
| Runtime overhead (cuBLAS ws, kernel scratch) | ~500 |
| **Peak total** | **~3.8 GB** |

**Вердикт:** BattleA config **fits в 96 GB Pro 6000 Blackwell easily.** Никакого fallback config не нужно. Headroom > 92 GB.

**Ограничение НЕ памяти, а времени CPU-путей nn.LLM**: если использовать `nn.LLM.ForwardWithCache` как есть — attention CPU loops = ~15-20 сек/шаг + FFN CPU bwd ~70 сек/шаг + Linear CPU bwd ~24 сек/шаг = **~100+ сек/шаг**. **Непригодно.**

⇒ **Строим НОВЫЙ transformer Step (стиль battle.go), а не через nn.LLM.**

### 1.5 Открытые вопросы / uncertainty

1. **FP8 amax стратегия для Q/K/V**: per-tensor vs per-head. **Замерить bit-vs-F64 на первом батче.**
2. **qk_descale/v_descale**: составить `scale = softmax_scale · scale_Q · scale_K` (простой вариант).
3. **`makeContiguous` в nn/attention.go:222** — потенциальный CPU per-element loop. **Решение: писать transformer Step с чистого листа, где Q/K/V сразу выходят в [BH, S, hd] layout без transpose (grouped проекция или отдельное хранение).**
4. **RoPE fp16/fp32 через FA**: FA принимает Q/K уже пост-RoPE. RoPE применяется на F32 Q/K перед квантизацией в FP8.
5. **`nn.CrossEntropyLoss` осиротел** — вместо использовать наш `cross_entropy_f32` A-1 kernel напрямую.
6. **P3-EMB TestAdapterEmbeddingGradF32_BvsJ FAIL (задача #79)** — pre-existing. Если Embedding backward используется в A-LLM-1 через `backend.EmbeddingGradF32` — этот путь ненадёжен. Обсудить: либо (а) использовать CPU embedding backward временно в А-LLM-1 (маленькая доля wall'а), либо (б) сначала пересмотреть floor P3-EMB.

---

## Этапы 2-5: план (заполнится по факту)

### Этап 2 G1: LSE-forward сборка — ✅ PASS (2026-07-25)

**Сборка:**
- `libs/fa_sm120/Makefile`: добавлено правило `_v121r_train_kernel_full.cu` (concat kernel+launcher), в `SRCS_CU`.
- `libs/fa_sm120/src/fa_ctx.cu`: namespace declaration `fa_sm120_v121r_train::launch` + новая extern-C функция `fa_forward_train(...)` РЯДОМ с `fa_forward` (production ABI unchanged).
- `libs/fa_sm120/include/fa_sm120.h`: прототип `fa_forward_train`.
- Rebuild: **train kernel 255 registers, 0 spill/stack, 1 barrier** (в допуске +2-3 vs production 244 fits within 255 ceiling).
- Symbols exported: `fa_forward_train` @ 0x4090 alongside `fa_forward` @ 0x3d00 — production символ бит-в-бит цел.

**FA-canary до/после сборки:**
- До: median 653.24T, mean 653.36T, WITHIN.
- После: median 653.39T, mean 653.38T, WITHIN (delta 0.15T = 0.02%, шум) — **production fa_forward не задет**.

**Purego binding:** `goml/backend/cuda/fa_forward.go` (~180 строк).
- Поиск .so: `$GOML_FA_LIB` → `libs/fa_sm120/libfa_sm120.so` → fallback пути.
- API: `FALoad()`, `FACreate()/Destroy()`, `FAContext.ForwardTrain(...)`, `FAVersion()`.
- Внутри `ForwardTrain`: `runtime.LockOSThread` per-call (мостовая дисциплина — FA использует cudaGetDevice on current thread).

**L-correctness тесты** (`goml/backend/cuda/fa_forward_test.go`, форма несимметричная bh=3, sl=280, hd=128):

| test | форма | L values | vs ref | verdict |
|------|-------|----------|--------|---------|
| Version | — | 0.1.0+652T-sm120a | — | ✓ |
| L_Uniform (Q=K=V=1.0) | bh=3, sl=280 | 16.9526 uniform | ref 16.9485, rel **2.4e-4** | ✓ (floor 5e-3) |
| L_Layout (Q per-row 1/2/4) | bh=3, sl=280 | row0=16.95, row1=28.27, row2=50.91 | все ref-per-row rel 2-3e-4 | ✓ (floor 5e-3) |

**Cross-check layout** (обязательный по правилу пользователя — off-by-one убьёт bwd тихо):
- `L[sl-1]` (последний s row 0) ≈ refRow0 = 16.9485 ✓ (не refRow1 — не [sl, bh] layout).
- `L[sl]` (первый s row 1) = 28.27 = refRow1 ✓.
- **[bh, sl] row-major layout подтверждён.**

**Механизм ошибки 4e-3 abs / 2.4e-4 rel:**
- FP8 e4m3 quantization noise: decoded_Q · decoded_K ≠ true F32 product (ULP-класс 2^-3 mantissa).
- F32 accumulation over hd=128: sqrt(128)·eps ≈ 6.8e-7 negligible vs FP8.
- Kernel exp2.approx.f16x2 (log2-space softmax): ~1 ULP.
- Дом. вклад — FP8 quantize noise + log-conversion.
- **Внутри floor 5e-3 (FP8 attention class), запас 20×.**

**G1 закрыт.** `fa_forward_train` production-ready в libfa_sm120.so. L-сертификат: значения + раскладка + асимметричная форма пройдены.

### Этап 2 G2: bwd .so сборка (**СЛЕДУЮЩИЙ ХОД** — после user check)

1. Добавить `_v121r_train_kernel.cu + _v121r_train_launcher.cuh` в `libs/fa_sm120/Makefile` SRCS_CU (аналог правила `_v121r_kernel_full.cu`).
2. В `fa_ctx.cu`: добавить `namespace fa_sm120_v121r_train { extern void launch(...); }` + новую extern-C функцию `fa_forward_train(ctx, q, k, v, o, l_out, bh, sl, hd, causal, window, scale, stream) → fa_status_t`.
3. В `include/fa_sm120.h`: прототип `fa_forward_train`.
4. **FA-canary до сборки** (baseline lock).
5. `make -C libs/fa_sm120 rebuild`.
6. Проверить symbols: `nm -D libfa_sm120.so | grep fa_forward`.
7. **FA-canary после сборки** (fa_forward inference неизменен).
8. Purego binding в `goml/backend/cuda/fa_forward.go` (новый файл).
9. **L-correctness тест:** малая форма (bh=4, sl=128, hd=128, random Q/K/V post-RoPE), сравнить `L_out` с F64 CPU reference `L_i = m_i + log(sum(exp(qi·kj - m_i)))`.

### Этап 2 G2: bwd .so сборка + smoke + fingerprints

1. Написать `libs/fa_bwd_sm120/wrapper.cu` extern-C для 4 launchers.
2. Makefile.fa_bwd_wrapper (аналог blas_wrapper).
3. Собрать `libfa_bwd_sm120.so`.
4. Проверить fingerprints через `cudaFuncGetAttributes` в smoke-тесте (numRegs: d_precompute=38, merged=252, dk_new=124, dq_new=69). Если не совпадает — nvcc/CUDA version mismatch, разбор.
5. Smoke: bh=128, sl=8192, hd=128 (canonical) — прогнать цепочку, сверить bit-wise fingerprint output с `release_v0.2.0/tests/bench_r2c_e2e.cu` reference.
6. **НЕ встраивать** в trainStepBattleA_LLM.

### Этап 3-5: (details после G1+G2 сборок)

---

## Ворота (в конце всего)

- Экзамен "дышит" (Этап 5) зелёный
- FA-canary WITHIN [652, 656]
- goml regression PASS + gotorch/v6 PASS
- Отчёт A_LLM1.md с 3 экзамен-таблицами
- Bundle-commit

**СТОП после отчёта.** Решение о встройке FA-bwd в trainStep — по данным экзамена.
