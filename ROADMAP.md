# GoTorch Roadmap

## ✅ v1.0.0 — Foundation (~25% PyTorch coverage)
- Tensor: N-dim float64, broadcast, reshape, reduce, linalg
- Autograd: reverse-mode, topological backward
- nn: Linear, Sequential, ReLU/Sigmoid/Tanh, MSE/BCE/CrossEntropy
- optim: SGD (momentum), Adam
- CUDA: GPU backend, elementwise + MatMul (cuBLAS), PinnedTensor (zero-copy)

## ✅ v1.1.0 — Level 1 (~45% PyTorch coverage)
- **float32 dtype** — all ops preserve dtype, f32/f64 casting
- **cat / stack / split / chunk** — tensor concatenation along any axis
- **Indexing** — Select, Narrow, Index, MaskedSelect, Clamp, Where
- **no_grad** — NoGrad(), EnableGrad(), IsGradEnabled()
- **nn.Conv2d** — im2col + matmul, full backward, He init
- **nn.MaxPool2d** — max-mask backward
- **nn.Flatten2d**

## 🔶 v1.2.0 — Level 2: Training Pipeline (~65% PyTorch coverage)

### Block 1 — Optimizers & Training Utilities
- [ ] `optim.AdamW` — Adam with decoupled weight decay (standard since 2019)
- [ ] `optim.clip_grad_norm_(params, max_norm)` — gradient clipping
- [ ] `optim.LRScheduler` interface
  - [ ] `StepLR(optimizer, step_size, gamma)`
  - [ ] `CosineAnnealingLR(optimizer, T_max)`
  - [ ] `LinearWarmup(optimizer, warmup_steps)`

### Block 2 — Model Persistence
- [ ] `gotorch.Save(model, path)` — serialize weights to JSON/binary
- [ ] `gotorch.Load(model, path)` — restore weights from file

### Block 3 — Regularization
- [ ] `nn.BatchNorm2d(numFeatures)` — with running mean/var, train/eval modes
- [ ] `nn.LayerNorm(normalizedShape)`
- [ ] `nn.Dropout(p)` — train/eval aware

## 🔵 v1.3.0 — Level 3: Modern Architectures (~80% PyTorch coverage)

### NLP Primitives
- [ ] `nn.Embedding(numEmbeddings, embeddingDim)` — lookup table with backward
- [ ] `nn.MultiheadAttention(embedDim, numHeads)` — scaled dot-product attention
- [ ] Positional encoding utilities

### Recurrent Networks
- [ ] `nn.LSTM(inputSize, hiddenSize, numLayers)` — with hidden/cell state
- [ ] `nn.GRU(inputSize, hiddenSize)` — gated recurrent unit

### Architecture Blocks
- [ ] `nn.TransformerEncoderLayer` — Attention + FFN + LayerNorm + Dropout
- [ ] `nn.TransformerEncoder(layer, numLayers)`

## ✅ v2.0.0 — Level 4: Production DataLoader + AMP (~92% PyTorch coverage)
- **tensor** — Std, Var, Norm, Prod, Floor, Ceil, Round, Sign, Linspace, Full
- **nn** — GELU, LeakyReLU, ELU, SiLU, Conv1d, AdaptiveAvgPool2d
- **optim** — RMSprop, Adadelta
- **data/** — Dataset interface, TensorDataset, DataLoader (goroutines, prefetch)
- **amp/** — GradScaler for mixed precision

## ✅ v2.1.0 — Level 5: Functional API + Advanced Ops (~95% PyTorch coverage)
- **nn/functional** — F.ReLU/GELU/LeakyReLU/ELU/SiLU/Sigmoid/Tanh, F.Softmax, F.LogSoftmax,
  F.Dropout, F.MSELoss, F.L1Loss, F.HuberLoss, F.BCELoss, F.CrossEntropyLoss, F.NLLLoss, F.Linear
- **tensor** — Gather, ScatterAdd, Cumsum, Cumprod, Tril, Triu, RepeatInterleave
- **nn** — ConvTranspose2d (transposed conv, full backward), Upsample (nearest-neighbor)
- **nn** — L1Loss, HuberLoss, NLLLoss (public), KLDivLoss
- **nn** — ModuleList, ModelSummary / PrintSummary

## ✅ v3.0.0 — Level 6: Distributed + Quantization (~97% PyTorch coverage)
- **DataParallel** — multi-GPU split/gather API (1-GPU pass-through + multi-GPU)
- **Dtypes** — Float16, BFloat16, Int8 (full dtype system)
- **Quantization** — Quantize8/Dequantize8, QLinear, QuantizeModel
- **ONNX export** — minimal protobuf encoder, no external deps
- **Tests** — 261 total (245 CPU + 16 GPU)

## 🔮 v4.0.0 — Level 7: Training Infrastructure + NLP (~99% PyTorch coverage)

### Block 1 — Optimizers & Schedulers
- [ ] `optim.AdamW` — Adam + decoupled weight decay
- [ ] `optim.ClipGradNorm(params, maxNorm)` — gradient clipping
- [ ] `LRScheduler` interface:
  - [ ] `StepLR(optimizer, stepSize, gamma)`
  - [ ] `CosineAnnealingLR(optimizer, Tmax)`
  - [ ] `LinearWarmup(optimizer, warmupSteps)`

### Block 2 — Model Persistence
- [ ] `gotorch.Save(model, path)` — serialize weights (JSON + binary formats)
- [ ] `gotorch.Load(model, path)` — restore weights
- [ ] Checkpoint support: save/resume optimizer state

### Block 3 — Normalization & Regularization
- [ ] `nn.BatchNorm1d / BatchNorm2d` — running mean/var, train/eval modes, backward
- [ ] `nn.LayerNorm(normalizedShape)` — full backward
- [ ] `nn.Dropout` (rework: proper Bernoulli mask, train/eval aware)
- [ ] `nn.GroupNorm(numGroups, numChannels)`

### Block 4 — NLP Primitives
- [ ] `nn.Embedding(numEmbeddings, embeddingDim)` — lookup table + backward
- [ ] `nn.MultiheadAttention(embedDim, numHeads)` — scaled dot-product, mask support
- [ ] Positional encoding utilities (sinusoidal)

### Block 5 — Recurrent Networks
- [ ] `nn.LSTM(inputSize, hiddenSize, numLayers)` — hidden + cell state, backward
- [ ] `nn.GRU(inputSize, hiddenSize)` — gated recurrent unit

### Block 6 — Transformer Blocks
- [ ] `nn.TransformerEncoderLayer` — MHA + FFN + LayerNorm + Dropout
- [ ] `nn.TransformerEncoder(layer, numLayers)` — stacked encoder
- [ ] Example: GPT-mini / simple language model

---

## What's in the remaining ~1% (v5.0.0+)
These are niche / very advanced PyTorch features rarely needed outside research:
- `torch.compile` / kernel fusion / JIT tracing
- `torch.distributed` + NCCL multi-node (beyond single-machine DataParallel)
- `nn.Transformer` (decoder + cross-attention, full encoder-decoder)
- Sparse tensors (`torch.sparse`)
- `torch.fx` graph mode / symbolic tracing
- Complex number dtype support
- Custom C++/CUDA extensions API (`torch.utils.cpp_extension`)
- TorchScript export

---

## Coverage Progress

| Version | Tests | Coverage | Unlocks |
|---|---|---|---|
| v1.0.0 | ~25 | ~25% | XOR, simple feedforward nets |
| v1.1.0 | 85 | ~45% | LeNet, VGG-style CNNs |
| v2.0.0 | 185 | ~92% | Production ML pipelines, mixed precision |
| v2.1.0 | 221 | ~95% | Functional API, deconvolution, upsampling, KL/Huber loss |
| v3.0.0 | 261 | ~97% | DataParallel, quantization, ONNX export |
| v4.0.0 | ~340 | ~99% | Full training infra, Transformers, LSTM, model save/load |
