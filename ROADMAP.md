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

## 🔮 v2.0.0 — Level 4: Production (~95% PyTorch coverage)
- float16 / bfloat16 + mixed precision training
- DataLoader — batching, shuffle, prefetch
- Multi-GPU (DataParallel)
- `torch.compile` equivalent (kernel fusion)
- Sparse tensors
- torch.distributed

---

## Coverage Progress

| Version | Coverage | Unlocks |
|---|---|---|
| v1.0.0 | ~25% | XOR, simple feedforward nets |
| v1.1.0 | ~45% | LeNet, VGG-style CNNs |
| v1.2.0 | ~65% | ResNet, stable deep training, model persistence |
| v1.3.0 | ~80% | Transformers, GPT-mini, seq2seq |
| v2.0.0 | ~95% | Production ML pipelines |
