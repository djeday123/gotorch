# gotorch v0.1.0

First tagged release of gotorch — a PyTorch-like deep learning framework
written entirely in Go. Targets CPU by default; pairs cleanly with goml
when a CUDA backend is required.

## Highlights

* **Correct autograd.** Ten silent backward-pass bugs found and fixed,
  every one of them verified by a numerical-gradient test:
  Sum/Mean (ignored upstream grad), Softmax (was identity), LayerNorm
  (missing 1/std), BatchNorm1d/GroupNorm (grad flow broken),
  MultiHeadAttention (Q/K/V received the same gradient), LSTM/GRU
  (gates lived outside the autograd graph — full BPTT now works).
* **Float32 preserved.** `MatMul` and `BatchMatMul` now return Float32
  when both inputs are Float32; previously the dtype silently widened.
* **Safe Div.** `Div.Backward` no longer blows up to NaN/Inf when the
  divisor is zero — uses a smooth reciprocal that gives 0 at b=0 and
  finite values everywhere else.
* **Concurrency-safe.** Replaced the broken `SwapInt32`-based NoGrad with
  counter-based depth tracking; removed the shared `cursor` field in
  DataLoader and added a `sync.Mutex`. `go test -race` is clean.
* **Performance:** Welford variance, heap-based TopK (O(n·k) → O(n log k)),
  in-place ops (49× faster on 1k vectors, zero allocations),
  ConvTranspose2d rewritten as matmul + col2im (10-50× faster), and a
  `sync.Pool`-backed tensor allocator with explicit `Release()`.

## What works

| Component | Coverage |
|---|---|
| Tensors, broadcasting, advanced indexing | ✅ |
| Reverse-mode autograd | ✅ |
| Linear, Conv1d/2d/ConvTranspose2d, MaxPool, Embedding | ✅ |
| BatchNorm1d/2d, LayerNorm, GroupNorm, Dropout | ✅ |
| MultiHeadAttention, TransformerEncoder/Decoder | ✅ |
| LSTM, GRU with proper BPTT | ✅ |
| Sequential, ModuleList, ModuleDict | ✅ |
| Loss functions (MSE/BCE/CE/NLL/L1/Huber/KLDiv) | ✅ |
| Optimizers (SGD/Adam/AdamW/RMSprop/Adadelta) + LR schedulers | ✅ |
| DataLoader with goroutine prefetch | ✅ |
| Mixed precision (`GradScaler`) | ✅ basic |
| Quantization (int8 `QLinear`) | ✅ inference only |
| ONNX export | ✅ basic |

## What does **not** work yet

* `torch.compile` / JIT / graph capture
* Distributed training (`DataParallel`, `DistributedDataParallel`)
* `torch.fft`, extensive `torch.linalg`, `torch.distributions`
* Native Float32 compute path (currently promotes to Float64 internally)
* vmap, jacobian, hessian, second-order autograd
* Multi-GPU

## Install

```bash
go get github.com/djeday123/gotorch@v0.1.0
```

A 10-line XOR example and 6 working examples (XOR → MNIST MLP → CNN →
LSTM-LM → Transformer → functional API) live in `examples/`.

## Honest scope

If you measure by PyTorch API surface, gotorch covers maybe **7 %** —
PyTorch is enormous. If you measure by what a typical supervised-learning
project actually uses (define a model, define a loss, train with AdamW,
save), gotorch covers **40-50 %**. Standard transformer / CNN / LSTM
training from scratch works end-to-end.

## Acknowledgements

The fixes in this release were caught by a combination of automated audit
agents and manual re-verification against the math of each operation. The
"agent flagged it, then I sat down and traced the gradient on paper" loop
turned out to remove a lot of false positives — see the commit messages
for which findings were real bugs and which were mis-identified.

See `CHANGELOG.md` for the full per-commit breakdown.
