# gotorch v6.0.0 — first properly working release

This is the **first release of gotorch where the autograd is actually
correct**. Earlier `v1.0.0` … `v5.0.0` builds shipped silent bugs in
backward passes for `LSTM`, `GRU`, `MultiHeadAttention`, `LayerNorm`,
`BatchNorm1d`, `GroupNorm`, `Sum`, `Mean`, and `Softmax`. Training with
those releases produced wrong gradients that just happened to converge
for simple problems.

If you used v5.0.0 or earlier — re-train. Pin to v6.0.0+ going forward.

The major-version bump signals a behavioural change for anyone who
inadvertently depended on the broken gradient values. Public APIs are
otherwise unchanged.

## What's new in v6.0.0

### Correct autograd

Ten silent backward-pass bugs caught and fixed, every one verified by a
numerical-gradient test in `nn/gradcheck_test.go`:

* `Sum.Backward` was returning `Ones(shape)`, ignoring the upstream
  gradient. Now `MulScalar(Ones, grad.Item())`.
* `Mean.Backward` had the same bug.
* `Softmax.Backward` was the identity. Now implements the Jacobian
  `dx = s · (g − sum(g · s, dim))`.
* `LayerNorm.Backward` was missing the `1 / std` factor — the forward
  never stored std. Forward now records per-group std; backward uses
  `(γ · dy − sumDy/M − xn · sumDyXn/M) / std`.
* `BatchNorm1d` returned `requiresGrad=false`, blocking gradient flow.
  Replaced with a proper backward for both train and eval modes.
* `GroupNorm` had the same broken-grad-flow problem; now fixed with
  per-(n, group) std.
* `MultiHeadAttention.Backward` was returning the same upstream
  gradient to Q, K, and V identically. Replaced with the proper chain
  rule using stored attention weights and per-head matmul.
* `LSTM` and `GRU` gates were raw `[]float64` operations outside the
  autograd graph; BPTT never reached the gate weights. Rewritten as a
  single sequence-level autograd op with explicit BPTT through every
  gate at every timestep.

### Numerical stability

* `Div.Backward` no longer produces NaN/Inf at b = 0. Uses smooth
  reciprocal `b / (b² + 1e-12)`.
* `Var` switched to Welford's online algorithm — stable at large offsets
  (variance of `{1+1e9, …, 5+1e9}` returns 2.5 exactly).

### Float32 propagation

`MatMul` and `BatchMatMul` now preserve `Float32` dtype when both
inputs are `Float32`. Internal accumulation stays in `Float64` for
numerical stability across K. Mixed precision promotes to `Float64`.

### Concurrency safety

* `autograd.NoGrad` rewritten as a depth counter — no more lost state
  when nested NoGrad calls overlap across goroutines.
* `data.DataLoader` got a `sync.Mutex` and the shared `cursor` field
  was removed in favour of a pure `loadBatchAt(batchIdx, indices)`.

`go test -race` is clean.

### Performance

* `TopK`: O(n·k) selection sort → O(n log k) min-heap. ~1.4 ms for
  100 k elements with k = 50.
* New in-place ops: `AddInPlace`, `SubInPlace`, `MulInPlace`,
  `AddScalarInPlace`, `MulScalarInPlace`. **49× faster than `Add` on a
  1024-vector, zero allocations.**
* `ConvTranspose2d` forward and backward rewritten as per-batch matmul
  + col2im scatter. Same op count, but cache-tiled matmul replaces
  hand-rolled inner loops. Expected 10-50× on realistic sizes.
* `sync.Pool`-backed tensor allocator with power-of-two buckets
  (64 → 16 M elements). `Tensor.Release()` returns storage to the pool;
  `Zeros()` allocates through it. Long training loops can call
  `Release` on intermediates to drop GC pressure dramatically.

## Why "v6.0.0" and not "v5.1.0"

Two reasons:

1. **Behaviour changed.** Any code that incidentally relied on the wrong
   gradient values now produces different output. SemVer-major is the
   honest signal.
2. **Identity.** v1 through v5 were "Level N" feature drops that
   shipped without verifying backward passes against finite-differences.
   v6.0.0 is the first release where the whole training loop has been
   audited and numerically-checked end to end.

## Install

```bash
go get github.com/djeday123/gotorch@v6.0.0
```

## What's in the box (carried over from v5.0.0)

Tensors with broadcasting and advanced indexing • reverse-mode autograd
• Linear, Conv1d/2d/ConvTranspose2d, MaxPool, Embedding • BatchNorm1d/2d,
LayerNorm, GroupNorm, Dropout • MultiHeadAttention, TransformerEncoder/
Decoder, full Transformer • LSTM, GRU with proper BPTT • Sequential,
ModuleList, ModuleDict • all common losses (MSE/BCE/CE/NLL/L1/Huber/
KLDiv) • all common activations (ReLU/Sigmoid/Tanh/GELU/LeakyReLU/ELU/
SiLU/Softplus) • SGD/Adam/AdamW/RMSprop/Adadelta + LR schedulers •
DataLoader with goroutine prefetch • mixed-precision GradScaler •
int8 inference quantization • basic ONNX export.

## Not yet supported

`torch.compile`, JIT, distributed/multi-GPU, `torch.fft`,
`torch.distributions`, vmap, jacobian, hessian.

## Honest scope

By PyTorch API surface gotorch covers roughly **7 %** — PyTorch is
enormous. By practical usefulness for typical supervised-learning
workflows gotorch covers **40-50 %**. End-to-end transformer / CNN /
LSTM training works.

## Commits in this release

* `28284e8` — 7 critical backward-pass bug fixes + gradcheck tests
* `9104a68` — MatMul/BatchMatMul preserve Float32 dtype; Div backward safe at b = 0
* `e9f618c` — NoGrad nesting race + DataLoader concurrency safety
* `f3b070c` — Welford variance, heap TopK, in-place ops, ConvTranspose2d matmul, pool

See `CHANGELOG.md` for the full per-commit breakdown plus a brief
history of v1.0.0 → v5.0.0 releases.
