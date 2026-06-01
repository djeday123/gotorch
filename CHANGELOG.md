# Changelog

All notable changes to gotorch are documented here. Format roughly follows
[Keep a Changelog](https://keepachangelog.com/). Versions use
[Semantic Versioning](https://semver.org/).

## [Unreleased]

— v5.1.0 cut on 2026-05-24; no unreleased changes pending at this time.

---

## [v5.1.0] — 2026-05-24

Stability and performance release on top of v5.0.0. Ten silent backward-pass
bugs found and fixed (each verified by a numerical-gradient test), Float32
dtype preservation in matmul, concurrency-safety repairs, and several
performance improvements that leave the public API backward-compatible.

If you trained with v5.0.0 or earlier and used `LSTM`, `GRU`,
`MultiHeadAttention`, `LayerNorm`, `BatchNorm1d`, `GroupNorm`, `Sum`,
`Mean`, or `Softmax` in a way that relied on gradients, your gradients
were silently wrong. They are now correct.

### Fixed — critical backward-pass bugs (commit `28284e8`)

* **`autograd/functions.go`**: `Sum.Backward` ignored the upstream gradient
  and always returned `Ones(shape)`. Now `MulScalar(Ones, grad.Item())`.
* **`autograd/functions.go`**: `Mean.Backward` had the same bug; now
  multiplies by `grad.Item() / n`.
* **`autograd/functions.go`**: `Softmax.Backward` was an identity
  passthrough. Now implements the real Jacobian
  `dx = s · (g − sum(g · s, dim))`.
* **`nn/norm.go`**: `LayerNorm.Backward` was missing the `1/std` factor
  (the forward never stored std). The forward now records per-group std;
  the backward uses `(γ · dy − sumDy/M − xn · sumDyXn/M) / std`.
* **`nn/norm.go`**: `BatchNorm1d` returned `requiresGrad=false` — gradients
  could not flow through it. Replaced with a full backward path for both
  training (per-batch stats) and eval (running stats) modes.
* **`nn/norm.go`**: `GroupNorm` had the same broken-grad-flow problem.
  Now has a proper backward with per-(n, group) std.
* **`nn/attention.go`**: `MultiHeadAttention.Backward` was returning the
  same upstream gradient to Q, K, and V identically. Replaced with the
  proper chain rule using stored attention weights and per-head matmul:
  `dVh = attnᵀ · dHead`, `dAttn = dHead · Vhᵀ`,
  `dScore = softmax_backward(dAttn, attn)`,
  `dQh = dScore · Kh · scale`, `dKh = dScoreᵀ · Qh · scale`.
* **`nn/rnn.go`**: `LSTM` and `GRU` gates were computed as raw `[]float64`
  operations outside the autograd graph, so backpropagation-through-time
  never reached the gate weights. Rewritten as a single sequence-level
  autograd op with explicit BPTT through all gates at every timestep.
  Per-timestep outputs exposed via a new `rowSliceBackward` op.

### Fixed — Float32 propagation and Div-by-zero (commit `9104a68`)

* **`tensor/linalg.go`**: `MatMul` and `BatchMatMul` now preserve `Float32`
  dtype when both inputs are `Float32`. Internal accumulation stays in
  `Float64` for numerical stability across K. Mixed precision promotes to
  `Float64`. Previously the result was always `Float64`, silently widening
  the data type.
* **`autograd/functions.go`**: `Div.Backward` no longer produces NaN/Inf
  when `b` contains exact zeros. The denominator becomes `b² + 1e-12`; the
  smooth reciprocal `b / (b² + 1e-12)` replaces `1/b`. At `b = 0` the
  gradient on `a` is exactly `0`; the gradient on `b` is large but finite.

### Fixed — concurrency safety (commit `e9f618c`)

* **`autograd/no_grad.go`**: Replaced the `SwapInt32`/`StoreInt32` pattern
  with separate depth counters (`noGradDepth`, `enableGradDepth`). The old
  approach lost state under nested NoGrad calls from different goroutines.
  New test `TestNoGradConcurrentNesting` (200 iterations × 2 goroutines)
  exercises the corner case and passes under `go test -race`.
* **`data/dataloader.go`**: Removed the shared `cursor` field. `loadBatch`
  is now the pure `loadBatchAt(batchIdx, indices)` with no shared state.
  A `sync.Mutex` serialises `Reset` / `HasNext` / `Next` / `NumBatches`.
  The prefetch worker receives `ch`, `done`, `total`, `indices` by value
  so `Reset` can rotate the channel safely.

### Added — performance and correctness improvements (commit `f3b070c`)

* **`tensor/extra.go`**: `Var` uses Welford's online algorithm on the
  all-elements path. Stable under large offsets — variance of
  `{1 + 1e9, …, 5 + 1e9}` returns 2.5 (unbiased N-1) exactly.
* **`tensor/extra.go`**: `TopK` switched from O(n·k) selection sort to
  O(n log k) min-heap of size k. 100k elements with k=50 ≈ 1.4 ms/call.
* **`tensor/ops.go`**: New in-place methods on `Tensor`: `AddInPlace`,
  `SubInPlace`, `MulInPlace`, `AddScalarInPlace`, `MulScalarInPlace`.
  Zero allocations, mutate the receiver. Require same shape, same dtype,
  and a contiguous receiver. On `[1024]`: `AddInPlace` 368 ns vs `Add`
  17 936 ns — **49× faster, 0 allocations vs 14**.
* **`nn/conv_transpose2d.go`**: Forward and backward rewritten as
  per-batch matmul + col2im scatter. Same multiplication count as the
  previous quintuple-nested-loop scalar path, but cache-tiled
  `tensor.MatMul` replaces hand-rolled inner loops. Expected 10-50× on
  realistic sizes.
* **`tensor/pool.go`**: New `sync.Pool`-backed allocator with
  power-of-two buckets (64 to 16 M elements). `AllocFloat64` /
  `FreeFloat64`, `AllocFloat32` / `FreeFloat32`, and `Tensor.Release()`
  for explicit reuse. `Zeros()` goes through the pool. Long training
  loops can call `Release` on intermediates to drop GC pressure.

### Compatibility note

Public APIs are unchanged. The only **behaviour** changes are bug-fix
gradients (above) — if you happen to have integration tests that pinned
old, wrong gradient values, they will need updating. The 221 existing
gotorch tests (plus 15 new ones added in this release) all pass.

### Known limitations carried forward from v5.0.0

* No `torch.compile`, JIT, or graph capture.
* No distributed / multi-GPU training (`DataParallel`, `DistributedDataParallel`).
* `cuda/` backend is narrow and CPU-first; goml uses a richer CUDA stack.
* Sparse tensors limited to COO format with few ops.
* `torch.fft`, `torch.distributions`, second-order autograd (vmap, hessian)
  are not implemented.
* Native Float32 compute path (without Float64 promotion in inner loops)
  is not yet present.

---

## Earlier tagged releases

For full history of v1.0.0 through v5.0.0 see git tags directly. Headline
features per "Level":

* **v5.0.0** — Level 8: TransformerDecoder, full Transformer, StackedLSTM/GRU,
  ModuleDict, SparseCOO (312 tests)
* **v4.0.0** — FlashAttention 153 T standalone bench
* **v3.0.0** — RMSNorm, SwiGLU, RoPE
* **v2.1.0** — Level 5: nn.functional, ConvTranspose2d, Upsample, Tril/Triu
* **v2.0.0** — Level 4: DataLoader, AMP, Conv1d, RMSprop
* **v1.1.0** — Level 1: Float32, Conv2d, NoGrad, Cat/Stack/Split
* **v1.0.0** — Foundation: tensor, autograd, Linear, SGD, Adam, CUDA

[Unreleased]: https://github.com/djeday123/gotorch/compare/v5.1.0...HEAD
[v5.1.0]: https://github.com/djeday123/gotorch/releases/tag/v5.1.0
