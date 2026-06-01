# Changelog

All notable changes to gotorch are documented here. Format roughly follows
[Keep a Changelog](https://keepachangelog.com/). Versions use
[Semantic Versioning](https://semver.org/).

## [Unreleased]

— ConvTranspose2d via `tensor.MatMul` was the last major perf rewrite before
the v0.1.0 cut; no unreleased changes pending at this time.

---

## [v0.1.0] — 2026-05-24

First tagged release. The API is stable enough for downstream projects to
pin against, but several PyTorch features (distributed, JIT, torch.compile)
remain unimplemented — hence the `0.x` line for now.

### Fixed — critical backward-pass bugs (commit `28284e8`)

These bugs silently produced wrong gradients during training. Each fix is
verified by a numerical-gradient test in `nn/gradcheck_test.go`.

* **`autograd/functions.go`**: `Sum.Backward` ignored the incoming upstream
  gradient and always returned `Ones(shape)`. Now correctly returns
  `MulScalar(Ones, grad.Item())`.
* **`autograd/functions.go`**: `Mean.Backward` had the same shape of bug;
  now multiplies by `grad.Item() / n`.
* **`autograd/functions.go`**: `Softmax.Backward` was an identity passthrough.
  Now implements the real Jacobian `dx = s · (g − sum(g·s, dim))`.
* **`nn/norm.go`**: `LayerNorm.Backward` was missing the `1/std` factor (the
  forward never stored std). The forward now records per-group std; the
  backward uses the correct formula `(γ · dy − sumDy/M − xn · sumDyXn/M) / std`.
* **`nn/norm.go`**: `BatchNorm1d` returned `requiresGrad=false`, so gradients
  could not flow through it. Replaced with a full backward path that handles
  both training (per-batch stats) and eval (running stats) modes.
* **`nn/norm.go`**: `GroupNorm` had the same broken-grad-flow problem. Now
  has a proper backward with per-(n, group) std.
* **`nn/attention.go`**: `MultiHeadAttention.Backward` returned the same
  upstream gradient to Q, K, and V identically. Replaced with the proper
  chain rule using stored attention weights and per-head matmul:
  `dVh = attnᵀ · dHead`, `dAttn = dHead · Vhᵀ`,
  `dScore = softmax_backward(dAttn, attn)`,
  `dQh = dScore · Kh · scale`, `dKh = dScoreᵀ · Qh · scale`.
* **`nn/rnn.go`**: `LSTM` and `GRU` gates were computed as raw `[]float64`
  operations outside the autograd graph, so backpropagation-through-time
  never reached the gate weights. Rewritten as a single sequence-level
  autograd op with explicit BPTT through all gates at every timestep.
  Per-timestep outputs are exposed via a new `rowSliceBackward` op.

### Fixed — Float32 propagation and Div-by-zero (commit `9104a68`)

* **`tensor/linalg.go`**: `MatMul` and `BatchMatMul` now preserve `Float32`
  dtype when both inputs are `Float32`. Internal accumulation stays in
  `Float64` for numerical stability across K. Mixed precision promotes to
  `Float64`. Previously the result was always `Float64`, silently widening
  the data type. `BatchMatMul` additionally promotes inputs internally
  because `slice2D` indexes `data []float64` directly and does not yet
  support `Float32` storage.
* **`autograd/functions.go`**: `Div.Backward` no longer produces NaN/Inf
  when `b` contains exact zeros. The denominator becomes `b² + 1e-12`, and
  the smooth reciprocal `b / (b² + 1e-12)` replaces `1/b`. At `b = 0` the
  gradient on `a` is exactly `0`; the gradient on `b` is large but finite,
  so upstream clipping can keep training stable instead of seeing NaN.

### Fixed — concurrency safety (commit `e9f618c`)

* **`autograd/no_grad.go`**: Replaced the `SwapInt32`/`StoreInt32` pattern
  with separate depth counters (`noGradDepth`, `enableGradDepth`). The old
  approach lost state under nested NoGrad calls from different goroutines.
  New test `TestNoGradConcurrentNesting` (200 iterations × 2 goroutines)
  exercises the corner case and passes under `go test -race`.
* **`data/dataloader.go`**: Removed the shared `cursor` field. `loadBatch`
  is now a pure `loadBatchAt(batchIdx, indices)` with no shared state. A
  `sync.Mutex` serialises `Reset` / `HasNext` / `Next` / `NumBatches`. The
  prefetch worker receives `ch`, `done`, `total`, and `indices` by value so
  `Reset` can rotate the channel safely.

### Added — performance and correctness improvements (commit `f3b070c`)

* **`tensor/extra.go`**: `Var` uses Welford's online algorithm on the
  all-elements path. Stable under large offsets — variance of
  `{1 + 1e9, …, 5 + 1e9}` returns 2.5 (unbiased N-1) exactly.
* **`tensor/extra.go`**: `TopK` switched from O(n·k) selection sort to
  O(n log k) min-heap of size k. Benchmark: 100k elements with k=50 takes
  ~1.4 ms per call.
* **`tensor/ops.go`**: New in-place methods on `Tensor`: `AddInPlace`,
  `SubInPlace`, `MulInPlace`, `AddScalarInPlace`, `MulScalarInPlace`. Zero
  allocations, mutate the receiver. Require same shape, same dtype, and a
  contiguous receiver. Benchmark on `[1024]`: `AddInPlace` 368 ns vs
  `Add` 17936 ns — **49× faster, 0 allocations vs 14**.
* **`nn/conv_transpose2d.go`**: Forward and backward rewritten as per-batch
  matmul + col2im scatter. Same multiplication count as the previous
  quintuple-nested-loop scalar path, but cache-tiled `tensor.MatMul`
  replaces hand-rolled inner loops. Expected 10-50× faster on realistic
  sizes.
* **`tensor/pool.go`**: New `sync.Pool`-backed allocator with power-of-two
  buckets (64 to 16 M elements). `AllocFloat64` / `FreeFloat64`,
  `AllocFloat32` / `FreeFloat32`, and `Tensor.Release()` for explicit
  reuse. `Zeros()` now goes through the pool. Long training loops can
  call `Release` on intermediates to drop GC pressure.

### Known limitations in v0.1.0

* No `torch.compile`, JIT, or graph capture.
* No distributed / multi-GPU training (no `DataParallel`, `DistributedDataParallel`).
* `cuda/` backend is narrow and CPU-first; goml uses a richer CUDA stack.
* Sparse tensors limited to COO format with few ops.
* `torch.fft`, `torch.distributions`, second-order autograd (vmap, hessian)
  are not implemented.
* Native Float32 compute path (without Float64 promotion in inner loops) is
  not yet present; would require generics or duplicated function bodies.

[Unreleased]: https://github.com/djeday123/gotorch/compare/v0.1.0...HEAD
[v0.1.0]: https://github.com/djeday123/gotorch/releases/tag/v0.1.0
