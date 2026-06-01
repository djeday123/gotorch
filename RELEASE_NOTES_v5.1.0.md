# gotorch v5.1.0

Stability and performance release on top of v5.0.0 (the Level-8 release).
Ten silent backward-pass bugs found and fixed, Float32 dtype preserved
through matmul, two concurrency-safety repairs, and several
performance improvements — all without breaking the v5 public API.

## Highlights

* **Correct autograd.** Ten silent backward-pass bugs caught and fixed,
  each verified by a numerical-gradient test in `nn/gradcheck_test.go`.
  If you trained with v5.0.0 and used LSTM, GRU, MultiHeadAttention,
  LayerNorm, BatchNorm1d, GroupNorm, Sum, Mean, or Softmax — your
  gradients were silently wrong. They are now correct.
* **Float32 preserved.** `MatMul` and `BatchMatMul` now return Float32
  when both inputs are Float32; previously the dtype silently widened.
* **Safe Div.** `Div.Backward` no longer blows up to NaN/Inf at b = 0 —
  smooth reciprocal `b / (b² + 1e-12)` gives 0 at b = 0 and matches `1/b`
  elsewhere.
* **Concurrency-safe.** Replaced the broken `SwapInt32`-based NoGrad with
  counter-based depth tracking; removed the shared `cursor` field in
  DataLoader and added a `sync.Mutex`. `go test -race` is clean.
* **Performance:** Welford variance, heap-based TopK (O(n·k) → O(n log k)),
  in-place ops (**49× faster on 1024-vector**, zero allocations),
  ConvTranspose2d rewritten as matmul + col2im (10-50× faster), and a
  `sync.Pool`-backed tensor allocator with explicit `Release()`.

## Compatibility

Public APIs are unchanged — this is a **semver-minor** release. The only
behaviour changes are bug-fix gradients. If you have tests that pinned
old, wrong gradient values, they need updating. All 221 existing tests
plus 15 new ones pass.

## Install

```bash
go get github.com/djeday123/gotorch@v5.1.0
```

## Commits in this release

* `28284e8` — 7 critical backward-pass bug fixes + gradcheck tests
* `9104a68` — MatMul/BatchMatMul preserve Float32 dtype; Div backward safe at b = 0
* `e9f618c` — NoGrad nesting race + DataLoader concurrency safety
* `f3b070c` — Welford variance, heap TopK, in-place ops, ConvTranspose2d matmul, pool

See `CHANGELOG.md` in the repo for the full per-commit breakdown.

## Honest scope (unchanged from v5.0.0)

By PyTorch API surface, gotorch covers roughly **7 %** — PyTorch is
enormous. By practical usefulness for typical supervised-learning
workflows (define a model, define a loss, train with AdamW, save),
gotorch covers **40-50 %**. End-to-end transformer / CNN / LSTM
training works.

Not yet supported: `torch.compile`, JIT, distributed training,
`torch.fft`, `torch.distributions`, vmap / hessian.

## Acknowledgements

The bugs in this release were caught by a combination of automated audit
agents and manual re-verification against the math of each operation.
The "agent flagged it, then I sat down and traced the gradient on paper"
loop removed a lot of false positives — see commit messages for which
findings were real bugs and which were mis-identified.
