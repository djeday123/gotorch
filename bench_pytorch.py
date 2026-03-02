"""
PyTorch transfer benchmarks — аналог Go pinned_test.go
Сравниваем: pageable H2D, pinned H2D, pageable D2H, pinned D2H
Данные: 1M float64 = 8 MB (как в Go бенчмарках)
"""
import torch
import time

N = 1_000_000
BYTES = N * 8
WARMUP = 50
RUNS = 500

def bench(label, fn, warmup=WARMUP, runs=RUNS):
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ns_per_op = elapsed / runs * 1e9
    mb_per_s  = (BYTES * runs) / elapsed / 1e6
    print(f"{label:<30} {ns_per_op:>10.0f} ns/op   {mb_per_s:>10.1f} MB/s")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print(f"Data: {N:,} float64 = {BYTES/1e6:.0f} MB\n")

# ── H2D ─────────────────────────────────────────────────────────────────────

# Pageable H2D: regular CPU tensor (pageable) → CUDA
t_pageable = torch.ones(N, dtype=torch.float64)  # NOT pin_memory → pageable
def h2d_pageable():
    t = t_pageable.cuda()
    torch.cuda.synchronize()

# Pinned H2D: pinned CPU tensor → CUDA
t_pinned = torch.ones(N, dtype=torch.float64, pin_memory=True)
def h2d_pinned():
    t = t_pinned.cuda(non_blocking=True)
    torch.cuda.synchronize()

# ── D2H ─────────────────────────────────────────────────────────────────────

t_gpu = torch.ones(N, dtype=torch.float64, device='cuda')

# Pageable D2H: CUDA → regular CPU tensor
def d2h_pageable():
    t = t_gpu.cpu()
    torch.cuda.synchronize()

# Pinned D2H: CUDA → pinned CPU tensor
t_pinned_dst = torch.empty(N, dtype=torch.float64, pin_memory=True)
def d2h_pinned():
    t_pinned_dst.copy_(t_gpu, non_blocking=True)
    torch.cuda.synchronize()

print("─── H2D (CPU → GPU) ───")
bench("Pageable H2D (numpy→cuda)", h2d_pageable)
bench("Pinned   H2D (pin→cuda) ", h2d_pinned)

print("\n─── D2H (GPU → CPU) ───")
bench("Pageable D2H (cuda→cpu) ", d2h_pageable)
bench("Pinned   D2H (cuda→pin) ", d2h_pinned)
