"""
MatMul-LayerNorm Warps Benchmark
================================
Benchmarks matmul_layernorm to compare performance with and without
the normalize_num_warps clamping (HELION_NORMALIZE_NUM_WARPS).

Usage:
    HELION_NORMALIZE_NUM_WARPS=1 HELION_PRINT_OUTPUT_CODE=1 python examples/layer_norm_warps_bench.py
    HELION_NORMALIZE_NUM_WARPS=0 HELION_PRINT_OUTPUT_CODE=1 python examples/layer_norm_warps_bench.py
"""

from __future__ import annotations

import time

import torch

import helion
import helion.language as hl
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE

M, K, N = 1024, 1024, 1024
NUM_WARPS = 32
BLOCK_SIZE_M = 1
BLOCK_SIZE_K = 64


@helion.kernel(
    static_shapes=True,
    config=helion.Config(
        block_sizes=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        num_warps=NUM_WARPS,
        num_stages=1,
        indexing="block_ptr",
    ),
)
def matmul_layernorm(
    x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    k2 = y.size(0)
    n = hl.specialize(y.size(1))
    assert k == k2, f"size mismatch {k} != {k2}"
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"
    assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {n}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            mm = torch.matmul(x[tile_m, tile_k], y[tile_k, :])
            acc = acc + mm
        eps = 1e-5
        sum_vals = acc.sum(dim=-1, keepdim=True)
        mean = sum_vals / n
        centered = acc - mean
        var = (centered * centered).sum(dim=-1, keepdim=True) / n
        normalized = centered * torch.rsqrt(var + eps)
        acc = normalized * (weight[:].to(torch.float32)) + (bias[:].to(torch.float32))
        out[tile_m, :] = acc
    return out


def main() -> None:
    x = torch.randn(M, K, device=DEVICE, dtype=HALF_DTYPE)
    y = torch.randn(K, N, device=DEVICE, dtype=HALF_DTYPE)
    weight = torch.randn(N, device=DEVICE, dtype=HALF_DTYPE)
    bias = torch.randn(N, device=DEVICE, dtype=HALF_DTYPE)

    # Warmup
    for _ in range(5):
        out = matmul_layernorm(x, y, weight, bias)
    torch.cuda.synchronize()

    # Benchmark
    iters = 200
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        matmul_layernorm(x, y, weight, bias)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed_us = (t1 - t0) / iters * 1e6
    print(f"Average kernel time: {elapsed_us:.1f} us  ({iters} iters)")

    # Correctness check
    ref_mm = torch.matmul(x, y)
    ref = torch.nn.functional.layer_norm(
        ref_mm.to(torch.float32),
        [N],
        weight.to(torch.float32),
        bias.to(torch.float32),
    ).to(HALF_DTYPE)
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)
    print("Correctness: PASS")


if __name__ == "__main__":
    main()
