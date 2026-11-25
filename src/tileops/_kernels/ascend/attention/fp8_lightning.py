"""Ascend FP8 grouped index scores with per-key scaling and device-side windows."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch
import torch.nn.functional as F

from tileops._kernels.attention.fp8_lightning_indexer import (
    FP8LightningIndexerFwdInterface,
)
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type


@functools.lru_cache(maxsize=32)
def _scores(batch, seq, sk, heads, groups, dim):
    # The Cube yields one row of head scores per key. Each Vector subblock
    # reduces half the keys, keeping per-head scores out of global memory.
    bc, bh = 128, 64
    hg = heads // groups
    hp = (hg + bh - 1) // bh * bh
    kp = (sk + bc - 1) // bc * bc
    dp = (dim + 31) // 32 * 32
    mad = 64 if dp % 64 == 0 else 32
    nt = kp // bc
    query_jobs = batch * seq * groups
    splits = min(nt, max(1, (64 + query_jobs - 1) // query_jobs))
    jobs = query_jobs * splits
    grid = min(32, jobs)
    fp8 = "float8_e4m3fn"

    @T.prim_func
    def main(
        q: T.Tensor((batch, seq, groups, hp, dp), fp8),
        k: T.Tensor((batch, kp, groups, dp), fp8),
        scales: T.Tensor((batch, kp, groups), "float32"),
        weights: T.Tensor((seq, groups, hp), "float32"),
        starts: T.Tensor((seq,), "int32"),
        ends: T.Tensor((seq,), "int32"),
        output: T.Tensor((batch, seq, sk, groups), "float32"),
    ):
        with T.MixedKernel(grid) as (bx, sub):
            keys = T.alloc_l1((bc, dp), fp8)
            queries = T.alloc_l1((bh, dp), fp8)
            a = T.alloc_l0a((bc, mad), fp8)
            b = T.alloc_l0b((bh, mad), fp8)
            c = T.alloc_l0c((bc, bh), "float32")
            dots = T.alloc_shared((bc // 2, bh), "float32")
            w = T.alloc_shared((bh,), "float32")
            scale = T.alloc_shared((bc // 2,), "float32")
            sums = T.alloc_shared((bc // 2,), "float32")
            for job in T.serial(bx, jobs, grid):
                query_job = job // splits
                bi = query_job // (groups * seq)
                qi = query_job // groups % seq
                g = query_job % groups
                split = job % splits
                first = T.max(0, starts[qi])
                last = T.min(sk, ends[qi])
                first_tile = first // bc
                num_tiles = T.if_then_else(last > first, T.ceildiv(last, bc) - first_tile, 0)
                begin = first_tile + num_tiles * split // splits
                end = first_tile + num_tiles * (split + 1) // splits
                if hp == bh:
                    T.copy(q[bi, qi, g, :, :], queries)
                    T.copy(weights[qi, g, :], w)
                for key_tile in T.serial(begin, end):
                    start = key_tile * bc
                    T.copy(k[bi, start : start + bc, g, :], keys)
                    T.copy(
                        scales[
                            bi,
                            start + sub * (bc // 2) : start + (sub + 1) * (bc // 2),
                            g,
                        ],
                        scale,
                    )
                    for head_tile in T.serial(hp // bh):
                        h = head_tile * bh
                        if hp != bh:
                            T.copy(q[bi, qi, g, h : h + bh, :], queries)
                            T.copy(weights[qi, g, h : h + bh], w)
                        for j in T.serial(dp // mad):
                            col = j * mad
                            T.copy(keys[:, col : col + mad], a)
                            T.copy(queries[:, col : col + mad], b)
                            T.gemm(a, b, c, transpose_B=True, clear_accum=j == 0)
                        T.dual_copy(c, dots)
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            one = T.simd.pset(32, "PAT_VL1")
                            ws = T.simd.vld(w[0])
                            for row in T.serial(bc // 2):
                                z = T.simd.vmul(
                                    T.simd.vld(dots[row, 0]),
                                    T.simd.vld(scale[row], dist="BRC_B32"),
                                    mask,
                                )
                                val = T.simd.vcadd(
                                    T.simd.vmul(T.simd.vmaxs(z, T.float32(0), mask), ws, mask),
                                    mask,
                                )
                                if head_tile == 0:
                                    T.simd.vsts(sums[row], val, one, dist="ONEPT_B32")
                                else:
                                    prev = T.simd.vld(sums[row], dist="BRC_B32")
                                    T.simd.vsts(
                                        sums[row],
                                        T.simd.vadd(prev, val, one),
                                        one,
                                        dist="ONEPT_B32",
                                    )
                    with T.SimtVF(threads=64):
                        for r in T.Parallel(bc // 2):
                            pos = start + sub * (bc // 2) + r
                            if first <= pos and pos < last:
                                output[bi, qi, pos, g] = sums[r]

    return tilelang.compile(main, target="ascend", out_idx=[])


class AscendFP8LightningIndexerKernel(Kernel, FP8LightningIndexerFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(self, IndexQ, IndexK, IndexKScale, Weights, Starts, Ends):
        c = self.call
        b, s, h, d = IndexQ.shape
        _, n, g, _ = IndexK.shape
        if not (b and s and n):
            return torch.empty((b, s, n, g), device=IndexQ.device, dtype=torch.float32)
        hg = h // g
        q = IndexQ.reshape(b, s, g, hg, d)
        w = Weights.reshape(s, g, hg)
        key = IndexK
        scale = IndexKScale
        if d % 32 or hg % 64:
            q = F.pad(q.view(torch.uint8), (0, (-d) % 32, 0, (-hg) % 64)).view(IndexQ.dtype)
        if hg % 64:
            w = F.pad(w, (0, (-hg) % 64))
        if n % 128 or d % 32:
            key = F.pad(key.view(torch.uint8), (0, (-d) % 32, 0, 0, 0, (-n) % 128)).view(
                IndexK.dtype
            )
        if n % 128:
            scale = F.pad(scale, (0, 0, 0, (-n) % 128))
        shape = (b, s, n, g)
        output = (
            torch.full(shape, -torch.inf, dtype=torch.float32, device=IndexQ.device)
            if c.clean_logits
            else torch.empty(shape, dtype=torch.float32, device=IndexQ.device)
        )
        _scores(b, s, n, h, g, d)(q, key, scale, w, Starts, Ends, output)
        return output


register_kernel_type("FP8LightningIndexerFwdOp", "ascend", AscendFP8LightningIndexerKernel)
