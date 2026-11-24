"""Batched FP8 contraction on Ascend Cube and per-tensor scaling in FP32."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.ascend.gemm.fp8 import _make_epilogue
from tileops._kernels.gemm.call_spec import (
    BMMFP8Call,
    BMMFP8FwdInterface,
    BMMFP8TransposeCall,
    BMMFP8TransposeFwdInterface,
)
from tileops._kernels.kernel_base import Entry, Kernel
from tileops.backend._registry import register_kernel_type


@functools.lru_cache(maxsize=32)
def _contract(batch, m, n, k):
    bm = min(128, (m + 15) // 16 * 16)
    bn = min(128, (n + 15) // 16 * 16)
    bk = next(v for v in (256, 128, 64, 32) if k % v == 0)
    mad = min(64, bk)
    mt, nt = (m + bm - 1) // bm, (n + bn - 1) // bn
    jobs = batch * mt * nt
    grid = min(32, jobs)
    dtype = "float8_e4m3fn"

    @T.prim_func
    def main(
        a: T.Tensor((batch, m, k), dtype),
        b: T.Tensor((batch, n, k), dtype),
        output: T.Tensor((batch, m, n), "float32"),
    ):
        with T.Kernel(grid) as bx:
            lhs = T.alloc_l1((bm, bk), dtype)
            rhs = T.alloc_l1((bn, bk), dtype)
            l0a = T.alloc_l0a((bm, mad), dtype)
            l0b = T.alloc_l0b((bn, mad), dtype)
            acc = T.alloc_l0c((bm, bn), "float32")
            T.annotate_buffer_versions({lhs: 2, rhs: 2, l0a: 2, l0b: 2})
            for wave in T.serial(T.ceildiv(jobs, grid)):
                task = wave * grid + bx
                if task < jobs:
                    bi = task // (mt * nt)
                    row = (task // nt % mt) * bm
                    col = task % nt * bn
                    rows, cols = T.min(bm, m - row), T.min(bn, n - col)
                    for step in T.Pipelined(k // bk, num_stages=2):
                        first = step * bk
                        T.copy(a[bi, row : row + rows, first : first + bk], lhs[:rows, :])
                        T.copy(b[bi, col : col + cols, first : first + bk], rhs[:cols, :])
                        for part in T.serial(bk // mad):
                            base = part * mad
                            T.copy(lhs[:rows, base : base + mad], l0a[:rows, :])
                            T.copy(rhs[:cols, base : base + mad], l0b[:cols, :])
                            T.gemm(
                                l0a[:rows, :],
                                l0b[:cols, :],
                                acc[:rows, :cols],
                                transpose_B=True,
                                clear_accum=(step == 0 and part == 0),
                            )
                    T.copy(
                        acc[:rows, :cols],
                        output[bi, row : row + rows, col : col + cols],
                    )

    return tilelang.compile(main, target="ascend", out_idx=[2])


@functools.lru_cache(maxsize=32)
def _transpose(batch, rows, cols):
    tile = 64
    rm, cn = (rows + tile - 1) // tile, (cols + tile - 1) // tile
    jobs = batch * rm * cn
    grid = min(64, jobs)

    @T.prim_func
    def main(
        src: T.Tensor((batch, rows, cols), "uint8"),
        dst: T.Tensor((batch, cols, rows), "uint8"),
    ):
        with T.Kernel(grid) as bx:
            x = T.alloc_shared((tile, tile), "uint8")
            y = T.alloc_shared((tile, tile), "uint8")
            for wave in T.serial(T.ceildiv(jobs, grid)):
                job = wave * grid + bx
                if job < jobs:
                    bi = job // (rm * cn)
                    r = job // cn % rm * tile
                    c = job % cn * tile
                    nr, nc = T.min(tile, rows - r), T.min(tile, cols - c)
                    T.copy(src[bi, r : r + nr, c : c + nc], x[:nr, :nc])
                    with T.SimtVF(threads=256):
                        for i, j in T.Parallel(tile, tile):
                            if i < nc and j < nr:
                                y[i, j] = x[j, i]
                    T.copy(y[:nc, :nr], dst[bi, c : c + nc, r : r + nr])

    return tilelang.compile(main, target="ascend", out_idx=[1])


class AscendBMMFP8Kernel(Kernel, BMMFP8FwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: BMMFP8Call):
        if call.dtype != torch.float8_e4m3fn:
            return "requires float8_e4m3fn"
        if call.k % 32:
            return f"requires K a multiple of 32, got {call.k}"
        if call.out_dtype not in (torch.float16, torch.bfloat16):
            return "requires a float16 or bfloat16 output"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: BMMFP8Call) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: BMMFP8Call):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(self, a, b, a_scale, b_scale):
        c = self.call
        shape = (c.batch, c.m, c.n)
        if min(shape) == 0:
            return torch.empty(shape, device=a.device, dtype=c.out_dtype)
        if c.k:
            acc = _contract(c.batch, c.m, c.n, c.k)(a, b)
        else:
            acc = torch.zeros(shape, device=a.device, dtype=torch.float32)
        epilogue = _make_epilogue(
            c.batch * c.m, c.n, str(c.out_dtype).removeprefix("torch."), False
        )
        return epilogue(acc.view(-1, c.n), a_scale.reshape(1, 1), b_scale.reshape(1, 1)).view(shape)


class AscendBMMFP8TransposeKernel(Kernel, BMMFP8TransposeFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: BMMFP8TransposeCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: BMMFP8TransposeCall):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(self, src):
        c = self.call
        if not src.numel():
            return src.new_empty((c.batch, c.cols, c.rows))
        return _transpose(c.batch, c.rows, c.cols)(src.view(torch.uint8)).view(src.dtype)


register_kernel_type("BMMFP8FwdOp", "ascend", AscendBMMFP8Kernel)
register_kernel_type("BMMFP8FwdOp", "ascend_transpose", AscendBMMFP8TransposeKernel)
