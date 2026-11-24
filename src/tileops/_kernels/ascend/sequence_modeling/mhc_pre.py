"""Ascend split projection, Sinkhorn normalization and stream mixing for mHC."""

import functools
import math

import tilelang
import tilelang.ascend.language as T

from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.sequence_modeling.mhc.call_spec import MHCPreCall, MHCPreFwdInterface
from tileops.backend._registry import register_kernel_type


@functools.lru_cache(maxsize=32)
def _projection(batch, width, mixes, dtype):
    chunk = 64
    parts = (width + chunk - 1) // chunk

    @T.prim_func
    def main(
        phi: T.Tensor((width, mixes), "float32"),
        x: T.Tensor((batch, width), dtype),
        partial: T.Tensor((batch, parts, mixes + 1), "float32"),
    ):
        with T.Kernel(batch * parts) as bx:
            b = bx // parts
            start = bx % parts * chunk
            count = T.min(chunk, width - start)
            xs = T.alloc_shared((chunk,), dtype)
            ws = T.alloc_shared((chunk, mixes), "float32")
            ps = T.alloc_shared((mixes + 1,), "float32")
            T.copy(x[b, start : start + count], xs[:count])
            T.copy(phi[start : start + count, :], ws[:count, :])
            with T.SimtVF(threads=128):
                for j in T.Parallel(mixes + 1):
                    acc = T.alloc_local((1,), "float32")
                    acc[0] = 0
                    for k in T.serial(count):
                        value = T.cast(xs[k], "float32")
                        if j < mixes:
                            acc[0] += value * ws[k, j]
                        else:
                            acc[0] += value * value
                    ps[j] = acc[0]
            T.copy(ps, partial[b, bx % parts, :])

    return tilelang.compile(main, target="ascend", out_idx=[2])


@functools.lru_cache(maxsize=32)
def _normalize(batch, n, width, pre, post, res, repeats, eps):
    mixes = n * n + 2 * n
    parts = (width + 63) // 64
    root_width = math.sqrt(width)

    @T.prim_func
    def main(
        partial: T.Tensor((batch, parts, mixes + 1), "float32"),
        bias: T.Tensor((mixes,), "float32"),
        coeff: T.Tensor((batch, n + n * n), "float32"),
        hpost: T.Tensor((batch, n), "float32"),
    ):
        with T.Kernel(batch) as b:
            sums = T.alloc_shared((mixes + 1,), "float32")
            mat = T.alloc_shared((n, n), "float32")
            row_scaled = T.alloc_shared((n, n), "float32")
            norm = T.alloc_shared((n,), "float32")
            with T.SimtVF(threads=128):
                for j in T.Parallel(mixes + 1):
                    acc = T.alloc_local((1,), "float32")
                    acc[0] = 0
                    for part in T.serial(parts):
                        acc[0] += partial[b, part, j]
                    sums[j] = acc[0]
            with T.SimtVF(threads=128):
                for i in T.Parallel(n):
                    denom = T.sqrt(sums[mixes]) / root_width + 0.0001
                    coeff[b, i] = T.sigmoid(pre * sums[i] / denom + bias[i])
                    hpost[b, i] = 2 * T.sigmoid(post * sums[n + i] / denom + bias[n + i])
                    logits = T.alloc_local((n,), "float32")
                    maximum = T.alloc_local((1,), "float32")
                    maximum[0] = T.min_value("float32")
                    for j in T.serial(n):
                        idx = 2 * n + i * n + j
                        logits[j] = res * sums[idx] / denom + bias[idx]
                        maximum[0] = T.max(maximum[0], logits[j])
                    for j in T.serial(n):
                        mat[i, j] = T.exp(logits[j] - maximum[0])
            for _ in T.serial(repeats):
                with T.SimtVF(threads=128):
                    for i in T.Parallel(n):
                        acc = T.alloc_local((1,), "float32")
                        acc[0] = 0
                        for j in T.serial(n):
                            acc[0] += mat[i, j]
                        for j in T.serial(n):
                            row_scaled[i, j] = mat[i, j] / (acc[0] + eps)
                with T.SimtVF(threads=128):
                    for j in T.Parallel(n):
                        acc = T.alloc_local((1,), "float32")
                        acc[0] = 0
                        for i in T.serial(n):
                            acc[0] += row_scaled[i, j]
                        norm[j] = acc[0] + eps
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(n, n):
                        mat[i, j] = row_scaled[i, j] / norm[j]
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(n, n):
                    coeff[b, n + i * n + j] = mat[i, j]

    return tilelang.compile(main, target="ascend", out_idx=[2, 3])


@functools.lru_cache(maxsize=32)
def _apply(batch, n, channels, dtype):
    width = 256
    tiles = (channels + width - 1) // width

    @T.prim_func
    def main(
        x: T.Tensor((batch, n * channels), dtype),
        coeff: T.Tensor((batch, n + n * n), "float32"),
        residual: T.Tensor((batch, n * channels), dtype),
        layer: T.Tensor((batch, channels), dtype),
    ):
        with T.Kernel(batch * tiles) as bx:
            b = bx // tiles
            start = bx % tiles * width
            with T.SimtVF(threads=128):
                for j in T.Parallel(width):
                    c = start + j
                    if c < channels:
                        for dest in T.serial(n + 1):
                            acc = T.alloc_local((1,), "float32")
                            acc[0] = 0
                            for src in T.serial(n):
                                acc[0] += coeff[b, dest * n + src] * T.cast(
                                    x[b, src * channels + c], "float32"
                                )
                            if dest == 0:
                                layer[b, c] = T.cast(acc[0], dtype)
                            else:
                                residual[b, (dest - 1) * channels + c] = T.cast(acc[0], dtype)

    return tilelang.compile(main, target="ascend", out_idx=[2, 3])


class AscendMHCPreKernel(Kernel, MHCPreFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: MHCPreCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: MHCPreCall):
        super().__init__(device_index=call.device.index)
        self.dtype = call.dtype
        b, n, c = call.batch, call.n_expand, call.c_x
        self.project = _projection(b, n * c, n * n + 2 * n, self.dtype_str)
        self.normalize = _normalize(
            b,
            n,
            n * c,
            call.alpha_pre,
            call.alpha_post,
            call.alpha_res,
            call.sinkhorn_repeat,
            call.sinkhorn_eps,
        )
        self.mix = _apply(b, n, c, self.dtype_str)

    def forward(self, phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat, sinkhorn_eps):
        partial = self.project(phi, x)
        coeff, h_post = self.normalize(partial, b)
        x_res, x_layer = self.mix(x, coeff)
        return x_res, x_layer, h_post


register_kernel_type("MHCPreFwdOp", "ascend", AscendMHCPreKernel)
