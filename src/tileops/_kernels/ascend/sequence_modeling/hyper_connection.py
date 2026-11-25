"""Manifold hyper-connection projection and weighted stream reductions on Ascend."""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.ascend.sequence_modeling.mhc_pre import _projection
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.sequence_modeling.hyper_connection.call_spec import (
    HyperConnectionHeadFwdInterface,
    HyperConnectionPostFwdInterface,
    HyperConnectionPreFwdInterface,
)
from tileops.backend._registry import register_kernel_type


@lru_cache(maxsize=64)
def _mix(rows, streams, channels, mix_dtype, post):
    width = min(256, channels)
    tiles = (channels + width - 1) // width
    outputs = streams if post else 1
    tasks = rows * outputs * tiles
    blocks = min(64, tasks)

    @T.prim_func
    def main(
        residual: T.Tensor((rows, streams, channels), "bfloat16"),
        weights: T.Tensor((rows, streams, outputs), mix_dtype),
        layer: T.Tensor((rows, channels), "bfloat16"),
        scales: T.Tensor((rows, outputs), mix_dtype),
        result: T.Tensor((rows, outputs, channels), "bfloat16"),
    ):
        with T.Kernel(blocks) as block:
            for step in T.serial(T.ceildiv(tasks, blocks)):
                bx = step * blocks + block
                if bx < tasks:
                    token = bx // (tiles * outputs)
                    dest = bx // tiles % outputs
                    column = bx % tiles * width
                    count = T.min(width, channels - column)
                    with T.SimtVF(threads=128):
                        for lane in T.Parallel(width):
                            if lane < count:
                                k = column + lane
                                acc = T.alloc_local((1,), "float32")
                                acc[0] = 0
                                if post:
                                    acc[0] = T.cast(layer[token, k], "float32") * T.cast(
                                        scales[token, dest], "float32"
                                    )
                                for src in T.serial(streams):
                                    acc[0] += T.cast(residual[token, src, k], "float32") * T.cast(
                                        weights[token, src, dest], "float32"
                                    )
                                result[token, dest, k] = T.cast(acc[0], "bfloat16")

    return tilelang.compile(main, target="ascend", out_idx=[4])


def _project(residual, fn, eps):
    n, c = residual.shape[-2:]
    rows = residual.numel() // (n * c)
    values = residual.contiguous().view(rows, n * c)
    if rows >= 256:
        # Bound intermediate storage for long prefills; the projection and inverse
        # RMS share the same promoted input, with no change to the stream mixtures.
        pieces = []
        for block in values.split(512, dim=0):
            promoted = block.float()
            pieces.append(
                (promoted @ fn.T) * torch.rsqrt(promoted.square().mean(-1, keepdim=True) + eps)
            )
        return values.view(rows, n, c), torch.cat(pieces)
    parts = _projection(rows, n * c, fn.shape[0], "bfloat16")(fn.T.contiguous(), values)
    summed = parts.sum(1)
    logits = summed[:, :-1] * torch.rsqrt(summed[:, -1:] / (n * c) + eps)
    return values.view(rows, n, c), logits


def _collapse(residual, weights):
    rows, streams, channels = residual.shape
    weights = weights.contiguous().view(rows, streams, 1)
    # The post-only inputs are unused in the collapse specialization.
    return _mix(rows, streams, channels, "float32", False)(
        residual,
        weights,
        residual.view(-1)[: rows * channels].view(rows, channels),
        weights.view(-1)[:rows].view(rows, 1),
    )


@lru_cache(maxsize=32)
def _four_stream_post(rows, channels, mix_dtype):
    # Reuse one input tile for the four destinations; every token owns its output.
    width = min(1024, channels)
    tiles = channels // width
    blocks = min(64, rows * tiles)

    @T.prim_func
    def main(
        residual: T.Tensor((rows, 4, channels), "bfloat16"),
        weights: T.Tensor((rows, 4, 4), mix_dtype),
        layer: T.Tensor((rows, channels), "bfloat16"),
        scales: T.Tensor((rows, 4), mix_dtype),
        result: T.Tensor((rows, 4, channels), "bfloat16"),
    ):
        with T.Kernel(blocks) as bx:
            rv = T.alloc_shared((4, width), "bfloat16")
            xv = T.alloc_shared((width,), "bfloat16")
            coeff = T.alloc_shared((4, 4), mix_dtype)
            scale = T.alloc_shared((4,), mix_dtype)
            out = T.alloc_shared((4, width), "bfloat16")
            for step in T.serial(T.ceildiv(rows * tiles, blocks)):
                task = step * blocks + bx
                if task < rows * tiles:
                    token = task // tiles
                    col = task % tiles * width
                    T.copy(residual[token, :, col : col + width], rv)
                    T.copy(layer[token, col : col + width], xv)
                    T.copy(weights[token, :, :], coeff)
                    T.copy(scales[token, :], scale)
                    with T.SimtVF(threads=256):
                        for dest, k in T.Parallel(4, width):
                            acc = T.alloc_local((1,), "float32")
                            acc[0] = T.cast(xv[k], "float32") * T.cast(scale[dest], "float32")
                            for src in T.unroll(4):
                                acc[0] += T.cast(rv[src, k], "float32") * T.cast(
                                    coeff[src, dest], "float32"
                                )
                            out[dest, k] = T.cast(acc[0], "bfloat16")
                    T.copy(out, result[token, :, col : col + width], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=[4])


class AscendHyperConnectionPostKernel(Kernel, HyperConnectionPostFwdInterface):
    devices = frozenset({"npu"})
    supported_archs = None

    def __init__(self, call):
        super().__init__(device_index=call.device.index)

    def forward(self, x, residual, post_layer_mix, comb_res_mix):
        if not residual.numel():
            return torch.empty_like(residual)
        n, c = residual.shape[-2:]
        rows = residual.numel() // (n * c)
        mix_dtype = self.dtype_to_str(comb_res_mix.dtype)
        if n == 4 and rows >= 64 and c % 1024 == 0:
            program = _four_stream_post(rows, c, mix_dtype)
        else:
            program = _mix(rows, n, c, mix_dtype, True)
        return program(
            residual.reshape(rows, n, c),
            comb_res_mix.reshape(rows, n, n),
            x.reshape(rows, c),
            post_layer_mix.reshape(rows, n),
        ).view(residual.shape)


class AscendHyperConnectionHeadKernel(Kernel, HyperConnectionHeadFwdInterface):
    devices = frozenset({"npu"})
    supported_archs = None

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(self, hidden_states, hc_fn, hc_scale, hc_base):
        flat, logits = _project(hidden_states, hc_fn, self.call.rms_norm_eps)
        weights = torch.sigmoid(logits * hc_scale + hc_base) + self.call.hc_eps
        return _collapse(flat, weights).view(*hidden_states.shape[:-2], hidden_states.shape[-1])


class AscendHyperConnectionPreKernel(Kernel, HyperConnectionPreFwdInterface):
    devices = frozenset({"npu"})
    supported_archs = None

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(self, residual, fn, hc_scale, hc_base):
        n = self.call.n
        leading = residual.shape[:-2]
        flat, logits = _project(residual, fn, self.call.rms_eps)
        pre = torch.sigmoid(logits[:, :n] * hc_scale[:1] + hc_base[:n]) + self.call.hc_pre_eps
        post = torch.sigmoid(logits[:, n : 2 * n] * hc_scale[1:2] + hc_base[n : 2 * n])
        matrix = (logits[:, 2 * n :] * hc_scale[2:] + hc_base[2 * n :]).view(-1, n, n)
        eps = self.call.hc_sinkhorn_eps
        matrix = matrix.softmax(-1) + eps
        matrix = matrix / (matrix.sum(-2, keepdim=True) + eps)
        for _ in range(self.call.sinkhorn_repeat - 1):
            matrix = matrix / (matrix.sum(-1, keepdim=True) + eps)
            matrix = matrix / (matrix.sum(-2, keepdim=True) + eps)
        return (
            (post * self.call.hc_post_mult_value).view(*leading, n, 1),
            matrix.view(*leading, n, n),
            _collapse(flat, pre).view(*leading, self.call.c),
        )


register_kernel_type("HyperConnectionPreFwdOp", "ascend", AscendHyperConnectionPreKernel)
register_kernel_type("HyperConnectionPostFwdOp", "ascend", AscendHyperConnectionPostKernel)
register_kernel_type("HyperConnectionHeadFwdOp", "ascend", AscendHyperConnectionHeadKernel)
