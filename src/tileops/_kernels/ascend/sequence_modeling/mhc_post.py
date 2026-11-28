"""Ascend vector kernel for the mHC post-layer residual mix."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.ascend.sequence_modeling import _mhc_fused

from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.sequence_modeling.mhc.call_spec import MHCPostCall, MHCPostFwdInterface
from tileops.backend._registry import register_kernel_type


@functools.lru_cache(maxsize=32)
def _build(batch: int, streams: int, channels: int, dtype: str):
    # Fill the 64 vector cores before adding another wave of output tiles.
    tiles_per_row = max(1, 64 // (batch * streams))
    width = min(2048, max(256, (channels + 64 * tiles_per_row - 1) // (64 * tiles_per_row) * 64))
    tiles = (channels + width - 1) // width

    @T.prim_func
    def main(
        layer: T.Tensor((batch, channels), dtype),
        weights: T.Tensor((batch, streams), "float32"),
        residual: T.Tensor((batch, streams * channels), dtype),
        output: T.Tensor((batch, streams * channels), dtype),
    ):
        with T.Kernel(batch * streams * tiles) as bx:
            row = bx // tiles
            b = row // streams
            start = bx % tiles * width
            offset = row % streams * channels + start
            count = T.min(width, channels - start)
            x = T.alloc_shared((width,), dtype)
            r = T.alloc_shared((width,), dtype)
            y = T.alloc_shared((width,), dtype)
            h = T.alloc_shared((8,), "float32")
            T.copy(weights[b, row % streams : row % streams + 1], h[:1])
            T.copy(layer[b, start : start + count], x[:count])
            T.copy(residual[b, offset : offset + count], r[:count])
            with T.SimdVF():
                mask = T.simd.pset(32)
                weight = T.simd.vld(h[0], dist="BRC_B32")
                for v in T.serial(width // 64):
                    a = T.simd.vcvt(T.simd.vld(x[v * 64], dist="UNPK_B16"), "float32", part=0)
                    z = T.simd.vcvt(T.simd.vld(r[v * 64], dist="UNPK_B16"), "float32", part=0)
                    value = T.simd.vadd(T.simd.vmul(a, weight, mask), z, mask)
                    result = T.simd.vcvt(value, dtype, round="ROUND_R", sat=False, part=0)
                    T.simd.vsts(y[v * 64], result, mask, dist="PK_B32")
            T.copy(y[:count], output[b, offset : offset + count])

    return tilelang.compile(main, target="ascend", out_idx=[3])


class AscendMHCPostKernel(Kernel, MHCPostFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: MHCPostCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: MHCPostCall):
        super().__init__(device_index=call.device.index)
        self.dtype = call.dtype
        self._fused = None
        batch, hidden = call.batch, call.c_x
        if call.n_expand == 4 and (batch, hidden) in _mhc_fused.MANIFEST_SHAPES and call.dtype == torch.bfloat16:
            self._fused_shape = (batch, 4, hidden)
            cores = batch * ((hidden + 255) // 256)
            kernel = _mhc_fused.post.compile(batch=batch, hidden=hidden, cores=cores)
            self._fused = _mhc_fused.export_launcher(kernel, call.device)
            return
        self._kernel = (
            _build(call.batch, call.n_expand, call.c_x, self.dtype_str)
            if call.batch and call.n_expand and call.c_x
            else None
        )

    def forward(self, x_layer_out, h_post, x_res):
        if self._fused is not None:
            output = x_res.new_empty(self._fused_shape)
            self._fused(x_layer_out, h_post, x_res.view(self._fused_shape), output)
            return output.view_as(x_res)
        if self._kernel is None:
            return x_res.new_empty(x_res.shape)
        return self._kernel(x_layer_out, h_post, x_res)


register_kernel_type("MHCPostFwdOp", "ascend", AscendMHCPostKernel)
