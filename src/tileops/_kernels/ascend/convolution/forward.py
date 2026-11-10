"""Channel-first Ascend convolutions with FP32 accumulation."""

import functools
import math

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.ascend.convolution.chunked import (
    _chunked_downsample_contract,
    _chunked_downsample_supported,
)
from tileops._kernels.ascend.convolution.downsample import (
    _small_downsample_contract,
    _small_downsample_supported,
)
from tileops._kernels.ascend.convolution.feature import (
    _feature_contract,
    _feature_supported,
)
from tileops._kernels.ascend.convolution.fused import _fused_contract, _fused_supported
from tileops._kernels.ascend.convolution.grouped import (
    _grouped_folded_contract,
    _grouped_folded_supported,
)
from tileops._kernels.ascend.convolution.native import (
    _native_bundle,
    _native_c0,
    _native_contract,
    _native_tuning,
    _pack_weights,
)
from tileops._kernels.ascend.convolution.packing import _pack_3d_weights, _simd_3d_supported
from tileops._kernels.ascend.convolution.pipeline import _pipelined_contract
from tileops._kernels.ascend.convolution.pointwise import _pointwise_contract
from tileops._kernels.ascend.convolution.streamed import (
    _streamed_contract,
    _streamed_pack,
    _streamed_supported,
)
from tileops._kernels.ascend.convolution.unfold import _unfold_kernel
from tileops._kernels.ascend.convolution.volume import (
    _volume_fused_contract,
    _volume_fused_supported,
)
from tileops._kernels.convolution.call_spec import Conv2dFwdInterface, Conv3dFwdInterface
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type


def _geometry(call):
    if hasattr(call, "d"):
        return (
            (call.d, call.h, call.w),
            (call.out_d, call.out_h, call.out_w),
            (call.kernel_d, call.kernel_h, call.kernel_w),
            call.stride,
            call.padding,
            call.dilation,
        )
    return (
        (1, call.h, call.w),
        (1, call.out_h, call.out_w),
        (1, call.kernel_h, call.kernel_w),
        (1, *call.stride),
        (0, *call.padding),
        (1, *call.dilation),
    )


@functools.lru_cache(maxsize=128)
def _contract(batch, groups, m, n, k, reduction, dtype, has_bias):
    bm = min(128 if m >= 128 and n >= 1024 else 64, max(16, tilelang.next_power_of_2(m)))
    bn, bk = 128, min(256, tilelang.next_power_of_2(k))
    if n >= 1024:
        bn = 256
    if n <= 64 and k <= 512:
        bn, bk = 64, k
    explicit_l0 = n <= 64 and k <= 512
    mt, nt = (m + bm - 1) // bm, (n + bn - 1) // bn
    tiles = batch * groups * mt * nt
    cores = min(32, tiles)

    @T.prim_func
    def main(
        W: T.Tensor((groups, m, reduction), dtype),
        P: T.Tensor((batch, groups, k, n), dtype),
        Bias: T.Tensor((groups, m), dtype),
        Y: T.Tensor((batch, groups, m, n), dtype),
    ):
        with T.Kernel(cores) as core:
            a = T.alloc_l1((bm, bk), dtype)
            b = T.alloc_l1((bn, bk), dtype)
            a0 = T.alloc_l0a((bm, bk if explicit_l0 else 16), dtype)
            b0 = T.alloc_l0b((bn, bk if explicit_l0 else 16), dtype)
            acc = T.alloc_l0c((bm, bn), "float32")
            epilogue = T.alloc_shared((bm // 2, bn), "float32")
            bias_ub = T.alloc_shared((1, bm // 2), dtype)
            output = T.alloc_shared((bm // 2, bn), dtype)
            T.annotate_buffer_versions({a: 2, b: 2, a0: 1, b0: 1})
            for step in T.serial(T.ceildiv(tiles, cores)):
                index = step * cores + core
                if index < tiles:
                    ni = index % nt * bn
                    mi = index // nt % mt * bm
                    group = index // (nt * mt) % groups
                    batch_idx = index // (nt * mt * groups)
                    for ki in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                        start = ki * bk
                        T.copy(W[group, mi : mi + bm, start : start + bk], a)
                        T.copy(
                            P[batch_idx, group, start : start + bk, ni : ni + bn],
                            b,
                            transpose=True,
                        )
                        if explicit_l0:
                            T.copy(a, a0)
                            T.copy(b, b0)
                            T.gemm(a0, b0, acc, transpose_B=True, clear_accum=ki == 0)
                        else:
                            T.gemm(a, b, acc, transpose_B=True, clear_accum=ki == 0)
                    if has_bias:
                        T.dual_copy(acc, epilogue)
                        T.dual_copy(Bias[group, mi : mi + bm], bias_ub)
                        with T.SimtVF(threads=256):
                            for i, j in T.Parallel(bm // 2, bn):
                                output[i, j] = T.cast(
                                    epilogue[i, j] + T.cast(bias_ub[0, i], "float32"),
                                    dtype,
                                )
                        T.dual_copy(output, Y[batch_idx, group, mi : mi + bm, ni : ni + bn])
                    else:
                        T.copy(acc, Y[batch_idx, group, mi : mi + bm, ni : ni + bn])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=128)
def _direct(n, ci, co, groups, size, out, window, stride, pad, dilation, dtype, has_bias):
    depth, height, width = size
    od, oh, ow = out
    kd, kh, kw = window
    sd, sh, sw = stride
    pd, ph, pw = pad
    dd, dh, dw = dilation
    spatial = od * oh * ow
    reduction = ci // groups * kd * kh * kw
    total, tile, cores = n * co * spatial, 256, 64

    @T.prim_func
    def main(
        X: T.Tensor((n, ci, depth, height, width), dtype),
        W: T.Tensor((co, reduction), dtype),
        B: T.Tensor((co,), dtype),
        Y: T.Tensor((total,), dtype),
    ):
        with T.Kernel(cores) as core, T.SimtVF(threads=256):
            for step in T.serial(T.ceildiv(total, cores * tile)):
                for j in T.Parallel(tile, coalesced_width=T.int32(1)):
                    index = (step * cores + core) * tile + j
                    if index < total:
                        pos = index % spatial
                        channel = index // spatial % co
                        batch = index // (spatial * co)
                        group = channel // (co // groups)
                        acc = T.alloc_local((1,), "float32")
                        acc[0] = T.float32(0)
                        for r in T.serial(reduction):
                            z = pos // (oh * ow) * sd - pd + r // (kh * kw) % kd * dd
                            y = pos // ow % oh * sh - ph + r // kw % kh * dh
                            x = pos % ow * sw - pw + r % kw * dw
                            if (
                                z >= 0
                                and z < depth
                                and y >= 0
                                and y < height
                                and x >= 0
                                and x < width
                            ):
                                acc[0] = acc[0] + T.cast(
                                    X[
                                        batch,
                                        group * (ci // groups) + r // (kd * kh * kw),
                                        z,
                                        y,
                                        x,
                                    ],
                                    "float32",
                                ) * T.cast(W[channel, r], "float32")
                        if has_bias:
                            acc[0] = acc[0] + T.cast(B[channel], "float32")
                        Y[index] = T.cast(acc[0], dtype)

    return tilelang.compile(main, target="ascend", out_idx=-1)


class _AscendConv(Kernel):
    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call):
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return "requires floating-point operands"
        return super().refusal(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        size, out, window, stride, pad, dilation = _geometry(call)
        self.out = out
        self.size = size
        self.spatial = math.prod(out)
        self.reduction = call.c_in_g * math.prod(window)
        self.pitch = (self.reduction + 15) // 16 * 16
        dtype = str(call.dtype).split(".")[-1]
        self.direct = None
        self.native = None
        self.pack_weight = None
        self.pointwise_native = None
        bundle = _native_bundle(
            call.c_in, call.c_out, call.groups, size, out, window, stride, pad, dilation
        )
        if (
            call.dtype == torch.float32
            or (call.groups == call.c_in and call.c_out == call.c_in)
            or (call.has_bias and call.c_out_g < 8 and (bundle is None or math.prod(window) == 1))
        ):
            self.direct = _direct(
                call.n,
                call.c_in,
                call.c_out,
                call.groups,
                size,
                out,
                window,
                stride,
                pad,
                dilation,
                dtype,
                call.has_bias,
            )
            return
        self.pointwise = (
            math.prod(window) == 1
            and stride == (1, 1, 1)
            and pad == (0, 0, 0)
            and size == out
            and self.pitch == self.reduction
        )
        if (
            self.pointwise
            and call.has_bias
            and call.c_out_g % 16 == 0
            and call.c_in_g % 16 == 0
            and call.c_out_g % min(128, 16384 // min(256, call.c_in_g), call.c_out_g) == 0
            and call.c_out_g >= call.c_in_g
            and (
                self.spatial > 512
                or (self.spatial <= 64 and call.c_in_g <= 512 and call.c_out_g % 64 == 0)
            )
        ):
            self.pointwise_native = _pointwise_contract(
                call.n,
                call.groups,
                call.c_out_g,
                self.spatial,
                call.c_in_g,
                dtype,
                call.has_bias,
            )
            return
        if (
            not call.has_bias
            and bundle is not None
            and _grouped_folded_supported(
                call.n, call.c_in, call.c_out, call.groups, size, out, window, stride, dilation
            )
        ):
            self.native = _grouped_folded_contract(
                call.n,
                call.c_in,
                call.c_out,
                call.groups,
                bundle,
                size,
                out,
                window,
                stride,
                pad,
                dilation,
                dtype,
                call.has_bias,
            )
            return
        for supported, builder in (
            (_small_downsample_supported, _small_downsample_contract),
            (_chunked_downsample_supported, _chunked_downsample_contract),
            (_fused_supported, _fused_contract),
            (_volume_fused_supported, _volume_fused_contract),
        ):
            if (
                not self.pointwise
                and bundle is not None
                and supported(
                    call.n, call.c_in, call.c_out, call.groups, size, out, window, stride, dilation
                )
            ):
                self.native = builder(
                    call.n,
                    call.c_in,
                    call.c_out,
                    call.groups,
                    bundle,
                    size,
                    out,
                    window,
                    stride,
                    pad,
                    dilation,
                    dtype,
                    call.has_bias,
                )
                return
        if (
            not self.pointwise
            and bundle is not None
            and _streamed_supported(
                call.n,
                call.c_in,
                call.c_out,
                call.groups,
                size,
                out,
                window,
                stride,
                pad,
                dilation,
            )
        ):
            self.pack_weight = _streamed_pack(call.c_out, call.c_in, *window[1:], dtype)
            self.native = _streamed_contract(
                call.n,
                call.c_in,
                call.c_out,
                size,
                out,
                window,
                stride,
                pad,
                dilation,
                dtype,
                call.has_bias,
            )
            return
        if not self.pointwise and bundle is not None:
            # Long 1D windows benefit from overlapping GM, L0 loads, and Cube work.
            use_l0_pipeline = (
                (not call.has_bias or call.c_out_g % 128 == 0)
                and window[:2] == (1, 1)
                and size[:2] == (1, 1)
                and 8 <= window[2] <= 64
                and call.c_in_g >= 64
                and call.c_out_g >= 128
            )
            use_feature = _feature_supported(
                call.n,
                call.c_in,
                call.c_out,
                call.groups,
                size,
                out,
                window,
                stride,
                dilation,
            )
            tuning = _native_tuning(
                call.n,
                call.c_in,
                call.c_out,
                call.groups,
                size,
                out,
                window,
                stride,
                dilation,
                call.has_bias,
            )
            use_simd_pack = tuning is not None or _simd_3d_supported(
                call.c_in, call.c_out, call.groups, window
            )
            use_bias_table = call.has_bias and use_simd_pack
            raw_bias = use_l0_pipeline or use_feature or use_bias_table
            native_builder = (
                _pipelined_contract
                if use_l0_pipeline
                else _feature_contract
                if use_feature
                else _native_contract
            )
            if use_simd_pack:
                self.pack_weight = _pack_3d_weights(
                    call.c_out, call.c_in, window[1], window[2], dtype, kd=window[0]
                )
            else:
                self.pack_weight = _pack_weights(
                    call.c_out,
                    call.c_in,
                    *window,
                    call.groups,
                    bundle,
                    dtype,
                    call.has_bias and not raw_bias,
                    _native_c0(call.c_in, call.groups, size, out, window, pad),
                )
            native_options = {}
            if native_builder is _native_contract:
                native_options["bias_table"] = use_bias_table
                if tuning is not None:
                    native_options.update(
                        block_m=tuning[0], block_n=tuning[1], reuse_weights=tuning[2]
                    )
            self.native = native_builder(
                call.n,
                call.c_in,
                call.c_out,
                call.groups,
                bundle,
                size,
                out,
                window,
                stride,
                pad,
                dilation,
                dtype,
                call.has_bias,
                **native_options,
            )
            return
        self.unfold = (
            None
            if self.pointwise
            else _unfold_kernel(
                call.n,
                call.c_in,
                call.groups,
                size,
                out,
                window,
                stride,
                pad,
                dilation,
                dtype,
            )
        )
        self.contract = _contract(
            call.n,
            call.groups,
            call.c_out_g,
            self.spatial,
            self.pitch,
            self.reduction,
            dtype,
            call.has_bias,
        )

    def forward(self, x, weight, bias=None):
        call = self.call
        if self.direct is not None:
            bias_arg = bias if bias is not None else weight.view(-1)[: call.c_out]
            result = self.direct(
                x.view(call.n, call.c_in, *self.size),
                weight.view(call.c_out, self.reduction),
                bias_arg,
            )
            return result.view(call.n, call.c_out, *self.out[-(weight.ndim - 2) :])
        if self.pointwise_native is not None:
            result = self.pointwise_native(
                x.view(call.n, call.groups, call.c_in_g, self.spatial),
                weight.view(call.groups, call.c_out_g, call.c_in_g),
                bias.view(call.groups, call.c_out_g),
            )
            return result.view(call.n, call.c_out, *self.out[-(weight.ndim - 2) :])
        if self.native is not None:
            bias_arg = bias if bias is not None else weight.view(-1)[: call.c_out]
            raw = weight.view(call.c_out, self.reduction)
            packed = raw if self.pack_weight is None else self.pack_weight(raw, bias_arg)
            result = self.native(x.view(call.n, call.c_in, -1), packed, bias_arg)
            return result.view(call.n, call.c_out, *self.out[-(weight.ndim - 2) :])
        patches = (
            x.view(call.n, call.groups, self.pitch, self.spatial)
            if self.pointwise
            else self.unfold(x.view(call.n, call.c_in, *self.size)).view(
                call.n, call.groups, self.pitch, self.spatial
            )
        )
        w = weight.view(call.c_out, self.reduction)
        bias_arg = bias if bias is not None else weight.view(-1)[: call.c_out]
        result = self.contract(
            w.view(call.groups, call.c_out_g, self.reduction),
            patches,
            bias_arg.view(call.groups, call.c_out_g),
        )
        rank = weight.ndim - 2
        return result.view(call.n, call.c_out, *self.out[-rank:])


class AscendConv2dKernel(_AscendConv, Conv2dFwdInterface):
    """Two-dimensional grouped convolution."""


class AscendConv3dKernel(_AscendConv, Conv3dFwdInterface):
    """Three-dimensional grouped convolution."""


register_kernel_type("Conv2dFwdOp", "ascend", AscendConv2dKernel)
register_kernel_type("Conv3dFwdOp", "ascend", AscendConv3dKernel)
