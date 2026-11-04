"""Ascend PReLU with shared/per-channel weights and bounded tails."""

import functools
from typing import ClassVar

import torch

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import PreluFwdInterface
from tileops.kernels.kernel_base import Kernel

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

if T is not None:

    @T.macro
    def _body(x, w, y, begin, width: int, channels: int, inner: int, dtype: str):
        with T.SimtVF(threads=min(width, 1024)):
            tx = T.get_thread_binding()
            for j in T.serial(width // min(width, 1024)):
                i = j * min(width, 1024) + tx
                channel = ((begin + i) // inner) % channels
                value = T.cast(x[i], "float32")
                slope = T.cast(w[channel], "float32")
                y[i] = T.cast(T.if_then_else(value > 0, value, value * slope), dtype)

    @T.macro
    def _channel_body(
        x, weights, y, first_channel, group: int, channels: int, inner: int, dtype: str
    ):
        with T.SimdVF():
            full_mask = T.simd.pset(32)
            for c in range(group):
                channel = (first_channel + c) % channels
                slope = (
                    T.simd.vld(weights[channel], dist="BRC_B32")
                    if dtype == "float32"
                    else T.simd.vcvt(
                        T.simd.vld(weights[channel], dist="BRC_B16"), "float32", part=0
                    )
                )
                for i in range((inner + 63) // 64):
                    mask = (
                        full_mask
                        if inner % 64 == 0
                        else T.simd.update_mask(T.min(64, inner - i * 64), 32)
                    )
                    offset = c * inner + i * 64
                    value = (
                        T.simd.vld(x[offset])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(x[offset], dist="UNPK_B16"), "float32", part=0)
                    )
                    product = T.simd.vmul(value, slope, mask)
                    positive = T.simd.vcmps(value, 0.0, mask, op="gt")
                    result = T.simd.vsel(value, product, positive)
                    if dtype == "float32":
                        T.simd.vsts(y[offset], result, mask)
                    else:
                        T.simd.vsts(y[offset], T.simd.vcvt(result, dtype), mask, dist="PK_B32")

    @functools.lru_cache(maxsize=256)
    def build_channels(n, dtype, channels, inner):
        count = n // inner
        group = 1
        target = 14336 if dtype == "float32" else 28672
        if inner % 16 == 0:
            while 2 * group * inner <= target and count % (2 * group) == 0:
                group *= 2
        count //= group
        span = inner * group
        grid = min(64, count)
        width = (span + 63) // 64 * 64
        weight_size = (channels + 63) // 64 * 64

        @T.prim_func
        def main(
            X: T.Tensor((n,), dtype), W: T.Tensor((channels,), dtype), Y: T.Tensor((n,), dtype)
        ):
            with T.Kernel(grid) as bx:
                weights = T.alloc_shared((weight_size,), dtype)
                x = T.alloc_shared((width,), dtype)
                y = T.alloc_shared((width,), dtype)
                T.annotate_buffer_versions({x: 2, y: 2})
                T.copy(W[:channels], weights[:channels])
                for step in T.Pipelined((count + grid - 1) // grid, num_stages=2):
                    block = step * grid + bx
                    if block < count:
                        begin = block * span
                        T.copy(X[begin : begin + span], x[:span])
                        _channel_body(x, weights, y, block * group, group, channels, inner, dtype)
                        T.copy(y[:span], Y[begin : begin + span], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[2])

    @functools.lru_cache(maxsize=256)
    def build(n, dtype, channels, inner):
        if (
            64 <= inner <= (8192 if dtype == "float32" else 16384)
            and channels <= 8192
            and n % inner == 0
        ):
            return build_channels(n, dtype, channels, inner)
        tile = 8192 if dtype != "float32" else 4096
        stages = 3
        full, tail = divmod(n, tile)
        grid = min(64, full) if full else 1
        padded = (tail + 63) // 64 * 64
        cached = channels <= 8192
        weight_size = (channels + 63) // 64 * 64 if cached else 64

        @T.prim_func
        def main(
            X: T.Tensor((n,), dtype), W: T.Tensor((channels,), dtype), Y: T.Tensor((n,), dtype)
        ):
            with T.Kernel(grid) as bx:
                weights = T.alloc_shared((weight_size,), dtype)
                if cached:
                    T.copy(W[:channels], weights[:channels])
                if full:
                    x = T.alloc_shared((tile,), dtype)
                    y = T.alloc_shared((tile,), dtype)
                    T.annotate_buffer_versions({x: stages, y: stages})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(X[begin : begin + tile], x)
                            if cached:
                                _body(x, weights, y, begin, tile, channels, inner, dtype)
                            else:
                                _body(x, W, y, begin, tile, channels, inner, dtype)
                            T.copy(y, Y[begin : begin + tile])
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        xt = T.alloc_shared((padded,), dtype)
                        yt = T.alloc_shared((padded,), dtype)
                        T.copy(X[full * tile : n], xt[:tail])
                        if cached:
                            _body(xt, weights, yt, full * tile, padded, channels, inner, dtype)
                        else:
                            _body(xt, W, yt, full * tile, padded, channels, inner, dtype)
                        T.copy(yt[:tail], Y[full * tile : n])

        return tilelang.compile(main, target="ascend", out_idx=[2])


class AscendPreluKernel(Kernel, PreluFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.n_total, call.dtype, call.num_channels, call.inner_size)

    def __init__(self, n_total, dtype, num_channels, inner_size):
        super().__init__()
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"Unsupported PReLU dtype: {dtype}")
        if n_total < 0 or num_channels < 0 or (n_total > 0 and num_channels == 0) or inner_size < 0:
            raise ValueError("Invalid PReLU dimensions")
        self.n_total = n_total
        self.dtype = dtype
        self.channels = num_channels
        self.inner = inner_size
        self.compiled = (
            build(n_total, str(dtype).split(".")[-1], num_channels, max(1, inner_size))
            if n_total
            else None
        )

    def forward(self, input, weight):
        if input.device.type != "npu" or weight.device != input.device:
            raise ValueError("PReLU input and weight must share an NPU device")
        if (
            input.dtype != self.dtype
            or weight.dtype != self.dtype
            or input.numel() != self.n_total
            or weight.numel() != self.channels
        ):
            raise ValueError("Input does not match compiled PReLU signature")
        if weight.ndim > 1:
            raise ValueError("PReLU weight must be scalar or one-dimensional")
        channel_count = input.shape[1] if input.ndim >= 2 else 1
        if self.channels not in (1, channel_count):
            raise ValueError("PReLU weight length does not match input channels")
        if not self.n_total:
            return torch.empty_like(input)
        return self.compiled(
            input.contiguous().reshape(-1), weight.contiguous().reshape(-1)
        ).reshape(input.shape)


if T is not None:
    register_kernel_type("PreluFwdOp", "ascend", AscendPreluKernel)
