"""Ascend masked fill with a runtime tensor scalar and bidirectional broadcasting."""

import functools
import math
from typing import ClassVar

import torch

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import MaskedFillTensorValueFwdInterface
from tileops.kernels.kernel_base import Kernel

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None


def _strides(shape, out_shape):
    aligned = (1,) * (len(out_shape) - len(shape)) + tuple(shape)
    return tuple(0 if d == 1 else math.prod(aligned[i + 1 :]) for i, d in enumerate(aligned))


if T is not None:

    @T.macro
    def _body(x, m, v, y, width: int, dtype: str):
        if dtype == "int16":
            with T.SimdVF():
                mask = T.simd.pset(16)
                fill = T.simd.vld(v[0], dist="BRC_B16")
                for i in range(width // 128):
                    condition = T.reinterpret(T.simd.vld(m[i * 128], dist="UNPK_B8"), "uint16x128")
                    choose = T.simd.vcmps(condition, T.uint16(0), mask, op="ne")
                    result = T.simd.vsel(fill, T.simd.vld(x[i * 128]), choose)
                    T.simd.vsts(y[i * 128], result, mask)
        elif dtype == "int32":
            with T.SimdVF():
                mask = T.simd.pset(32)
                fill = T.simd.vld(v[0], dist="BRC_B32")
                for i in range(width // 64):
                    condition = T.reinterpret(T.simd.vld(m[i * 64], dist="UNPK4_B8"), "uint32x64")
                    choose = T.simd.vcmps(condition, T.uint32(0), mask, op="ne")
                    result = T.simd.vsel(fill, T.simd.vld(x[i * 64]), choose)
                    T.simd.vsts(y[i * 64], result, mask)
        else:
            with T.SimtVF(threads=min(width, 1024)):
                for i in T.Parallel(width):
                    y[i] = T.if_then_else(m[i] != 0, v[0], x[i])

    @functools.lru_cache(maxsize=256)
    def build(xs, ms, out_shape, dtype, itemsize):
        n = math.prod(out_shape)
        nx = math.prod(xs)
        nm = math.prod(ms)
        # Wider full tiles amortize DMA overhead. Keep room for separate tail buffers.
        tile = (32768 if itemsize == 4 and n % 8192 == 0 else 16384) // itemsize
        stages = 3
        same = xs == out_shape and ms == out_shape
        full, tail = divmod(n, tile)
        grid = min(64, full) if full else 1
        padded = (tail + 127) // 128 * 128
        xstr = _strides(xs, out_shape)
        mstr = _strides(ms, out_shape)
        outstr = tuple(math.prod(out_shape[j + 1 :]) for j in range(len(out_shape)))

        def offset(idx, strides):
            result = 0
            for dim, stride, source_stride in zip(out_shape, outstr, strides):  # noqa: B905 - broadcasting deliberately permits unequal ranks
                if source_stride:
                    result = result + (idx // stride % dim) * source_stride
            return result

        if same:

            @T.prim_func
            def main(
                X: T.Tensor((nx,), dtype),
                M: T.Tensor((nm,), "uint8"),
                V: T.Tensor((1,), dtype),
                Y: T.Tensor((n,), dtype),
            ):
                with T.Kernel(grid) as bx:
                    value = T.alloc_shared((64,), dtype)
                    T.copy(V[:1], value[:1])
                    if full:
                        x = T.alloc_shared((tile,), dtype)
                        m = T.alloc_shared((tile,), "uint8")
                        y = T.alloc_shared((tile,), dtype)
                        T.annotate_buffer_versions({x: stages, m: stages, y: stages})
                        for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                            block = step * grid + bx
                            if block < full:
                                begin = block * tile
                                T.copy(X[begin : begin + tile], x)
                                T.copy(M[begin : begin + tile], m)
                                _body(x, m, value, y, tile, dtype)
                                T.copy(y, Y[begin : begin + tile], l2_cache_ctrl=0)
                    if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                        if bx == 0:
                            x_tail = T.alloc_shared((padded,), dtype)
                            m_tail = T.alloc_shared((padded,), "uint8")
                            y_tail = T.alloc_shared((padded,), dtype)
                            T.copy(X[full * tile : n], x_tail[:tail])
                            T.copy(M[full * tile : n], m_tail[:tail])
                            _body(x_tail, m_tail, value, y_tail, padded, dtype)
                            T.copy(y_tail[:tail], Y[full * tile : n], l2_cache_ctrl=0)
        else:
            blocks = min(64, (n + 255) // 256)

            @T.prim_func
            def main(
                X: T.Tensor((nx,), dtype),
                M: T.Tensor((nm,), "uint8"),
                V: T.Tensor((1,), dtype),
                Y: T.Tensor((n,), dtype),
            ):
                with T.Kernel(blocks) as bx:  # noqa: SIM117 - preserve validated TileLang DSL structure
                    with T.SimtVF(threads=256):
                        tx = T.get_thread_binding()
                        for step in T.serial((n + blocks * 256 - 1) // (blocks * 256)):
                            idx = (step * blocks + bx) * 256 + tx
                            if idx < n:
                                Y[idx] = T.if_then_else(
                                    M[offset(idx, mstr)] != 0, V[0], X[offset(idx, xstr)]
                                )

        return tilelang.compile(main, target="ascend", out_idx=[3])


class AscendMaskedFillTensorKernel(Kernel, MaskedFillTensorValueFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total, dtype):
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.compiled = None
        if dtype not in (
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise TypeError(f"Unsupported masked fill dtype: {dtype}")
        self.storage = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
            dtype.itemsize
        ]

    def forward(self, input, mask, value):
        if (
            input.device.type != "npu"
            or mask.device != input.device
            or value.device != input.device
        ):
            raise ValueError("Masked fill tensors must share an NPU device")
        if (
            input.dtype != self.dtype
            or value.dtype != self.dtype
            or mask.dtype != torch.bool
            or value.ndim != 0
        ):
            raise ValueError("Invalid masked fill input signature")
        shape = tuple(torch.broadcast_shapes(input.shape, mask.shape))
        if math.prod(shape) != self.n_total:
            raise ValueError("Output element count differs from compiled signature")
        if not self.n_total:
            return torch.empty(shape, dtype=self.dtype, device=input.device)
        self.compiled = build(
            tuple(input.shape),
            tuple(mask.shape),
            shape,
            str(self.storage).split(".")[-1],
            self.dtype.itemsize,
        )
        out = self.compiled(
            input.contiguous().reshape(-1).view(self.storage),
            mask.contiguous().reshape(-1).view(torch.uint8),
            value.contiguous().reshape(-1).view(self.storage),
        )
        return out.view(self.dtype).reshape(shape)


if T is not None:
    register_kernel_type("MaskedFillTensorFwdOp", "ascend", AscendMaskedFillTensorKernel)
