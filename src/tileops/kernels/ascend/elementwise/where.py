"""Ascend three-way broadcast selection, preserving selected operand bits."""

import functools
import math
from typing import ClassVar

import torch

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import WhereFwdInterface
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
    def _select(c, x, z, y, width: int, itemsize: int):
        if itemsize == 2:
            with T.SimdVF():
                mask = T.simd.pset(16)
                for i in range(width // 128):
                    condition = T.reinterpret(T.simd.vld(c[i * 128], dist="UNPK_B8"), "uint16x128")
                    choose = T.simd.vcmps(condition, T.uint16(0), mask, op="ne")
                    result = T.simd.vsel(T.simd.vld(x[i * 128]), T.simd.vld(z[i * 128]), choose)
                    T.simd.vsts(y[i * 128], result, mask)
        elif itemsize == 4:
            with T.SimdVF():
                mask = T.simd.pset(32)
                for i in range(width // 64):
                    condition = T.reinterpret(T.simd.vld(c[i * 64], dist="UNPK4_B8"), "uint32x64")
                    choose = T.simd.vcmps(condition, T.uint32(0), mask, op="ne")
                    result = T.simd.vsel(T.simd.vld(x[i * 64]), T.simd.vld(z[i * 64]), choose)
                    T.simd.vsts(y[i * 64], result, mask)
        else:
            with T.SimtVF(threads=min(width, 1024)):
                for i in T.Parallel(width):
                    y[i] = T.if_then_else(c[i] != 0, x[i], z[i])

    @functools.lru_cache(maxsize=256)
    def build(cs, xs, zs, shape, dtype, itemsize):
        n = math.prod(shape)
        nc, nx, nz = math.prod(cs), math.prod(xs), math.prod(zs)
        tile, stages = 16384 // itemsize, 3
        full, tail = divmod(n, tile)
        grid = min(64, full) if full else 1
        padded = (tail + 127) // 128 * 128
        # Reserve UB for SIMT runtime scratch in addition to pipeline and tail buffers.
        while (3 * stages * tile + 3 * padded) * itemsize + stages * tile + padded > 192 * 1024:
            tile //= 2
            full, tail = divmod(n, tile)
            grid = min(64, full) if full else 1
            padded = (tail + 127) // 128 * 128
        cstr, xstr, zstr = (_strides(s, shape) for s in (cs, xs, zs))
        ostr = tuple(math.prod(shape[j + 1 :]) for j in range(len(shape)))

        def offset(idx, strides):
            result = 0
            for dim, stride, source_stride in zip(shape, ostr, strides):  # noqa: B905 - broadcasting deliberately permits unequal ranks
                if source_stride:
                    result = result + (idx // stride % dim) * source_stride
            return result

        if cs == xs == zs == shape:

            @T.prim_func
            def main(
                C: T.Tensor((nc,), "uint8"),
                X: T.Tensor((nx,), dtype),
                Z: T.Tensor((nz,), dtype),
                Y: T.Tensor((n,), dtype),
            ):
                with T.Kernel(grid) as bx:
                    if full:
                        c = T.alloc_shared((tile,), "uint8")
                        x = T.alloc_shared((tile,), dtype)
                        z = T.alloc_shared((tile,), dtype)
                        y = T.alloc_shared((tile,), dtype)
                        T.annotate_buffer_versions({c: stages, x: stages, z: stages, y: stages})
                        for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                            block = step * grid + bx
                            if block < full:
                                begin = block * tile
                                T.copy(C[begin : begin + tile], c)
                                T.copy(X[begin : begin + tile], x)
                                T.copy(Z[begin : begin + tile], z)
                                _select(c, x, z, y, tile, itemsize)
                                T.copy(y, Y[begin : begin + tile], l2_cache_ctrl=0)
                    if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                        if bx == 0:
                            ct = T.alloc_shared((padded,), "uint8")
                            xt = T.alloc_shared((padded,), dtype)
                            zt = T.alloc_shared((padded,), dtype)
                            yt = T.alloc_shared((padded,), dtype)
                            T.copy(C[full * tile : n], ct[:tail])
                            T.copy(X[full * tile : n], xt[:tail])
                            T.copy(Z[full * tile : n], zt[:tail])
                            _select(ct, xt, zt, yt, padded, itemsize)
                            T.copy(yt[:tail], Y[full * tile : n], l2_cache_ctrl=0)
        else:
            blocks = min(64, (n + 255) // 256)

            @T.prim_func
            def main(
                C: T.Tensor((nc,), "uint8"),
                X: T.Tensor((nx,), dtype),
                Z: T.Tensor((nz,), dtype),
                Y: T.Tensor((n,), dtype),
            ):
                with T.Kernel(blocks) as bx:  # noqa: SIM117 - preserve validated TileLang DSL structure
                    with T.SimtVF(threads=256):
                        tx = T.get_thread_binding()
                        for step in T.serial((n + blocks * 256 - 1) // (blocks * 256)):
                            idx = (step * blocks + bx) * 256 + tx
                            if idx < n:
                                Y[idx] = T.if_then_else(
                                    C[offset(idx, cstr)] != 0,
                                    X[offset(idx, xstr)],
                                    Z[offset(idx, zstr)],
                                )

        return tilelang.compile(main, target="ascend", out_idx=[3])


class AscendWhereKernel(Kernel, WhereFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total, dtype):
        super().__init__()
        self.n_total, self.dtype, self.compiled = n_total, dtype, None
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
            raise TypeError(f"Unsupported where dtype: {dtype}")
        self.storage = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
            dtype.itemsize
        ]

    def forward(self, condition, input, other):
        if condition.dtype != torch.bool:
            raise ValueError("condition dtype is not bool")
        if (
            input.device.type != "npu"
            or condition.device != input.device
            or other.device != input.device
        ):
            raise ValueError("Where tensors must share an NPU device")
        if input.dtype != self.dtype or other.dtype != self.dtype:
            raise ValueError("Where operand dtypes must match the compiled signature")
        shape = tuple(torch.broadcast_shapes(condition.shape, input.shape, other.shape))
        if math.prod(shape) != self.n_total:
            raise ValueError("Output element count differs from compiled signature")
        if not self.n_total:
            return torch.empty(shape, dtype=self.dtype, device=input.device)
        self.compiled = build(
            tuple(condition.shape),
            tuple(input.shape),
            tuple(other.shape),
            shape,
            str(self.storage).split(".")[-1],
            self.dtype.itemsize,
        )
        result = self.compiled(
            condition.contiguous().reshape(-1).view(torch.uint8),
            input.contiguous().reshape(-1).view(self.storage),
            other.contiguous().reshape(-1).view(self.storage),
        )
        return result.view(self.dtype).reshape(shape)


if T is not None:
    register_kernel_type("WhereFwdOp", "ascend", AscendWhereKernel)
