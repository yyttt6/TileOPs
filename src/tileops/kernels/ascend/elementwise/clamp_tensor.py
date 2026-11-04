"""Ascend tensor-bound clamp with exact tails and direct broadcast indexing."""

import functools
import math
from typing import ClassVar

import torch

from tileops.kernels.elementwise.call_spec import ClampTensorFwdInterface
from tileops.kernels.kernel_base import Kernel

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None


def _strides(shape, output):
    aligned = (1,) * (len(output) - len(shape)) + tuple(shape)
    return tuple(0 if d == 1 else math.prod(aligned[i + 1 :]) for i, d in enumerate(aligned))


if T is not None:

    @T.macro
    def _body(x, lo, hi, y, width: int, dtype: str, has_min: bool, has_max: bool):
        with T.SimdVF():
            lanes = 64 if dtype == "float32" else 128
            mask = T.simd.pset(32 if dtype == "float32" else 16)
            for i in range(width // lanes):
                value = T.simd.vld(x[i * lanes])
                if has_min:
                    lower = T.simd.vld(lo[i * lanes])
                    selected = T.simd.vsel(lower, value, T.simd.vcmp(value, lower, mask, "le"))
                    low = T.simd.vsel(lower, selected, T.simd.vcmp(lower, lower, mask, "ne"))
                else:
                    low = value
                if has_max:
                    upper = T.simd.vld(hi[i * lanes])
                    selected_hi = T.simd.vsel(upper, low, T.simd.vcmp(low, upper, mask, "ge"))
                    result = T.simd.vsel(upper, selected_hi, T.simd.vcmp(upper, upper, mask, "ne"))
                else:
                    result = low
                if dtype == "float32":
                    T.simd.vsts(y[i * lanes], result, mask)
                else:
                    T.simd.vsts(y[i * lanes], result, mask, dist="NORM_B16")

    @functools.lru_cache(maxsize=256)
    def build(xs, ls, us, shape, dtype, has_min, has_max):
        n, nx = math.prod(shape), math.prod(xs)
        nl, nu = math.prod(ls), math.prod(us)
        same = xs == shape and (not has_min or ls == shape) and (not has_max or us == shape)
        if same:
            tile, stages = 8192, 2
            size = 4 if dtype == "float32" else 2
            buffers = 2 + int(has_min) + int(has_max)
            while buffers * (tile * stages + (n % tile + 127) // 128 * 128) * size > 248 * 1024:
                tile //= 2
            full, tail = divmod(n, tile)
            padded = (tail + 127) // 128 * 128
            grid = min(64, full) if full else 1

            @T.prim_func
            def main(
                X: T.Tensor((nx,), dtype),
                L: T.Tensor((nl,), dtype),
                U: T.Tensor((nu,), dtype),
                Y: T.Tensor((n,), dtype),
            ):
                with T.Kernel(grid) as bx:
                    if full:
                        x = T.alloc_shared((tile,), dtype)
                        lo = T.alloc_shared((tile if has_min else 128,), dtype)
                        hi = T.alloc_shared((tile if has_max else 128,), dtype)
                        y = T.alloc_shared((tile,), dtype)
                        T.annotate_buffer_versions({x: stages, y: stages})
                        if has_min:
                            T.annotate_buffer_versions({lo: stages})
                        if has_max:
                            T.annotate_buffer_versions({hi: stages})
                        for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                            block = step * grid + bx
                            if block < full:
                                begin = block * tile
                                T.copy(X[begin : begin + tile], x)
                                if has_min:
                                    T.copy(L[begin : begin + tile], lo)
                                if has_max:
                                    T.copy(U[begin : begin + tile], hi)
                                _body(x, lo, hi, y, tile, dtype, has_min, has_max)
                                T.copy(y, Y[begin : begin + tile], l2_cache_ctrl=0)
                    if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                        if bx == 0:
                            xt = T.alloc_shared((padded,), dtype)
                            lt = T.alloc_shared((padded if has_min else 128,), dtype)
                            ut = T.alloc_shared((padded if has_max else 128,), dtype)
                            yt = T.alloc_shared((padded,), dtype)
                            T.copy(X[full * tile : n], xt[:tail])
                            if has_min:
                                T.copy(L[full * tile : n], lt[:tail])
                            if has_max:
                                T.copy(U[full * tile : n], ut[:tail])
                            _body(xt, lt, ut, yt, padded, dtype, has_min, has_max)
                            T.copy(yt[:tail], Y[full * tile : n], l2_cache_ctrl=0)
        else:
            xstr, lstr, ustr = (_strides(s, shape) for s in (xs, ls, us))
            ostr = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))

            def offset(idx, strides):
                result = 0
                for dim, stride, source in zip(shape, ostr, strides):  # noqa: B905 - broadcasting deliberately permits unequal ranks
                    if source:
                        result = result + (idx // stride % dim) * source
                return result

            blocks = min(64, (n + 255) // 256)

            @T.prim_func
            def main(
                X: T.Tensor((nx,), dtype),
                L: T.Tensor((nl,), dtype),
                U: T.Tensor((nu,), dtype),
                Y: T.Tensor((n,), dtype),
            ):
                with T.Kernel(blocks) as bx:  # noqa: SIM117 - preserve validated TileLang DSL structure
                    with T.SimtVF(threads=256):
                        tx = T.get_thread_binding()
                        for step in T.serial((n + blocks * 256 - 1) // (blocks * 256)):
                            idx = (step * blocks + bx) * 256 + tx
                            if idx < n:
                                value = T.cast(X[offset(idx, xstr)], "float32")
                                bits = T.reinterpret(value, "uint32")
                                if has_min:
                                    lower = T.cast(L[offset(idx, lstr)], "float32")
                                    lb = T.reinterpret(lower, "uint32")
                                    low_bits = T.if_then_else(
                                        T.isnan(lower), lb, T.if_then_else(value <= lower, lb, bits)
                                    )
                                else:
                                    low_bits = bits
                                low = T.reinterpret(low_bits, "float32")
                                if has_max:
                                    upper = T.cast(U[offset(idx, ustr)], "float32")
                                    ub = T.reinterpret(upper, "uint32")
                                    result_bits = T.if_then_else(
                                        T.isnan(upper),
                                        ub,
                                        T.if_then_else(low >= upper, ub, low_bits),
                                    )
                                else:
                                    result_bits = low_bits
                                Y[idx] = T.cast(T.reinterpret(result_bits, "float32"), dtype)

        return tilelang.compile(main, target="ascend", out_idx=[3])


class AscendClampTensorKernel(Kernel, ClampTensorFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.n_total, call.dtype, call.has_min, call.has_max)

    def __init__(self, n_total, dtype, has_min, has_max):
        super().__init__()
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"Unsupported clamp dtype: {dtype}")
        if not (has_min or has_max):
            raise ValueError("ClampOut requires at least one bound")
        self.n_total, self.dtype = n_total, dtype
        self.has_min, self.has_max = has_min, has_max
        self.compiled = None

    def forward(self, input, min=None, max=None):
        if (min is not None, max is not None) != (self.has_min, self.has_max):
            raise ValueError("Clamp bound presence differs from the compiled signature")
        operands = [v for v in (input, min, max) if v is not None]
        if input.device.type != "npu" or any(
            v.device != input.device or v.dtype != self.dtype for v in operands
        ):
            raise ValueError("Clamp operands must share the compiled dtype and NPU device")
        shape = tuple(torch.broadcast_shapes(*(v.shape for v in operands)))
        if math.prod(shape) != self.n_total:
            raise ValueError("Clamp output size differs from the compiled signature")
        if not self.n_total:
            return torch.empty(shape, dtype=self.dtype, device=input.device)
        lower = input if min is None else min
        upper = input if max is None else max
        self.compiled = build(
            tuple(input.shape),
            tuple(lower.shape),
            tuple(upper.shape),
            shape,
            str(self.dtype).removeprefix("torch."),
            self.has_min,
            self.has_max,
        )
        return self.compiled(
            input.contiguous().reshape(-1),
            lower.contiguous().reshape(-1),
            upper.contiguous().reshape(-1),
        ).reshape(shape)


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("ClampTensorFwdOp", "ascend", AscendClampTensorKernel)
