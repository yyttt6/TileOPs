"""Ascend floor with native vector rounding and bounded tails and bounded tails."""

import functools
from typing import ClassVar

import torch

import tileops.backend
from tileops.kernels.elementwise.call_spec import UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Kernel

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

_NATIVE_FLOOR_PRELUDE = r"""#include <c_api/asc_simd.h>
template <typename V>
__simd_callee__ inline void tileops_floor_native(V &value, vector_bool mask) {
    asc_floor(value, value, mask);
}
"""

if T is not None:

    @T.macro
    def _body(x, y, width: int, dtype: str, floating: bool):
        if floating:
            with T.SimdVF():
                bits = 32 if dtype == "float32" else 16
                lanes = 256 // (bits // 8)
                mask = T.simd.pset(bits)
                int_dtype = "int32" if dtype == "float32" else "int16"
                int_vector = "int32x64" if dtype == "float32" else "int16x128"
                float_vector = dtype + ("x64" if dtype == "float32" else "x128")
                sign_mask = T.simd.vdup(T.cast(-(1 << (bits - 1)), int_dtype), int_dtype)
                value = T.simd.alloc_local(1, dtype)
                for i in range(width // lanes):
                    original = T.simd.vld(x[i * lanes])
                    sign = T.simd.vand(T.reinterpret(original, int_vector), sign_mask, mask)
                    value[0] = original
                    T.evaluate(T.call_extern("void", "tileops_floor_native", value[0], mask))
                    # Native vtrc clears -0; rounding cannot change the sign.
                    result_bits = T.simd.vor(T.reinterpret(value[0], int_vector), sign, mask)
                    T.simd.vsts(y[i * lanes], T.reinterpret(result_bits, float_vector), mask)
        else:
            with T.SimdVF():
                mask = T.simd.pset(8)
                for i in range(width // 256):
                    T.simd.vsts(y[i * 256], T.simd.vld(x[i * 256]), mask)

    @functools.lru_cache(maxsize=256)
    def build(n, dtype, floating, tile=16384, blocks=64, stages=3):
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        padded_tail = (tail + 255) // 256 * 256

        @T.prim_func
        def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
            with T.Kernel(grid, prelude=_NATIVE_FLOOR_PRELUDE) as bx:
                if full:
                    x = T.alloc_shared((tile,), dtype)
                    y = x
                    T.annotate_buffer_versions({x: stages})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(X[begin : begin + tile], x, l2_cache_ctrl=4)
                            _body(x, y, tile, dtype, floating)
                            T.copy(y, Y[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        xt = T.alloc_shared((padded_tail,), dtype)
                        yt = xt
                        T.copy(X[full * tile : n], xt[:tail], l2_cache_ctrl=4)
                        _body(xt, yt, padded_tail, dtype, floating)
                        T.copy(yt[:tail], Y[full * tile : n], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)


class AscendFloorKernel(Kernel, UnaryElementwiseFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def refusal(cls, call):
        return None

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total, dtype):
        super().__init__()
        if dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("unsupported floor dtype")
        if T is None:
            raise ImportError("tilelang.ascend is required")
        self.n_total, self.dtype = n_total, dtype
        self.storage_dtype = dtype if dtype.is_floating_point else torch.uint8
        self.storage_size = n_total if dtype.is_floating_point else n_total * dtype.itemsize
        if n_total:
            tile = 16384 if self.storage_dtype.itemsize <= 2 else 8192
            stages = 3
            if dtype.is_floating_point:
                tile *= 2
            buffers = 1 if dtype.is_floating_point else 2
            while (
                buffers * tile * stages + buffers * ((self.storage_size % tile + 255) // 256 * 256)
            ) * self.storage_dtype.itemsize > 248 * 1024:
                tile //= 2
            self.compiled = build(
                self.storage_size,
                str(self.storage_dtype).split(".")[-1],
                dtype.is_floating_point,
                tile,
                64,
                stages,
            )

    def forward(self, input):
        if input.device.type != "npu" or input.dtype != self.dtype or input.numel() != self.n_total:
            raise ValueError("input must match compiled NPU kernel")
        if not self.n_total:
            return torch.empty_like(input)
        flat = input.contiguous().reshape(-1).view(self.storage_dtype)
        return self.compiled(flat).view(self.dtype).reshape(input.shape)


tileops.backend.register_kernel_type("FloorFwdOp", "ascend", AscendFloorKernel)
