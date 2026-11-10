"""Ascend ReLU with bit-preserving SIMD selection and bounded parallel tiles."""

import functools
from typing import ClassVar

import torch

import tileops.backend._registry
from tileops._kernels.elementwise.call_spec import UnaryElementwiseFwdInterface
from tileops._kernels.kernel_base import Compilation, Kernel

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

if T is not None:

    @T.macro
    def _body(x, y, width: int, dtype: str, floating: str):
        if floating:
            with T.SimdVF():
                if dtype == "int16":
                    mask = T.simd.pset(16)
                    zero = T.simd.vdup(T.int16(0), "int16")
                    for i in range(width // 128):
                        bits = T.simd.vld(x[i * 128])
                        value = T.reinterpret(bits, floating + "x128")
                        negative = T.simd.vcmps(value, T.cast(0, floating), mask, op="lt")
                        T.simd.vsts(y[i * 128], T.simd.vsel(zero, bits, negative), mask)
                else:
                    mask = T.simd.pset(32)
                    zero = T.simd.vdup(T.int32(0), "int32")
                    for i in range(width // 64):
                        bits = T.simd.vld(x[i * 64])
                        value = T.reinterpret(bits, "float32x64")
                        negative = T.simd.vcmps(value, T.float32(0), mask, op="lt")
                        T.simd.vsts(y[i * 64], T.simd.vsel(zero, bits, negative), mask)
        elif dtype == "uint8":
            with T.SimdVF():
                mask = T.simd.pset(8)
                for i in range(width // 256):
                    T.simd.vsts(y[i * 256], T.simd.vld(x[i * 256]), mask)
        else:
            with T.SimtVF(threads=256):
                for i in T.Parallel(width):
                    y[i] = T.if_then_else(x[i] < 0, T.cast(0, dtype), x[i])

    def program(n, dtype, floating, tile=16384, blocks=64, stages=3):
        """The compiled-shape program for one call; the only place its DSL is written."""
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        padded_tail = (tail + 255) // 256 * 256

        @T.prim_func
        def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
            with T.Kernel(grid) as bx:
                if full:
                    x = T.alloc_shared((tile,), dtype)
                    y = T.alloc_shared((tile,), dtype)
                    T.annotate_buffer_versions({x: stages, y: stages})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(X[begin : begin + tile], x)
                            _body(x, y, tile, dtype, floating)
                            T.copy(y, Y[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        xt = T.alloc_shared((padded_tail,), dtype)
                        yt = T.alloc_shared((padded_tail,), dtype)
                        T.copy(X[full * tile : n], xt[:tail])
                        _body(xt, yt, padded_tail, dtype, floating)
                        T.copy(yt[:tail], Y[full * tile : n], l2_cache_ctrl=0)

        return main

    @functools.lru_cache(maxsize=256)
    def build(n, dtype, floating, tile=16384, blocks=64, stages=3):
        return tilelang.compile(
            program(n, dtype, floating, tile, blocks, stages), target="ascend", out_idx=-1
        )

    def _tile(n_total, dtype):
        """The tile and stage count one call runs with, by its length and element width."""
        tile = 16384 if dtype.itemsize <= 2 else 8192 if dtype.itemsize == 4 else 4096
        stages = 3
        while (
            2 * tile * stages + 2 * ((n_total % tile + 255) // 256 * 256)
        ) * dtype.itemsize > 248 * 1024:
            tile //= 2
        return tile, stages


class AscendReLUKernel(Kernel, UnaryElementwiseFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def refusal(cls, call):
        return None

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.n_total, call.dtype)

    @classmethod
    def compilations(cls, tensors, params):
        """The one program a ReLU call builds, from the manifest's shape and dtype.

        Reading ``(shape, dtype)`` keeps this independent of any tensor: a pre-compilation
        pass reads only host metadata; the current CI pilot still holds the device reservation --
        no construction, no device query, no launch.
        """
        shape, dtype_name = tensors["input"]
        dtype = getattr(torch, dtype_name)
        n_total = 1
        for extent in shape:
            n_total *= extent
        if T is None or not n_total or not cls._serves(dtype):
            return ()
        storage = str(
            (torch.int32 if dtype == torch.float32 else torch.int16)
            if dtype.is_floating_point
            else dtype
        ).split(".")[-1]
        floating = str(dtype).split(".")[-1] if dtype.is_floating_point else ""
        tile, stages = _tile(n_total, dtype)
        return (
            Compilation(
                program(n_total, storage, floating, tile, 64, stages),
                out_idx=-1,
                target="ascend",
            ),
        )

    @staticmethod
    def _serves(dtype):
        return dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        )

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
            raise TypeError("unsupported relu dtype")
        if T is None:
            raise ImportError("tilelang.ascend is required")
        self.n_total, self.dtype = n_total, dtype
        self.storage_dtype = (
            (torch.int32 if dtype == torch.float32 else torch.int16)
            if dtype.is_floating_point
            else dtype
        )
        if n_total:
            tile, stages = _tile(n_total, dtype)
            self.compiled = build(
                n_total,
                str(self.storage_dtype).split(".")[-1],
                str(dtype).split(".")[-1] if dtype.is_floating_point else "",
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


tileops.backend._registry.register_kernel_type("ReLUFwdOp", "ascend", AscendReLUKernel)
