"""Ascend ALiBi generation with bounded output tiles and exact tail stores."""

import functools
from typing import ClassVar

import torch

from tileops.kernels.elementwise.call_spec import AlibiFwdInterface
from tileops.kernels.kernel_base import Kernel

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

if T is not None:

    @T.macro
    def _body(output, width: int, begin, seq_len: int, heads: int, dtype: str, uniform_head: bool):
        if seq_len == 1:
            with T.SimdVF():
                lanes = 64 if dtype == "float32" else 128
                mask = T.simd.pset(32 if dtype == "float32" else 16)
                zero = T.simd.vdup(T.cast(-0.0, dtype), dtype)
                for i in range(width // lanes):
                    if dtype == "float32":
                        T.simd.vsts(output[i * lanes], zero, mask)
                    else:
                        T.simd.vsts(output[i * lanes], zero, mask, dist="NORM_B16")
        else:
            with T.SimtVF(threads=min(2048, width)):
                if uniform_head:
                    tile_head = begin // (seq_len * seq_len)
                    common_slope = T.exp2(
                        T.cast(-8.0, "float32")
                        * T.cast(tile_head + 1, "float32")
                        / T.cast(heads, "float32")
                    )
                for chunk in T.serial((width + min(2048, width) - 1) // min(2048, width)):
                    for lane in T.Parallel(min(2048, width)):
                        i = chunk * min(2048, width) + lane
                        if i < width:
                            flat = begin + i
                            head = flat // (seq_len * seq_len)
                            row = flat // seq_len % seq_len
                            col = flat % seq_len
                            if uniform_head:
                                slope = common_slope
                            else:
                                exponent = (
                                    T.cast(-8.0, "float32")
                                    * T.cast(head + 1, "float32")
                                    / T.cast(heads, "float32")
                                )
                                slope = T.exp2(exponent)
                            distance = T.abs(T.cast(row - col, "float32"))
                            output[i] = T.cast(-slope * distance, dtype)

    @functools.lru_cache(maxsize=128)
    def build(seq_len, heads, dtype):
        count = heads * seq_len * seq_len
        tile, stages = (16384 if dtype == "float32" else 32768), 2
        full, tail = divmod(count, tile)
        padded = (tail + 127) // 128 * 128
        grid = min(64, full) if full else 1

        @T.prim_func
        def main(Y: T.Tensor((count,), dtype)):
            with T.Kernel(grid) as bx:
                if full:
                    y = T.alloc_shared((tile,), dtype)
                    T.annotate_buffer_versions({y: stages})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            _body(
                                y, tile, begin, seq_len, heads, dtype, seq_len * seq_len % tile == 0
                            )
                            T.copy(y, Y[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        yt = T.alloc_shared((padded,), dtype)
                        _body(yt, padded, full * tile, seq_len, heads, dtype, False)
                        T.copy(yt[:tail], Y[full * tile : count], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[0])


class AscendAlibiKernel(Kernel, AlibiFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.seq_len, call.num_heads, call.dtype, call.device)

    def __init__(self, seq_len, num_heads, dtype, device=None):
        super().__init__()
        if seq_len <= 0 or num_heads <= 0:
            raise ValueError("ALiBi requires positive sequence length and head count")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"Unsupported ALiBi dtype: {dtype}")
        self.seq_len, self.num_heads, self.dtype = seq_len, num_heads, dtype
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("npu", torch.npu.current_device())
        )
        self.compiled = build(seq_len, num_heads, str(dtype).removeprefix("torch."))

    def forward(self):
        with torch.npu.device(self.device):
            return self.compiled().reshape(self.num_heads, self.seq_len, self.seq_len)


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("AlibiFwdOp", "ascend", AscendAlibiKernel)
