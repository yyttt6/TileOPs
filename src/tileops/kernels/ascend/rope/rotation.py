"""Fused Rotary Position Embedding rotations for Ascend 950."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.ascend.rope.staged import build_staged_rotation
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.rope import (
    RoPENeoxFwdInterface,
    RoPENeoxPositionIdsFwdInterface,
    RoPENonNeoxFwdInterface,
)

__all__ = ["AscendRoPEInterleavedKernel", "AscendRoPENeoxKernel", "AscendRoPEPositionsKernel"]


@functools.lru_cache(maxsize=64)
def _rotation_kernel(tokens, heads, dim, rotary, table_rows, dtype, interleaved, indexed):
    token_width = heads * dim
    if (
        heads > 1
        and rotary == dim
        and token_width <= 8192
        and 8192 % token_width == 0
        and tokens % (8192 // token_width) == 0
        and (indexed or table_rows % (8192 // token_width) == 0)
    ):
        return build_staged_rotation(tokens, heads, dim, table_rows, dtype, interleaved, indexed)
    # Bound the working set while distributing both short and long sequences over Vector cores.
    tile = 2048
    total = tokens * heads * dim
    blocks = min(64, max(1, (total + tile - 1) // tile))
    half = rotary // 2
    positions_size = tokens if indexed else 1

    @T.prim_func
    def main(
        X: T.Tensor((total,), dtype),
        Cos: T.Tensor((table_rows * half,), dtype),
        Sin: T.Tensor((table_rows * half,), dtype),
        Positions: T.Tensor((positions_size,), "int32"),
        Y: T.Tensor((total,), dtype),
        Invalid: T.Tensor((positions_size,), "int32"),
    ):
        with T.Kernel(blocks) as bx:
            result = T.alloc_shared((tile,), dtype)
            for step in T.serial(T.ceildiv(total, blocks * tile)):
                begin = (step * blocks + bx) * tile
                if begin < total:
                    with T.SimtVF(threads=256):
                        for j in T.Parallel(tile, coalesced_width=T.int32(1)):
                            index = begin + j
                            if index < total:
                                token = index // (heads * dim)
                                col = index % dim
                                position = token
                                if tokens > table_rows and not indexed:
                                    position = token % table_rows
                                if indexed:
                                    raw_position = Positions[token]
                                    position = T.min(T.max(raw_position, 0), table_rows - 1)
                                    if index % (heads * dim) == 0:
                                        Invalid[token] = T.cast(
                                            raw_position < 0 or raw_position >= table_rows, "int32"
                                        )
                                if col < rotary:
                                    lane = (
                                        col // 2
                                        if interleaved
                                        else T.if_then_else(col < half, col, col - half)
                                    )
                                    other = (
                                        (col ^ 1)
                                        if interleaved
                                        else T.if_then_else(col < half, col + half, col - half)
                                    )
                                    subtract = col % 2 == 0 if interleaved else col < half
                                    a = T.cast(X[index], "float32")
                                    b = T.cast(X[index - col + other], "float32")
                                    c = T.cast(Cos[position * half + lane], "float32")
                                    s = T.cast(Sin[position * half + lane], "float32")
                                    result[j] = T.cast(
                                        a * c + T.if_then_else(subtract, -b, b) * s, dtype
                                    )
                                else:
                                    result[j] = X[index]
                    count = T.min(tile, total - begin)
                    T.copy(result[:count], Y[begin : begin + count])

    return tilelang.compile(main, target="ascend", out_idx=[4, 5])


class _AscendRoPEKernel(Kernel):
    """Rotate half-split heads with the caller's cached frequency tables."""

    devices = frozenset({"npu"})
    interleaved = False

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self._optimized = None
        # Keep the existing multi-head interleaved path; contiguous half-split
        # and single-head layouts use the independently validated batched kernels.
        if not self.interleaved or call.num_heads == 1:
            from tileops.kernels.ascend.rope.rope import (
                AscendRoPEInterleavedKernel as BatchedInterleaved,
            )
            from tileops.kernels.ascend.rope.rope import AscendRoPENeoxKernel as BatchedNeox

            implementation = BatchedInterleaved if self.interleaved else BatchedNeox
            self._optimized = implementation(call)
            return
        self._run = _rotation_kernel(
            call.batch * call.seq_len,
            call.num_heads,
            call.head_dim,
            call.head_dim,
            call.seq_len,
            self.dtype_to_str(call.dtype),
            self.interleaved,
            False,
        )
        self._positions = torch.empty(1, dtype=torch.int32, device=call.device)

    def forward(self, x, cos, sin):
        self._require_devices(x=x, cos=cos, sin=sin)
        if self._optimized is not None:
            return self._optimized(x, cos, sin)
        result, _ = self._run(x.view(-1), cos.view(-1), sin.view(-1), self._positions)
        return result.view_as(x)


class AscendRoPENeoxKernel(_AscendRoPEKernel, RoPENeoxFwdInterface):
    """Rotate heads by pairing their first and second halves."""


class AscendRoPEInterleavedKernel(_AscendRoPEKernel, RoPENonNeoxFwdInterface):
    """Rotate adjacent pairs with the caller's cached frequency tables."""

    interleaved = True


class AscendRoPEPositionsKernel(Kernel, RoPENeoxPositionIdsFwdInterface):
    """Rotate packed tokens and report invalid positions without out-of-range reads."""

    devices = frozenset({"npu"})

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self._optimized = None
        # Table gathering pays off for full-rotation half-precision prefill.
        # Retain the fused path for decode, FP32 and partial rotations.
        if (
            call.dtype in (torch.float16, torch.bfloat16)
            and call.num_tokens >= 128
            and call.rotary_dim == call.head_dim
        ):
            from tileops.kernels.ascend.rope.position_ids import AscendRoPEPositionIdsKernel

            self._optimized = AscendRoPEPositionIdsKernel(call)
            return
        self._run = _rotation_kernel(
            call.num_tokens,
            call.num_heads,
            call.head_dim,
            call.rotary_dim,
            call.max_position,
            self.dtype_to_str(call.dtype),
            False,
            True,
        )
        self._invalid = None

    def forward(self, x, cos, sin, position_ids):
        self._require_devices(x=x, cos=cos, sin=sin, position_ids=position_ids)
        if self._optimized is not None:
            return self._optimized(x, cos, sin, position_ids)
        result, self._invalid = self._run(x.view(-1), cos.view(-1), sin.view(-1), position_ids)
        return result.view_as(x)

    def take_out_of_range(self):
        if self._optimized is not None:
            return self._optimized.take_out_of_range()
        if self._invalid is None:
            return False
        invalid, self._invalid = self._invalid, None
        return bool(invalid.cpu().any())


register_kernel_type("RoPEFwdOp", "ascend", AscendRoPENeoxKernel)
register_kernel_type("RoPELlama31FwdOp", "ascend", AscendRoPENeoxKernel)
register_kernel_type("YaRNFwdOp", "ascend", AscendRoPENeoxKernel)
register_kernel_type("LongRoPEFwdOp", "ascend", AscendRoPENeoxKernel)
register_kernel_type("RoPEFwdOp", "ascend_interleaved", AscendRoPEInterleavedKernel)
register_kernel_type("RoPENeoxPositionIdsFwdOp", "ascend", AscendRoPEPositionsKernel)
