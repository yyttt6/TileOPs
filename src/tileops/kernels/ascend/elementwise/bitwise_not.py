"""Ascend 950 implementations for BitwiseNotFwdOp."""

from functools import lru_cache
from typing import ClassVar, Optional

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["AscendBitwiseNotKernel"]
_NUM_BLOCKS = 64
_NUM_STAGES = 2
_TILE_THREADS_BY_DTYPE = {
    "bool": (2048, 2048),
    "uint8": (2048, 2048),
    "int8": (2048, 2048),
    "int16": (2048, 2048),
    "int32": (8192, 2048),
    "int64": (4096, 2048),
}
_FORCE_INT64_PAIR_FALLBACK = False


def _flat(t: "torch.Tensor") -> "torch.Tensor":
    """``t``, flattened: the op ``.contiguous()``s its input but does not reshape it."""
    return t.reshape(-1)


def _not_op(dtype_str: str):
    """The NOT body for one TileLang element-type string."""
    if dtype_str == "bool":

        def op(v):
            return v == T.cast(0, "bool")

        return op
    if dtype_str == "uint8":

        def op(v):
            return T.bitwise_xor(v, T.cast(255, "uint8"))

        return op

    def op(v):
        return T.bitwise_xor(v, T.cast(-1, dtype_str))

    return op


@lru_cache(maxsize=None)
def _build_kernel(n_total: int, dtype_str: str):
    """Compile the pipelined load/NOT/store program for ``n_total`` elements of ``dtype_str``."""
    tile, threads = _TILE_THREADS_BY_DTYPE[dtype_str]
    group = _NUM_BLOCKS * tile
    if n_total <= 0 or n_total % group:
        raise ValueError(
            f"AscendBitwiseNotKernel: n_total={n_total} must be a positive multiple of num_blocks * tile = {_NUM_BLOCKS} * {tile} = {group} elements for dtype {dtype_str!r} (no tail handling"
        )
    iterations = n_total // group
    op_func = _not_op(dtype_str)

    @T.prim_func
    def main(A: T.Tensor((n_total,), dtype_str), C: T.Tensor((n_total,), dtype_str)):
        with T.Kernel(_NUM_BLOCKS) as bx:
            a = T.alloc_shared((tile,), dtype_str)
            c = T.alloc_shared((tile,), dtype_str)
            T.annotate_buffer_versions({a: _NUM_STAGES, c: _NUM_STAGES})
            for step in T.Pipelined(iterations, num_stages=_NUM_STAGES):
                begin = (step * _NUM_BLOCKS + bx) * tile
                T.copy(A[begin : begin + tile], a)
                with T.SimtVF(threads=threads):
                    for i in T.Parallel(tile):
                        c[i] = op_func(a[i])
                T.copy(c, C[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


class AscendBitwiseNotKernel(Kernel, UnaryElementwiseFwdInterface):
    """Ascend implementation of ``BitwiseNotFwdOp``: ``y = ~x``, bandwidth-bound, no Cube."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return (call, lambda: cls(call.n_total, call.dtype))

    def __init__(self, n_total: int, dtype: torch.dtype, config: Optional[dict] = None) -> None:
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self._int64_as_pair = dtype == torch.int64 and _FORCE_INT64_PAIR_FALLBACK
        if self._int64_as_pair:
            self.kernel = _build_kernel(n_total * 2, "int32")
        else:
            self.kernel = _build_kernel(n_total, self.dtype_to_str(dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got a tensor on {input.device}. Another target's backend serves other devices."
            )
        shape = input.shape
        flat_in = _flat(input)
        if self._int64_as_pair:
            result = self.kernel(flat_in.view(torch.int32)).view(torch.int64)
        else:
            result = self.kernel(flat_in)
        return result.reshape(shape)


register_kernel_type("BitwiseNotFwdOp", "ascend", AscendBitwiseNotKernel)
