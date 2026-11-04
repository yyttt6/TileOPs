"""Ascend Rotary Position Embedding implementations."""

from tileops.kernels.ascend.rope.rotation import (
    AscendRoPEInterleavedKernel,
    AscendRoPENeoxKernel,
    AscendRoPEPositionsKernel,
)

__all__ = [
    "AscendRoPEInterleavedKernel",
    "AscendRoPENeoxKernel",
    "AscendRoPEPositionsKernel",
    "build_staged_rotation",
]

from tileops.kernels.ascend.rope.staged import build_staged_rotation
