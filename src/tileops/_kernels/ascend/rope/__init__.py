"""Ascend Rotary Position Embedding implementations."""

from tileops._kernels.ascend.rope.cos_sin import AscendRoPECosSinKernel
from tileops._kernels.ascend.rope.rotation import (
    AscendRoPEInterleavedKernel,
    AscendRoPENeoxKernel,
    AscendRoPEPositionsKernel,
)

__all__ = [
    "AscendRoPECosSinKernel",
    "AscendRoPEInterleavedKernel",
    "AscendRoPENeoxKernel",
    "AscendRoPEPositionsKernel",
    "build_staged_rotation",
]

from tileops._kernels.ascend.rope.staged import build_staged_rotation
