"""Ascend implementations of attention kernel interfaces."""

from tileops.kernels.ascend.attention.nsa_compressed_varlen import (
    AscendNSACompressedFwdVarlenKernel,
)
from tileops.kernels.ascend.attention.nsa_topk_varlen import AscendNSATopKVarlenKernel
from tileops.kernels.ascend.attention.nsa_varlen_unified import AscendNSAFwdVarlenKernel
from tileops.kernels.ascend.attention.topk_select import AscendTopKSelectKernel

__all__ = [
    "AscendNSACompressedFwdVarlenKernel",
    "AscendNSAFwdVarlenKernel",
    "AscendNSATopKVarlenKernel",
    "AscendTopKSelectKernel",
]
