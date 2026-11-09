"""Ascend implementations of attention kernel interfaces."""

from tileops.kernels.ascend.attention.merge_states import AscendMergeAttentionStatesKernel
from tileops.kernels.ascend.attention.nsa_compressed_varlen import (
    AscendNSACompressedFwdVarlenKernel,
)
from tileops.kernels.ascend.attention.nsa_topk_varlen import AscendNSATopKVarlenKernel
from tileops.kernels.ascend.attention.nsa_varlen_unified import AscendNSAFwdVarlenKernel

__all__ = [
    "AscendMergeAttentionStatesKernel",
    "AscendNSACompressedFwdVarlenKernel",
    "AscendNSAFwdVarlenKernel",
    "AscendNSATopKVarlenKernel",
]
