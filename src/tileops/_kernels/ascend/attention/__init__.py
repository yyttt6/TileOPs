from . import paged_cache_write
from . import mla_cache_write

"""Ascend implementations of attention kernel interfaces."""

from tileops._kernels.ascend.attention.nsa_compressed_varlen import (
    AscendNSACompressedFwdVarlenKernel,
)
from tileops._kernels.ascend.attention.nsa_topk_varlen import AscendNSATopKVarlenKernel
from tileops._kernels.ascend.attention.nsa_varlen_unified import AscendNSAFwdVarlenKernel

__all__ = [
    "AscendNSACompressedFwdVarlenKernel",
    "AscendNSAFwdVarlenKernel",
    "AscendNSATopKVarlenKernel",
]
