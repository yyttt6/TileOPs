"""Ascend attention implementations and registrations."""

from tileops._kernels.ascend.attention import mla_cache_write, paged_cache_write
from tileops._kernels.ascend.attention.nsa_compressed_varlen import (
    AscendNSACompressedFwdVarlenKernel,
)
from tileops._kernels.ascend.attention.nsa_topk_varlen import (
    AscendNSATopKVarlenKernel,
)
from tileops._kernels.ascend.attention.nsa_varlen_unified import (
    AscendNSAFwdVarlenKernel,
)

__all__ = [
    "AscendNSACompressedFwdVarlenKernel",
    "AscendNSAFwdVarlenKernel",
    "AscendNSATopKVarlenKernel",
]
