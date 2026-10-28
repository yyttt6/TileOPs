"""Ascend attention implementations and registrations."""

from tileops._kernels.ascend.attention import mla_cache_write, paged_cache_write
from tileops._kernels.ascend.attention.gqa_dense_prefill import AscendGQADenseKernel
from tileops._kernels.ascend.attention.merge_states import AscendMergeAttentionStatesKernel
from tileops._kernels.ascend.attention.nsa_compressed_varlen import (
    AscendNSACompressedFwdVarlenKernel,
)
from tileops._kernels.ascend.attention.nsa_topk_varlen import (
    AscendNSATopKVarlenKernel,
)
from tileops._kernels.ascend.attention.nsa_varlen_unified import (
    AscendNSAFwdVarlenKernel,
)
from tileops._kernels.ascend.attention.paged_cache_gather import (
    AscendPagedKVCacheGatherGenericKernel,
    AscendPagedKVCacheGatherKernel,
)
from tileops._kernels.ascend.attention.topk_select import AscendTopKSelectKernel

__all__ = [
    "AscendGQADenseKernel",
    "AscendMergeAttentionStatesKernel",
    "AscendNSACompressedFwdVarlenKernel",
    "AscendNSAFwdVarlenKernel",
    "AscendNSATopKVarlenKernel",
    "AscendPagedKVCacheGatherGenericKernel",
    "AscendPagedKVCacheGatherKernel",
    "AscendTopKSelectKernel",
]
