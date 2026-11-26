"""Ascend attention implementations and registrations."""

from tileops._kernels.ascend.attention import mla_cache_write, paged_cache_write
from tileops._kernels.ascend.attention.dsa_dense import AscendDSADenseKernel
from tileops._kernels.ascend.attention.dsa_paged import AscendDSAPagedKernel
from tileops._kernels.ascend.attention.fp8_lightning import AscendFP8LightningIndexerKernel
from tileops._kernels.ascend.attention.gated_pool_compress import AscendGatedPoolCompressKernel
from tileops._kernels.ascend.attention.gqa_dense_bwd import (
    AscendGQABwdPreprocess,
    AscendGQADenseBwdKernel,
)
from tileops._kernels.ascend.attention.gqa_dense_prefill import AscendGQADenseKernel
from tileops._kernels.ascend.attention.gqa_paged import AscendGQAPagedKernel
from tileops._kernels.ascend.attention.gqa_varlen import AscendGQAVarlenKernel
from tileops._kernels.ascend.attention.merge_states import AscendMergeAttentionStatesKernel
from tileops._kernels.ascend.attention.mha_paged_decode import AscendMHAPagedDecodeKernel
from tileops._kernels.ascend.attention.mla_bwd import (
    AscendMLADenseBwdKernel,
    AscendMLAVarlenBwdKernel,
)
from tileops._kernels.ascend.attention.mla_decode import (
    AscendMLADenseDecodeKernel,
    AscendMLAPagedKernel,
)
from tileops._kernels.ascend.attention.mla_varlen_fwd import AscendMLAVarlenKernel
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
    "AscendDSADenseKernel",
    "AscendDSAPagedKernel",
    "AscendFP8LightningIndexerKernel",
    "AscendGQABwdPreprocess",
    "AscendGQADenseBwdKernel",
    "AscendGQADenseKernel",
    "AscendGQAPagedKernel",
    "AscendGQAVarlenKernel",
    "AscendGatedPoolCompressKernel",
    "AscendMHAPagedDecodeKernel",
    "AscendMLADenseBwdKernel",
    "AscendMLADenseDecodeKernel",
    "AscendMLAPagedKernel",
    "AscendMLAVarlenBwdKernel",
    "AscendMLAVarlenKernel",
    "AscendMergeAttentionStatesKernel",
    "AscendNSACompressedFwdVarlenKernel",
    "AscendNSAFwdVarlenKernel",
    "AscendNSATopKVarlenKernel",
    "AscendPagedKVCacheGatherGenericKernel",
    "AscendPagedKVCacheGatherKernel",
    "AscendTopKSelectKernel",
]
