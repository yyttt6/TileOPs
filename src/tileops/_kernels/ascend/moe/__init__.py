"""Ascend implementations of MoE routing boundaries."""

from tileops._kernels.ascend.moe.indexed_expert import (
    AscendIndexedExpertDownKernel,
    AscendIndexedExpertGateUpKernel,
    AscendIndexedRouteStatsKernel,
    AscendIndexedRouteStatsParallelKernel,
    AscendIndexedWeightedReduceKernel,
)
from tileops._kernels.ascend.moe.indexed_expert_fused import (
    AscendIndexedExpertDownFusedKernel,
    AscendIndexedExpertGateUpFusedKernel,
)
from tileops._kernels.ascend.moe.indexed_expert_grouped import (
    AscendIndexedExpertDownGroupedKernel,
    AscendIndexedExpertDownVectorWeightsKernel,
    AscendIndexedExpertGateUpGroupedKernel,
    AscendIndexedExpertGateUpVectorWeightsKernel,
)
from tileops._kernels.ascend.moe.indexed_expert_packed import (
    AscendIndexedExpertDownPackedKernel,
    AscendIndexedExpertGateUpPackedKernel,
)
from tileops._kernels.ascend.moe.indexed_expert_preload import (
    AscendIndexedExpertGateUpPreloadKernel,
)
from tileops._kernels.ascend.moe.pre_permute import AscendMoEPrePermuteKernel

__all__ = [
    "AscendIndexedExpertDownFusedKernel",
    "AscendIndexedExpertDownGroupedKernel",
    "AscendIndexedExpertDownKernel",
    "AscendIndexedExpertDownPackedKernel",
    "AscendIndexedExpertDownVectorWeightsKernel",
    "AscendIndexedExpertGateUpFusedKernel",
    "AscendIndexedExpertGateUpGroupedKernel",
    "AscendIndexedExpertGateUpKernel",
    "AscendIndexedExpertGateUpPackedKernel",
    "AscendIndexedExpertGateUpPreloadKernel",
    "AscendIndexedExpertGateUpVectorWeightsKernel",
    "AscendIndexedRouteStatsKernel",
    "AscendIndexedRouteStatsParallelKernel",
    "AscendIndexedWeightedReduceKernel",
    "AscendMoEPrePermuteKernel",
]
