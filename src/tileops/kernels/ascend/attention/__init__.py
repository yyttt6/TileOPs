"""Ascend implementations of attention state boundaries."""

from tileops.kernels.ascend.attention.merge_states import AscendMergeAttentionStatesKernel

__all__ = ["AscendMergeAttentionStatesKernel"]
