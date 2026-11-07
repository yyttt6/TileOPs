"""Ascend implementations of attention kernel interfaces."""

from .paged_cache_gather import AscendPagedKVCacheGatherKernel

__all__ = ["AscendPagedKVCacheGatherKernel"]
