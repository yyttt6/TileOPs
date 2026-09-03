"""Attention indexing / sparse selection family (manifest ``attention_indexing``)."""

from __future__ import annotations

from .._registry import register
from ..kernels.attention_indexing import build_topk_selector_kernel


@register("TopkSelectorFwdOp")
def build_topk_selector(index_score, starts, ends, *, topk):
    """Manifest signature: ``(index_score, starts, ends)`` positional, ``topk`` by keyword.

    ``topk`` is a compile-time constant on this boundary: ``TopkSelectorFwdOp`` takes it
    in its constructor (``TileOPs/src/tileops/ops/topk_selector.py:26-32``) and
    ``Op._manifest_params()`` reads it off the instance, so the kernel can size UB with it.
    """
    if index_score is None or starts is None or ends is None:
        raise ValueError("TopkSelectorFwdOp requires index_score, starts and ends")
    return build_topk_selector_kernel(
        tuple(index_score.shape),
        index_score.dtype,
        tuple(starts.shape),
        tuple(ends.shape),
        topk,
    )


__all__ = ["build_topk_selector"]
