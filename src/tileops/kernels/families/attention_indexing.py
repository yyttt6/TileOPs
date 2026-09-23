"""Attention indexing / sparse selection family (manifest ``attention_indexing``)."""

from __future__ import annotations

from .._registry import register
from ..attention_indexing import build_topk_selector_kernel
from ..attention_indexing_fp8 import build_fp8_lightning_indexer_kernel


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


@register("FP8LightningIndexerFwdOp")
def build_fp8_lightning_indexer(
    index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, index_k_scale=None,
    *, clean_logits=True, config=None,
):
    """Manifest signature: the six ``signature.inputs`` positionally, params by keyword.

    ``index_k_scale`` is the manifest's one optional input, so it arrives as ``None``
    when the workload does not declare ``index_k_scale_shape``.  ``config`` is the
    manifest's escape hatch for a kernel-config override; this builder has a single
    static schedule and rejects a non-empty one rather than ignoring it.

    float8_e4m3fn tensors reach the kernel already decoded to float16 by
    :func:`tileops.kernels.attention_indexing_fp8.decode_fp8_e4m3fn`, because the
    Cube on 910B1 has no fp8 path (``tileops/perf/profile.py:73-89``) and
    ``Tensor.to(torch.float8_e4m3fn)`` aborts on this build with aclnnInplaceCopy
    561103.  The decode is bit-exact (R352 §fp8) and is inside our timing window.
    """
    if index_q is None or index_k is None or weights is None:
        raise ValueError("FP8LightningIndexerFwdOp requires index_q, index_k and weights")
    if cu_seqlen_ks is None or cu_seqlen_ke is None:
        raise ValueError("FP8LightningIndexerFwdOp requires cu_seqlen_ks and cu_seqlen_ke")
    if config:
        raise ValueError(
            "the registered Ascend FP8LightningIndexerFwdOp kernel has one static "
            f"schedule and does not implement manifest config={config!r}"
        )
    return build_fp8_lightning_indexer_kernel(
        tuple(index_q.shape),
        tuple(index_k.shape),
        tuple(weights.shape),
        tuple(cu_seqlen_ks.shape),
        tuple(cu_seqlen_ke.shape),
        None if index_k_scale is None else tuple(index_k_scale.shape),
        index_q.dtype,
        bool(clean_logits),
    )


__all__ = ["build_topk_selector", "build_fp8_lightning_indexer"]
