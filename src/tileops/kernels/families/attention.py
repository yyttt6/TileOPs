"""Dense MHA and GQA registrations for the Ascend CV attention template."""

from __future__ import annotations

import torch

from .._registry import register
from ..attention import (
    _require_fp16_gqa,
    build_attention_kernel,
    build_grouped_query_attention_kernel,
)


def _validate_gqa_dtype(q, op_name):
    _require_fp16_gqa(q.dtype, op_name)


def _packed_uniform_lengths(cu_seqlens, total, name):
    """Return ``(batch, seqlen)`` without padding packed input on the host."""
    if cu_seqlens is None or cu_seqlens.dtype != torch.int32:
        raise TypeError(f"{name} must be an int32 tensor")
    values = [int(v) for v in cu_seqlens.detach().cpu().tolist()]
    if len(values) < 2 or values[0] != 0 or values[-1] != int(total):
        raise ValueError(f"{name} must start at 0 and end at packed length {total}")
    lengths = [b - a for a, b in zip(values, values[1:])]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError(f"{name} must describe positive sequence lengths")
    if len(set(lengths)) != 1:
        raise ValueError(
            f"packed attention requires uniform {name} lengths on Ascend; "
            "non-uniform varlen is not padded or silently reshaped"
        )
    return len(lengths), lengths[0]


def _window_mask(batch, seq_len, is_causal, window_size_left, window_size_right, device):
    import torch

    q = torch.arange(seq_len, device=device).view(seq_len, 1)
    k = torch.arange(seq_len, device=device).view(1, seq_len)
    visible = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    if is_causal:
        visible &= k <= q
    if window_size_left >= 0:
        visible &= k >= q - int(window_size_left)
    if window_size_right >= 0:
        visible &= k <= q + int(window_size_right)
    return visible.float().unsqueeze(0).expand(batch, -1, -1).contiguous()


def _validate_common(q, k, v, op_name):
    if q is None or k is None or v is None:
        raise ValueError(f"{op_name} requires q, k and v tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError(f"{op_name} inputs must share a device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError(f"{op_name} inputs must share dtype")
    if len(q.shape) != 4 or len(k.shape) != 4 or len(v.shape) != 4:
        raise ValueError(f"{op_name} expects rank-4 BSHD inputs")


@register("MultiHeadAttentionFwdOp")
def build_multi_head_attention(q, k, v, *, is_causal=True):
    _validate_common(q, k, v, "MultiHeadAttentionFwdOp")
    if len(q.shape) != 4 or tuple(k.shape) != tuple(q.shape) or tuple(v.shape) != tuple(q.shape):
        raise ValueError(f"dense MHA expects equal BSHD inputs, got {q.shape}, {k.shape}, {v.shape}")
    batch, seq_len, heads, dim = (int(x) for x in q.shape)
    if batch <= 0 or seq_len <= 0 or heads <= 0:
        raise ValueError(f"dense MHA dimensions must be positive, got {q.shape}")
    return build_attention_kernel(batch, seq_len, heads, dim, bool(is_causal), q.dtype)


@register("GroupedQueryAttentionFwdOp")
def build_grouped_query_attention(q, k, v, *, is_causal=True):
    _validate_common(q, k, v, "GroupedQueryAttentionFwdOp")
    _validate_gqa_dtype(q, "GroupedQueryAttentionFwdOp")
    batch, seq_len, heads, dim = (int(x) for x in q.shape)
    kv_batch, kv_seq_len, heads_kv, kv_dim = (int(x) for x in k.shape)
    if tuple(v.shape) != tuple(k.shape):
        raise ValueError(f"GQA expects equal K/V shapes, got {k.shape}, {v.shape}")
    if (kv_batch, kv_seq_len, kv_dim) != (batch, seq_len, dim):
        raise ValueError(f"GQA expects matching B/S/D dimensions, got {q.shape}, {k.shape}")
    if batch <= 0 or seq_len <= 0 or heads <= 0 or heads_kv <= 0:
        raise ValueError(f"GQA dimensions must be positive, got {q.shape}, {k.shape}")
    if heads % heads_kv != 0:
        raise ValueError(f"GQA requires H % H_kv == 0, got H={heads}, H_kv={heads_kv}")
    return build_grouped_query_attention_kernel(
        batch, seq_len, heads, heads_kv, dim, bool(is_causal), q.dtype
    )


def _build_packed_gqa(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    *,
    max_seqlen_q,
    max_seqlen_kv,
    is_causal=True,
    q_scale=None,
    k_scale=None,
    v_scale=None,
    op_name="GroupedQueryAttentionPrefillFwdOp",
):
    del q_scale, k_scale, v_scale
    if q is None or k is None or v is None:
        raise ValueError(f"{op_name} requires q, k and v tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError(f"{op_name} inputs must share a device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError(f"{op_name} inputs must share dtype")
    _validate_gqa_dtype(q, op_name)
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"{op_name} expects packed THD q/k/v tensors")
    batch_q, seq_q = _packed_uniform_lengths(cu_seqlens_q, q.shape[0], "cu_seqlens_q")
    batch_k, seq_k = _packed_uniform_lengths(cu_seqlens_kv, k.shape[0], "cu_seqlens_kv")
    if batch_q != batch_k or seq_q > int(max_seqlen_q) or seq_k > int(max_seqlen_kv):
        raise ValueError(f"{op_name} packed lengths exceed declared max sequence lengths")
    if seq_q != seq_k:
        raise ValueError(f"{op_name} requires equal packed Q/K lengths on Ascend")
    heads, dim = int(q.shape[1]), int(q.shape[2])
    heads_kv = int(k.shape[1])
    if int(k.shape[2]) != dim or tuple(v.shape) != tuple(k.shape):
        raise ValueError(f"{op_name} q/k/v packed head dimensions do not match")
    base = build_grouped_query_attention_kernel(
        batch_q, seq_q, heads, heads_kv, dim, bool(is_causal), q.dtype
    )

    def invoke(q_in, k_in, v_in, cu_q=None, cu_kv=None):
        del cu_q, cu_kv
        return base(q_in, k_in, v_in, None, None).view(q_in.shape[0], heads, dim)

    return invoke


@register("GroupedQueryAttentionPrefillFwdOp")
def build_grouped_query_attention_prefill(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    q_scale=None,
    k_scale=None,
    v_scale=None,
    *,
    max_seqlen_q,
    max_seqlen_kv,
    is_causal=True,
    sm_scale=None,
    softcap=None,
    window_size_left=-1,
    window_size_right=-1,
    backend="auto",
    validate_uniform_cu_seqlens=True,
    dtype=None,
):
    del sm_scale, softcap, window_size_left, window_size_right, backend
    del validate_uniform_cu_seqlens, dtype
    return _build_packed_gqa(
        q, k, v, cu_seqlens_q, cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv,
        is_causal=is_causal, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale,
    )


@register("GroupedQueryAttentionSlidingWindowFwdOp")
def build_grouped_query_attention_sliding_window(
    q, k, v, *, is_causal=True, window_size_left=-1, window_size_right=-1
):
    _validate_common(q, k, v, "GroupedQueryAttentionSlidingWindowFwdOp")
    _validate_gqa_dtype(q, "GroupedQueryAttentionSlidingWindowFwdOp")
    if q.ndim != 4 or tuple(v.shape) != tuple(k.shape):
        raise ValueError("GroupedQueryAttentionSlidingWindowFwdOp expects BSHD q and KV tensors")
    batch, seq_len, heads, dim = (int(x) for x in q.shape)
    if tuple(k.shape[:2]) != (batch, seq_len) or int(k.shape[3]) != dim:
        raise ValueError("GroupedQueryAttentionSlidingWindowFwdOp shape mismatch")
    base = build_grouped_query_attention_kernel(
        batch, seq_len, heads, int(k.shape[2]), dim, bool(is_causal), q.dtype
    )

    def invoke(q_in, k_in, v_in):
        mask = _window_mask(
            batch, seq_len, bool(is_causal), int(window_size_left), int(window_size_right), q_in.device
        )
        qp = q_in.view(batch * seq_len, heads, dim)
        kp = k_in.view(batch * seq_len, int(k_in.shape[2]), dim)
        vp = v_in.view(batch * seq_len, int(v_in.shape[2]), dim)
        return base(qp, kp, vp, None, None, mask).view_as(q_in)

    return invoke


@register("GroupedQueryAttentionSlidingWindowVarlenFwdOp")
def build_grouped_query_attention_sliding_window_varlen(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    *,
    is_causal=True,
    window_size_left=-1,
    window_size_right=-1,
    max_seqlen_q,
):
    # Keep cu_seqlens genuinely packed. The current fixed-shape CV kernel can
    # execute uniform packed requests; non-uniform requests are rejected rather
    # than padded, which would change the FLOPs contract.
    batch, seq_len = _packed_uniform_lengths(cu_seqlens_q, q.shape[0], "cu_seqlens_q")
    batch_k, seq_k = _packed_uniform_lengths(cu_seqlens_k, k.shape[0], "cu_seqlens_k")
    if batch != batch_k or seq_len != seq_k or seq_len > int(max_seqlen_q):
        raise ValueError("GroupedQueryAttentionSlidingWindowVarlenFwdOp requires matching uniform packed lengths")
    _validate_gqa_dtype(q, "GroupedQueryAttentionSlidingWindowVarlenFwdOp")
    heads, dim, heads_kv = int(q.shape[1]), int(q.shape[2]), int(k.shape[1])
    base = build_grouped_query_attention_kernel(batch, seq_len, heads, heads_kv, dim, bool(is_causal), q.dtype)

    def invoke(q_in, k_in, v_in, cu_q=None, cu_k=None):
        del cu_q, cu_k
        mask = _window_mask(batch, seq_len, bool(is_causal), int(window_size_left), int(window_size_right), q_in.device)
        return base(q_in, k_in, v_in, None, None, mask)

    return invoke


__all__ = [
    "build_grouped_query_attention",
    "build_multi_head_attention",
    "build_grouped_query_attention_prefill",
    "build_grouped_query_attention_sliding_window",
    "build_grouped_query_attention_sliding_window_varlen",
]
