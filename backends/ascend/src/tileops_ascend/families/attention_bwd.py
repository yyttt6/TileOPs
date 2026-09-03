"""Dense MHA backward registration for the standalone backward pilot."""

import torch

from .._registry import register
from ..kernels.attention_bwd import (
    build_attention_bwd_kernel,
    build_attention_bwd_role_kernel,
)


@register("MultiHeadAttentionBwdOp")
def build_multi_head_attention_bwd(q, k, v, o, do, lse, *, is_causal=True):
    tensors = (q, k, v, o, do, lse)
    if any(x is None for x in tensors):
        raise ValueError("MultiHeadAttentionBwdOp requires q, k, v, o, do and lse")
    if any(x.device != q.device for x in tensors):
        raise ValueError("MultiHeadAttentionBwdOp inputs must share a device")
    if q.dtype is not torch.float16:
        raise TypeError(
            f"MultiHeadAttentionBwdOp requires dtype=torch.float16; received {q.dtype}. "
            "bf16 is explicitly unsupported by the current attention Cube workspace ABI."
        )
    if any(x.dtype != q.dtype for x in (k, v, o, do)):
        raise TypeError("q, k, v, o and do must share dtype")
    if lse.dtype != torch.float32:
        raise TypeError("lse must be float32")
    q_shape = tuple(int(x) for x in q.shape)
    return build_attention_bwd_kernel(
        q_shape,
        tuple(int(x) for x in k.shape),
        tuple(int(x) for x in v.shape),
        tuple(int(x) for x in o.shape),
        tuple(int(x) for x in do.shape),
        tuple(int(x) for x in lse.shape),
        q.dtype,
        bool(is_causal),
    )


@register("GroupedQueryAttentionBwdOp")
def build_grouped_query_attention_bwd(q, k, v, o, do, lse, *, is_causal=True):
    """Expose the dense MHA specialization through TileOPs' GQA delegate."""
    tensors = (q, k, v, o, do, lse)
    if any(x is None for x in tensors):
        raise ValueError("GroupedQueryAttentionBwdOp requires q, k, v, o, do and lse")
    if any(x.device != q.device for x in tensors):
        raise ValueError("GroupedQueryAttentionBwdOp inputs must share a device")
    if q.dtype is not torch.float16:
        raise TypeError(
            f"GroupedQueryAttentionBwdOp requires dtype=torch.float16; received {q.dtype}. "
            "bf16 is explicitly unsupported by the current attention Cube workspace ABI."
        )
    if any(x.dtype != q.dtype for x in (k, v, o, do)):
        raise TypeError("q, k, v, o and do must share dtype")
    if lse.dtype != torch.float32:
        raise TypeError("lse must be float32")
    q_shape = tuple(int(x) for x in q.shape)
    k_shape = tuple(int(x) for x in k.shape)
    v_shape = tuple(int(x) for x in v.shape)
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
        raise ValueError("GroupedQueryAttentionBwdOp expects rank-4 BSHD tensors")
    if k_shape != v_shape:
        raise ValueError("GroupedQueryAttentionBwdOp K/V shapes must match")
    if q_shape[:2] != k_shape[:2] or q_shape[3] != k_shape[3]:
        raise ValueError("GroupedQueryAttentionBwdOp Q/K/V batch, sequence, and dim must match")
    if k_shape[2] <= 0 or q_shape[2] % k_shape[2]:
        raise ValueError(
            f"GroupedQueryAttentionBwdOp requires heads % heads_kv == 0; "
            f"received heads={q_shape[2]}, heads_kv={k_shape[2]}"
        )
    if tuple(o.shape) != q_shape or tuple(do.shape) != q_shape:
        raise ValueError("GroupedQueryAttentionBwdOp O/dO must match Q shape")
    role_kernel = build_attention_bwd_role_kernel(
        q_shape,
        k_shape,
        v_shape,
        tuple(int(x) for x in o.shape),
        tuple(int(x) for x in do.shape),
        tuple(int(x) for x in lse.shape),
        q.dtype,
        bool(is_causal),
    )

    def invoke(*args):
        if len(args) == 2:
            # TileOPs' preprocess result is an opaque tensor to the main role.
            # Carry O through that slot; the split kernel recomputes delta itself.
            return args[0]
        if len(args) == 9:
            return role_kernel(*args)
        raise TypeError(f"unexpected GQA backward role ABI with {len(args)} arguments")

    return invoke


__all__ = ["build_grouped_query_attention_bwd", "build_multi_head_attention_bwd"]
