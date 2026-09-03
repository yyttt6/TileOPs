"""Decode attention registration; kept separate from the prefill family."""

import torch

from .._registry import register
from ..kernels.attention_decode import (
    build_gqa_decode_kernel,
    build_gqa_decode_paged_kernel,
    build_gqa_prefill_paged_kernel,
    build_mha_decode_kernel,
    build_mha_decode_paged_kernel,
    build_mla_decode_kernel,
)


@register("MultiHeadAttentionDecodeWithKVCacheFwdOp")
def build_multi_head_attention_decode(q, k, v):
    if q is None or k is None or v is None:
        raise ValueError("decode MHA requires q, k and v tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("decode MHA inputs must share a device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("decode MHA inputs must share dtype")
    if len(q.shape) != 4 or len(k.shape) != 4 or tuple(v.shape) != tuple(k.shape):
        raise ValueError("decode MHA expects q=[B,1,H,D], k/v=[B,N,H,D]")
    batch, seq_q, heads, dim = (int(x) for x in q.shape)
    kb, max_seqlen_kv, kh, kd = (int(x) for x in k.shape)
    if seq_q != 1 or (batch, heads, dim) != (kb, kh, kd):
        raise ValueError(
            f"decode MHA shape mismatch: q={tuple(q.shape)}, k={tuple(k.shape)}"
        )
    kernel = build_mha_decode_kernel(batch, heads, max_seqlen_kv, dim, q.dtype)

    def invoke(q_in, k_in, v_in):
        cache_seqlens = torch.full(
            (batch,), max_seqlen_kv, dtype=torch.int32, device=q_in.device
        )
        return kernel(q_in, k_in, v_in, cache_seqlens)

    return invoke


@register("MultiHeadAttentionDecodePagedWithKVCacheFwdOp")
def build_multi_head_attention_decode_paged(
    q, k, v, real_seqlen_kv, block_table, *, page_size, is_causal=False
):
    if any(tensor is None for tensor in (q, k, v, real_seqlen_kv, block_table)):
        raise ValueError("paged decode MHA requires q, k, v, cache lengths and block table")
    if len({tensor.device for tensor in (q, k, v, real_seqlen_kv, block_table)}) != 1:
        raise ValueError("paged decode MHA inputs must share a device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("paged decode MHA q/k/v must share dtype")
    if real_seqlen_kv.dtype != torch.int32 or block_table.dtype != torch.int32:
        raise TypeError("paged decode MHA metadata must be int32")
    if len(q.shape) != 4 or len(k.shape) != 3 or tuple(v.shape) != tuple(k.shape):
        raise ValueError("paged decode MHA expects q=[B,1,H,D], k/v=[N,H,D]")
    batch, seq_q, heads, dim = (int(x) for x in q.shape)
    physical_tokens, kh, kd = (int(x) for x in k.shape)
    if seq_q != 1 or (batch, heads, dim) != (int(real_seqlen_kv.shape[0]), kh, kd):
        raise ValueError("paged decode MHA shape mismatch")
    if tuple(block_table.shape) != (batch, (physical_tokens + int(page_size) - 1) // int(page_size)):
        raise ValueError("paged decode MHA block_table shape mismatch")
    if tuple(real_seqlen_kv.shape) != (batch,):
        raise ValueError("paged decode MHA real_seqlen_kv shape mismatch")
    kernel = build_mha_decode_paged_kernel(
        batch,
        heads,
        seq_q,
        physical_tokens,
        dim,
        int(page_size),
        bool(is_causal),
        q.dtype,
    )

    def invoke(q_in, k_in, v_in, lengths_in, table_in):
        return kernel(q_in, k_in, v_in, lengths_in, table_in)

    return invoke


@register("MultiHeadLatentAttentionDecodeWithKVCacheFwdOp")
def build_multi_head_latent_attention_decode(q, q_pe, k, k_pe, *, pe_dim):
    tensors = (q, q_pe, k, k_pe)
    if any(tensor is None for tensor in tensors):
        raise ValueError("decode MLA requires q, q_pe, k and k_pe tensors")
    if (
        len({tensor.device for tensor in tensors}) != 1
        or len({tensor.dtype for tensor in tensors}) != 1
    ):
        raise ValueError("decode MLA inputs must share device and dtype")
    if (
        len(q.shape) != 3
        or len(q_pe.shape) != 3
        or len(k.shape) != 4
        or len(k_pe.shape) != 4
    ):
        raise ValueError("decode MLA expects q/q_pe rank 3 and k/k_pe rank 4")
    batch, heads, dim = (int(x) for x in q.shape)
    kb, max_seqlen_kv, heads_kv, kd = (int(x) for x in k.shape)
    if tuple(q_pe.shape) != (batch, heads, int(pe_dim)):
        raise ValueError(f"q_pe shape mismatch: {tuple(q_pe.shape)}")
    if tuple(k_pe.shape) != (batch, max_seqlen_kv, heads_kv, int(pe_dim)):
        raise ValueError(f"k_pe shape mismatch: {tuple(k_pe.shape)}")
    if (batch, dim) != (kb, kd):
        raise ValueError(
            f"decode MLA shape mismatch: q={tuple(q.shape)}, k={tuple(k.shape)}"
        )
    kernel = build_mla_decode_kernel(
        batch, heads, heads_kv, max_seqlen_kv, dim, int(pe_dim), q.dtype
    )

    def invoke(q_in, q_pe_in, k_in, k_pe_in):
        cache_seqlens = torch.full(
            (batch,), max_seqlen_kv, dtype=torch.int32, device=q_in.device
        )
        return kernel(q_in, q_pe_in, k_in, k_pe_in, cache_seqlens)

    return invoke


@register("GroupedQueryAttentionDecodeWithKVCacheFwdOp")
def build_grouped_query_attention_decode(q, k, v):
    if q is None or k is None or v is None:
        raise ValueError("decode GQA requires q, k and v tensors")
    from ..kernels.attention import _require_fp16_gqa

    _require_fp16_gqa(q.dtype, "GroupedQueryAttentionDecodeWithKVCacheFwdOp")
    if (
        q.device != k.device
        or q.device != v.device
        or q.dtype != k.dtype
        or q.dtype != v.dtype
    ):
        raise ValueError("decode GQA inputs must share device and dtype")
    if len(q.shape) != 3 or len(k.shape) != 4 or tuple(v.shape) != tuple(k.shape):
        raise ValueError("decode GQA expects q=[B,H,D], k/v=[B,N,H_kv,D]")
    batch, heads, dim = (int(x) for x in q.shape)
    kb, max_seqlen_kv, heads_kv, kd = (int(x) for x in k.shape)
    if (batch, dim) != (kb, kd) or heads % heads_kv:
        raise ValueError(
            f"decode GQA shape mismatch: q={tuple(q.shape)}, k={tuple(k.shape)}"
        )
    kernel = build_gqa_decode_kernel(
        batch, heads, heads_kv, max_seqlen_kv, dim, q.dtype
    )

    def invoke(q_in, k_in, v_in):
        cache_seqlens = torch.full(
            (batch,), max_seqlen_kv, dtype=torch.int32, device=q_in.device
        )
        return kernel(q_in, k_in, v_in, cache_seqlens)

    return invoke


@register("GroupedQueryAttentionDecodePagedWithKVCacheFwdOp")
def build_grouped_query_attention_decode_paged(
    q,
    k,
    v,
    real_seqlen_kv,
    block_table,
    *,
    page_size,
    sm_scale=None,
    softcap=None,
):
    tensors = (q, k, v, real_seqlen_kv, block_table)
    if any(tensor is None for tensor in tensors):
        raise ValueError("paged decode GQA requires q, k, v, cache lengths and block table")
    from ..kernels.attention import _require_fp16_gqa

    _require_fp16_gqa(q.dtype, "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp")
    if sm_scale is not None and float(sm_scale) != int(q.shape[-1]) ** -0.5:
        raise ValueError(f"paged decode GQA requires default sm_scale; received {sm_scale}")
    if softcap not in (None, 0, 0.0):
        raise ValueError(f"paged decode GQA requires softcap=None or 0; received {softcap}")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("paged decode GQA inputs must share a device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("paged decode GQA q/k/v must share dtype")
    if real_seqlen_kv.dtype != torch.int32 or block_table.dtype != torch.int32:
        raise TypeError("paged decode GQA metadata must be int32")
    if len(q.shape) != 3 or len(k.shape) != 3 or tuple(v.shape) != tuple(k.shape):
        raise ValueError("paged decode GQA expects q=[B,H,D], k/v=[N,H_kv,D]")
    batch, heads, dim = (int(x) for x in q.shape)
    physical_tokens, heads_kv, kd = (int(x) for x in k.shape)
    if dim != kd or heads_kv <= 0 or heads % heads_kv:
        raise ValueError(
            f"paged decode GQA shape mismatch: q={tuple(q.shape)}, k={tuple(k.shape)}"
        )
    if tuple(real_seqlen_kv.shape) != (batch,):
        raise ValueError("paged decode GQA real_seqlen_kv shape mismatch")
    expected_pages = (physical_tokens + int(page_size) - 1) // int(page_size)
    if tuple(block_table.shape) != (batch, expected_pages):
        raise ValueError(
            "paged decode GQA block_table shape mismatch: "
            f"expected {(batch, expected_pages)}, got {tuple(block_table.shape)}"
        )
    kernel = build_gqa_decode_paged_kernel(
        batch, heads, heads_kv, physical_tokens, dim, int(page_size), q.dtype
    )

    def invoke(q_in, k_in, v_in, lengths_in, table_in):
        return kernel(q_in, k_in, v_in, lengths_in, table_in)

    return invoke


@register("GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp")
def build_grouped_query_attention_prefill_paged(
    q,
    k_new,
    v_new,
    k_pages,
    v_pages,
    k_scale,
    v_scale,
    cu_seqlens_q,
    cache_seqlens,
    block_table,
    *,
    max_pages_per_req,
    page_size,
    max_seqlen_q,
    cache_dtype=None,
    is_causal=True,
    sm_scale=None,
    softcap=None,
    fuse_rope=False,
    rope_base=10000.0,
    max_position=None,
    rotary_dim=None,
):
    del rope_base, max_position, rotary_dim
    tensors = (
        q,
        k_new,
        v_new,
        k_pages,
        v_pages,
        k_scale,
        v_scale,
        cu_seqlens_q,
        cache_seqlens,
        block_table,
    )
    if any(tensor is None for tensor in tensors):
        raise ValueError("paged prefill GQA requires all ten manifest inputs")
    from ..kernels.attention import _require_fp16_gqa

    _require_fp16_gqa(q.dtype, "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp")
    if fuse_rope:
        raise ValueError("paged prefill GQA on Ascend does not yet support fuse_rope=True")
    if softcap not in (None, 0, 0.0):
        raise ValueError(f"paged prefill GQA requires softcap=None or 0; received {softcap}")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("paged prefill GQA inputs must share a device")
    if any(tensor.dtype != q.dtype for tensor in (k_new, v_new, k_pages, v_pages)):
        raise TypeError("paged prefill GQA q/new-KV/cache pages must share dtype")
    if cache_dtype is not None and cache_dtype != q.dtype:
        raise TypeError(
            f"paged prefill GQA requires cache_dtype={q.dtype}; received {cache_dtype}"
        )
    if k_scale.dtype != torch.float32 or v_scale.dtype != torch.float32:
        raise TypeError("paged prefill GQA k_scale/v_scale must be float32")
    if any(
        tensor.dtype != torch.int32
        for tensor in (cu_seqlens_q, cache_seqlens, block_table)
    ):
        raise TypeError("paged prefill GQA sequence metadata must be int32")
    if len(q.shape) != 3 or len(k_new.shape) != 3 or tuple(v_new.shape) != tuple(k_new.shape):
        raise ValueError("paged prefill GQA expects packed q/new-KV rank-3 tensors")
    total_q, heads, dim = (int(x) for x in q.shape)
    kt, heads_kv, kd = (int(x) for x in k_new.shape)
    physical_tokens = int(k_pages.shape[0]) if len(k_pages.shape) == 3 else -1
    if (
        kt != total_q
        or kd != dim
        or tuple(k_pages.shape) != (physical_tokens, heads_kv, dim)
        or tuple(v_pages.shape) != tuple(k_pages.shape)
        or heads_kv <= 0
        or heads % heads_kv
    ):
        raise ValueError(
            "paged prefill GQA shape mismatch: "
            f"q={tuple(q.shape)}, k_new={tuple(k_new.shape)}, k_pages={tuple(k_pages.shape)}"
        )
    batch = int(cache_seqlens.shape[0])
    if tuple(cu_seqlens_q.shape) != (batch + 1,):
        raise ValueError("paged prefill GQA cu_seqlens_q shape mismatch")
    if tuple(block_table.shape) != (batch, int(max_pages_per_req)):
        raise ValueError("paged prefill GQA block_table shape mismatch")
    if tuple(k_scale.shape) != (1,) or tuple(v_scale.shape) != (1,):
        raise ValueError("paged prefill GQA k_scale/v_scale must have shape [1]")
    kernel = build_gqa_prefill_paged_kernel(
        batch,
        heads,
        heads_kv,
        total_q,
        int(max_seqlen_q),
        physical_tokens,
        int(max_pages_per_req),
        int(page_size),
        dim,
        bool(is_causal),
        q.dtype,
        sm_scale,
    )

    def invoke(
        q_in,
        k_new_in,
        v_new_in,
        k_pages_in,
        v_pages_in,
        k_scale_in,
        v_scale_in,
        cu_in,
        cache_in,
        table_in,
    ):
        del k_scale_in, v_scale_in
        return kernel(
            q_in,
            k_new_in,
            v_new_in,
            k_pages_in,
            v_pages_in,
            cu_in,
            cache_in,
            table_in,
        )

    return invoke


__all__ = [
    "build_grouped_query_attention_decode",
    "build_grouped_query_attention_decode_paged",
    "build_grouped_query_attention_prefill_paged",
    "build_multi_head_attention_decode",
    "build_multi_head_latent_attention_decode",
]
