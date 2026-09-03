"""Position-encoding family registrations for the Ascend backend."""

from __future__ import annotations

import torch

from .._registry import register
from ..kernels.position_encoding import build_position_ids_kernel, build_rope_kernel


def _build_rope(x, *, layout, rotation, **params):
    if x is None:
        raise ValueError("RoPE requires an input tensor")
    shape = tuple(int(v) for v in x.shape)
    if layout == "1d":
        if len(shape) != 2:
            raise ValueError(f"RoPE 1d layout expects [seq_len, head_dim], got {shape}")
        seq_len, head_dim = shape
        batch, num_heads = 1, 1
    elif layout == "2d":
        if len(shape) != 4:
            raise ValueError(f"RoPE 2d layout expects [batch, seq_len, num_heads, head_dim], got {shape}")
        batch, seq_len, num_heads, head_dim = shape
    else:
        raise ValueError(f"unsupported RoPE layout {layout!r}")
    return build_rope_kernel(
        x,
        layout=layout,
        rotation=rotation,
        seq_len=seq_len,
        head_dim=head_dim,
        dtype=x.dtype,
        batch=batch,
        num_heads=num_heads,
        **params,
    )


@register("RopeNeoxFwdOp")
def build_rope_neox(x, *, layout="1d", base=10000.0, seq_len=None, head_dim=None, dtype=None, batch=None, num_heads=None):
    del seq_len, head_dim, dtype, batch, num_heads
    return _build_rope(x, layout=layout, rotation="neox", base=base)


@register("RopeNonNeoxFwdOp")
def build_rope_non_neox(x, *, layout="1d", base=10000.0, seq_len=None, head_dim=None, dtype=None, batch=None, num_heads=None):
    del seq_len, head_dim, dtype, batch, num_heads
    return _build_rope(x, layout=layout, rotation="non_neox", base=base)


@register("RopeLlama31FwdOp")
def build_rope_llama31(x, *, layout="1d", base=10000.0, scale_factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position=8192, seq_len=None, head_dim=None, dtype=None, batch=None, num_heads=None):
    del seq_len, head_dim, dtype, batch, num_heads
    return _build_rope(x, layout=layout, rotation="neox", base=base, scale_factor=scale_factor, low_freq_factor=low_freq_factor, high_freq_factor=high_freq_factor, original_max_position=original_max_position)


@register("RopeYarnFwdOp")
def build_rope_yarn(x, *, layout="1d", base=10000.0, scale=16.0, original_max_position=4096, beta_fast=32.0, beta_slow=1.0, attn_factor=1.0, seq_len=None, head_dim=None, dtype=None, batch=None, num_heads=None):
    del seq_len, head_dim, dtype, batch, num_heads
    return _build_rope(x, layout=layout, rotation="neox", base=base, scale=scale, original_max_position=original_max_position, beta_fast=beta_fast, beta_slow=beta_slow, attn_factor=attn_factor)


@register("RopeLongRopeFwdOp")
def build_rope_longrope(x, *, layout="1d", base=10000.0, rescale_factors=None, max_position_embeddings=4096, original_max_position_embeddings=4096, seq_len=None, head_dim=None, dtype=None, batch=None, num_heads=None):
    del seq_len, head_dim, dtype, batch, num_heads
    if rescale_factors is not None:
        if not isinstance(rescale_factors, torch.Tensor):
            raise TypeError("rescale_factors must be a torch.Tensor or None")
        expected = (int(x.shape[-1]) // 2,)
        if tuple(rescale_factors.shape) != expected:
            raise ValueError(f"rescale_factors shape must be {expected}, got {tuple(rescale_factors.shape)}")
    return _build_rope(x, layout=layout, rotation="neox", base=base, max_position_embeddings=max_position_embeddings, original_max_position_embeddings=original_max_position_embeddings)


@register("RopeNeoxPositionIdsFwdOp")
def build_rope_neox_position_ids(x, position_ids, *, max_position, base=10000.0, rotary_dim=None):
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported RoPE dtype {x.dtype}")
    return build_position_ids_kernel(x, position_ids, max_position=max_position, base=base, rotary_dim=rotary_dim)


__all__ = [
    "build_rope_llama31",
    "build_rope_longrope",
    "build_rope_neox",
    "build_rope_neox_position_ids",
    "build_rope_non_neox",
    "build_rope_yarn",
]
