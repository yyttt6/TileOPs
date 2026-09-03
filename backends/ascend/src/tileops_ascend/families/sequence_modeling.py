"""Sequence-modeling family."""

import math

import torch

from .._registry import register
from ..kernels.sequence_modeling import (
    build_engram_bwd_kernel,
    build_engram_fwd_kernel,
    build_mhc_post_kernel,
    build_mhc_pre_kernel,
)


def _validate_engram_tensor(spec, shape, dtype, name):
    if tuple(int(v) for v in spec.shape) != tuple(shape):
        raise ValueError(f"{name} shape must be {tuple(shape)}, got {tuple(spec.shape)}")
    if spec.dtype != dtype:
        raise TypeError(f"{name} dtype must be {dtype}, got {spec.dtype}")


@register("EngramGateConvFwdOp")
def build_engram_fwd(H, k, v, rms_w_h, rms_w_v, conv_w, *, M, seq_len, d, eps=1e-6):
    if H.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("EngramGateConvFwdOp supports float16/bfloat16")
    shape = (int(M), int(seq_len), int(d))
    _validate_engram_tensor(H, shape, H.dtype, "H")
    _validate_engram_tensor(k, shape, H.dtype, "k")
    _validate_engram_tensor(v, shape, H.dtype, "v")
    _validate_engram_tensor(rms_w_h, (int(d),), H.dtype, "rms_w_h")
    _validate_engram_tensor(rms_w_v, (int(d),), H.dtype, "rms_w_v")
    _validate_engram_tensor(conv_w, (4, int(d)), H.dtype, "conv_w")
    return build_engram_fwd_kernel(M, seq_len, d, eps, H.dtype)


@register("EngramGateConvBwdOp")
def build_engram_bwd(
    dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha, rrms_h, rrms_k, rrms_v,
    *, M, seq_len, d, eps=1e-6,
):
    if dY.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("EngramGateConvBwdOp supports float16/bfloat16")
    shape = (int(M), int(seq_len), int(d))
    for spec, name in ((dY, "dY"), (H, "H"), (k, "k"), (v, "v"), (vhat, "vhat")):
        _validate_engram_tensor(spec, shape, dY.dtype, name)
    _validate_engram_tensor(rms_w_h, (int(d),), dY.dtype, "rms_w_h")
    _validate_engram_tensor(rms_w_v, (int(d),), dY.dtype, "rms_w_v")
    _validate_engram_tensor(conv_w, (4, int(d)), dY.dtype, "conv_w")
    for spec, name in ((alpha, "alpha"), (rrms_h, "rrms_h"), (rrms_k, "rrms_k"), (rrms_v, "rrms_v")):
        _validate_engram_tensor(spec, (int(M), int(seq_len)), torch.float32, name)
    return build_engram_bwd_kernel(M, seq_len, d, eps, dY.dtype)


@register("MHCPostFwdOp")
def build_mhc_post(x_layer_out, h_post, x_res):
    if x_layer_out is None or h_post is None or x_res is None:
        raise ValueError("MHCPostFwdOp requires x_layer_out, h_post, and x_res")
    if len(x_layer_out.shape) != 2 or len(h_post.shape) != 2 or len(x_res.shape) != 2:
        raise ValueError("MHCPostFwdOp expects 2D inputs")
    batch, c_x = (int(v) for v in x_layer_out.shape)
    if int(h_post.shape[0]) != batch or int(x_res.shape[0]) != batch:
        raise ValueError("MHCPostFwdOp inputs must have matching batch dimensions")
    n_expand = int(h_post.shape[1])
    if tuple(x_res.shape) != (batch, n_expand * c_x):
        raise ValueError("MHCPostFwdOp x_res shape must be [B, N*C]")
    if x_layer_out.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("MHCPostFwdOp x_layer_out must be floating point")
    if h_post.dtype != torch.float32 or x_res.dtype != x_layer_out.dtype:
        raise TypeError("MHCPostFwdOp expects h_post=float32 and matching x_layer_out/x_res dtype")
    if len({x_layer_out.device, h_post.device, x_res.device}) != 1:
        raise ValueError("MHCPostFwdOp tensors must share a device")
    return build_mhc_post_kernel(batch, n_expand, c_x, x_layer_out.dtype)


@register("MHCPreFwdOp")
def build_mhc_pre(
    phi, x, b, *, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat, sinkhorn_eps=0.02
):
    if phi.dtype != torch.float32 or b.dtype != torch.float32 or x.dtype != torch.bfloat16:
        raise TypeError("MHCPreFwdOp expects phi/b=float32 and x=bfloat16")
    if len(phi.shape) != 2 or len(x.shape) != 2 or len(b.shape) != 1:
        raise ValueError("MHCPreFwdOp expects phi/x/b ranks 2/2/1")
    batch, x_dim = (int(v) for v in x.shape)
    if int(phi.shape[0]) != x_dim or int(b.shape[0]) != int(phi.shape[1]):
        raise ValueError("MHCPreFwdOp phi/x/b dimensions are inconsistent")
    phi_dim = int(phi.shape[1])
    n_expand = int(math.isqrt(phi_dim + 1) - 1)
    if n_expand <= 0 or n_expand * n_expand + 2 * n_expand != phi_dim:
        raise ValueError("MHCPreFwdOp phi.shape[1] does not encode n_expand")
    if x_dim % n_expand:
        raise ValueError("MHCPreFwdOp x dimension must be divisible by n_expand")
    return build_mhc_pre_kernel(
        batch, n_expand, x_dim // n_expand, alpha_pre, alpha_post, alpha_res,
        sinkhorn_repeat, sinkhorn_eps,
    )


__all__ = ["build_engram_bwd", "build_engram_fwd", "build_mhc_post", "build_mhc_pre"]
