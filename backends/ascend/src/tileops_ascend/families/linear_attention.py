"""Linear-attention family registrations owned by the T038 work stream."""

from __future__ import annotations

from .._registry import register
from ..kernels.gated_deltanet import build_gated_deltanet_kernel


@register("GatedDeltaNetBTHDFwdOp")
def build_gated_deltanet_bthd(q, k, v, g, beta, *, chunk_size=64):
    specs = (q, k, v, g, beta)
    if any(spec is None for spec in specs):
        raise ValueError("GatedDeltaNetBTHDFwdOp requires q, k, v, g, and beta")
    if any(spec.device != q.device for spec in specs):
        raise ValueError("GatedDeltaNetBTHDFwdOp inputs must share one device")
    if any(spec.dtype != q.dtype for spec in specs):
        raise TypeError("GatedDeltaNetBTHDFwdOp inputs must share one dtype")
    return build_gated_deltanet_kernel(
        tuple(q.shape), tuple(k.shape), tuple(v.shape), tuple(g.shape), tuple(beta.shape),
        q.dtype, int(chunk_size)
    )

