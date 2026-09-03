"""Cumsum and cumprod pilot builders."""

from .._registry import register
from ..kernels.scan import build_scan


@register("CumsumFwdOp")
def build_cumsum(x, *, dim=-1):
    return build_scan(x, dim=dim, op_kind="sum")


@register("CumprodFwdOp")
def build_cumprod(x, *, dim=-1):
    return build_scan(x, dim=dim, op_kind="prod")


__all__ = ["build_cumsum", "build_cumprod"]
