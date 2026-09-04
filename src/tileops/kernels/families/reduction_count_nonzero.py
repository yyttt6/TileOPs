"""Ascend registration for the CountNonzero reduction family member."""

from __future__ import annotations

from .._registry import register
from ..reduction_count_nonzero import build_count_nonzero_kernel


@register("CountNonzeroFwdOp")
def build_count_nonzero(x, *, dim=None):
    if x is None:
        raise ValueError("CountNonzeroFwdOp requires an input tensor")
    return build_count_nonzero_kernel(tuple(x.shape), x.dtype, dim)


__all__ = ["build_count_nonzero"]
