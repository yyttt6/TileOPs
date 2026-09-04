"""Indexed reduction family (archetype 4)."""

from __future__ import annotations

from .._registry import register
from ..indexed_reduce import build_argmax_kernel, build_argmin_kernel


@register("ArgmaxFwdOp")
def build_argmax(x, *, dim=None, keepdim=False):
    if x is None:
        raise ValueError("ArgmaxFwdOp requires an input tensor")
    return build_argmax_kernel(tuple(x.shape), x.dtype, dim, keepdim)


@register("ArgminFwdOp")
def build_argmin(x, *, dim=None, keepdim=False):
    if x is None:
        raise ValueError("ArgminFwdOp requires an input tensor")
    return build_argmin_kernel(tuple(x.shape), x.dtype, dim, keepdim)


__all__ = ["build_argmax", "build_argmin"]
