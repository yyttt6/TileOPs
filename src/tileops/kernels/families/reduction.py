"""Reduction family (archetype 2: AIV single-pass row reduction)."""

from __future__ import annotations

from .._registry import register
from ..reduction import build_reduction_kernel


def _build(x, dim, keepdim, *, op_kind: str, op_name: str, correction: int = 1):
    if x is None:
        raise ValueError(f"{op_name} requires an input tensor")
    return build_reduction_kernel(
        tuple(x.shape),
        x.dtype,
        dim,
        keepdim,
        op_kind=op_kind,
        op_name=op_name,
        correction=correction,
    )


@register("SumFwdOp")
def build_sum(x, *, dim=None, keepdim=False):
    return _build(x, dim, keepdim, op_kind="sum", op_name="SumFwdOp")


@register("MeanFwdOp")
def build_mean(x, *, dim=None, keepdim=False):
    return _build(x, dim, keepdim, op_kind="mean", op_name="MeanFwdOp")


@register("ProdFwdOp")
def build_prod(x, *, dim=-1, keepdim=False):
    return _build(x, dim, keepdim, op_kind="prod", op_name="ProdFwdOp")


@register("VarFwdOp")
def build_var(x, *, dim=None, correction=1, keepdim=False):
    return _build(x, dim, keepdim, op_kind="var", op_name="VarFwdOp", correction=correction)


@register("StdFwdOp")
def build_std(x, *, dim=None, correction=1, keepdim=False):
    return _build(x, dim, keepdim, op_kind="std", op_name="StdFwdOp", correction=correction)


@register("VarMeanFwdOp")
def build_var_mean(x, *, dim=None, correction=1, keepdim=False):
    return _build(x, dim, keepdim, op_kind="var_mean", op_name="VarMeanFwdOp", correction=correction)


@register("LogSumExpFwdOp")
def build_logsumexp(x, *, dim=-1, keepdim=False):
    return _build(x, dim, keepdim, op_kind="logsumexp", op_name="LogSumExpFwdOp")


@register("AmaxFwdOp")
def build_amax(x, *, dim=None, keepdim=False):
    return _build(x, dim, keepdim, op_kind="amax", op_name="AmaxFwdOp")


@register("AminFwdOp")
def build_amin(x, *, dim=None, keepdim=False):
    return _build(x, dim, keepdim, op_kind="amin", op_name="AminFwdOp")


@register("AllFwdOp")
def build_all(x, *, dim=None, keepdim=False):
    return _build(x, dim, keepdim, op_kind="all", op_name="AllFwdOp")


@register("AnyFwdOp")
def build_any(x, *, dim=None, keepdim=False):
    return _build(x, dim, keepdim, op_kind="any", op_name="AnyFwdOp")


def _build_norm(x, ord, dim, keepdim, *, expected_ord, op_kind: str, op_name: str):
    if ord != expected_ord:
        raise ValueError(f"{op_name} only supports ord={expected_ord!r}, got {ord!r}")
    return _build(x, dim, keepdim, op_kind=op_kind, op_name=op_name)


@register("L1NormFwdOp")
def build_l1_norm(x, *, ord=1, dim=None, keepdim=False):
    return _build_norm(
        x,
        ord,
        dim,
        keepdim,
        expected_ord=1,
        op_kind="l1",
        op_name="L1NormFwdOp",
    )


@register("L2NormFwdOp")
def build_l2_norm(x, *, ord=2, dim=None, keepdim=False):
    return _build_norm(
        x,
        ord,
        dim,
        keepdim,
        expected_ord=2,
        op_kind="l2",
        op_name="L2NormFwdOp",
    )


@register("InfNormFwdOp")
def build_inf_norm(x, *, ord=float("inf"), dim=None, keepdim=False):
    return _build_norm(
        x,
        ord,
        dim,
        keepdim,
        expected_ord=float("inf"),
        op_kind="inf",
        op_name="InfNormFwdOp",
    )


__all__ = [
    "build_all",
    "build_amax",
    "build_amin",
    "build_any",
    "build_inf_norm",
    "build_l1_norm",
    "build_l2_norm",
    "build_mean",
    "build_prod",
    "build_var",
    "build_std",
    "build_var_mean",
    "build_logsumexp",
    "build_sum",
]
