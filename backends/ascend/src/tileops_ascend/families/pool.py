"""Pooling family (archetype 6: AIV window gather)."""

from __future__ import annotations

from .._registry import register
from ..kernels.pool import build_max_pool2d_kernel
from ..kernels.pool_avg import build_avg_pool_kernel, build_max_pool3d_kernel
from ..kernels.pool_indices import (
    build_adaptive_pool2d_kernel,
    build_adaptive_max_pool2d_indices_kernel,
    build_max_pool_indices_kernel,
)
from ..kernels.pool_max1d import build_max_pool1d_kernel


@register("MaxPool2dFwdOp")
def build_max_pool2d(
    input,
    *,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    """Build the value-only NCHW max-pool kernel declared by the manifest."""
    if input is None:
        raise ValueError("MaxPool2dFwdOp requires an input tensor")
    return build_max_pool2d_kernel(
        tuple(input.shape),
        input.dtype,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


@register("MaxPool1dFwdOp")
def build_max_pool1d(
    input,
    *,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    """Build value-only NCL max-pooling through the dedicated AIV template."""
    if input is None:
        raise ValueError("MaxPool1dFwdOp requires an input tensor")
    return build_max_pool1d_kernel(
        tuple(input.shape),
        input.dtype,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


def _build_avg(
    op_name,
    ndim,
    input,
    *,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    if input is None:
        raise ValueError(f"{op_name} requires an input tensor")
    return build_avg_pool_kernel(
        tuple(input.shape),
        input.dtype,
        ndim=ndim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@register("AvgPool1dFwdOp")
def build_avg_pool1d(
    input,
    *,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
):
    return _build_avg(
        "AvgPool1dFwdOp",
        1,
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=None,
    )


@register("AvgPool2dFwdOp")
def build_avg_pool2d(
    input,
    *,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    return _build_avg(
        "AvgPool2dFwdOp",
        2,
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@register("AvgPool3dFwdOp")
def build_avg_pool3d(
    input,
    *,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    return _build_avg(
        "AvgPool3dFwdOp",
        3,
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@register("MaxPool3dFwdOp")
def build_max_pool3d(
    input,
    *,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    if input is None:
        raise ValueError("MaxPool3dFwdOp requires an input tensor")
    return build_max_pool3d_kernel(
        tuple(input.shape),
        input.dtype,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


def _build_max_pool_indices(
    ndim,
    input,
    *,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    if input is None:
        raise ValueError(f"MaxPool{ndim}dIndicesFwdOp requires an input tensor")
    return build_max_pool_indices_kernel(
        tuple(input.shape),
        input.dtype,
        ndim=ndim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


@register("MaxPool1dIndicesFwdOp")
def build_max_pool1d_indices(
    input,
    *,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    return _build_max_pool_indices(
        1,
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


@register("MaxPool2dIndicesFwdOp")
def build_max_pool2d_indices(
    input,
    *,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    return _build_max_pool_indices(
        2,
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


@register("MaxPool3dIndicesFwdOp")
def build_max_pool3d_indices(
    input,
    *,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    return _build_max_pool_indices(
        3,
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


@register("AdaptiveMaxPool2dIndicesFwdOp")
def build_adaptive_max_pool2d_indices(input, *, output_size):
    if input is None:
        raise ValueError("AdaptiveMaxPool2dIndicesFwdOp requires an input tensor")
    return build_adaptive_max_pool2d_indices_kernel(
        tuple(input.shape), input.dtype, output_size=output_size
    )


def _build_adaptive_pool2d(op_name, reduction, input, *, output_size):
    if input is None:
        raise ValueError(f"{op_name} requires an input tensor")
    return build_adaptive_pool2d_kernel(
        tuple(input.shape),
        input.dtype,
        output_size=output_size,
        reduction=reduction,
    )


@register("AdaptiveAvgPool2dFwdOp")
def build_adaptive_avg_pool2d(input, *, output_size):
    return _build_adaptive_pool2d(
        "AdaptiveAvgPool2dFwdOp", "avg", input, output_size=output_size
    )


@register("AdaptiveMaxPool2dFwdOp")
def build_adaptive_max_pool2d(input, *, output_size):
    return _build_adaptive_pool2d(
        "AdaptiveMaxPool2dFwdOp", "max", input, output_size=output_size
    )


__all__ = [
    "build_adaptive_avg_pool2d",
    "build_adaptive_max_pool2d",
    "build_avg_pool1d",
    "build_avg_pool2d",
    "build_avg_pool3d",
    "build_max_pool1d",
    "build_max_pool2d",
    "build_max_pool3d",
    "build_max_pool1d_indices",
    "build_max_pool2d_indices",
    "build_max_pool3d_indices",
    "build_adaptive_max_pool2d_indices",
]
