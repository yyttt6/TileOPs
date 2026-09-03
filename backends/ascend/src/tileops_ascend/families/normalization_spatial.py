"""Spatial normalization family (GroupNorm, InstanceNorm, BatchNorm).

This module is intentionally separate from ``families.two_pass``.  The latter
owns the row-wise normalization registrations while that file is being
migrated; the PM should import this module after removing the corresponding
fail-closed registrations there.
"""

import math

import torch

from .._registry import register
from ..kernels.normalization_spatial import (
    launch_ada_norm,
    launch_batch,
    launch_batch_backward,
    launch_fused_add_norm,
    launch_group,
    launch_instance_infer,
)


def _check_float(op_name, x, *, min_rank=3):
    if x is None or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"{op_name} supports float16, bfloat16, and float32 inputs")
    if len(x.shape) < min_rank:
        raise ValueError(f"{op_name} expects input rank >= {min_rank}")


@register("GroupNormFwdOp")
def build_group_norm(x, weight=None, bias=None, *, num_groups, eps=1e-5):
    _check_float("GroupNormFwdOp", x)
    n, c = int(x.shape[0]), int(x.shape[1])
    if not isinstance(num_groups, int) or num_groups <= 0 or c % num_groups:
        raise ValueError(f"C={c} must be divisible by num_groups={num_groups}")
    if (weight is None) != (bias is None):
        raise ValueError("weight and bias must be passed together")
    if weight is not None:
        for name, tensor in (("weight", weight), ("bias", bias)):
            if tensor.dtype != x.dtype or tuple(tensor.shape) != (c,):
                raise ValueError(f"{name} must have shape ({c},) and dtype {x.dtype}")

    def launch(inp, scale=weight, shift=bias):
        return launch_group(inp, scale, shift, groups=num_groups, eps=eps)

    return launch


@register("InstanceNormFwdOp")
def build_instance_norm(
    x,
    running_mean=None,
    running_var=None,
    weight=None,
    bias=None,
    *,
    use_input_stats=True,
    momentum=0.1,
    eps=1e-5,
):
    _check_float("InstanceNormFwdOp", x)
    c = int(x.shape[1])
    if (weight is None) != (bias is None):
        raise ValueError("weight and bias must be passed together")
    if (running_mean is None) != (running_var is None):
        raise ValueError("running_mean and running_var must be passed together")
    if use_input_stats and running_mean is not None:
        raise ValueError("running stats are only valid when use_input_stats=False")
    if weight is not None:
        for name, tensor in (("weight", weight), ("bias", bias)):
            if tensor.dtype != x.dtype or tuple(tensor.shape) != (c,):
                raise ValueError(f"{name} must have shape ({c},) and dtype {x.dtype}")
    if not use_input_stats:
        if running_mean is None or running_mean.dtype != torch.float32 or tuple(running_mean.shape) != (c,):
            raise ValueError("running_mean must be fp32 with shape (C,)")
        if running_var is None or running_var.dtype != torch.float32 or tuple(running_var.shape) != (c,):
            raise ValueError("running_var must be fp32 with shape (C,)")

    if use_input_stats:
        def launch(inp, _running_mean=None, _running_var=None, scale=None, shift=None):
            return launch_group(inp, scale, shift, groups=int(inp.shape[1]), eps=eps)
    else:
        def launch(inp, rm, rv, scale=None, shift=None):
            return launch_instance_infer(inp, rm, rv, scale, shift, eps=eps)
    return launch


@register("BatchNormFwdOp")
def build_batch_norm(
    x,
    running_mean,
    running_var,
    weight,
    bias,
    *,
    training=False,
    momentum=0.1,
    eps=1e-5,
):
    _check_float("BatchNormFwdOp", x, min_rank=2)
    c = int(x.shape[1])
    for name, tensor in (("running_mean", running_mean), ("running_var", running_var), ("weight", weight), ("bias", bias)):
        if tensor is None or tuple(tensor.shape) != (c,):
            raise ValueError(f"{name} must have shape ({c},)")
    if running_mean.dtype != torch.float32 or running_var.dtype != torch.float32:
        raise ValueError("running statistics must be float32")
    if weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError("BatchNorm affine tensors must be float32")

    def launch(inp, rm=running_mean, rv=running_var, scale=weight, shift=bias):
        return launch_batch(
            inp,
            rm,
            rv,
            scale,
            shift,
            training=training,
            eps=eps,
            momentum=momentum,
            return_stats=training,
        )

    return launch


def _build_ada(x, scale, shift, gate, *, eps, op_name):
    _check_float(op_name, x, min_rank=1)
    for name, tensor in (("scale", scale), ("shift", shift)):
        if tensor is None or tuple(tensor.shape) != tuple(x.shape) or tensor.dtype != x.dtype:
            raise ValueError(f"{op_name} {name} must match input shape and dtype")
    if gate is not None and (tuple(gate.shape) != tuple(x.shape) or gate.dtype != x.dtype):
        raise ValueError(f"{op_name} gate must match input shape and dtype")

    def launch(inp, scale_in, shift_in, gate_in=gate):
        return launch_ada_norm(inp, scale_in, shift_in, gate_in, eps=eps)

    return launch


@register("AdaLayerNormFwdOp")
def build_ada_layer_norm(x, scale, shift, *, eps=1e-5):
    return _build_ada(
        x,
        scale,
        shift,
        None,
        eps=eps,
        op_name="AdaLayerNormFwdOp",
    )


@register("AdaLayerNormZeroFwdOp")
def build_ada_layer_norm_zero(x, scale, shift, gate, *, eps=1e-5):
    return _build_ada(
        x,
        scale,
        shift,
        gate,
        eps=eps,
        op_name="AdaLayerNormZeroFwdOp",
    )


def _build_fused_add(x, residual, weight, bias, *, kind, eps, op_name):
    _check_float(op_name, x, min_rank=1)
    if kind == "rms" and x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"{op_name} supports float16 and bfloat16 inputs")
    if residual is None or tuple(residual.shape) != tuple(x.shape) or residual.dtype != x.dtype:
        raise ValueError(f"{op_name} residual must match input shape and dtype")
    n = int(x.shape[-1])
    if weight is None or tuple(weight.shape) != (n,) or weight.dtype != x.dtype:
        raise ValueError(f"{op_name} weight must have shape ({n},) and dtype {x.dtype}")
    if kind == "layer":
        if bias is None or tuple(bias.shape) != (n,) or bias.dtype != x.dtype:
            raise ValueError(f"{op_name} bias must match weight shape and dtype")

    def launch(inp, residual_in, scale=weight, shift=bias):
        return launch_fused_add_norm(
            inp,
            residual_in,
            scale,
            shift,
            kind=kind,
            eps=eps,
        )

    return launch


@register("FusedAddRMSNormFwdOp")
def build_fused_add_rms_norm(x, residual, weight, *, eps=1e-6):
    return _build_fused_add(
        x,
        residual,
        weight,
        None,
        kind="rms",
        eps=eps,
        op_name="FusedAddRMSNormFwdOp",
    )


@register("FusedAddLayerNormFwdOp")
def build_fused_add_layer_norm(x, residual, weight, bias, *, eps=1e-5):
    return _build_fused_add(
        x,
        residual,
        weight,
        bias,
        kind="layer",
        eps=eps,
        op_name="FusedAddLayerNormFwdOp",
    )


@register("BatchNormBwdOp")
def build_batch_norm_bwd(grad_out, x, weight, mean, rstd):
    _check_float("BatchNormBwdOp", grad_out, min_rank=2)
    if x is None or tuple(x.shape) != tuple(grad_out.shape) or x.dtype != grad_out.dtype:
        raise ValueError("BatchNormBwdOp x must match grad_out shape and dtype")
    c = int(grad_out.shape[1])
    for name, tensor in (("weight", weight), ("mean", mean), ("rstd", rstd)):
        if tensor is None or tuple(tensor.shape) != (c,):
            raise ValueError(f"BatchNormBwdOp {name} must have shape ({c},)")
    if weight.dtype != torch.float32 or mean.dtype != torch.float32 or rstd.dtype != torch.float32:
        raise TypeError("BatchNormBwdOp weight, mean, and rstd must be float32")
    if any(t.device != grad_out.device for t in (x, weight, mean, rstd)):
        raise ValueError("BatchNormBwdOp inputs must share a device")

    def launch(grad_out_in, x_in, weight_in, mean_in, rstd_in):
        return launch_batch_backward(grad_out_in, x_in, weight_in, mean_in, rstd_in)

    return launch
