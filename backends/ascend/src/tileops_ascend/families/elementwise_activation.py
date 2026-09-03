"""T033 activation, fused-gated, and multi-input registrations."""

from __future__ import annotations

import torch

from .._registry import register
from ..kernels.elementwise_activation import build_activation_kernel


def _check_tensor(name, value):
    if value is None:
        raise ValueError(f"{name} requires a tensor input")
    return value


def _unary(name, kind, params=()):
    @register(name)
    def builder(input, **kwargs):
        input = _check_tensor(name, input)
        values = tuple(kwargs.get(key, default) for key, default in params)
        return build_activation_kernel(
            (tuple(input.shape),),
            input.dtype,
            op_kind=kind,
            params=values,
            op_name=name,
        )

    return builder


build_relu = _unary("ReluFwdOp", "relu")
build_silu = _unary("SiluFwdOp", "silu")
build_hardswish = _unary("HardswishFwdOp", "hardswish")
build_hardsigmoid = _unary("HardsigmoidFwdOp", "hardsigmoid")
build_mish = _unary("MishFwdOp", "mish")
build_selu = _unary("SeluFwdOp", "selu")
build_leaky_relu = _unary("LeakyReluFwdOp", "leaky_relu", (("negative_slope", 0.01),))
build_elu = _unary("EluFwdOp", "elu", (("alpha", 1.0),))
build_hardtanh = _unary(
    "HardtanhFwdOp", "hardtanh", (("min_val", -1.0), ("max_val", 1.0))
)
build_softplus = _unary(
    "SoftplusFwdOp", "softplus", (("beta", 1.0), ("threshold", 20.0))
)


@register("GeluFwdOp")
def build_gelu(input, *, approximate="none"):
    input = _check_tensor("GeluFwdOp", input)
    if approximate not in {"none", "tanh"}:
        raise ValueError("GeluFwdOp approximate must be 'none' or 'tanh'")
    return build_activation_kernel(
        (tuple(input.shape),),
        input.dtype,
        op_kind=f"gelu_{approximate}",
        op_name="GeluFwdOp",
    )


@register("ClampFwdOp")
def build_clamp(input, min=None, max=None):
    input = _check_tensor("ClampFwdOp", input)
    if min is None and max is None:
        raise ValueError("ClampFwdOp requires at least one of min or max")
    tensors = [input]
    kind = "clamp"
    if min is not None:
        if min.dtype != input.dtype:
            raise TypeError(
                f"ClampFwdOp min dtype must match input; received {min.dtype}"
            )
        tensors.append(min)
    elif max is not None:
        kind = "clamp_max"
        tensors.append(max)
    if max is not None and min is not None:
        if max.dtype != input.dtype:
            raise TypeError(
                f"ClampFwdOp max dtype must match input; received {max.dtype}"
            )
        tensors.append(max)
    elif max is not None and min is None:
        kind = "clamp_max"
    elif min is not None:
        kind = "clamp_min"
    return build_activation_kernel(
        tuple(tuple(t.shape) for t in tensors),
        input.dtype,
        op_kind=kind,
        op_name="ClampFwdOp",
    )


@register("ClampScalarFwdOp")
def build_clamp_scalar(input, *, min=None, max=None):
    input = _check_tensor("ClampScalarFwdOp", input)
    if min is None and max is None:
        raise ValueError("ClampScalarFwdOp requires at least one of min or max")
    lo = -float("inf") if min is None else float(min)
    hi = float("inf") if max is None else float(max)
    return build_activation_kernel(
        (tuple(input.shape),),
        input.dtype,
        op_kind="clamp_scalar",
        params=(lo, hi),
        op_name="ClampScalarFwdOp",
    )


@register("NanToNumFwdOp")
def build_nan_to_num(input, *, nan=0.0, posinf=None, neginf=None):
    input = _check_tensor("NanToNumFwdOp", input)
    finfo = torch.finfo(input.dtype)
    pos = float(finfo.max) if posinf is None else float(posinf)
    neg = -float(finfo.max) if neginf is None else float(neginf)
    return build_activation_kernel(
        (tuple(input.shape),),
        input.dtype,
        op_kind="nan_to_num",
        params=(float(nan), pos, neg),
        op_name="NanToNumFwdOp",
    )


def _build_gated(name, kind):
    @register(name)
    def builder(x):
        x = _check_tensor(name, x)
        shape = tuple(x.shape)
        if len(shape) != 2 or shape[1] % 2:
            raise ValueError(f"{name} requires a 2-D [M, 2*N] input; received {shape}")
        return build_activation_kernel(
            (shape,),
            x.dtype,
            op_kind=kind,
            output_shape=(shape[0], shape[1] // 2),
            op_name=name,
        )

    return builder


build_gelu_and_mul = _build_gated("GeluAndMulFwdOp", "gelu_and_mul")
build_gelu_tanh_and_mul = _build_gated("GeluTanhAndMulFwdOp", "gelu_tanh_and_mul")
build_silu_and_mul = _build_gated("SiluAndMulFwdOp", "silu_and_mul")


@register("LerpTensorFwdOp")
def build_lerp_tensor(input, end, weight):
    input = _check_tensor("LerpTensorFwdOp", input)
    end = _check_tensor("LerpTensorFwdOp", end)
    weight = _check_tensor("LerpTensorFwdOp", weight)
    if input.dtype != end.dtype or input.dtype != weight.dtype:
        raise TypeError("LerpTensorFwdOp requires all tensor dtypes to match")
    return build_activation_kernel(
        (tuple(input.shape), tuple(end.shape), tuple(weight.shape)),
        input.dtype,
        op_kind="lerp",
        op_name="LerpTensorFwdOp",
    )


__all__ = [name for name in globals() if name.startswith("build_")]
