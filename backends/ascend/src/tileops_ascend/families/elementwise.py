"""Elementwise family (archetype 1: AIV 1:1, UB tiling + pipelined copy).

Owned by the elementwise work stream. Add one ``build_*`` per op, each decorated
with ``@register("<ManifestOpName>")`` and written to that op's manifest signature.
"""

from __future__ import annotations

import torch

from .._registry import register
from ..kernels.elementwise_binary import SUPPORTED_DTYPES, build_binary_kernel
from ..kernels.elementwise_binary_batch import FLOAT_DTYPES, build_batch_binary
from ..kernels.elementwise_predicate import build_predicate_binary, build_where_kernel
from ..kernels.generative import build_alibi_kernel, build_sinusoidal_kernel
from ..kernels.elementwise_mixed import (
    _FLOAT_DTYPES as MIXED_FLOAT_DTYPES,
    _MASKED_DTYPES,
    _INTEGER_DTYPES,
    build_bitwise_kernel,
    build_masked_fill_kernel,
    build_prelu_kernel,
)

__all__: list[str] = []


@register("AddFwdOp")
def build_add(input, other, *, alpha=1):
    if input is None or other is None:
        raise ValueError("AddFwdOp requires both input and other tensors")
    if input.device != other.device:
        raise ValueError(
            f"AddFwdOp requires both inputs on one device; received {input.device} and {other.device}"
        )
    if input.dtype != other.dtype:
        raise TypeError(
            f"AddFwdOp requires input and other to have the same dtype; received "
            f"input={input.dtype}, other={other.dtype}"
        )
    return build_binary_kernel(
        tuple(input.shape),
        tuple(other.shape),
        input.dtype,
        alpha,
        op_kind="add",
        supported_dtypes=SUPPORTED_DTYPES,
        op_name="AddFwdOp",
    )


def _batch_builder(op_name, op_kind, supported_dtypes=FLOAT_DTYPES, output_dtype=None):
    def builder(input, other, **kwargs):
        if input is None or other is None:
            raise ValueError(f"{op_name} requires both input operands")
        if input.device != other.device:
            raise ValueError(
                f"{op_name} requires both inputs on one device; received {input.device} and {other.device}"
            )
        if input.dtype != other.dtype:
            raise TypeError(
                f"{op_name} requires matching dtypes; received input={input.dtype}, other={other.dtype}"
            )
        return build_batch_binary(
            tuple(input.shape),
            tuple(other.shape),
            input.dtype,
            op_kind=op_kind,
            supported_dtypes=supported_dtypes,
            output_dtype=output_dtype,
            op_name=op_name,
            **kwargs,
        )

    return builder


@register("SubFwdOp")
def build_sub(input, other, *, alpha=1):
    if input is None or other is None:
        raise ValueError("SubFwdOp requires both input and other tensors")
    if input.device != other.device or input.dtype != other.dtype:
        raise TypeError(
            f"SubFwdOp requires matching device and dtype; received {input.device}/{input.dtype} and {other.device}/{other.dtype}"
        )
    return build_binary_kernel(
        tuple(input.shape),
        tuple(other.shape),
        input.dtype,
        alpha,
        op_kind="sub",
        supported_dtypes=SUPPORTED_DTYPES,
        op_name="SubFwdOp",
    )


@register("MulFwdOp")
def build_mul(input, other):
    return _batch_builder("MulFwdOp", "mul")(input, other)


@register("DivFwdOp")
def build_div(input, other, *, rounding_mode=None):
    if rounding_mode not in (None, "trunc", "floor"):
        raise ValueError(
            f"DivFwdOp rounding_mode must be None, 'trunc', or 'floor'; received {rounding_mode!r}"
        )
    op_kind = {
        None: "div",
        "trunc": "trunc_div",
        "floor": "floor_divide",
    }[rounding_mode]
    return _batch_builder("DivFwdOp", op_kind)(input, other)


@register("RemainderFwdOp")
def build_remainder(input, other):
    """PyTorch remainder (floor-mod; result follows the divisor sign)."""

    return _batch_builder("RemainderFwdOp", "remainder")(input, other)


@register("FloorDivideFwdOp")
def build_floor_divide(input, other):
    return _batch_builder("FloorDivideFwdOp", "floor_divide")(input, other)


@register("PowFwdOp")
def build_pow(input, exponent):
    return _batch_builder("PowFwdOp", "pow")(input, exponent)


@register("LerpFwdOp")
def build_lerp(input, end, *, weight=0.5):
    if not isinstance(weight, (int, float)) or isinstance(weight, bool):
        raise TypeError(f"LerpFwdOp weight must be a float, received {weight!r}")
    return _batch_builder("LerpFwdOp", "lerp")(input, end, scalar=float(weight))


@register("MaximumFwdOp")
def build_maximum(input, other):
    return _batch_builder("MaximumFwdOp", "maximum")(input, other)


@register("MinimumFwdOp")
def build_minimum(input, other):
    return _batch_builder("MinimumFwdOp", "minimum")(input, other)


def _predicate_builder(op_name, op_kind):
    def builder(input, other):
        if input.device != other.device or input.dtype != other.dtype:
            raise TypeError(f"{op_name} requires matching device and dtype")
        return build_predicate_binary(
            tuple(input.shape),
            tuple(other.shape),
            input.dtype,
            op_kind=op_kind,
            op_name=op_name,
        )

    return builder


for _name, _kind in (
    ("EqFwdOp", "EQ"),
    ("NeFwdOp", "NE"),
    ("GtFwdOp", "GT"),
    ("GeFwdOp", "GE"),
    ("LtFwdOp", "LT"),
    ("LeFwdOp", "LE"),
    ("LogicalAndFwdOp", "logical_and"),
    ("LogicalOrFwdOp", "logical_or"),
):
    register(_name)(_predicate_builder(_name, _kind))


@register("WhereFwdOp")
def build_where(condition, input, other):
    _check_device("WhereFwdOp", condition, input, other)
    if condition.dtype != torch.bool:
        raise TypeError(f"WhereFwdOp condition must be bool; received {condition.dtype}")
    if input.dtype != other.dtype:
        raise TypeError(
            f"WhereFwdOp input and other must match; received {input.dtype}/{other.dtype}"
        )
    return build_where_kernel(
        tuple(condition.shape), tuple(input.shape), tuple(other.shape), input.dtype
    )


@register("AlibiFwdOp")
def build_alibi(probe, *, seq_len, num_heads, dtype):
    if tuple(probe.shape) != (int(num_heads), int(seq_len), int(seq_len)):
        raise ValueError("AlibiFwdOp probe shape does not match parameters")
    return build_alibi_kernel(tuple(probe.shape), dtype)


@register("SinusoidalFwdOp")
def build_sinusoidal(probe, *, seq_len, d_model, dtype):
    if int(d_model) % 2:
        raise ValueError("SinusoidalFwdOp requires an even d_model")
    if tuple(probe.shape) != (int(seq_len), int(d_model)):
        raise ValueError("SinusoidalFwdOp probe shape does not match parameters")
    return build_sinusoidal_kernel(tuple(probe.shape), dtype)


def _check_device(op_name, *tensors):
    if any(tensor is None for tensor in tensors):
        raise ValueError(f"{op_name} requires all tensor inputs")
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors[1:]):
        raise ValueError(f"{op_name} requires all inputs on one device")


@register("MaskedFillFwdOp")
def build_masked_fill(input, mask, value):
    _check_device("MaskedFillFwdOp", input, mask, value)
    if input.dtype not in _MASKED_DTYPES:
        raise TypeError(f"MaskedFillFwdOp does not support dtype {input.dtype}")
    if mask.dtype != torch.bool:
        raise TypeError(f"MaskedFillFwdOp mask must be bool, received {mask.dtype}")
    if value.dtype != input.dtype:
        raise TypeError(f"MaskedFillFwdOp value dtype must match input; received {value.dtype}")
    return build_masked_fill_kernel(tuple(input.shape), tuple(mask.shape), input.dtype, value_tensor=value)


@register("MaskedFillScalarFwdOp")
def build_masked_fill_scalar(input, mask, *, value=0):
    _check_device("MaskedFillScalarFwdOp", input, mask)
    if input.dtype not in _MASKED_DTYPES:
        raise TypeError(f"MaskedFillScalarFwdOp does not support dtype {input.dtype}")
    if mask.dtype != torch.bool:
        raise TypeError(f"MaskedFillScalarFwdOp mask must be bool, received {mask.dtype}")
    if not isinstance(value, (bool, int, float)):
        raise TypeError(f"MaskedFillScalarFwdOp value must be a scalar, received {type(value).__name__}")
    return build_masked_fill_kernel(tuple(input.shape), tuple(mask.shape), input.dtype, value=value)


@register("PreluFwdOp")
def build_prelu(input, weight):
    _check_device("PreluFwdOp", input, weight)
    if input.dtype not in MIXED_FLOAT_DTYPES or weight.dtype not in MIXED_FLOAT_DTYPES:
        raise TypeError(f"PreluFwdOp supports float16, bfloat16, and float32; received {input.dtype}/{weight.dtype}")
    return build_prelu_kernel(tuple(input.shape), tuple(weight.shape), input.dtype, weight_dtype=weight.dtype)


def _bitwise_builder(op_name, op_kind):
    def builder(input, other):
        _check_device(op_name, input, other)
        if input.dtype != other.dtype:
            raise TypeError(f"{op_name} requires matching dtypes; received input={input.dtype}, other={other.dtype}")
        if input.dtype not in _INTEGER_DTYPES:
            raise TypeError(f"{op_name} supports bool and integer dtypes; received {input.dtype}")
        return build_bitwise_kernel(tuple(input.shape), tuple(other.shape), input.dtype, op_kind=op_kind)

    return builder


for _name, _kind in (
    ("BitwiseAndFwdOp", "and"),
    ("BitwiseOrFwdOp", "or"),
    ("BitwiseXorFwdOp", "xor"),
):
    register(_name)(_bitwise_builder(_name, _kind))
