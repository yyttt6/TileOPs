"""Ascend registrations for the unary math manifest family."""

from __future__ import annotations

import torch

from .._registry import register
from ..kernels.elementwise_unary import FLOAT_DTYPES, INT_DTYPES, build_unary_kernel


_FLOAT = FLOAT_DTYPES
_FLOAT_INT = FLOAT_DTYPES + (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_PRED = FLOAT_DTYPES + INT_DTYPES
_BITWISE = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _build(input, *, op_kind, supported, output_dtype=None, op_name):
    if input is None:
        raise ValueError(f"{op_name} requires an input tensor")
    if op_kind in {"floor", "ceil", "trunc", "round"} and input.dtype not in FLOAT_DTYPES:
        op_kind = "identity"
    return build_unary_kernel(tuple(input.shape), input.dtype, op_kind=op_kind,
                              supported_dtypes=supported, output_dtype=output_dtype,
                              op_name=op_name)


def _register(name, op_kind, supported, output_dtype=None):
    def decorator(fn):
        return register(name)(fn)
    @decorator
    def builder(input):
        return _build(input, op_kind=op_kind, supported=supported,
                      output_dtype=output_dtype, op_name=name)
    return builder


build_abs = _register("AbsFwdOp", "abs", _FLOAT_INT)
build_bitwise_not = _register("BitwiseNotFwdOp", "bitwise_not", _BITWISE)
build_ceil = _register("CeilFwdOp", "ceil", _FLOAT_INT)
build_cos = _register("CosFwdOp", "cos", _FLOAT)
build_erf = _register("ErfFwdOp", "erf", _FLOAT)
build_exp = _register("ExpFwdOp", "exp", _FLOAT)
build_expm1 = _register("Expm1FwdOp", "expm1", _FLOAT)
build_floor = _register("FloorFwdOp", "floor", _FLOAT_INT)
build_isfinite = _register("IsfiniteFwdOp", "isfinite", _PRED, torch.bool)
build_isinf = _register("IsinfFwdOp", "isinf", _PRED, torch.bool)
build_isnan = _register("IsnanFwdOp", "isnan", _PRED, torch.bool)
build_log1p = _register("Log1pFwdOp", "log1p", _FLOAT)
build_log = _register("LogFwdOp", "log", _FLOAT)
build_logical_not = _register("LogicalNotFwdOp", "logical_not", _PRED, torch.bool)
build_neg = _register("NegFwdOp", "neg", _FLOAT_INT)
build_reciprocal = _register("ReciprocalFwdOp", "reciprocal", _FLOAT)


@register("RoundFwdOp")
def build_round(input, *, decimals=0):
    if int(decimals) != 0:
        raise NotImplementedError("RoundFwdOp decimals != 0 is outside the Ascend unary intrinsic template")
    return _build(input, op_kind="round", supported=_FLOAT_INT, op_name="RoundFwdOp")


build_rsqrt = _register("RsqrtFwdOp", "rsqrt", _FLOAT)
build_sigmoid = _register("SigmoidFwdOp", "sigmoid", _FLOAT)
build_sign = _register("SignFwdOp", "sign", _FLOAT_INT)
build_sin = _register("SinFwdOp", "sin", _FLOAT)
build_sqrt = _register("SqrtFwdOp", "sqrt", _FLOAT)
build_tanh = _register("TanhFwdOp", "tanh", _FLOAT)
build_trunc = _register("TruncFwdOp", "trunc", _FLOAT_INT)
