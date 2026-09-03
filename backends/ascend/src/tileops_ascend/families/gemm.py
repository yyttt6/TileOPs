"""GemmFwdOp registration for the Ascend Cube path."""

from __future__ import annotations

from .._registry import register
from ..kernels.gemm import (
    build_bmm_kernel,
    build_gemm_kernel,
    build_gemm_w4a16_kernel,
    build_grouped_gemm_kernel,
)


@register("GemmFwdOp")
def build_gemm(a, b, *, trans_a=False, trans_b=True):
    if a is None or b is None:
        raise ValueError("GemmFwdOp requires both a and b tensors")
    if a.device != b.device:
        raise ValueError(f"GemmFwdOp inputs must share a device, got {a.device} and {b.device}")
    if a.dtype != b.dtype:
        raise TypeError(f"GemmFwdOp inputs must share dtype, got {a.dtype} and {b.dtype}")
    return build_gemm_kernel(tuple(a.shape), tuple(b.shape), a.dtype, trans_a, trans_b)


@register("BmmFwdOp")
def build_bmm(a, b):
    if a is None or b is None:
        raise ValueError("BmmFwdOp requires both a and b tensors")
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError("BmmFwdOp inputs must share device and dtype")
    return build_bmm_kernel(tuple(a.shape), tuple(b.shape), a.dtype)


@register("GroupedGemmFwdOp")
def build_grouped_gemm(
    a,
    b,
    batch_sizes,
    batch_offsets,
    batch_padded_offsets,
    *,
    transpose_a=False,
    transpose_b=True,
):
    tensors = (a, b, batch_sizes, batch_offsets, batch_padded_offsets)
    if any(tensor is None for tensor in tensors):
        raise ValueError("GroupedGemmFwdOp requires all five input tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("GroupedGemmFwdOp inputs must share a device")
    if a.dtype != b.dtype:
        raise TypeError("GroupedGemmFwdOp a and b must share dtype")
    if batch_sizes.dtype != batch_offsets.dtype or batch_sizes.dtype != batch_padded_offsets.dtype:
        raise TypeError("GroupedGemmFwdOp metadata tensors must share int32 dtype")
    import torch

    if batch_sizes.dtype != torch.int32:
        raise TypeError("GroupedGemmFwdOp metadata tensors must use int32")
    if batch_sizes.ndim != 1 or batch_offsets.shape != batch_sizes.shape or batch_padded_offsets.shape != batch_sizes.shape:
        raise ValueError("GroupedGemmFwdOp metadata tensors must be matching 1D tensors")
    return build_grouped_gemm_kernel(
        tuple(a.shape),
        tuple(b.shape),
        batch_sizes.shape[0],
        a.dtype,
        transpose_a,
        transpose_b,
    )


@register("GemmW4A16FwdOp")
def build_gemm_w4a16(
    activation,
    packed_weight,
    weight_scale,
    weight_zero,
    *,
    group_size=128,
):
    tensors = (activation, packed_weight, weight_scale, weight_zero)
    if any(tensor is None for tensor in tensors):
        raise ValueError("GemmW4A16FwdOp requires all four input tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("GemmW4A16FwdOp inputs must share a device")
    return build_gemm_w4a16_kernel(
        tuple(activation.shape),
        tuple(packed_weight.shape),
        tuple(weight_scale.shape),
        tuple(weight_zero.shape),
        activation.dtype,
        packed_weight.dtype,
        weight_scale.dtype,
        weight_zero.dtype,
        group_size,
    )
