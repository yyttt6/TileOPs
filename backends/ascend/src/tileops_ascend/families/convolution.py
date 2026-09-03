"""Convolution family registration for Ascend."""

import torch

from .._registry import register
from ..kernels.convolution import build_conv2d_kernel, build_conv_nd_kernel


def _pair(name, value):
    if isinstance(value, int) and not isinstance(value, bool):
        result = (value, value)
    elif isinstance(value, tuple) and len(value) == 2:
        result = tuple(value)
    else:
        raise TypeError(f"{name} must be an int or a length-2 tuple")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in result):
        raise TypeError(f"{name} entries must be ints")
    return int(result[0]), int(result[1])


def _padding_pair(padding, stride, kernel, dilation):
    if padding == "valid":
        return (0, 0)
    if padding == "same":
        if stride != (1, 1):
            raise ValueError("Conv2dFwdOp padding='same' requires stride == 1")
        effective = tuple(
            dilation_axis * (kernel_axis - 1) + 1
            for kernel_axis, dilation_axis in zip(kernel, dilation, strict=True)
        )
        if any(value % 2 == 0 for value in effective):
            raise ValueError(
                "Conv2dFwdOp padding='same' requires odd effective kernel sizes"
            )
        return effective[0] // 2, effective[1] // 2
    if isinstance(padding, str):
        raise ValueError("Conv2dFwdOp padding must be int, pair, 'valid', or 'same'")
    return _pair("padding", padding)


@register("Conv2dFwdOp")
def build_conv2d(
    input,
    weight,
    bias,
    *,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    """Validate metadata and build the NCHW Conv2d kernel."""
    if input is None or weight is None:
        raise ValueError("Conv2dFwdOp requires input and weight tensors")
    if len(input.shape) != 4 or len(weight.shape) != 4:
        raise ValueError("Conv2dFwdOp expects 4D NCHW input and OIHW weight")
    tensors = (input, weight) if bias is None else (input, weight, bias)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("Conv2dFwdOp tensors must share a device")
    if any(tensor.dtype != input.dtype for tensor in tensors):
        raise TypeError("Conv2dFwdOp tensors must share a dtype")
    if input.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(f"Conv2dFwdOp does not support {input.dtype}")
    if not isinstance(groups, int) or isinstance(groups, bool) or groups <= 0:
        raise ValueError("Conv2dFwdOp groups must be a positive int")

    n, c_in, h_in, w_in = (int(value) for value in input.shape)
    c_out, c_in_g, kernel_h, kernel_w = (int(value) for value in weight.shape)
    if min(n, c_in, h_in, w_in, c_out, c_in_g, kernel_h, kernel_w) <= 0:
        raise ValueError("Conv2dFwdOp requires non-empty dimensions")
    if c_in % groups or c_out % groups:
        raise ValueError("Conv2dFwdOp channels must be divisible by groups")
    if c_in_g != c_in // groups:
        raise ValueError(
            f"Conv2dFwdOp expected weight.shape[1]={c_in // groups}, got {c_in_g}"
        )
    if bias is not None and tuple(bias.shape) != (c_out,):
        raise ValueError(f"Conv2dFwdOp expects bias shape ({c_out},)")

    stride_pair = _pair("stride", stride)
    dilation_pair = _pair("dilation", dilation)
    if min(*stride_pair, *dilation_pair) <= 0:
        raise ValueError("Conv2dFwdOp stride and dilation must be positive")
    padding_pair = _padding_pair(
        padding,
        stride_pair,
        (kernel_h, kernel_w),
        dilation_pair,
    )
    if min(*padding_pair) < 0:
        raise ValueError("Conv2dFwdOp padding must be non-negative")
    out_h = (
        h_in + 2 * padding_pair[0] - dilation_pair[0] * (kernel_h - 1) - 1
    ) // stride_pair[0] + 1
    out_w = (
        w_in + 2 * padding_pair[1] - dilation_pair[1] * (kernel_w - 1) - 1
    ) // stride_pair[1] + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Conv2dFwdOp output spatial dimensions must be positive")

    return build_conv2d_kernel(
        tuple(input.shape),
        tuple(weight.shape),
        input.dtype,
        stride=stride_pair,
        padding=padding_pair,
        dilation=dilation_pair,
        groups=groups,
        has_bias=bias is not None,
    )


__all__ = ["build_conv1d", "build_conv2d", "build_conv3d"]


def _scalar_param(name, value):
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, tuple) and len(value) == 1 and isinstance(value[0], int) and not isinstance(value[0], bool):
        return int(value[0])
    raise TypeError(f"{name} must be an int or a length-1 tuple")


def _triple(name, value):
    if isinstance(value, int) and not isinstance(value, bool):
        return (value, value, value)
    if isinstance(value, tuple) and len(value) == 3 and all(isinstance(x, int) and not isinstance(x, bool) for x in value):
        return tuple(value)
    raise TypeError(f"{name} must be an int or a length-3 tuple")


def _padding_nd(padding, stride, kernel, dilation, op_name):
    ndim = len(kernel)
    if padding == "valid":
        return (0,) * ndim
    if padding == "same":
        if any(s != 1 for s in stride):
            raise ValueError(f"{op_name} padding='same' requires stride == 1")
        effective = tuple(d * (k - 1) + 1 for k, d in zip(kernel, dilation, strict=True))
        if any(k % 2 == 0 for k in effective):
            raise ValueError(f"{op_name} padding='same' requires odd effective kernels")
        return tuple(k // 2 for k in effective)
    if isinstance(padding, int) and not isinstance(padding, bool):
        return (padding,) * ndim
    if isinstance(padding, tuple) and len(padding) == ndim and all(isinstance(x, int) and not isinstance(x, bool) for x in padding):
        return tuple(padding)
    raise TypeError(f"{op_name} padding must be an int, a length-{ndim} tuple, 'valid', or 'same'")


def _build_conv_nd(op_name, input, weight, bias, *, stride, padding, dilation, groups, ndim):
    if input is None or weight is None:
        raise ValueError(f"{op_name} requires input and weight tensors")
    if len(input.shape) != ndim + 2 or len(weight.shape) != ndim + 2:
        raise ValueError(f"{op_name} expects {ndim + 2}D input and weight")
    tensors = (input, weight) if bias is None else (input, weight, bias)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError(f"{op_name} tensors must share a device")
    if any(tensor.dtype != input.dtype for tensor in tensors):
        raise TypeError(f"{op_name} tensors must share a dtype")
    if input.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(f"{op_name} does not support {input.dtype}")
    if not isinstance(groups, int) or isinstance(groups, bool) or groups <= 0:
        raise ValueError(f"{op_name} groups must be a positive int")
    n, c_in, *spatial = (int(v) for v in input.shape)
    c_out, c_in_g, *kernel = (int(v) for v in weight.shape)
    if min(n, c_in, c_out, c_in_g, *spatial, *kernel) <= 0:
        raise ValueError(f"{op_name} requires non-empty dimensions")
    if c_in % groups or c_out % groups or c_in_g != c_in // groups:
        raise ValueError(f"{op_name} channels and weight.shape[1] must match groups")
    if bias is not None and tuple(bias.shape) != (c_out,):
        raise ValueError(f"{op_name} expects bias shape ({c_out},)")
    stride = stride if ndim == 3 else (_scalar_param("stride", stride),)
    dilation = dilation if ndim == 3 else (_scalar_param("dilation", dilation),)
    stride = tuple(stride)
    dilation = tuple(dilation)
    if any(v <= 0 for v in (*stride, *dilation)):
        raise ValueError(f"{op_name} stride and dilation must be positive")
    padding = _padding_nd(padding, stride, tuple(kernel), dilation, op_name)
    if any(v < 0 for v in padding):
        raise ValueError(f"{op_name} padding must be non-negative")
    out = tuple((s + 2 * p - d * (k - 1) - 1) // st + 1 for s, k, st, p, d in zip(spatial, kernel, stride, padding, dilation, strict=True))
    if any(v <= 0 for v in out):
        raise ValueError(f"{op_name} output dimensions must be positive")
    return build_conv_nd_kernel(tuple(input.shape), tuple(weight.shape), input.dtype, stride=stride, padding=padding, dilation=dilation, groups=groups, has_bias=bias is not None)


@register("Conv1dFwdOp")
def build_conv1d(input, weight, bias, *, stride=1, padding=0, dilation=1, groups=1):
    return _build_conv_nd("Conv1dFwdOp", input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups, ndim=1)


@register("Conv3dFwdOp")
def build_conv3d(input, weight, bias, *, stride=1, padding=0, dilation=1, groups=1):
    return _build_conv_nd("Conv3dFwdOp", input, weight, bias, stride=_triple("stride", stride), padding=padding, dilation=_triple("dilation", dilation), groups=groups, ndim=3)
