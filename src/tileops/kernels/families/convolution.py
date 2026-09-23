"""Convolution family registration for Ascend."""

import torch

from .._registry import register
from ..convolution import build_conv2d_kernel, build_conv_nd_kernel


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


# ======================================================================================
# R355 / T355 #142 -- Conv2dFp8FwdOp.  ``conv2d_fp8`` is the only row of the 150-row
# scope with no operator at all; this gives it one.
#
# 910B1 has no FP8 unit, so an e4m3fn byte is a container: it is decoded to fp16 on the
# vector unit and the convolution runs on the existing fp16 path.  Three launches per
# call -- decode(input), decode(weight), conv2d -- and ``launches_per_call`` says so
# rather than hiding it, exactly as the soft-FP8 GEMM family does.
#
# 🚨 THE DECODE IS DUPLICATED ON PURPOSE.  ``kernels/families/gemm.py`` has an
# equivalent one.  F067 measured that sharing a file across parallel rounds got all
# three of them discarded, and T355 section 5 makes the duplication an instruction, so
# this is a fresh copy and not an import.  The bit identity it rests on is the same and
# is re-derived here: fp16's exponent bias is 15 and e4m3fn's is 7, and both formats
# lay the sign at the top, so ``(u & 0x80) << 8 | (u & 0x7F) << 7`` read as fp16 IS the
# e4m3fn value scaled by 2^-8 -- for normals AND subnormals, because e4m3's subnormals
# land exactly on fp16's.  The caller supplies the 2^8.
#
# Three framework constraints are baked into the shape of the macro, each measured:
#   * there is no uint8 -> int16 cast on dav-2201, so the byte goes through half;
#   * a float -> int cast issued with CAST_NONE silently yields zeros, so CAST_RINT;
#   * ``T.tile.bitwise_and`` has no scalar form, hence the filled ``m80`` tile -- and it
#     is INT16 because R355's probe measured the int32 form covering only half its tile
#     (R355-data/probe/p2-intrinsics.json, wrong=256/512 starting at element 256).
# ======================================================================================

import math                                                      # noqa: E402
from functools import lru_cache                                  # noqa: E402

import tilelang                                                  # noqa: E402
import tilelang.language as T                                    # noqa: E402

from ..common import (                                           # noqa: E402
    LAUNCH_BLOCK_CAP,
    grid_repeat_count,
    launch_block_count,
)

#: Two AIV contexts per launch block, as every vector template in this tree uses.
_FP8_VEC = 2
#: 8192 halves = 16 KiB per live buffer; four of them plus the byte tile fit the
#: 180224-byte VECCALC watermark with room to spare.
_FP8_DECODE_TILE = 8192


@T.macro
def _fp8_decode(u8s, hs, i16s, s16s, m80, f16s, count):
    """e4m3fn byte -> fp16 value * 2^-8, in place in ``i16s`` (aliased by ``f16s``).

    MUST be a ``T.macro``.  A plain Python helper is evaluated by the TVMScript parser
    but its expression statements are never emitted into the enclosing frame, and the
    failure is SILENT -- T351 measured a decode that generated Fill / copy / Muls and
    not one of the eight decode instructions, returning garbage instead of failing to
    compile.
    """
    T.tile.cast(hs, u8s, "CAST_NONE", count)
    T.tile.cast(i16s, hs, "CAST_RINT", count)
    T.tile.bitwise_and(s16s, i16s, m80)
    T.tile.sub(i16s, i16s, s16s)              # i16s &= 0x7F  (s16s is 0 or 128)
    T.tile.bitwise_lshift(s16s, s16s, 8)      # 0x80 << 8 == the fp16 sign bit
    T.tile.bitwise_lshift(i16s, i16s, 7)
    T.tile.bitwise_or(i16s, i16s, s16s)
    T.reinterpretcast(f16s, i16s, "half")


@lru_cache(maxsize=64)
def _compile_fp8_decode(numel: int, tile: int = _FP8_DECODE_TILE):
    """Flat e4m3fn bytes -> fp16 values.  One elementwise pass, no scaling beyond 2^8."""
    tile = min(tile, numel)
    while numel % (_FP8_VEC * tile) and tile > 256:
        tile //= 2
    block_total = tile * _FP8_VEC
    logical_blocks = max(1, math.ceil(numel / block_total))
    launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
    grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)

    @tilelang.jit(out_idx=[-1], pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    })
    def factory():
        @T.prim_func
        def main(U: T.Tensor((numel,), "uint8"), OUT: T.Tensor((numel,), "float16")):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                u8s = T.alloc_ub((tile,), "uint8")
                hs = T.alloc_ub((tile,), "float16")
                i16s = T.alloc_ub((tile,), "int16")
                s16s = T.alloc_ub((tile,), "int16")
                m80 = T.alloc_ub((tile,), "int16")
                f16s = T.alloc_ub((tile,), "float16")
                with T.Scope("V"):
                    T.tile.fill(m80, 128)
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            start = logical_cid * block_total + vid * tile
                            if start + tile <= numel:
                                T.copy(U[start], u8s)
                                T.barrier_all()
                                _fp8_decode(u8s, hs, i16s, s16s, m80, f16s, tile)
                                T.tile.mul(f16s, f16s, 256.0)
                                T.barrier_all()
                                T.copy(f16s, OUT[start])
                            else:
                                # No ``T.tile.fill`` here: AscendC's Duplicate has no
                                # uint8 overload ("current api support dtype
                                # combination is ... half / bfloat16_t / int16_t /
                                # uint16_t / int32_t / uint32_t / float"), so the pad
                                # is written lane by lane -- the same shape
                                # elementwise_unary.py uses for its uint8 tail.
                                for lane in T.serial(tile):
                                    if start + lane < numel:
                                        u8s[lane] = U[start + lane]
                                    else:
                                        u8s[lane] = T.uint8(0)
                                T.barrier_all()
                                _fp8_decode(u8s, hs, i16s, s16s, m80, f16s, tile)
                                T.tile.mul(f16s, f16s, 256.0)
                                T.barrier_all()
                                for lane in T.serial(tile):
                                    if start + lane < numel:
                                        OUT[start + lane] = f16s[lane]
        return main
    return factory()


@register("Conv2dFp8FwdOp")
def build_conv2d_fp8(input, weight, *, stride=1, padding=0, dilation=1, groups=1):
    """float8_e4m3fn NCHW convolution: decode both operands to fp16, then the fp16 path.

    Reuses ``build_conv2d_kernel`` unchanged, so the convolution this measures IS the
    convolution ``Conv2dFwdOp`` measures -- which is the point: the number it produces
    is comparable with the rest of the conv cluster instead of being a second, private
    implementation.
    """
    import torch as _torch

    if input is None or weight is None:
        raise ValueError("Conv2dFp8FwdOp requires input and weight tensors")
    if len(input.shape) != 4 or len(weight.shape) != 4:
        raise ValueError("Conv2dFp8FwdOp expects 4D NCHW input and OIHW weight")
    if input.dtype != _torch.float8_e4m3fn or weight.dtype != _torch.float8_e4m3fn:
        raise TypeError(
            f"Conv2dFp8FwdOp is float8_e4m3fn only, got {input.dtype} and {weight.dtype}"
        )
    if not isinstance(groups, int) or isinstance(groups, bool) or groups <= 0:
        raise ValueError("Conv2dFp8FwdOp groups must be a positive int")

    in_shape = tuple(int(v) for v in input.shape)
    w_shape = tuple(int(v) for v in weight.shape)
    n, c_in, h_in, w_in = in_shape
    c_out, c_in_g, kernel_h, kernel_w = w_shape
    if min(in_shape + w_shape) <= 0:
        raise ValueError("Conv2dFp8FwdOp requires non-empty dimensions")
    if c_in % groups or c_out % groups:
        raise ValueError("Conv2dFp8FwdOp channels must be divisible by groups")
    if c_in_g != c_in // groups:
        raise ValueError(
            f"Conv2dFp8FwdOp expected weight.shape[1]={c_in // groups}, got {c_in_g}"
        )
    stride_pair = _pair("stride", stride)
    dilation_pair = _pair("dilation", dilation)
    padding_pair = _padding_pair(padding, stride_pair, (kernel_h, kernel_w), dilation_pair)
    if min(*stride_pair, *dilation_pair) <= 0 or min(*padding_pair) < 0:
        raise ValueError("Conv2dFp8FwdOp stride/dilation must be positive and padding non-negative")

    decode_in = _compile_fp8_decode(math.prod(in_shape))
    decode_w = _compile_fp8_decode(math.prod(w_shape))
    conv = build_conv2d_kernel(
        in_shape, w_shape, _torch.float16,
        stride=stride_pair, padding=padding_pair, dilation=dilation_pair,
        groups=groups, has_bias=False,
    )

    def invoke(x, w):
        # ``.view`` only relabels the bytes.  Nothing here calls ``.to()`` on a device
        # fp8 tensor -- aclnnInplaceCopy has no e4m3 path (561103).
        x16 = decode_in(x.reshape(-1).view(_torch.uint8)).reshape(in_shape)
        w16 = decode_w(w.reshape(-1).view(_torch.uint8)).reshape(w_shape)
        return conv(x16, w16)

    invoke.launches_per_call = 2 + 1
    invoke.conv_path_kind = getattr(conv, "path_kind", None)
    return invoke
