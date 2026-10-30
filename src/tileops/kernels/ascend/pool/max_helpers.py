"""Pure geometry helpers shared by the six Ascend maximum-pooling variants."""

import math


def output_shape(size, window, stride, pad, dilation, ceil_mode):
    result = []
    for n, k, s, p, d in zip(size, window, stride, pad, dilation, strict=True):
        out = (n + 2 * p - d * (k - 1) - 1 + (s - 1 if ceil_mode else 0)) // s + 1
        if ceil_mode and (out - 1) * s >= n + p:
            out -= 1
        result.append(max(out, 0))
    return tuple(result)


def normalize_geometry(size, window, stride, pad, dilation, ceil_mode):
    """Embed 1D/2D in 3D; the innermost axis always remains contiguous."""
    ndim = len(size)
    if ndim not in (1, 2, 3) or any(len(x) != ndim for x in (window, stride, pad, dilation)):
        raise ValueError("MaxPool requires matching 1D, 2D or 3D spatial parameters")
    if any(x <= 0 for group in (size, window, stride, dilation) for x in group):
        raise ValueError("MaxPool sizes, windows, strides and dilations must be positive")
    if any(p < 0 or p > k // 2 for p, k in zip(pad, window, strict=True)):
        raise ValueError("MaxPool padding must be between zero and half the window")
    out = output_shape(size, window, stride, pad, dilation, ceil_mode)
    if min(out) <= 0:
        raise ValueError("MaxPool has an empty spatial output")
    if math.prod(window) >= 2**31:
        raise ValueError("MaxPool window exceeds the int32 tap-selector range")
    prefix = 3 - ndim
    return (
        (1,) * prefix + tuple(size),
        (1,) * prefix + tuple(window),
        (1,) * prefix + tuple(stride),
        (0,) * prefix + tuple(pad),
        (1,) * prefix + tuple(dilation),
        (1,) * prefix + out,
        out,
    )


def first_spatial_index(origin, size, dilation):
    """PyTorch's all-minus-infinity index, including empty dilated windows."""
    coords = [a + max(0, (-a + d - 1) // d) * d for a, d in zip(origin, dilation, strict=True)]
    result = 0
    for coord, extent in zip(coords, size, strict=True):
        result = result * extent + coord
    return result


def staged_1d_plan(rows, size, window, stride, pad, dilation, out, dtype, indices):
    """Bounded full-row DMA/SIMD path, only when every real window is in bounds."""
    width, taps, step, spacing, outputs = size[2], window[2], stride[2], dilation[2], out[2]
    if (
        size[:2] != (1, 1)
        or window[:2] != (1, 1)
        or out[:2] != (1, 1)
        or pad != (0, 0, 0)
        or not 1 < taps <= 16
        or outputs <= 1
        or width % 64
        or width > 8192
        or (outputs - 1) * step + (taps - 1) * spacing >= width
    ):
        return None
    padded_outputs = (outputs + 63) // 64 * 64
    itemsize = 4 if dtype == "float32" else 2
    bytes_per_row = width * (itemsize + (4 if itemsize == 2 else 0))
    bytes_per_row += padded_outputs * (itemsize + (4 if indices else 0))
    row_tile = 4
    while row_tile > 1 and (rows % row_tile or row_tile * bytes_per_row > 160 * 1024):
        row_tile //= 2
    if row_tile * bytes_per_row > 160 * 1024:
        return None
    return dict(
        row_tile=row_tile,
        padded_outputs=padded_outputs,
        estimated_ub_bytes=row_tile * bytes_per_row + 512,
        tiles=rows // row_tile,
    )
