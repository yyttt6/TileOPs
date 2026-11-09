"""Constants, reductions and device facts shared by the Ascend NSA kernels."""

import functools
from typing import Optional

import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_cube_cores

from tileops._kernels.constants import VECTOR_ACCESS_BYTES

__all__ = [
    "FLOAT32_MAX",
    "VECTOR_LANES",
    "calibrated_cube_blocks",
    "check_inputs",
    "normalize_cast_rows",
    "reduce_max_rows",
    "reduce_sum_rows",
]

# The greatest finite float32, the identity of a masked maximum. Every copy of an
# underflowing or all-masked row reduces against this same value.
FLOAT32_MAX = -3.4028234663852886e38

# Floats one SIMD load covers; a reduced row is a whole number of these.
VECTOR_LANES = 64


@functools.lru_cache(maxsize=1)
def calibrated_cube_blocks(fallback: int = 36) -> int:
    """This SKU's live Cube core count, or *fallback* when the driver cannot report it."""
    try:
        queried = get_num_cube_cores(device_id=0)
    except Exception:
        queried = None
    return int(queried) if queried else fallback


def check_inputs(kernel, expected) -> None:
    """Raise unless every ``(name, tensor, shape, dtype)`` entry matches one specialization.

    Every tensor must share the first input's device, whose type must be among
    *kernel*'s ``devices``, as the op layer requires before running a built kernel.
    """
    kernel._require_devices(**{name: tensor for name, tensor, _shape, _dtype in expected})
    device = expected[0][1].device
    for name, tensor, shape, dtype in expected:
        if tensor.device != device:
            raise ValueError(
                f"{type(kernel).__name__}: {name} is on {tensor.device}; "
                f"every input must be on {device}"
            )
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
            raise ValueError(
                f"{type(kernel).__name__}: {name} is {tuple(tensor.shape)} {tensor.dtype}; "
                f"this specialization takes {tuple(shape)} {dtype}"
            )


def reduce_max_rows(rows: int, width: int):
    """A macro reducing each of *rows* FP32 rows of *width* floats to one scalar.

    *width* is 64, 128 or 256; every load address is a build-time constant. The result
    is stored at the row's own four-byte scalar address with an explicit one-element
    extent.
    """
    if width not in (VECTOR_LANES, 2 * VECTOR_LANES, 4 * VECTOR_LANES):
        raise ValueError(f"a reduced row must contain 64, 128 or 256 floats; got {width}")

    @T.macro
    def macro(src, dst):
        with T.SimdVF():
            mask = T.simd.pset(32, "PAT_ALL")
            one = T.simd.pset(32, "PAT_VL1")
            for row in T.serial(rows):
                acc = T.simd.vdup(T.float32(FLOAT32_MAX), "float32", mask)
                acc = T.simd.vmax(acc, T.simd.vcmax(T.simd.vld(src[row, 0]), mask), mask)
                if width >= 2 * VECTOR_LANES:
                    acc = T.simd.vmax(
                        acc,
                        T.simd.vcmax(T.simd.vld(src[row, VECTOR_LANES]), mask),
                        mask,
                    )
                if width >= 4 * VECTOR_LANES:
                    acc = T.simd.vmax(
                        acc,
                        T.simd.vcmax(T.simd.vld(src[row, 2 * VECTOR_LANES]), mask),
                        mask,
                    )
                    acc = T.simd.vmax(
                        acc,
                        T.simd.vcmax(T.simd.vld(src[row, 3 * VECTOR_LANES]), mask),
                        mask,
                    )
                T.simd.vsts(dst[row], acc, one, dist="ONEPT_B32", extent=1)

    return macro


def reduce_sum_rows(rows: int, width: int):
    """Sum each FP32 row of 64, 128 or 256 floats into one scalar."""
    if width not in (VECTOR_LANES, 2 * VECTOR_LANES, 4 * VECTOR_LANES):
        raise ValueError(f"a reduced row must contain 64, 128 or 256 floats; got {width}")

    @T.macro
    def macro(src, dst):
        with T.SimdVF():
            mask = T.simd.pset(32, "PAT_ALL")
            one = T.simd.pset(32, "PAT_VL1")
            for row in T.serial(rows):
                acc = T.simd.vdup(T.float32(0.0), "float32", mask)
                acc = T.simd.vadd(acc, T.simd.vcadd(T.simd.vld(src[row, 0]), mask), mask)
                if width >= 2 * VECTOR_LANES:
                    acc = T.simd.vadd(
                        acc,
                        T.simd.vcadd(T.simd.vld(src[row, VECTOR_LANES]), mask),
                        mask,
                    )
                if width >= 4 * VECTOR_LANES:
                    acc = T.simd.vadd(
                        acc,
                        T.simd.vcadd(T.simd.vld(src[row, 2 * VECTOR_LANES]), mask),
                        mask,
                    )
                    acc = T.simd.vadd(
                        acc,
                        T.simd.vcadd(T.simd.vld(src[row, 3 * VECTOR_LANES]), mask),
                        mask,
                    )
                T.simd.vsts(dst[row], acc, one, dist="ONEPT_B32", extent=1)

    return macro


def normalize_cast_rows(rows: int, width: int, dtype: str):
    """Normalize dense FP32 rows and pack each result into its storage dtype.

    Each denominator is a single FP32 value. Broadcast loads read that value
    alone, and packed stores write exactly 64 half-sized elements per chunk.
    Nonpositive denominators produce zeros without evaluating an invalid divide.
    """
    if rows <= 0 or width not in (VECTOR_LANES, 2 * VECTOR_LANES):
        raise ValueError("normalization requires positive rows and a width of 64 or 128")
    if dtype not in ("float16", "bfloat16"):
        raise ValueError("normalization output must be float16 or bfloat16")

    @T.macro
    def macro(src, denominator, dst):
        with T.SimdVF():
            mask = T.simd.pset(32, "PAT_ALL")
            zero = T.simd.vdup(T.float32(0.0), "float32", mask)
            one = T.simd.vdup(T.float32(1.0), "float32", mask)
            for row in T.serial(rows):
                divisor = T.simd.vld(denominator[row], dist="BRC_B32")
                positive = T.simd.vcmps(divisor, T.float32(0.0), mask, op="gt")
                safe_divisor = T.simd.vsel(divisor, one, positive)
                for chunk in T.serial(width // VECTOR_LANES):
                    values = T.simd.vld(src[row, chunk * VECTOR_LANES])
                    normalized = T.simd.vdiv(values, safe_divisor, mask, precision="exact")
                    normalized = T.simd.vsel(normalized, zero, positive)
                    converted = T.simd.vcvt(normalized, dtype, mask, round="ROUND_R", sat=False)
                    T.simd.vsts(
                        dst[row, chunk * VECTOR_LANES],
                        converted,
                        mask,
                        dist="PK_B32",
                        extent=VECTOR_LANES,
                    )

    return macro


def vector_aligned(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Copy a read-only input only when its storage misses a vector boundary."""
    return t.clone() if t is not None and t.data_ptr() % VECTOR_ACCESS_BYTES else t
