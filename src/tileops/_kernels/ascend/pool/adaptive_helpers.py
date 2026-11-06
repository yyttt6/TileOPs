"""Host-only tiling arithmetic shared by the Ascend adaptive pooling kernels."""

from math import gcd


def bin_bounds(index: int, size: int, output: int) -> tuple[int, int]:
    return index * size // output, ((index + 1) * size + output - 1) // output


def max_bin_extent(size: int, output: int) -> int:
    # Exact maximum, including enlargement and divisible shapes, without an O(output)
    # host loop. The bin residues repeat every output / gcd(size, output).
    return (size + output - gcd(size, output) - 1) // output + 1


def adaptive_tiling(h: int, w: int, oh: int, ow: int) -> tuple[int, int, int, int]:
    kh, kw = max_bin_extent(h, oh), max_bin_extent(w, ow)
    reduction = max(64, 1 << (min(kh * kw, 512) - 1).bit_length())
    return 16, reduction, kh, kw


def mean_tiling(width: int, chunk: int) -> tuple[int, int]:
    # 32 x 512 fp32 is 64 KiB: leave room for staging, state and compiler scratch.
    return min(512, max(64, 1 << (width - 1).bit_length())), min(32, chunk)
