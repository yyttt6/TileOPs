"""Ascend 950 implementation of the scaled Walsh-Hadamard transform.

A row is ``n = base_order * m`` elements with ``m`` a power of two, and the transform is
``x @ kron(B, H_m) / sqrt(n)``: every stage is an add/subtract butterfly of one bit inside
a segment, the base product mixes the segments, and the sum accumulates in float32 and
rounds to the input dtype once, at the end.

A row of at most ``_ROW_ELEMENTS`` elements is transformed whole in the UB by one program:
``rows_per_task`` rows are loaded, converted to float32, transformed, and stored.  A longer
row takes passes over global memory: a chunk pass transforms the low bits of every segment
into a float32 buffer, one in-place butterfly pass takes each remaining bit, and a closing
pass scales and rounds (base order 1) or mixes the segments with the base matrix (above 1).

Every program specializes on the element count, as the other Ascend kernels in this
package do, so none states a dynamic extent.  Each grid divides its task count exactly,
so no vector region sits inside a branch.
"""

import functools
import math
from typing import ClassVar, Optional, Tuple

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned
from tileops.kernels.transform.call_spec import (
    HADAMARD_BASES,
    HadamardTransformCall,
    HadamardTransformFwdInterface,
)

__all__ = ["AscendHadamardTransformKernel"]

# A row of at most this many elements is transformed whole in the UB: a float32
# accumulator plus an input and an output tile of that size take eight bytes an element,
# 96 KiB at the limit, well inside the UB.
_ROW_ELEMENTS = 12288
# A chunk pass holds one input tile and one float32 accumulator below 48 KiB.
_CHUNK_ELEMENTS = 8192
# The base pass reads every segment once; the segment tiles together take the storage of
# one chunk pass.
_BASE_ELEMENTS = 8192
_MAX_THREADS = 2048
_FALLBACK_VECTOR_CORES = 64


def _num_vector_cores(device_index: Optional[int]) -> int:
    """Number of vector cores, queried on the device this specialization serves."""
    try:
        cores = get_num_vector_cores(device_id=device_index or 0)
        if cores:
            return int(cores)
    except Exception:
        pass
    return _FALLBACK_VECTOR_CORES


def _largest_divisor_leq(n: int, cap: int) -> int:
    """The largest divisor of *n* that does not exceed *cap*; always >= 1 for n >= 1."""
    cap = max(1, min(n, cap))
    for candidate in range(cap, 0, -1):
        if n % candidate == 0:
            return candidate
    return 1


def _power_of_two_leq(n: int) -> int:
    """The largest power of two ``<= n`` (``n >= 1``)."""
    return 1 << (n.bit_length() - 1)


def _threads(count: int) -> int:
    """Lanes for one region: one an element, up to the SIMT cap."""
    return min(_MAX_THREADS, max(1, count))


def _base_signs(base_order: int) -> Tuple[Tuple[float, ...], ...]:
    """``B`` read the way the kernels need it: one tuple of +-1 factors per output column.

    Entry ``[c][a]`` is the sign row ``a`` of the input contributes to output column ``c``,
    i.e. ``HADAMARD_BASES[base_order][a][c]``.
    """
    rows = HADAMARD_BASES[base_order]
    return tuple(
        tuple(1.0 if rows[a][c] == "+" else -1.0 for a in range(base_order))
        for c in range(base_order)
    )


@T.macro
def _to_float(src, dst, count: int, threads: int):
    """``dst[:count] = float32(src[:count])``."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(count):
            dst[i] = T.cast(src[i], "float32")


@T.macro
def _scale_round(src, dst, count: int, scale: float, threads: int):
    """``dst[:count] = dtype(src[:count] * scale)``, the one rounding of a base-order-1 call."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(count):
            dst[i] = T.cast(src[i] * T.float32(scale), dst.dtype)


@T.macro
def _butterflies(acc, strides, k: int, count: int, threads: int):
    """Every stride of *strides* in turn, low bit first: a stage is one add/subtract a pair.

    Strides below a segment's length keep every pair inside one segment, so the same
    program serves a row of one segment and a row of ``base_order`` of them.
    """
    if k < len(strides):
        stride = strides[k]
        with T.SimtVF(threads=threads):
            for i in T.Parallel(count // 2):
                left = i // stride * (2 * stride) + i % stride
                a = acc[left]
                b = acc[left + stride]
                acc[left] = a + b
                acc[left + stride] = a - b
        _butterflies(acc, strides, k + 1, count, threads)


@T.macro
def _high_stage(src, dst, groups: int, regs: int, lanes: int, stride_regs: int):
    """One stage whose stride is at least one register: register *r* pairs with *r+stride*.

    *groups* counts the independent segments in the tile and *regs* the registers of one
    segment; a stage never pairs across a segment boundary.
    """
    with T.SimdVF():
        for group in T.serial(groups):
            for g in T.serial(regs // (2 * stride_regs)):
                for j in T.serial(stride_regs):
                    left = (group * regs + g * 2 * stride_regs + j) * lanes
                    right = (group * regs + g * 2 * stride_regs + j + stride_regs) * lanes
                    a = T.simd.vld(src[left])
                    b = T.simd.vld(src[right])
                    T.simd.vsts(dst[left], T.simd.vadd(a, b))
                    T.simd.vsts(dst[right], T.simd.vsub(a, b))


@T.macro
def _split_stage(src, dst, groups: int, regs: int, lanes: int):
    """One stage below one register: split adjacent lanes, add/subtract, keep them split."""
    with T.SimdVF():
        for group in T.serial(groups):
            for p in T.serial(regs // 2):
                left = (group * regs + 2 * p) * lanes
                right = (group * regs + 2 * p + 1) * lanes
                a = T.simd.vld(src[left])
                b = T.simd.vld(src[right])
                even, odd = T.simd.vdintlv(a, b)
                T.simd.vsts(dst[left], T.simd.vadd(even, odd))
                T.simd.vsts(dst[right], T.simd.vsub(even, odd))


@T.macro
def _join_stage(src, dst, groups: int, regs: int, lanes: int):
    """Undo the split order the last stage left: one ``vdintlv`` per register pair."""
    with T.SimdVF():
        for group in T.serial(groups):
            for p in T.serial(regs // 2):
                left = (group * regs + 2 * p) * lanes
                right = (group * regs + 2 * p + 1) * lanes
                a = T.simd.vld(src[left])
                b = T.simd.vld(src[right])
                even, odd = T.simd.vdintlv(a, b)
                T.simd.vsts(dst[left], even)
                T.simd.vsts(dst[right], odd)


@T.macro
def _tile_butterfly(left, right, tile: int, threads: int):
    """One butterfly stage whose partners are the same position of two *tile*-wide blocks."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(tile):
            a = left[i]
            b = right[i]
            left[i] = a + b
            right[i] = a - b


@T.macro
def _tile_butterfly_final(left, right, left_out, right_out, tile: int, scale: float, threads: int):
    """The last base-order-1 stage: butterfly, scale and round once, into the output."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(tile):
            a = left[i]
            b = right[i]
            left_out[i] = T.cast((a + b) * T.float32(scale), left_out.dtype)
            right_out[i] = T.cast((a - b) * T.float32(scale), right_out.dtype)


def _segment_terms(source, tile: int, signs, scale: float, j):
    """Output column *j* of one row part: ``sum_a signs[a] * source[a * tile + j] / sqrt(n)``."""
    first = signs[0]
    total = source[j] if first > 0 else -source[j]
    for a, sign in enumerate(signs[1:], start=1):
        value = source[a * tile + j]
        total = total + value if sign > 0 else total - value
    return total * T.float32(scale)


def _row_terms(acc, base_offset, m: int, signs, scale: float, j):
    """Output column *j* of one row: ``sum_a signs[a] * acc[base_offset + a * m + j] / sqrt(n)``."""
    first = signs[0]
    total = acc[base_offset + j] if first > 0 else -acc[base_offset + j]
    for a, sign in enumerate(signs[1:], start=1):
        value = acc[base_offset + a * m + j]
        total = total + value if sign > 0 else total - value
    return total * T.float32(scale)


@T.macro
def _row_columns(
    acc, out, base: int, m: int, n: int, positions: int, columns, scale: float, c: int, threads: int
):
    """Column *c* of the base product of every row in one tile, one region a column."""
    if c < base:
        with T.SimtVF(threads=threads):
            for i in T.Parallel(positions):
                row = i // m
                j = i % m
                out[row * n + c * m + j] = T.cast(
                    _row_terms(acc, row * n, m, columns[c], scale, j), out.dtype
                )
        _row_columns(acc, out, base, m, n, positions, columns, scale, c + 1, threads)


@T.macro
def _load_segments(source, destination, a: int, base: int, begin, m: int, tile: int):
    """Load the *tile*-wide part of every segment of one row part into the float32 tile."""
    if a < base:
        start = begin + a * m
        T.copy(source[start : start + tile], destination[a * tile : (a + 1) * tile])
        _load_segments(source, destination, a + 1, base, begin, m, tile)


@T.macro
def _store_columns(
    source, out, y, begin, base: int, tile: int, m: int, columns, scale: float, c: int, threads: int
):
    """Column *c* of the base product of one row part: mix, scale, round once and store."""
    if c < base:
        with T.SimtVF(threads=threads):
            for j in T.Parallel(tile):
                out[j] = T.cast(_segment_terms(source, tile, columns[c], scale, j), out.dtype)
        T.copy(out, y[begin + c * m : begin + c * m + tile])
        _store_columns(source, out, y, begin, base, tile, m, columns, scale, c + 1, threads)


@functools.lru_cache(maxsize=64)
def _build_row_program(n: int, base: int, dtype: str, total: int, cores: int):
    """One program transforming whole rows in the UB, the batch split over the grid."""
    rows = total // n
    m = n // base
    rows_per_task = _largest_divisor_leq(rows, max(1, _ROW_ELEMENTS // n))
    count = rows_per_task * n
    positions = rows_per_task * m
    tasks = rows // rows_per_task
    blocks = _largest_divisor_leq(tasks, cores)
    iterations = tasks // blocks
    strides = tuple(1 << bit for bit in range(m.bit_length() - 1))
    columns = _base_signs(base) if base > 1 else ()
    scale = 1.0 / math.sqrt(n)
    threads = _threads(count)
    half_threads = _threads(count // 2)
    position_threads = _threads(positions)

    @T.prim_func
    def main(X: T.Tensor((total,), dtype), Y: T.Tensor((total,), dtype)):
        with T.Kernel(blocks) as bx:
            x_ub = T.alloc_shared((count,), dtype)
            acc = T.alloc_shared((count,), "float32")
            out_ub = T.alloc_shared((count,), dtype)
            for step in T.serial(iterations):
                task = step * blocks + bx
                begin = task * count
                T.copy(X[begin : begin + count], x_ub)
                _to_float(x_ub, acc, count, threads)
                if m > 1:
                    _butterflies(acc, strides, 0, count, half_threads)
                if base == 1:
                    _scale_round(acc, out_ub, count, scale, threads)
                if base > 1:
                    _row_columns(
                        acc, out_ub, base, m, n, positions, columns, scale, 0, position_threads
                    )
                T.copy(out_ub, Y[begin : begin + count])

    return tilelang.compile(main, target="ascend", out_idx=-1)


# One full-width float32 register holds this many lanes.
_REGISTER_LANES = 64


def _register_plan(m: int, regs: int):
    """The register strides and the number of split stages a row of *regs* registers takes."""
    high = []
    stride_regs = 1
    while stride_regs < regs:
        high.append(stride_regs)
        stride_regs *= 2
    return tuple(high), m.bit_length() - 1 - (regs.bit_length() - 1)


@functools.lru_cache(maxsize=64)
def _build_register_program(n: int, base: int, dtype: str, total: int, cores: int):
    """One row transformed by full-width vector registers, the batch split over the grid."""
    rows = total // n
    m = n // base
    rows_per_task = _largest_divisor_leq(rows, max(1, _ROW_ELEMENTS // n))
    count = rows_per_task * n
    tasks = rows // rows_per_task
    blocks = _largest_divisor_leq(tasks, cores)
    iterations = tasks // blocks
    regs = m // _REGISTER_LANES
    # One butterfly segment is ``m`` elements; the tile holds ``count // m`` of them.
    groups = count // m
    high, splits = _register_plan(m, regs)
    columns = _base_signs(base) if base > 1 else ()
    scale = 1.0 / math.sqrt(n)
    threads = _threads(count)
    positions = rows_per_task * m
    position_threads = _threads(positions)

    @T.prim_func
    def main(X: T.Tensor((total,), dtype), Y: T.Tensor((total,), dtype)):
        with T.Kernel(blocks) as bx:
            x_ub = T.alloc_shared((count,), dtype)
            acc = T.alloc_shared((count,), "float32")
            alt = T.alloc_shared((count,), "float32")
            out_ub = T.alloc_shared((count,), dtype)
            for step in T.serial(iterations):
                task = step * blocks + bx
                begin = task * count
                T.copy(X[begin : begin + count], x_ub)
                _to_float(x_ub, acc, count, threads)
                passes = [(stride_regs, "high") for stride_regs in high]
                passes += [(-1, "split")] * splits
                if splits:
                    passes.append((-1, "join"))
                for index, (stride_regs, kind) in enumerate(passes):
                    even = index % 2 == 0
                    if kind == "high":
                        if even:
                            _high_stage(acc, alt, groups, regs, _REGISTER_LANES, stride_regs)
                        else:
                            _high_stage(alt, acc, groups, regs, _REGISTER_LANES, stride_regs)
                    elif kind == "split":
                        if even:
                            _split_stage(acc, alt, groups, regs, _REGISTER_LANES)
                        else:
                            _split_stage(alt, acc, groups, regs, _REGISTER_LANES)
                    else:
                        if even:
                            _join_stage(acc, alt, groups, regs, _REGISTER_LANES)
                        else:
                            _join_stage(alt, acc, groups, regs, _REGISTER_LANES)
                live = alt if len(passes) % 2 == 1 else acc
                if base > 1:
                    _row_columns(
                        live, out_ub, base, m, n, positions, columns, scale, 0, position_threads
                    )
                else:
                    _scale_round(live, out_ub, count, scale, threads)
                T.copy(out_ub, Y[begin : begin + count])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _build_chunk_program(n: int, base: int, dtype: str, total: int, cores: int):
    """The pass transforming the low bits of every segment into a float32 copy."""
    tile = min(n // base, _CHUNK_ELEMENTS)
    tasks = total // tile
    blocks = _largest_divisor_leq(tasks, cores)
    iterations = tasks // blocks
    strides = tuple(1 << bit for bit in range(tile.bit_length() - 1))
    threads = _threads(tile)
    half_threads = _threads(tile // 2)

    @T.prim_func
    def main(X: T.Tensor((total,), dtype), Y: T.Tensor((total,), "float32")):
        with T.Kernel(blocks) as bx:
            x_ub = T.alloc_shared((tile,), dtype)
            acc = T.alloc_shared((tile,), "float32")
            for step in T.serial(iterations):
                task = step * blocks + bx
                begin = task * tile
                T.copy(X[begin : begin + tile], x_ub)
                _to_float(x_ub, acc, tile, threads)
                if tile > 1:
                    _butterflies(acc, strides, 0, tile, half_threads)
                T.copy(acc, Y[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _build_inplace_program(total: int, stride: int, tile: int, cores: int):
    """One in-place butterfly pass of *stride* over the float32 rows, *tile* wide a block."""
    pairs_per_group = stride // tile
    tasks = total // (2 * tile)
    blocks = _largest_divisor_leq(tasks, cores)
    iterations = tasks // blocks
    threads = _threads(tile)

    @T.prim_func
    def main(buffer: T.Tensor((total,), "float32")):
        with T.Kernel(blocks) as bx:
            left_ub = T.alloc_shared((tile,), "float32")
            right_ub = T.alloc_shared((tile,), "float32")
            for step in T.serial(iterations):
                task = step * blocks + bx
                group = task // pairs_per_group
                slot = task % pairs_per_group
                left = group * (2 * stride) + slot * tile
                right = left + stride
                T.copy(buffer[left : left + tile], left_ub)
                T.copy(buffer[right : right + tile], right_ub)
                _tile_butterfly(left_ub, right_ub, tile, threads)
                T.copy(left_ub, buffer[left : left + tile])
                T.copy(right_ub, buffer[right : right + tile])

    return tilelang.compile(main, target="ascend", out_idx=[])


@functools.lru_cache(maxsize=64)
def _build_final_program(n: int, total: int, stride: int, tile: int, dtype: str, cores: int):
    """The last butterfly pass of base order 1: scale and round once, into the output."""
    pairs_per_group = stride // tile
    tasks = total // (2 * tile)
    blocks = _largest_divisor_leq(tasks, cores)
    iterations = tasks // blocks
    scale = 1.0 / math.sqrt(n)
    threads = _threads(tile)

    @T.prim_func
    def main(buffer: T.Tensor((total,), "float32"), Y: T.Tensor((total,), dtype)):
        with T.Kernel(blocks) as bx:
            left_ub = T.alloc_shared((tile,), "float32")
            right_ub = T.alloc_shared((tile,), "float32")
            left_out = T.alloc_shared((tile,), dtype)
            right_out = T.alloc_shared((tile,), dtype)
            for step in T.serial(iterations):
                task = step * blocks + bx
                group = task // pairs_per_group
                slot = task % pairs_per_group
                left = group * (2 * stride) + slot * tile
                right = left + stride
                T.copy(buffer[left : left + tile], left_ub)
                T.copy(buffer[right : right + tile], right_ub)
                _tile_butterfly_final(left_ub, right_ub, left_out, right_out, tile, scale, threads)
                T.copy(left_out, Y[left : left + tile])
                T.copy(right_out, Y[right : right + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _build_base_program(n: int, base: int, dtype: str, total: int, cores: int):
    """The closing pass above base order 1: mix the segments, scale and round once."""
    m = n // base
    tile = min(m, _power_of_two_leq(max(1, _BASE_ELEMENTS // base)))
    parts_per_row = m // tile
    rows = total // n
    tasks = rows * parts_per_row
    blocks = _largest_divisor_leq(tasks, cores)
    iterations = tasks // blocks
    columns = _base_signs(base)
    scale = 1.0 / math.sqrt(n)
    threads = _threads(tile)

    @T.prim_func
    def main(buffer: T.Tensor((total,), "float32"), Y: T.Tensor((total,), dtype)):
        with T.Kernel(blocks) as bx:
            src_ub = T.alloc_shared((base * tile,), "float32")
            out_ub = T.alloc_shared((tile,), dtype)
            for step in T.serial(iterations):
                task = step * blocks + bx
                row = task // parts_per_row
                part = task % parts_per_row
                begin = row * n + part * tile
                _load_segments(buffer, src_ub, 0, base, begin, m, tile)
                _store_columns(src_ub, out_ub, Y, begin, base, tile, m, columns, scale, 0, threads)

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _programs(n: int, base: int, dtype: str, total: int, cores: int):
    """The programs that transform *total* elements, run in order, return the output."""
    if n <= _ROW_ELEMENTS:
        # The register path splits lanes by pairing two registers, so a segment needs at
        # least two full registers; a shorter one goes to the elementwise path.
        if n // base >= 2 * _REGISTER_LANES:
            program = _build_register_program(n, base, dtype, total, cores)
        else:
            program = _build_row_program(n, base, dtype, total, cores)
        return lambda x: program(x)

    tile = min(n // base, _CHUNK_ELEMENTS)
    chunk = _build_chunk_program(n, base, dtype, total, cores)
    strides = tuple(1 << bit for bit in range(tile.bit_length() - 1, (n // base).bit_length() - 1))
    if base == 1:
        passes = tuple(
            _build_inplace_program(total, stride, tile, cores) for stride in strides[:-1]
        )
        closing = _build_final_program(n, total, strides[-1], tile, dtype, cores)
    else:
        passes = tuple(_build_inplace_program(total, stride, tile, cores) for stride in strides)
        closing = _build_base_program(n, base, dtype, total, cores)

    def run(x):
        buffer = chunk(x)
        for program in passes:
            program(buffer)
        return closing(buffer)

    return run


class AscendHadamardTransformKernel(Kernel, HadamardTransformFwdInterface):
    """Scaled last-axis Walsh-Hadamard transform on Ascend NPUs."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: HadamardTransformCall) -> Entry:
        index = call.device.index if call.device is not None else None
        return (call.n, call.base_order, call.dtype, index), lambda: cls(
            call.n, call.base_order, call.dtype, device_index=index
        )

    def __init__(
        self,
        n: int,
        base_order: int,
        dtype: torch.dtype,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        m = n // base_order
        if n < base_order or n % base_order or m & (m - 1):
            raise ValueError(
                f"{type(self).__name__} needs n = base_order * a power of two, got "
                f"n={n} at base order {base_order}"
            )
        self.n = n
        self.base_order = base_order
        self.dtype = dtype
        self._cores = _num_vector_cores(device_index)
        # This entry is a pipeline of programs specialized per element count, not one
        # tunable TileLang program.
        self.kernel = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform every row of ``x`` without modifying the input."""
        self._require_devices(x=x)
        if x.numel() == 0:
            return torch.empty_like(x)
        x = vector_aligned(x.contiguous())
        total = x.numel()
        run = _programs(self.n, self.base_order, self.dtype_str, total, self._cores)
        return run(x.view(-1)).view(x.shape)


register_kernel_type("HadamardTransformFwdOp", "ascend", AscendHadamardTransformKernel)
