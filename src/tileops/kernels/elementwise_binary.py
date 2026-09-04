"""Reusable Ascend vector template for broadcast binary elementwise ops."""

import math
import re
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count


SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# The lowering reserves 196352 bytes of VECCALC for every kernel in this
# template family (``pipe.InitBuffer(ascend_ub, 196352)`` in the generated
# artifact).  The already-shipped ``same`` geometry -- tile 8192, bfloat16
# with a non-unit alpha, i.e. 22 bytes of live UB per element -- occupies
# 180224 of them, so that watermark is the budget every other path reuses.
UB_BUDGET_BYTES = 180224

ELEMENT_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "int32": 4}


def tile_cap(bytes_per_element: int, hard_cap: int = 32768) -> int:
    """Largest 256-element-aligned tile whose live UB buffers fit the budget."""
    raw = UB_BUDGET_BYTES // max(1, int(bytes_per_element))
    return max(256, min(int(hard_cap), raw - raw % 256))


# --- Geometry (R199; aligns this template with the R194 unary refinements) ---
# R183 wired the grid-stride loop into this template but left
# ``launch_blocks == logical_blocks``, so ``grid_repeats`` was always 1 and the
# loop never strided.  Per-block dispatch is ~0.11 us and used to be the entire
# cost: R194's probe held the tile fixed and moved only the launch width,
# 32768 blocks -> 192 blocks was 13.1x (15.7 -> 205.8 GB/s).  48 is the low end
# of the measured 48-192 plateau (PROJECT_STATE 13.57).
LAUNCH_BLOCK_CAP = 48

# Tile widths stay a multiple of this.  256 is what ``tile_cap`` already rounds
# to, and it keeps every ``T.tile.compare`` operand a whole number of 256-byte
# blocks for the templates that import from here (PROJECT_STATE 13.55 trap 3).
TILE_GRAIN = 256


def fit_same_tile(cap: int, n_total: int, hard_cap: int = 8192,
                  grain: int = TILE_GRAIN) -> int:
    """Tile width for a contiguous path, applying all three R194 clamps.

    ``cap`` is what the live UB buffers leave; the caller computes it.  On top
    of that:

    * ``ceil(n_total / 2)`` -- never widen past what the tensor needs.  A tile
      wider than half the tensor cannot fill both vector contexts of a block,
      so the second context falls into the per-lane scalar tail over a full
      tile width.  Without this clamp ``n_total = 4097`` picked tile 8192,
      ``full`` was False on the only block, and the entire tensor went scalar:
      measured 187-217 us against a 52-113 us vendor baseline.
    * Prefer a tile that divides ``n_total``, so no ragged tile exists at all.
      A single ragged tile sits on the critical path of the whole launch --
      R194 measured 2.65x (113 us -> 300 us) for fp32 exp at 16M caused by one
      partial tile.  Give up at most half the width chasing a divisor; below
      that the narrower DMA costs more than the tail it avoids.

    Returns a multiple of ``grain``, at least ``grain``.
    """
    ceiling = min(int(hard_cap), int(cap), max(grain, math.ceil(n_total / 2)))
    ceiling = max(grain, ceiling - ceiling % grain)
    candidate = ceiling
    while candidate >= max(grain, ceiling // 2):
        if n_total % (2 * candidate) == 0:
            return candidate
        candidate -= grain
    return ceiling


_UB_INIT_RE = re.compile(r"InitBuffer\(ascend_ub, (\d+)\)")
_UB_SLOT_RE = re.compile(r"GetWithOffset<(\w+)>\((\d+), (\d+)\)")
_CTYPE_BYTES = {
    "int8_t": 1, "uint8_t": 1,
    "half": 2, "bfloat16_t": 2, "int16_t": 2, "uint16_t": 2,
    "float": 4, "int32_t": 4, "uint32_t": 4,
    "double": 8, "int64_t": 8, "uint64_t": 8,
}


def ub_usage(kernel_source: str) -> tuple[int, int] | None:
    """Return ``(highest_used_byte, unified_buffer_bytes)`` for an artifact.

    Some vector intrinsics take an implicit scratch buffer that never appears
    in the Python source -- ``T.tile.pow`` lowers to ``AscendC::Power`` with a
    65536-byte ``tmp_ub``.  The lowering places it in the same arena as the
    declared tiles and does **not** check the total, so a tile that is too
    wide silently puts a real buffer past the end of the unified buffer and
    the kernel faults at run time with aicore error 507015.  Reading the
    emitted offsets back is the only reliable accounting.

    Returns ``None`` when the artifact does not follow the expected shape, so
    a lowering change degrades to "unchecked" rather than to a false failure.
    """
    init = _UB_INIT_RE.search(kernel_source)
    if init is None:
        return None
    slots = _UB_SLOT_RE.findall(kernel_source)
    if not slots:
        return None
    end = 0
    for ctype, count, offset in slots:
        end = max(end, int(offset) + int(count) * _CTYPE_BYTES.get(ctype, 8))
    return end, int(init.group(1))


def ub_fits(kernel_source: str) -> bool:
    """True when every emitted UB slot lands inside the unified buffer."""
    usage = ub_usage(kernel_source)
    return True if usage is None else usage[0] <= usage[1]


def cap_ladder(cap: int) -> list[int]:
    """Descending tile-budget candidates for the compile-and-check retry."""
    ladder, value = [], max(256, int(cap))
    while value > 256:
        ladder.append(value)
        value = max(256, (value // 2) - (value // 2) % 256)
    ladder.append(256)
    return ladder


def pack_broadcast_units(
    repeat: int, unit_count: int, cap: int, broadcast_element_bytes: int
) -> tuple[int, int]:
    """Return ``(units_per_tile, tile)`` for the suffix-broadcast geometry.

    A broadcast *unit* is one source scalar's contiguous run of ``repeat``
    output elements.  The shipped geometry used half a unit per vector
    context, which makes every GM<->UB transfer ``repeat/2`` elements wide no
    matter how much UB is free.  Packing whole units instead keeps the
    transfer as wide as the UB budget allows; the broadcast operand then
    needs one scalar per packed unit rather than one per tile.

    Packing writes the broadcast operand through UB buffer *regions* at byte
    offsets of ``slot * repeat * broadcast_element_bytes``.  The Ascend
    broadcast intrinsic faults (aicore error 507015) on a region that is not
    32-byte aligned, so a unit that is not a whole number of 32-byte blocks
    stays at one unit per tile, where the only region is the buffer base.

    A divisor of ``unit_count`` is preferred over the raw cap so the grid
    stays free of a predicated partial tile, but only when that divisor is
    still more than half the cap -- otherwise the packing win is larger than
    the cost of one scalar-predicated tail tile.
    """
    repeat = int(repeat)
    if (repeat * int(broadcast_element_bytes)) % 32:
        return 1, repeat
    limit = max(1, min(int(unit_count), int(cap) // max(1, repeat)))
    units = limit
    for candidate in range(limit, limit // 2, -1):
        if int(unit_count) % candidate == 0:
            units = candidate
            break
    return units, units * repeat


def _broadcast_strides(
    shape: tuple[int, ...], out_shape: tuple[int, ...]
) -> tuple[int, ...]:
    padded = (1,) * (len(out_shape) - len(shape)) + tuple(shape)
    raw = [1] * len(out_shape)
    for axis in range(len(out_shape) - 2, -1, -1):
        raw[axis] = raw[axis + 1] * padded[axis + 1]
    return tuple(
        0 if padded[axis] == 1 and out_shape[axis] != 1 else raw[axis]
        for axis in range(len(out_shape))
    )


def _offset_expr(index, out_shape: tuple[int, ...], strides: tuple[int, ...]):
    """Build an output-linear-index -> operand-linear-offset expression."""
    offset = 0
    out_stride = 1
    for axis in range(len(out_shape) - 1, -1, -1):
        coord = (index // out_stride) % out_shape[axis]
        if strides[axis]:
            offset = offset + coord * strides[axis]
        out_stride *= out_shape[axis]
    return offset


def _is_contiguous(shape: tuple[int, ...], strides: tuple[int, ...]) -> bool:
    expected = 1
    for size, stride in zip(reversed(shape), reversed(strides)):
        if stride != expected:
            return False
        expected *= size
    return True


def _tail_broadcast_info(
    shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    strides: tuple[int, ...],
) -> tuple[int, int] | None:
    """Return ``(repeat, unit_count)`` for a contiguous suffix broadcast.

    A source such as ``[256, 1, 1]`` has one active (non-zero-stride) axis
    followed by a contiguous output suffix.  Each source element can then be
    loaded once and broadcast over that suffix.  Leading broadcast axes are
    allowed; interior broadcast axes are not, because they would make a
    source tile non-contiguous.
    """
    padded = (1,) * (len(out_shape) - len(shape)) + tuple(shape)
    active = [axis for axis, stride in enumerate(strides) if stride]
    if not active:
        return None
    first_active, last_active = active[0], active[-1]
    if active != list(range(first_active, last_active + 1)):
        return None
    repeat = math.prod(out_shape[last_active + 1 :])
    if repeat <= 1:
        return None
    # Keep the specialized UB tile bounded; larger repeats use the generic
    # path rather than forcing an oversized physical buffer.
    if repeat > 4096:
        return None
    unit_count = math.prod(out_shape[: last_active + 1])
    if unit_count <= 0 or math.prod(padded) != math.prod(shape):
        return None
    return repeat, unit_count


@lru_cache(maxsize=64)
def _compile_binary(
    a_numel: int,
    b_numel: int,
    out_shape: tuple[int, ...],
    a_strides: tuple[int, ...],
    b_strides: tuple[int, ...],
    dtype: str,
    alpha: float,
    op_kind: str,
    mode: str,
    repeat: int,
    unit_count: int,
    broadcast_side: str,
):
    if op_kind not in {"add", "sub"}:
        raise ValueError(f"unsupported elementwise binary expression {op_kind!r}")
    out_numel = math.prod(out_shape)
    # Each Ascend kernel block exposes two vector contexts.  The first two
    # paths use one contiguous tile per context; the generic path keeps the
    # old scalar index lowering but distributes its tiles across the grid.
    element_bytes = ELEMENT_BYTES[dtype]
    if dtype == "bfloat16":
        # a_ub/b_ub/c_ub plus the fp32 scratch trio, and alpha_calc when the
        # scalar multiply is live.
        live_bytes = 3 * element_bytes + 3 * 4 + (4 if alpha != 1.0 else 0)
    else:
        live_bytes = 3 * element_bytes
    # bfloat16 lands the broadcast in the fp32 scratch buffer, so the
    # alignment of the packed regions is judged in fp32 bytes there.
    broadcast_bytes = 4 if dtype == "bfloat16" else element_bytes

    def geometry(cap: int):
        """Resolve the tile geometry for one UB budget candidate."""
        if mode == "same":
            # 8192 stays the hard cap the shipped fast path proved; the three
            # R194 clamps decide how far below it this tensor should sit.
            return 1, fit_same_tile(cap, out_numel)
        if mode == "tail_broadcast":
            return pack_broadcast_units(repeat, unit_count, cap, broadcast_bytes)
        return 1, min(256, cap)

    def build(units_per_tile: int, tile: int):
        block_total = tile * 2
        if mode == "tail_broadcast":
            total_tiles = max(1, math.ceil(unit_count / units_per_tile))
            block_count = max(1, math.ceil(total_tiles / 2))
        else:
            block_count = max(1, math.ceil(out_numel / block_total))
        # Cap the launch width so the grid-stride loop actually strides; without
        # the cap this is ``launch_blocks == block_count`` and ``grid_repeats``
        # is always 1 (PROJECT_STATE 13.57).
        launch_blocks = launch_block_count(min(block_count, LAUNCH_BLOCK_CAP))
        grid_repeats = grid_repeat_count(block_count, launch_blocks)

        @T.prim_func
        def main(
            A: T.Tensor((a_numel,), dtype),
            B: T.Tensor((b_numel,), dtype),
            C: T.Tensor((out_numel,), dtype),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                a_ub = T.alloc_ub((tile,), dtype)
                b_ub = T.alloc_ub((tile,), dtype)
                c_ub = T.alloc_ub((tile,), dtype)
                a_calc = T.alloc_ub((tile,), "float32")
                b_calc = T.alloc_ub((tile,), "float32")
                c_calc = T.alloc_ub((tile,), "float32")
                alpha_calc = T.alloc_ub((tile,), "float32")
                scalar_calc = T.alloc_ub((1,), "float32")
                # Kept allocated for all compile-time paths so the frontend
                # does not treat the broadcast scalar as a branch-local name.
                scalar_ub = T.alloc_ub((1,), dtype)
                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < block_count:
                            base = logical_cid * block_total
                            start = base + vid * tile
                            # ``block_total`` is two tiles wide, so this is the
                            # flat index of the tile this vector context owns.
                            tile_index = logical_cid * 2 + vid
                            unit = tile_index * units_per_tile
                            full = start + tile <= out_numel
                            if mode == "same":
                                full = start + tile <= out_numel
                                if full:
                                    T.copy(A[start], a_ub)
                                    T.copy(B[start], b_ub)
                                else:
                                    T.tile.fill(a_ub, 0.0)
                                    T.tile.fill(b_ub, 0.0)
                                    for lane in T.serial(tile):
                                        idx = start + lane
                                        if idx < out_numel:
                                            a_ub[lane] = A[idx]
                                            b_ub[lane] = B[idx]
                                T.barrier_all()
                            elif mode == "tail_broadcast":
                                # ``tile`` is a whole multiple of ``repeat`` and
                                # the packed units are consecutive, so the only
                                # bound left is the end of the output.
                                valid = start < out_numel
                                full = start + tile <= out_numel
                                if broadcast_side == "b":
                                    if valid:
                                        if full:
                                            T.copy(A[start], a_ub)
                                        else:
                                            T.tile.fill(a_ub, 0.0)
                                        if not full:
                                            for lane in T.serial(tile):
                                                idx = start + lane
                                                if idx < out_numel:
                                                    a_ub[lane] = A[idx]
                                        # One source scalar per packed unit;
                                        # the loop bound is compile-time.
                                        for slot in range(units_per_tile):
                                            scalar_ub[0] = B[(unit + slot) % b_numel]
                                            if dtype == "bfloat16":
                                                T.tile.cast(
                                                    scalar_calc,
                                                    scalar_ub,
                                                    "CAST_NONE",
                                                    1,
                                                )
                                                T.tile.broadcast(
                                                    b_calc[
                                                        slot
                                                        * repeat : (slot + 1)
                                                        * repeat
                                                    ]
                                                    if units_per_tile > 1
                                                    else b_calc,
                                                    scalar_calc,
                                                )
                                            else:
                                                T.tile.broadcast(
                                                    b_ub[
                                                        slot
                                                        * repeat : (slot + 1)
                                                        * repeat
                                                    ]
                                                    if units_per_tile > 1
                                                    else b_ub,
                                                    scalar_ub,
                                                )
                                    else:
                                        T.tile.fill(a_ub, 0.0)
                                        T.tile.fill(b_ub, 0.0)
                                else:
                                    if valid:
                                        if full:
                                            T.copy(B[start], b_ub)
                                        else:
                                            T.tile.fill(b_ub, 0.0)
                                        if not full:
                                            for lane in T.serial(tile):
                                                idx = start + lane
                                                if idx < out_numel:
                                                    b_ub[lane] = B[idx]
                                        for slot in range(units_per_tile):
                                            scalar_ub[0] = A[(unit + slot) % a_numel]
                                            if dtype == "bfloat16":
                                                T.tile.cast(
                                                    scalar_calc,
                                                    scalar_ub,
                                                    "CAST_NONE",
                                                    1,
                                                )
                                                T.tile.broadcast(
                                                    a_calc[
                                                        slot
                                                        * repeat : (slot + 1)
                                                        * repeat
                                                    ]
                                                    if units_per_tile > 1
                                                    else a_calc,
                                                    scalar_calc,
                                                )
                                            else:
                                                T.tile.broadcast(
                                                    a_ub[
                                                        slot
                                                        * repeat : (slot + 1)
                                                        * repeat
                                                    ]
                                                    if units_per_tile > 1
                                                    else a_ub,
                                                    scalar_ub,
                                                )
                                    else:
                                        T.tile.fill(a_ub, 0.0)
                                        T.tile.fill(b_ub, 0.0)
                                T.barrier_all()
                            else:
                                T.tile.fill(a_ub, 0.0)
                                T.tile.fill(b_ub, 0.0)
                                for lane in T.serial(tile):
                                    idx = start + lane
                                    if idx < out_numel:
                                        a_idx = _offset_expr(idx, out_shape, a_strides)
                                        b_idx = _offset_expr(idx, out_shape, b_strides)
                                        a_ub[lane] = A[a_idx]
                                        b_ub[lane] = B[b_idx]
                                T.barrier_all()

                            if dtype == "bfloat16":
                                if not (
                                    mode == "tail_broadcast" and broadcast_side == "a"
                                ):
                                    T.tile.cast(a_calc, a_ub, "CAST_NONE", tile)
                                if not (
                                    mode == "tail_broadcast" and broadcast_side == "b"
                                ):
                                    T.tile.cast(b_calc, b_ub, "CAST_NONE", tile)
                                if alpha != 1.0:
                                    T.tile.fill(alpha_calc, alpha)
                                    T.tile.mul(b_calc, b_calc, alpha_calc)
                                if op_kind == "add":
                                    T.tile.add(c_calc, a_calc, b_calc)
                                else:
                                    T.tile.sub(c_calc, a_calc, b_calc)
                                T.tile.cast(c_ub, c_calc, "CAST_RINT", tile)
                            else:
                                if alpha != 1.0:
                                    T.tile.mul(b_ub, b_ub, alpha)
                                if op_kind == "add":
                                    T.tile.add(c_ub, a_ub, b_ub)
                                else:
                                    T.tile.sub(c_ub, a_ub, b_ub)
                            T.barrier_all()

                            if mode == "tail_broadcast":
                                if full:
                                    T.copy(c_ub, C[start])
                                else:
                                    for lane in T.serial(tile):
                                        idx = start + lane
                                        if idx < out_numel:
                                            C[idx] = c_ub[lane]
                            elif mode == "same":
                                if full:
                                    T.copy(c_ub, C[start])
                                else:
                                    for lane in T.serial(tile):
                                        idx = start + lane
                                        if idx < out_numel:
                                            C[idx] = c_ub[lane]
                            else:
                                for lane in T.serial(tile):
                                    idx = start + lane
                                    if idx < out_numel:
                                        C[idx] = c_ub[lane]

        return tilelang.jit(out_idx=[-1])(lambda: main)()

    ladder = cap_ladder(tile_cap(live_bytes))
    last = None
    for candidate_cap in ladder:
        units_per_tile, tile = geometry(candidate_cap)
        last = (units_per_tile, tile)
        compiled = build(units_per_tile, tile)
        if ub_fits(compiled.get_kernel_source()):
            return compiled
        if len(ladder) > 1 and (units_per_tile, tile) == geometry(ladder[-1]):
            # Shrinking the budget no longer moves the geometry (an unaligned
            # broadcast unit pins the tile), so retrying cannot help.
            break
    usage = ub_usage(compiled.get_kernel_source())
    raise ValueError(
        f"elementwise binary {op_kind}/{dtype} mode={mode} needs "
        f"{usage[0] if usage else '?'} UB bytes at tile {last[1]}, over the "
        f"{usage[1] if usage else '?'} available; no smaller tile fits"
    )


def build_binary_kernel(
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    dtype: torch.dtype,
    alpha,
    *,
    op_kind: str = "add",
    supported_dtypes: tuple[torch.dtype, ...] = SUPPORTED_DTYPES,
    op_name: str = "elementwise binary op",
):
    """Build a callable ``(input, other) -> output`` for a binary expression."""
    if dtype not in supported_dtypes:
        supported = ", ".join(str(item) for item in supported_dtypes)
        raise TypeError(
            f"{op_name} does not support dtype {dtype}; supported dtypes are [{supported}], "
            f"but received input dtype {dtype}"
        )
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise TypeError(
            f"{op_name} parameter alpha must be int or float, but received {alpha!r} "
            f"of type {type(alpha).__name__}"
        )
    if not math.isfinite(float(alpha)):
        raise ValueError(
            f"{op_name} parameter alpha must be finite, but received {alpha!r}"
        )
    try:
        out_shape = tuple(torch.broadcast_shapes(tuple(a_shape), tuple(b_shape)))
    except RuntimeError as exc:
        raise ValueError(
            f"{op_name} cannot broadcast input={tuple(a_shape)} and other={tuple(b_shape)}"
        ) from exc

    a_shape = tuple(a_shape)
    b_shape = tuple(b_shape)
    a_strides = _broadcast_strides(a_shape, out_shape)
    b_strides = _broadcast_strides(b_shape, out_shape)
    out_numel = math.prod(out_shape)
    a_numel = math.prod(a_shape)
    b_numel = math.prod(b_shape)
    direct_a = a_numel == out_numel and _is_contiguous(out_shape, a_strides)
    direct_b = b_numel == out_numel and _is_contiguous(out_shape, b_strides)
    mode = "generic"
    repeat = 1
    unit_count = 1
    broadcast_side = ""
    if direct_a and direct_b:
        mode = "same"
    elif direct_a:
        info = _tail_broadcast_info(b_shape, out_shape, b_strides)
        if info is not None:
            mode, (repeat, unit_count), broadcast_side = "tail_broadcast", info, "b"
    elif direct_b:
        info = _tail_broadcast_info(a_shape, out_shape, a_strides)
        if info is not None:
            mode, (repeat, unit_count), broadcast_side = "tail_broadcast", info, "a"
    compiled = _compile_binary(
        math.prod(a_shape),
        math.prod(b_shape),
        out_shape,
        a_strides,
        b_strides,
        str(dtype).replace("torch.", ""),
        float(alpha),
        op_kind,
        mode,
        repeat,
        unit_count,
        broadcast_side,
    )

    def invoke(input_tensor: torch.Tensor, other_tensor: torch.Tensor):
        if tuple(input_tensor.shape) != a_shape or tuple(other_tensor.shape) != b_shape:
            raise ValueError(
                f"{op_name} kernel shape mismatch: expected input={a_shape}, other={b_shape}; "
                f"received input={tuple(input_tensor.shape)}, other={tuple(other_tensor.shape)}"
            )
        input_flat = (
            input_tensor if input_tensor.ndim == 1 else input_tensor.reshape(-1)
        )
        other_flat = (
            other_tensor if other_tensor.ndim == 1 else other_tensor.reshape(-1)
        )
        result = compiled(input_flat, other_flat)
        return result if out_shape == (result.numel(),) else result.reshape(out_shape)

    return invoke
