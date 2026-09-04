"""Batch-only vector implementations for archetype-1 binary operators.

The established ``elementwise_binary`` template intentionally has a small
surface (add/sub).  This file keeps its addressing contract while adding the
other expressions that have the same two-input, one-output dataflow.  The
kernel never materializes a broadcasted operand on the host.
"""

import math
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count
from .elementwise_binary import (
    ELEMENT_BYTES,
    LAUNCH_BLOCK_CAP,
    SUPPORTED_DTYPES,
    _broadcast_strides,
    _is_contiguous,
    _offset_expr,
    _tail_broadcast_info,
    cap_ladder,
    fit_same_tile,
    pack_broadcast_units,
    tile_cap,
    ub_fits,
    ub_usage,
)


FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Expressions whose lowering runs through the fp32 scratch trio even when the
# operands are already fp32/fp16.
_FP32_PATH_KINDS = frozenset({"trunc_div", "floor_divide", "remainder"})
_ROUNDING_KINDS = frozenset({"trunc_div", "floor_divide", "remainder"})
_RECIPROCAL_KINDS = frozenset({"floor_divide", "remainder"})


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _live_bytes_per_element(in_dtype: str, out_dtype: str, op_kind: str) -> int:
    """Upper bound on UB bytes per tile element for one compiled variant.

    Mirrors the branch structure of the kernel body: buffers that no live
    branch touches are dropped by the lowering (visible in the generated
    artifact as missing ``GetWithOffset`` slots), so counting only the
    reachable ones is what lets the batch template use the same tile width
    as the add/sub template instead of budgeting for every scratch buffer.
    """
    total = 2 * ELEMENT_BYTES[in_dtype] + ELEMENT_BYTES[out_dtype]
    if in_dtype == "bfloat16" or op_kind in _FP32_PATH_KINDS:
        total += 3 * 4  # a_calc, b_calc, c_calc
        if op_kind in _RECIPROCAL_KINDS:
            total += 4  # reciprocal_calc
        if op_kind in _ROUNDING_KINDS:
            total += 4 + 4  # rounded_calc, rounded_i32
    return total


def _validate_shapes(a_shape, b_shape):
    try:
        out_shape = tuple(torch.broadcast_shapes(tuple(a_shape), tuple(b_shape)))
    except RuntimeError as exc:
        raise ValueError(
            f"cannot broadcast input={tuple(a_shape)} and other={tuple(b_shape)}"
        ) from exc
    return tuple(a_shape), tuple(b_shape), out_shape


@lru_cache(maxsize=128)
def _compile_batch(
    a_numel: int,
    b_numel: int,
    out_shape: tuple[int, ...],
    a_strides: tuple[int, ...],
    b_strides: tuple[int, ...],
    in_dtype: str,
    out_dtype: str,
    op_kind: str,
    scalar: float,
    mode: str,
    repeat: int,
    unit_count: int,
    broadcast_side: str,
    diagnostic_sentinel: bool,
):
    out_numel = math.prod(out_shape)
    live_bytes = _live_bytes_per_element(in_dtype, out_dtype, op_kind)
    # bfloat16 lands the broadcast in the fp32 scratch buffer, so the
    # alignment of the packed regions is judged in fp32 bytes there.
    broadcast_bytes = 4 if in_dtype == "bfloat16" else ELEMENT_BYTES[in_dtype]

    def geometry(cap: int):
        """Resolve the tile geometry for one UB budget candidate."""
        if mode == "tail_broadcast":
            return pack_broadcast_units(repeat, unit_count, cap, broadcast_bytes)
        if mode == "same":
            # Parity with the add/sub template's proven `same` geometry,
            # clamped to what this expression's live scratch buffers leave.
            # The rounding kinds have the smallest ``cap`` here, and they are
            # the ones the divisor search rescues: fp16 floor_divide at 16M had
            # cap 5888, so tile 5888 and 8388608 % 11776 != 0 -- every launch
            # carried a ragged tile.
            return 1, fit_same_tile(cap, out_numel)
        return 1, min(512, cap)

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
            A: T.Tensor((a_numel,), in_dtype),
            B: T.Tensor((b_numel,), in_dtype),
            C: T.Tensor((out_numel,), out_dtype),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                a_ub = T.alloc_ub((tile,), in_dtype)
                b_ub = T.alloc_ub((tile,), in_dtype)
                c_ub = T.alloc_ub((tile,), out_dtype)
                a_calc = T.alloc_ub((tile,), "float32")
                b_calc = T.alloc_ub((tile,), "float32")
                c_calc = T.alloc_ub((tile,), "float32")
                reciprocal_calc = T.alloc_ub((tile,), "float32")
                rounded_calc = T.alloc_ub((tile,), "float32")
                rounded_i32 = T.alloc_ub((tile,), "int32")
                # One-element staging for the suffix-broadcast source scalar,
                # kept allocated on every compile-time path so the frontend
                # does not treat it as a branch-local name.
                scalar_ub = T.alloc_ub((1,), in_dtype)
                scalar_calc = T.alloc_ub((1,), "float32")
                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < block_count:
                            start = logical_cid * block_total + vid * tile
                            # ``block_total`` is two tiles wide, so this is the
                            # flat index of the tile this vector context owns.
                            tile_index = logical_cid * 2 + vid
                            unit = tile_index * units_per_tile
                            full = start + tile <= out_numel
                            if diagnostic_sentinel:
                                T.tile.fill(c_ub, 123.0)
                                if mode == "same" and full:
                                    T.copy(c_ub, C[start])
                                else:
                                    for lane in T.serial(tile):
                                        idx = start + lane
                                        if idx < out_numel:
                                            C[idx] = c_ub[lane]
                                T.barrier_all()
                            if mode == "same":
                                if full:
                                    T.copy(A[start], a_ub)
                                    T.copy(B[start], b_ub)
                                else:
                                    T.tile.fill(a_ub, 0)
                                    T.tile.fill(b_ub, 0)
                                    for lane in T.serial(tile):
                                        idx = start + lane
                                        if idx < out_numel:
                                            a_ub[lane] = A[idx]
                                            b_ub[lane] = B[idx]
                            elif mode == "tail_broadcast":
                                # ``tile`` is a whole multiple of ``repeat`` and
                                # the packed units are consecutive, so the only
                                # bound left is the end of the output.
                                full = start + tile <= out_numel
                                valid = start < out_numel
                                if broadcast_side == "b":
                                    if valid:
                                        if full:
                                            T.copy(A[start], a_ub)
                                        else:
                                            T.tile.fill(a_ub, 0)
                                            for lane in T.serial(tile):
                                                idx = start + lane
                                                if idx < out_numel:
                                                    a_ub[lane] = A[idx]
                                        # One source scalar per packed unit;
                                        # the loop bound is compile-time.
                                        for slot in range(units_per_tile):
                                            scalar_ub[0] = B[
                                                (unit + slot) % b_numel
                                            ]
                                            if in_dtype == "bfloat16":
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
                                        T.tile.fill(a_ub, 0)
                                        T.tile.fill(b_ub, 0)
                                else:
                                    if valid:
                                        if full:
                                            T.copy(B[start], b_ub)
                                        else:
                                            T.tile.fill(b_ub, 0)
                                            for lane in T.serial(tile):
                                                idx = start + lane
                                                if idx < out_numel:
                                                    b_ub[lane] = B[idx]
                                        # One source scalar per packed unit;
                                        # the loop bound is compile-time.
                                        for slot in range(units_per_tile):
                                            scalar_ub[0] = A[
                                                (unit + slot) % a_numel
                                            ]
                                            if in_dtype == "bfloat16":
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
                                        T.tile.fill(a_ub, 0)
                                        T.tile.fill(b_ub, 0)
                            else:
                                T.tile.fill(a_ub, 0)
                                T.tile.fill(b_ub, 0)
                                for lane in T.serial(tile):
                                    idx = start + lane
                                    if idx < out_numel:
                                        a_ub[lane] = A[
                                            _offset_expr(idx, out_shape, a_strides)
                                        ]
                                        b_ub[lane] = B[
                                            _offset_expr(idx, out_shape, b_strides)
                                        ]

                            T.barrier_all()

                            if op_kind in {
                                "mul",
                                "div",
                                "trunc_div",
                                "floor_divide",
                                "remainder",
                                "pow",
                                "maximum",
                                "minimum",
                                "lerp",
                            }:
                                if in_dtype == "bfloat16" or op_kind in {
                                    "trunc_div",
                                    "floor_divide",
                                    "remainder",
                                }:
                                    # A bfloat16 suffix broadcast lands directly
                                    # in the fp32 scratch buffer (dav-2201 has no
                                    # native bf16 scalar path), so that operand
                                    # must not be re-derived from its UB tile.
                                    bcast_is_fp32 = (
                                        mode == "tail_broadcast"
                                        and in_dtype == "bfloat16"
                                    )
                                    if in_dtype == "float32":
                                        T.copy(a_ub, a_calc)
                                        T.copy(b_ub, b_calc)
                                    else:
                                        if not (bcast_is_fp32 and broadcast_side == "a"):
                                            T.tile.cast(a_calc, a_ub, "CAST_NONE", tile)
                                        if not (bcast_is_fp32 and broadcast_side == "b"):
                                            T.tile.cast(b_calc, b_ub, "CAST_NONE", tile)
                                    if op_kind == "mul":
                                        T.tile.mul(c_calc, a_calc, b_calc)
                                    elif op_kind in {
                                        "div",
                                        "trunc_div",
                                        "floor_divide",
                                        "remainder",
                                    }:
                                        if op_kind in {"div", "trunc_div"}:
                                            T.tile.div(c_calc, a_calc, b_calc)
                                        else:
                                            # Direct vector div can round a quotient
                                            # just below an integer up to that integer
                                            # before CAST_TRUNC/CAST_FLOOR.  Reciprocal
                                            # followed by multiply retains the residual
                                            # needed by floor-mod correction.
                                            T.tile.reciprocal(reciprocal_calc, b_calc)
                                            T.tile.mul(c_calc, a_calc, reciprocal_calc)
                                        if op_kind in {
                                            "trunc_div",
                                            "floor_divide",
                                            "remainder",
                                        }:
                                            T.tile.cast(
                                                rounded_i32,
                                                c_calc,
                                                "CAST_TRUNC"
                                                if op_kind == "trunc_div"
                                                else "CAST_FLOOR",
                                                tile,
                                            )
                                            T.tile.cast(
                                                rounded_calc,
                                                rounded_i32,
                                                "CAST_NONE",
                                                tile,
                                            )
                                        if op_kind in {
                                            "trunc_div",
                                            "floor_divide",
                                            "remainder",
                                        }:
                                            if op_kind in {"floor_divide", "remainder"}:
                                                # The approximate quotient can cross an
                                                # integer boundary in either direction.
                                                # A plain q*b comparison loses a true
                                                # one-ULP residual when that product
                                                # rounds to a.  Split b into high/low
                                                # fp32 parts (Dekker two-product) and
                                                # retain that residual in the scalar
                                                # correction loop.
                                                for lane in T.serial(tile):
                                                    reciprocal_calc[lane] = (
                                                        4097.0 * b_calc[lane]
                                                    )
                                                    reciprocal_calc[lane] = (
                                                        reciprocal_calc[lane]
                                                        - (
                                                            reciprocal_calc[lane]
                                                            - b_calc[lane]
                                                        )
                                                    )
                                                    c_calc[lane] = (
                                                        b_calc[lane]
                                                        - reciprocal_calc[lane]
                                                    )
                                                    reciprocal_calc[lane] = (
                                                        rounded_calc[lane]
                                                        * reciprocal_calc[lane]
                                                    )
                                                    c_calc[lane] = (
                                                        rounded_calc[lane]
                                                        * c_calc[lane]
                                                    )
                                                    reciprocal_calc[lane] = (
                                                        a_calc[lane]
                                                        - reciprocal_calc[lane]
                                                    ) - c_calc[lane]
                                                    if b_calc[lane] > 0.0:
                                                        if reciprocal_calc[lane] < 0.0:
                                                            rounded_calc[lane] = (
                                                                rounded_calc[lane] - 1.0
                                                            )
                                                        elif (
                                                            reciprocal_calc[lane]
                                                            >= b_calc[lane]
                                                        ):
                                                            rounded_calc[lane] = (
                                                                rounded_calc[lane] + 1.0
                                                            )
                                                    elif b_calc[lane] < 0.0:
                                                        if reciprocal_calc[lane] > 0.0:
                                                            rounded_calc[lane] = (
                                                                rounded_calc[lane] - 1.0
                                                            )
                                                        elif (
                                                            reciprocal_calc[lane]
                                                            <= b_calc[lane]
                                                        ):
                                                            rounded_calc[lane] = (
                                                                rounded_calc[lane] + 1.0
                                                            )
                                            if op_kind == "remainder":
                                                T.tile.mul(c_calc, rounded_calc, b_calc)
                                                T.tile.sub(c_calc, a_calc, c_calc)
                                            else:
                                                T.copy(rounded_calc, c_calc)
                                    elif op_kind == "pow":
                                        T.tile.pow(c_calc, a_calc, b_calc)
                                    elif op_kind == "maximum":
                                        T.tile.max(c_calc, a_calc, b_calc)
                                    elif op_kind == "minimum":
                                        T.tile.min(c_calc, a_calc, b_calc)
                                    elif op_kind == "lerp":
                                        T.tile.sub(c_calc, b_calc, a_calc)
                                        T.tile.mul(c_calc, c_calc, scalar)
                                        T.tile.add(c_calc, a_calc, c_calc)
                                    else:
                                        T.tile.add(c_calc, a_calc, b_calc)
                                    # CAST_RINT is the required fp32->fp16/bf16
                                    # conversion on dav-2201, but for a float32
                                    # remainder it would round the fractional result
                                    # to an integer.  Preserve the fp32 value directly.
                                    if (
                                        op_kind == "remainder"
                                        and out_dtype == "float32"
                                    ):
                                        T.copy(c_calc, c_ub)
                                    else:
                                        T.tile.cast(c_ub, c_calc, "CAST_RINT", tile)
                                else:
                                    if op_kind == "mul":
                                        T.tile.mul(c_ub, a_ub, b_ub)
                                    elif op_kind == "div":
                                        T.tile.div(c_ub, a_ub, b_ub)
                                    elif op_kind == "pow":
                                        T.tile.pow(c_ub, a_ub, b_ub)
                                    elif op_kind == "maximum":
                                        T.tile.max(c_ub, a_ub, b_ub)
                                    elif op_kind == "minimum":
                                        T.tile.min(c_ub, a_ub, b_ub)
                                    elif op_kind == "lerp":
                                        T.tile.sub(c_ub, b_ub, a_ub)
                                        T.tile.mul(c_ub, c_ub, scalar)
                                        T.tile.add(c_ub, a_ub, c_ub)
                                    else:
                                        T.tile.add(c_ub, a_ub, b_ub)

                            T.barrier_all()

                            if mode == "tail_broadcast" or mode == "same":
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

        return tilelang.jit(
            out_idx=[-1],
            pass_configs=(
                {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}
                if diagnostic_sentinel
                else None
            ),
        )(lambda: main)()

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
        f"batch binary {op_kind}/{in_dtype} mode={mode} needs "
        f"{usage[0] if usage else '?'} UB bytes at tile {last[1]}, over the "
        f"{usage[1] if usage else '?'} available; no smaller tile fits"
    )


def build_batch_binary(
    a_shape,
    b_shape,
    dtype: torch.dtype,
    *,
    op_kind: str,
    supported_dtypes=SUPPORTED_DTYPES,
    output_dtype: torch.dtype | None = None,
    scalar: float = 0.5,
    op_name: str = "elementwise binary op",
    diagnostic_sentinel: bool = False,
):
    supported_kinds = {
        "mul",
        "div",
        "trunc_div",
        "floor_divide",
        "remainder",
        "pow",
        "maximum",
        "minimum",
        "lerp",
    }
    if op_kind not in supported_kinds:
        raise ValueError(
            f"{op_name} has unsupported expression {op_kind!r}; expected one of {sorted(supported_kinds)}"
        )
    if dtype not in supported_dtypes:
        names = ", ".join(str(item) for item in supported_dtypes)
        raise TypeError(
            f"{op_name} does not support dtype {dtype}; supported dtypes are [{names}], received {dtype}"
        )
    a_shape, b_shape, out_shape = _validate_shapes(a_shape, b_shape)
    out_dtype = output_dtype or dtype
    a_strides = _broadcast_strides(a_shape, out_shape)
    b_strides = _broadcast_strides(b_shape, out_shape)
    a_numel, b_numel = math.prod(a_shape), math.prod(b_shape)
    out_numel = math.prod(out_shape)
    direct_a = a_numel == out_numel and _is_contiguous(out_shape, a_strides)
    direct_b = b_numel == out_numel and _is_contiguous(out_shape, b_strides)
    mode, repeat, unit_count, side = "generic", 1, 1, ""
    if direct_a and direct_b:
        mode = "same"
    elif direct_a:
        info = _tail_broadcast_info(b_shape, out_shape, b_strides)
        if info is not None:
            mode, (repeat, unit_count), side = "tail_broadcast", info, "b"
    elif direct_b:
        info = _tail_broadcast_info(a_shape, out_shape, a_strides)
        if info is not None:
            mode, (repeat, unit_count), side = "tail_broadcast", info, "a"
    compiled = _compile_batch(
        a_numel,
        b_numel,
        out_shape,
        a_strides,
        b_strides,
        _dtype_name(dtype),
        _dtype_name(out_dtype),
        op_kind,
        float(scalar),
        mode,
        repeat,
        unit_count,
        side,
        diagnostic_sentinel,
    )

    def invoke(a: torch.Tensor, b: torch.Tensor):
        if tuple(a.shape) != a_shape or tuple(b.shape) != b_shape:
            raise ValueError(
                f"{op_name} kernel shape mismatch: expected {a_shape}/{b_shape}, received {tuple(a.shape)}/{tuple(b.shape)}"
            )
        result = compiled(a.reshape(-1), b.reshape(-1))
        return result if out_shape == (result.numel(),) else result.reshape(out_shape)

    return invoke
