"""Comparison and logical binary kernels with byte-backed bool output."""

import math
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count
from .elementwise_binary import (
    LAUNCH_BLOCK_CAP,
    TILE_GRAIN,
    UB_BUDGET_BYTES,
    _broadcast_strides,
    _is_contiguous,
    _offset_expr,
    cap_ladder,
    fit_same_tile,
    ub_fits,
    ub_usage,
)


PREDICATE_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


def _storage_dtype(dtype: torch.dtype) -> str:
    return "uint8" if dtype == torch.bool else str(dtype).replace("torch.", "")


_ELEMENT_BYTES = {
    "bool": 1, "uint8": 1, "int8": 1,
    "float16": 2, "bfloat16": 2, "int16": 2,
    "float32": 4, "int32": 4,
    "int64": 8,
}

# Beyond this width the R194 probe measured no further bandwidth gain, and the
# emitted UB offsets are what actually decide whether a tile fits (ub_fits).
TILE_HARD_CAP = 16384


@lru_cache(maxsize=128)
def _compile_predicate(
    a_numel: int,
    b_numel: int,
    out_shape: tuple[int, ...],
    a_strides: tuple[int, ...],
    b_strides: tuple[int, ...],
    in_dtype: str,
    out_dtype: str,
    semantic_dtype: str,
    op_kind: str,
    mode: str,
):
    out_numel = math.prod(out_shape)
    compare_mode = "NE" if op_kind in {"logical_and", "logical_or"} else op_kind
    scalar_integer = semantic_dtype in {
        "bool",
        "uint8",
        "int8",
        "int16",
        "int64",
    } or (semantic_dtype == "int32" and op_kind in {"logical_and", "logical_or"})

    # --- Geometry (R199; the three R194 refinements, PROJECT_STATE 13.57) ---
    # Which buffers each variant actually reads or writes.  Read off the branch
    # structure of the kernel body below -- a wrong entry here is a silent wrong
    # value, not a crash, so the unused ones are still allocated at one grain
    # rather than dropped and hoped for:
    #
    #   a_ub, b_ub  in_dtype   always
    #   c_ub        uint8      always
    #   a_calc,b_calc fp32     only the ``semantic_dtype == "bfloat16"`` branch
    #   mask        uint8/8    every branch except ``scalar_integer``
    #   mask2       uint8/8    only logical_and / logical_or, and not scalar_integer
    needs_calc = semantic_dtype == "bfloat16"
    needs_mask = not scalar_integer
    needs_mask2 = needs_mask and op_kind in {"logical_and", "logical_or"}
    in_bytes = _ELEMENT_BYTES[in_dtype]
    # Eighths of a byte per element: the packed masks cost 1/8 byte each and the
    # integer arithmetic has to stay exact.
    live_eighths = 8 * (
        in_bytes                                   # a_ub
        + in_bytes                                 # b_ub
        + 1                                        # c_ub (uint8)
        + (8 if needs_calc else 0)                 # a_calc + b_calc (fp32 each)
    ) + (1 if needs_mask else 0) + (1 if needs_mask2 else 0)
    ub_cap = (UB_BUDGET_BYTES * 8) // live_eighths

    def build(tile: int):
        """Compile one candidate tile width."""
        # ``T.tile.compare`` wants operands that are a whole number of 256-byte
        # blocks (PROJECT_STATE 13.55 trap 3).  TILE_GRAIN is 256 and every
        # dtype that reaches a compare is at least 2 bytes wide, so
        # tile * in_bytes is a multiple of 512; assert rather than trust it.
        assert tile % TILE_GRAIN == 0, tile
        assert (tile * in_bytes) % 256 == 0, (tile, in_bytes)
        scratch_calc = tile if needs_calc else TILE_GRAIN
        scratch_mask = tile if needs_mask else TILE_GRAIN
        scratch_mask2 = tile if needs_mask2 else TILE_GRAIN
        block_total = tile * 2
        logical_blocks = max(1, math.ceil(out_numel / block_total))
        launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
        grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)

        @tilelang.jit(out_idx=[-1])
        def kernel():
            @T.prim_func
            def main(
                A: T.Tensor((a_numel,), in_dtype),
                B: T.Tensor((b_numel,), in_dtype),
                C: T.Tensor((out_numel,), out_dtype),
            ):
                with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                    a_ub = T.alloc_ub((tile,), in_dtype)
                    b_ub = T.alloc_ub((tile,), in_dtype)
                    a_calc = T.alloc_ub((scratch_calc,), "float32")
                    b_calc = T.alloc_ub((scratch_calc,), "float32")
                    mask = T.alloc_ub((scratch_mask // 8,), "uint8")
                    mask2 = T.alloc_ub((scratch_mask2 // 8,), "uint8")
                    c_ub = T.alloc_ub((tile,), "uint8")
                    with T.Scope("V"):
                        # Bounded grid-stride: ``launch_blocks`` blocks walk the
                        # tensor instead of one block per tile.  ``T.barrier_all()``
                        # inside the loop body is an in-core pipeline barrier, not a
                        # device fence, so it is safe here (verified in
                        # elementwise_binary.py).
                        for grid_repeat in T.serial(grid_repeats):
                            logical_cid = cid + grid_repeat * launch_blocks
                            if logical_cid < logical_blocks:
                                start = logical_cid * block_total + vid * tile
                                full = start + tile <= out_numel
                                if in_dtype in {"uint8", "int8", "int64"}:
                                    for lane in T.serial(tile):
                                        a_ub[lane] = 0
                                        b_ub[lane] = 0
                                else:
                                    T.tile.fill(a_ub, 0)
                                    T.tile.fill(b_ub, 0)
                                if mode == "same" and full:
                                    T.copy(A[start], a_ub)
                                    T.copy(B[start], b_ub)
                                else:
                                    for lane in T.serial(tile):
                                        idx = start + lane
                                        if idx < out_numel:
                                            a_ub[lane] = A[_offset_expr(idx, out_shape, a_strides)]
                                            b_ub[lane] = B[_offset_expr(idx, out_shape, b_strides)]
                                T.barrier_all()

                                if scalar_integer:
                                    for lane in T.serial(tile):
                                        if op_kind == "EQ":
                                            c_ub[lane] = T.if_then_else(
                                                a_ub[lane] == b_ub[lane], 1, 0
                                            )
                                        elif op_kind == "NE":
                                            c_ub[lane] = T.if_then_else(
                                                a_ub[lane] != b_ub[lane], 1, 0
                                            )
                                        elif op_kind == "GT":
                                            c_ub[lane] = T.if_then_else(
                                                a_ub[lane] > b_ub[lane], 1, 0
                                            )
                                        elif op_kind == "GE":
                                            c_ub[lane] = T.if_then_else(
                                                a_ub[lane] >= b_ub[lane], 1, 0
                                            )
                                        elif op_kind == "LT":
                                            c_ub[lane] = T.if_then_else(
                                                a_ub[lane] < b_ub[lane], 1, 0
                                            )
                                        elif op_kind == "LE":
                                            c_ub[lane] = T.if_then_else(
                                                a_ub[lane] <= b_ub[lane], 1, 0
                                            )
                                        elif op_kind == "logical_and":
                                            c_ub[lane] = T.if_then_else(
                                                (a_ub[lane] != 0) & (b_ub[lane] != 0), 1, 0
                                            )
                                        else:
                                            c_ub[lane] = T.if_then_else(
                                                (a_ub[lane] != 0) | (b_ub[lane] != 0), 1, 0
                                            )
                                elif semantic_dtype == "bfloat16":
                                    T.tile.cast(a_calc, a_ub, "CAST_NONE", tile)
                                    T.tile.cast(b_calc, b_ub, "CAST_NONE", tile)
                                    if op_kind in {"logical_and", "logical_or"}:
                                        T.tile.compare(mask, a_calc, 0.0, "NE")
                                        T.tile.compare(mask2, b_calc, 0.0, "NE")
                                    else:
                                        T.tile.compare(mask, a_calc, b_calc, compare_mode)
                                else:
                                    T.tile.compare(mask, a_ub, b_ub, compare_mode)
                                    if op_kind in {"logical_and", "logical_or"}:
                                        T.tile.compare(mask, a_ub, a_ub[0] - a_ub[0], "NE")
                                        T.tile.compare(mask2, b_ub, b_ub[0] - b_ub[0], "NE")
                                if not scalar_integer:
                                    if op_kind in {"logical_and", "logical_or"}:
                                        for byte in T.serial(tile // 8):
                                            mask[byte] = (
                                                mask[byte] & mask2[byte]
                                                if op_kind == "logical_and"
                                                else mask[byte] | mask2[byte]
                                            )
                                    for lane in T.serial(tile):
                                        c_ub[lane] = (mask[lane // 8] >> (lane % 8)) & 1
                                T.barrier_all()
                                if full:
                                    T.copy(c_ub, C[start])
                                else:
                                    for lane in T.serial(tile):
                                        idx = start + lane
                                        if idx < out_numel:
                                            C[idx] = c_ub[lane]

            return main
        return kernel()

    # Declared buffers adding up to under budget is not the same as fitting:
    # some intrinsics take an implicit scratch buffer that never appears in the
    # Python source, and the lowering places it in the same arena without
    # checking the total.  Read the emitted offsets back and step down the
    # ladder until they land inside the unified buffer (T195 acceptance 4).
    floor_tile = fit_same_tile(TILE_GRAIN, out_numel, hard_cap=TILE_HARD_CAP)
    compiled = None
    for candidate_cap in cap_ladder(ub_cap):
        tile = fit_same_tile(candidate_cap, out_numel, hard_cap=TILE_HARD_CAP)
        compiled = build(tile)
        if ub_fits(compiled.get_kernel_source()):
            return compiled
        if tile == floor_tile:
            break
    usage = ub_usage(compiled.get_kernel_source()) if compiled else None
    raise ValueError(
        f"elementwise predicate {op_kind}/{in_dtype} mode={mode} needs "
        f"{usage[0] if usage else '?'} UB bytes, over the "
        f"{usage[1] if usage else '?'} available; no smaller tile fits"
    )


def build_predicate_binary(a_shape, b_shape, dtype, *, op_kind, op_name):
    if dtype not in PREDICATE_DTYPES:
        raise TypeError(f"{op_name} does not support dtype {dtype}")
    try:
        out_shape = tuple(torch.broadcast_shapes(tuple(a_shape), tuple(b_shape)))
    except RuntimeError as exc:
        raise ValueError(f"{op_name} inputs cannot broadcast") from exc
    a_shape, b_shape = tuple(a_shape), tuple(b_shape)
    a_strides = _broadcast_strides(a_shape, out_shape)
    b_strides = _broadcast_strides(b_shape, out_shape)
    out_numel = math.prod(out_shape)
    mode = (
        "same"
        if (
            math.prod(a_shape) == out_numel
            and math.prod(b_shape) == out_numel
            and _is_contiguous(out_shape, a_strides)
            and _is_contiguous(out_shape, b_strides)
        )
        else "generic"
    )
    compiled = _compile_predicate(
        math.prod(a_shape),
        math.prod(b_shape),
        out_shape,
        a_strides,
        b_strides,
        _storage_dtype(dtype),
        "uint8",
        str(dtype).replace("torch.", ""),
        op_kind,
        mode,
    )

    def invoke(a, b):
        if tuple(a.shape) != a_shape or tuple(b.shape) != b_shape:
            raise ValueError(f"{op_name} kernel shape mismatch")
        if dtype == torch.bool:
            a = a.view(torch.uint8)
            b = b.view(torch.uint8)
        result = compiled(a.reshape(-1), b.reshape(-1))
        result = result.view(torch.bool)
        return result if out_shape == (result.numel(),) else result.reshape(out_shape)

    return invoke


@lru_cache(maxsize=128)
def _compile_where(
    condition_numel: int,
    input_numel: int,
    other_numel: int,
    out_shape: tuple[int, ...],
    condition_strides: tuple[int, ...],
    input_strides: tuple[int, ...],
    other_strides: tuple[int, ...],
    dtype: str,
    mode: str,
):
    """Compile a three-input, mixed-dtype where kernel.

    The condition is byte-backed bool while both selected operands retain the
    output dtype.  Arithmetic is promoted to fp32 in UB before the predicated
    select and cast back once, avoiding scalar half/bfloat16 expressions in the
    AIC path.  Generic broadcasts use the same offset expression as binary
    predicates; no host expansion is involved.
    """
    out_numel = math.prod(out_shape)
    # --- Geometry (R199; the three R194 refinements, PROJECT_STATE 13.57) ---
    # Every buffer below is live on every path: the fp32 trio is written either
    # by T.tile.cast (fp16/bf16) or by T.copy (fp32), so there is no conditional
    # scratch to table up here.
    #   condition_ub uint8 | input_ub, other_ub, output_ub dtype
    #   input32, other32, output32 fp32
    live_bytes = 1 + 3 * _ELEMENT_BYTES[dtype] + 3 * 4
    ub_cap = UB_BUDGET_BYTES // live_bytes
    tile = fit_same_tile(ub_cap, out_numel, hard_cap=TILE_HARD_CAP)
    block_total = tile * 2
    logical_blocks = max(1, math.ceil(out_numel / block_total))
    # Without the cap this was ``launch_blocks == logical_blocks`` and
    # ``grid_repeats == 1``: the loop below existed but never strided.
    launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
    grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)
    all_contiguous = mode == "same"

    @tilelang.jit(out_idx=[-1])
    def kernel():
        @T.prim_func
        def main(
            CONDITION: T.Tensor((condition_numel,), "uint8"),
            INPUT: T.Tensor((input_numel,), dtype),
            OTHER: T.Tensor((other_numel,), dtype),
            OUTPUT: T.Tensor((out_numel,), dtype),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                condition_ub = T.alloc_ub((tile,), "uint8")
                input_ub = T.alloc_ub((tile,), dtype)
                other_ub = T.alloc_ub((tile,), dtype)
                input32 = T.alloc_ub((tile,), "float32")
                other32 = T.alloc_ub((tile,), "float32")
                output32 = T.alloc_ub((tile,), "float32")
                output_ub = T.alloc_ub((tile,), dtype)
                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            start = logical_cid * block_total + vid * tile
                            full = start + tile <= out_numel
                            if all_contiguous and full:
                                T.copy(CONDITION[start : start + tile], condition_ub)
                                T.copy(INPUT[start : start + tile], input_ub)
                                T.copy(OTHER[start : start + tile], other_ub)
                            else:
                                for lane in T.serial(tile):
                                    condition_ub[lane] = 0
                                T.tile.fill(input_ub, 0)
                                T.tile.fill(other_ub, 0)
                                for lane in T.serial(tile):
                                    idx = start + lane
                                    if idx < out_numel:
                                        condition_ub[lane] = CONDITION[
                                            _offset_expr(idx, out_shape, condition_strides)
                                        ]
                                        input_ub[lane] = INPUT[
                                            _offset_expr(idx, out_shape, input_strides)
                                        ]
                                        other_ub[lane] = OTHER[
                                            _offset_expr(idx, out_shape, other_strides)
                                        ]
                            T.barrier_all()
                            if dtype == "float32":
                                T.copy(input_ub, input32)
                                T.copy(other_ub, other32)
                            else:
                                T.tile.cast(input32, input_ub, "CAST_NONE", tile)
                                T.tile.cast(other32, other_ub, "CAST_NONE", tile)
                            for lane in T.serial(tile):
                                output32[lane] = T.if_then_else(
                                    condition_ub[lane] != 0,
                                    input32[lane],
                                    other32[lane],
                                )
                            if dtype == "float32":
                                T.copy(output32, output_ub)
                            else:
                                T.tile.cast(output_ub, output32, "CAST_RINT", tile)
                            T.barrier_all()
                            if full:
                                T.copy(output_ub, OUTPUT[start : start + tile])
                            elif start < out_numel:
                                T.copy(output_ub[0 : out_numel - start], OUTPUT[start:out_numel])

        return main

    return kernel()


def build_where_kernel(condition_shape, input_shape, other_shape, dtype):
    """Build ``torch.where(condition, input, other)`` with full broadcasting."""
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"WhereFwdOp does not support dtype {dtype}")
    try:
        out_shape = tuple(
            torch.broadcast_shapes(tuple(condition_shape), tuple(input_shape), tuple(other_shape))
        )
    except RuntimeError as exc:
        raise ValueError("WhereFwdOp inputs cannot broadcast") from exc
    condition_shape = tuple(condition_shape)
    input_shape = tuple(input_shape)
    other_shape = tuple(other_shape)
    condition_strides = _broadcast_strides(condition_shape, out_shape)
    input_strides = _broadcast_strides(input_shape, out_shape)
    other_strides = _broadcast_strides(other_shape, out_shape)
    out_numel = math.prod(out_shape)
    mode = (
        "same"
        if all(
            math.prod(shape) == out_numel and _is_contiguous(out_shape, strides)
            for shape, strides in (
                (condition_shape, condition_strides),
                (input_shape, input_strides),
                (other_shape, other_strides),
            )
        )
        else "generic"
    )
    compiled = _compile_where(
        math.prod(condition_shape),
        math.prod(input_shape),
        math.prod(other_shape),
        out_shape,
        condition_strides,
        input_strides,
        other_strides,
        str(dtype).replace("torch.", ""),
        mode,
    )

    def invoke(condition, input, other):
        if tuple(condition.shape) != condition_shape or tuple(input.shape) != input_shape or tuple(other.shape) != other_shape:
            raise ValueError("WhereFwdOp kernel shape mismatch")
        condition = condition.view(torch.uint8) if condition.dtype == torch.bool else condition
        result = compiled(condition.reshape(-1), input.reshape(-1), other.reshape(-1))
        return result if tuple(out_shape) == (result.numel(),) else result.reshape(out_shape)

    return invoke
