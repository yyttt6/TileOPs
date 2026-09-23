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
    _tail_broadcast_info,
    cap_ladder,
    fit_same_tile,
    pack_broadcast_units,
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


#: R346.  The 1-bit-mask -> 1-byte-bool expansion runs its select/cast through
#: float16 buffers for EVERY input dtype.  That is forced by measurement, not
#: taste: on dav-2201 ``cast float16 -> uint8`` lowers in all four rounding modes
#: while ``cast float32 -> uint8`` and ``cast float32 -> int8`` do not lower at
#: all (R346-data/probe/p1_intrinsics.log).  A compare mask is a packed bitmask
#: with no dtype of its own, so the expansion width is free to be float16.
_EXPAND_DTYPE = "float16"

#: Integer storages whose entire value range is exact in float16, so the compare
#: can be vectorised by casting the operands first.  int16/int32/int64 are NOT
#: here: 2048 and up is not exact in float16, and ``T.tile.compare`` has no int16
#: lowering at all (measured, p1_intrinsics.log ``int_compare/int16``).
_F16_EXACT = frozenset({"bool", "uint8", "int8"})

#: R346.  ``T.tile.broadcast`` does not lower at bfloat16 (measured,
#: probe/p1b_broadcast.log ``broadcast1d/bfloat16``), so a bfloat16 suffix
#: broadcast cannot blow its source scalar up in the OPERAND buffer.  It blows it
#: up in the fp32 COMPARE buffer instead -- bfloat16 already widens to fp32 for
#: the compare, and ``broadcast1d/float32`` is measured correct.  Getting the
#: scalar there needs a vector cast, because dav-2201 has no SCALAR bf16 cast
#: instruction (convolution.py:901), so the per-unit scalars are staged into one
#: 256-byte buffer and cast in a single ``T.tile.cast``.  128 lanes = 256 bytes at
#: bfloat16, which is the block the cast wants.
_STAGE_SLOTS = 8

#: Elements between two slots' staging cells.  The Ascend broadcast intrinsic
#: faults with aicore error 507015 on an operand that is not 32-byte aligned
#: (the same 507015 ``_compile_where``'s geometry comment records, and measured
#: again this round: a uint8 staging cell at byte offset ``slot * 1`` took the
#: whole ``logical_and``/bool broadcast down).  32 ELEMENTS is 32 bytes at the
#: narrowest storage and a whole number of 32-byte blocks at every wider one, so
#: ``stage[slot * _STAGE_STRIDE]`` is aligned for every dtype -- and because the
#: stride is the same in both the input and the widened buffer, the single
#: ``T.tile.cast`` between them still maps slot to slot.
_STAGE_STRIDE = 32


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
    repeat: int = 1,
    unit_count: int = 1,
    broadcast_side: str = "",
):
    """Compile one comparison/logical kernel with a VECTOR bit->byte expansion.

    R346.  ``T.tile.compare`` produces a PACKED one-bit-per-lane mask while the
    output is byte-backed bool, so something has to widen 1 bit to 1 byte.  The
    shipped kernel did it with::

        for lane in T.serial(tile):
            c_ub[lane] = (mask[lane // 8] >> (lane % 8)) & 1

    on EVERY path -- contiguous, full tile, every shape.  That is one scalar
    iteration per output element, and the emitted AscendC confirms it: the
    ``hidden-state-prefill`` fp16 kernel carries an unconditional
    ``for (lane_1 = 0; lane_1 < 16384; ++lane_1) c_ub.SetValue(...)``
    (R346-data/src-dump/before/pred-eq-prefill-fp16.cce).  8.4M output elements
    = 8.4M scalar iterations per call, which is why every case of all eight
    operators sat between 0.003 and 0.035 of the vendor.

    The replacement is ``T.tile.select`` between two constant float16 tiles
    followed by ``T.tile.cast`` down to the output byte -- the same select the
    masked-fill vector path uses, measured end to end at wrong=0/512 in
    ``probe/p1_intrinsics.log``.  Two more per-lane loops go with it: the
    ``logical_and``/``logical_or`` byte-wise mask merge becomes one
    ``T.tile.bitwise_and``/``_or`` on the packed masks, and a bool operand is
    widened once with ``T.tile.cast`` instead of being compared lane by lane.
    """
    out_numel = math.prod(out_shape)
    compare_mode = "NE" if op_kind in {"logical_and", "logical_or"} else op_kind
    logical = op_kind in {"logical_and", "logical_or"}
    # Which semantics still have to run the shipped per-lane scalar compute.
    # Read off the intrinsic probe, not guessed:
    #   int16  -- no ``T.tile.compare`` lowering, and not exact in float16
    #   int64  -- no compare and no cast at that width
    #   int32  -- ``T.tile.compare`` DOES lower on an int32 buffer and it is
    #             SILENTLY WRONG for every ordering predicate.  R346 measured it
    #             on both trees with the same seed (R346-data/p7-before.json vs
    #             p7-after.json, 512x1024 of randint(-3,4)):
    #                 NE 524288/524288 wrong, GT 310002, GE 213958,
    #                 LT 310330, LE 214286 wrong; only EQ is right.
    #             The counts are IDENTICAL on both trees, so this is a
    #             PRE-EXISTING defect, not something this round introduced -- but
    #             ``PREDICATE_DTYPES`` advertises int32 and
    #             ``t301a_kernels.build_compare`` calls straight into it, so it
    #             is a live wrong answer.  The manifest never caught it because
    #             every predicate workload is float16/bfloat16/float32.
    #             Correctness outranks speed: int32 takes the scalar path, which
    #             the same probe shows is exact at int16 and int64.
    scalar_integer = semantic_dtype in {"int16", "int32", "int64"}
    if scalar_integer:
        calc_dtype = None
    elif semantic_dtype == "bfloat16":
        calc_dtype = "float32"          # shipped behaviour, unchanged
    elif semantic_dtype in _F16_EXACT:
        calc_dtype = _EXPAND_DTYPE      # R346: new vector path for bool/uint8/int8
    else:
        calc_dtype = None               # float16 / float32 / int32 compare directly

    # --- Geometry (R199; the three R194 refinements, PROJECT_STATE 13.57) ---
    # Which buffers each variant actually reads or writes.  Read off the branch
    # structure of the kernel body below -- a wrong entry here is a silent wrong
    # value, not a crash, so the unused ones are still allocated at one grain
    # rather than dropped and hoped for:
    #
    #   a_ub, b_ub          in_dtype   always
    #   c_ub                uint8      always
    #   a_calc, b_calc      calc_dtype only when ``calc_dtype`` is set
    #   mask                uint8/8    every branch except ``scalar_integer``
    #   mask2               uint8/8    only logical_and / logical_or, and not scalar_integer
    #   sel_ub, one_ub, zero_ub  float16  the expansion, every non-scalar branch
    needs_calc = calc_dtype is not None
    calc_bytes = _ELEMENT_BYTES[calc_dtype] if needs_calc else 0
    needs_mask = not scalar_integer
    needs_mask2 = needs_mask and logical
    needs_expand = not scalar_integer
    # R346.  bfloat16 is the one dtype whose OPERAND buffer cannot take
    # ``T.tile.broadcast`` at all (measured, probe/p1b_broadcast.log); it
    # broadcasts into the fp32 COMPARE buffer instead, which bfloat16 has to fill
    # anyway, and which ``broadcast1d/float32`` is measured correct on.
    bcast_via_calc = (
        mode == "tail_broadcast"
        and needs_calc
        and calc_dtype != in_dtype
        and semantic_dtype == "bfloat16"
    )
    # Plain Python bool, on purpose: see the comment at its use site.
    is_broadcast = bool(broadcast_side)
    expand_bytes = _ELEMENT_BYTES[_EXPAND_DTYPE]
    in_bytes = _ELEMENT_BYTES[in_dtype]
    # Element granularity.  ``same``/``generic`` keep the shipped 256-element
    # grain so their tile choice is unchanged.  The suffix-broadcast tile is
    # ``units * repeat``, which has no reason to be a multiple of 256, so it gets
    # the real constraint instead: whole 256-byte blocks for the compare operand
    # AND for the float16 expansion buffers.
    grain = TILE_GRAIN if mode != "tail_broadcast" else max(128, 256 // in_bytes)
    # Eighths of a byte per element: the packed masks cost 1/8 byte each and the
    # integer arithmetic has to stay exact.
    live_eighths = 8 * (
        in_bytes                                   # a_ub
        + in_bytes                                 # b_ub
        + 1                                        # c_ub (uint8)
        + (2 * calc_bytes if needs_calc else 0)    # a_calc + b_calc
        + (3 * expand_bytes if needs_expand else 0)  # sel_ub + one_ub + zero_ub
    ) + (1 if needs_mask else 0) + (1 if needs_mask2 else 0)
    ub_cap = (UB_BUDGET_BYTES * 8) // live_eighths

    def build(tile: int, units_per_tile: int = 1):
        """Compile one candidate tile width."""
        # ``T.tile.compare``/``T.tile.select`` want operands that are a whole
        # number of 256-byte blocks (PROJECT_STATE 13.55 trap 3).  Assert the
        # BYTE counts rather than a single element count, because three
        # different widths are live at once (in_dtype, calc_dtype, float16).
        assert tile % grain == 0, (tile, grain)
        assert (tile * in_bytes) % 256 == 0, (tile, in_bytes)
        if needs_calc:
            assert (tile * calc_bytes) % 256 == 0, (tile, calc_bytes)
        if needs_expand:
            assert (tile * expand_bytes) % 256 == 0, (tile, expand_bytes)
        scratch_calc = tile if needs_calc else grain
        scratch_mask = tile if needs_mask else grain
        scratch_mask2 = tile if needs_mask2 else grain
        scratch_expand = tile if needs_expand else grain
        block_total = tile * 2
        if mode == "tail_broadcast":
            total_tiles = max(1, math.ceil(unit_count / units_per_tile))
            logical_blocks = max(1, math.ceil(total_tiles / 2))
        else:
            logical_blocks = max(1, math.ceil(out_numel / block_total))
        launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
        grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)
        # --- Ragged tail (R334's rule, same as elementwise_binary.py) --------
        # Every context start is a multiple of ``tile``, so the ONE ragged tile
        # is at ``(out_numel // tile) * tile`` and ``tail`` is compile-time.  A
        # partial-extent copy moves WHOLE 32-byte blocks, so only the
        # block-aligned prefix may ride ``T.copy``; the at-most-one-block
        # remainder is a bounded scalar loop instead of the O(tile) one the
        # shipped code ran on every lane of the tile no matter how short the
        # tail was.  For the STORE a too-long copy would be an over-WRITE past
        # the end of the output, which is why the store uses its own
        # ``out_vec`` and never ``tail``.
        tail = out_numel % tile
        in_blk = max(1, 32 // in_bytes)
        in_vec = tail - tail % in_blk
        out_blk = 32
        out_vec = tail - tail % out_blk

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
                    a_calc = T.alloc_ub((scratch_calc,), calc_dtype or "float32")
                    b_calc = T.alloc_ub((scratch_calc,), calc_dtype or "float32")
                    mask = T.alloc_ub((scratch_mask // 8,), "uint8")
                    mask2 = T.alloc_ub((scratch_mask2 // 8,), "uint8")
                    sel_ub = T.alloc_ub((scratch_expand,), _EXPAND_DTYPE)
                    one_ub = T.alloc_ub((scratch_expand,), _EXPAND_DTYPE)
                    zero_ub = T.alloc_ub((scratch_expand,), _EXPAND_DTYPE)
                    # One-element staging for the suffix-broadcast source scalar,
                    # allocated on every compile-time path so the frontend does
                    # not treat it as a branch-local name.
                    # Suffix-broadcast scalar staging: ONE CELL PER SLOT (see the
                    # race comment in the load block below).  Kept allocated on
                    # every compile-time path so the frontend never sees a
                    # branch-local name.  At most 128*(in_bytes + calc_bytes)
                    # bytes; it is not a per-element cost and so does not enter
                    # ``live_eighths``.
                    stage_in = T.alloc_ub(
                        (_STAGE_SLOTS * _STAGE_STRIDE,), in_dtype
                    )
                    stage_calc = T.alloc_ub(
                        (_STAGE_SLOTS * _STAGE_STRIDE,), calc_dtype or "float32"
                    )
                    c_ub = T.alloc_ub((tile,), "uint8")
                    with T.Scope("V"):
                        if needs_expand:
                            # Hoisted out of the grid-stride loop on purpose:
                            # the two select operands are loop invariants, so
                            # this is two vector fills per BLOCK, not per tile.
                            T.tile.fill(one_ub, 1.0)
                            T.tile.fill(zero_ub, 0.0)
                            T.barrier_all()
                        # Bounded grid-stride: ``launch_blocks`` blocks walk the
                        # tensor instead of one block per tile.  ``T.barrier_all()``
                        # inside the loop body is an in-core pipeline barrier, not a
                        # device fence, so it is safe here (verified in
                        # elementwise_binary.py).
                        for grid_repeat in T.serial(grid_repeats):
                            logical_cid = cid + grid_repeat * launch_blocks
                            if logical_cid < logical_blocks:
                                start = logical_cid * block_total + vid * tile
                                # ``block_total`` is two tiles wide, so this is the
                                # flat index of the tile this context owns.
                                unit = (logical_cid * 2 + vid) * units_per_tile
                                full = start + tile <= out_numel
                                if mode == "tail_broadcast":
                                    # R334 §2.3's geometry: the broadcast operand
                                    # contributes ONE scalar per packed unit, blown
                                    # up by ``T.tile.broadcast``, while the direct
                                    # operand rides a full-width ``T.copy``.  The
                                    # generic path this replaces re-read BOTH
                                    # operands one lane at a time.
                                    if start < out_numel:
                                        # R346 RACE.  Every per-unit source scalar
                                        # gets its OWN staging cell and the Scalar
                                        # work is fenced off from the Vector work.
                                        # Reusing one cell across slots -- which is
                                        # what the shipped suffix-broadcast code in
                                        # ``_compile_where`` does -- is a WAR hazard:
                                        # ``T.tile.broadcast`` for slot k reads the
                                        # cell on the Vector pipe while the Scalar
                                        # store for slot k+1 is already overwriting
                                        # it.  Measured, R346-data/p9-prefix.json:
                                        # 12 calls of ONE compiled kernel on ONE pair
                                        # of tensors gave n_wrong =
                                        # 1440,1362,1362,...,1363,... (fp16 EQ) and
                                        # 3868793,3858855,3824389,... (fp16 GT) --
                                        # a different answer nearly every call.
                                        if broadcast_side != "a":
                                            T.copy(A[start], a_ub)
                                        else:
                                            for slot in range(units_per_tile):
                                                stage_in[slot * _STAGE_STRIDE] = A[
                                                    (unit + slot) % a_numel
                                                ]
                                        if broadcast_side != "b":
                                            T.copy(B[start], b_ub)
                                        else:
                                            for slot in range(units_per_tile):
                                                stage_in[slot * _STAGE_STRIDE] = B[
                                                    (unit + slot) % b_numel
                                                ]
                                        # NOT ``if broadcast_side:`` -- the TVM
                                        # script parser only folds a Python *bool*
                                        # (or lowers a PrimExpr); a bare ``str``
                                        # test raises "expected Object but got str"
                                        # and the whole tail_broadcast build silently
                                        # falls back to the generic per-lane gather.
                                        # That cost R346 one round of measurements
                                        # (src-dump/after.txt at 04:48 shows the
                                        # diagnostic and a ``for lane = 0..14336``
                                        # in what should have been a DMA-only tile).
                                        if is_broadcast:
                                            T.barrier_all()
                                            if bcast_via_calc:
                                                # bfloat16 cannot take
                                                # ``T.tile.broadcast`` at all, so it
                                                # widens the staged scalars with ONE
                                                # vector cast and broadcasts in fp32.
                                                T.tile.cast(
                                                    stage_calc, stage_in, "CAST_NONE",
                                                    _STAGE_SLOTS * _STAGE_STRIDE,
                                                )
                                                T.barrier_all()
                                            for slot in range(units_per_tile):
                                                if bcast_via_calc:
                                                    if broadcast_side == "a":
                                                        T.tile.broadcast(
                                                            a_calc[
                                                                slot * repeat
                                                                : (slot + 1) * repeat
                                                            ]
                                                            if units_per_tile > 1
                                                            else a_calc,
                                                            stage_calc[
                                                                slot * _STAGE_STRIDE
                                                                : slot * _STAGE_STRIDE + 1
                                                            ],
                                                        )
                                                    else:
                                                        T.tile.broadcast(
                                                            b_calc[
                                                                slot * repeat
                                                                : (slot + 1) * repeat
                                                            ]
                                                            if units_per_tile > 1
                                                            else b_calc,
                                                            stage_calc[
                                                                slot * _STAGE_STRIDE
                                                                : slot * _STAGE_STRIDE + 1
                                                            ],
                                                        )
                                                else:
                                                    if broadcast_side == "a":
                                                        T.tile.broadcast(
                                                            a_ub[
                                                                slot * repeat
                                                                : (slot + 1) * repeat
                                                            ]
                                                            if units_per_tile > 1
                                                            else a_ub,
                                                            stage_in[
                                                                slot * _STAGE_STRIDE
                                                                : slot * _STAGE_STRIDE + 1
                                                            ],
                                                        )
                                                    else:
                                                        T.tile.broadcast(
                                                            b_ub[
                                                                slot * repeat
                                                                : (slot + 1) * repeat
                                                            ]
                                                            if units_per_tile > 1
                                                            else b_ub,
                                                            stage_in[
                                                                slot * _STAGE_STRIDE
                                                                : slot * _STAGE_STRIDE + 1
                                                            ],
                                                        )
                                elif mode == "same":
                                    if full:
                                        T.copy(A[start], a_ub)
                                        T.copy(B[start], b_ub)
                                    else:
                                        # Lanes at or past ``tail`` are never
                                        # stored, so they are left as they are
                                        # rather than zeroed: the shipped
                                        # ``T.tile.fill``/per-lane zeroing ran on
                                        # every tile including the full ones, and
                                        # a compare only reads bits, so stale
                                        # lanes cannot trap.
                                        if tail:
                                            # A Python int and a PrimExpr may not
                                            # share an ``and``; keep the
                                            # compile-time test on the outside.
                                            if start < out_numel:
                                                if in_vec:
                                                    T.copy(
                                                        A[start : start + in_vec],
                                                        a_ub[0:in_vec],
                                                    )
                                                    T.copy(
                                                        B[start : start + in_vec],
                                                        b_ub[0:in_vec],
                                                    )
                                                for lane in T.serial(in_blk):
                                                    if in_vec + lane < tail:
                                                        a_ub[in_vec + lane] = A[
                                                            start + in_vec + lane
                                                        ]
                                                        b_ub[in_vec + lane] = B[
                                                            start + in_vec + lane
                                                        ]
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
                                else:
                                    if needs_calc:
                                        # When the broadcast rode the fp32 buffer
                                        # directly there is nothing in ``*_ub`` for
                                        # that side to widen -- casting it here
                                        # would overwrite the broadcast with
                                        # garbage.
                                        if not (bcast_via_calc and broadcast_side == "a"):
                                            T.tile.cast(a_calc, a_ub, "CAST_NONE", tile)
                                        if not (bcast_via_calc and broadcast_side == "b"):
                                            T.tile.cast(b_calc, b_ub, "CAST_NONE", tile)
                                        if logical:
                                            T.tile.compare(mask, a_calc, 0.0, "NE")
                                            T.tile.compare(mask2, b_calc, 0.0, "NE")
                                        else:
                                            T.tile.compare(mask, a_calc, b_calc, compare_mode)
                                    else:
                                        if logical:
                                            T.tile.compare(mask, a_ub, a_ub[0] - a_ub[0], "NE")
                                            T.tile.compare(mask2, b_ub, b_ub[0] - b_ub[0], "NE")
                                        else:
                                            T.tile.compare(mask, a_ub, b_ub, compare_mode)
                                    if logical:
                                        # Was ``for byte in T.serial(tile // 8)``.
                                        # The packed masks are uint8 and uint8 is a
                                        # width where the vector bitwise is
                                        # *verified* correct (p1_intrinsics.log
                                        # ``mask_bitwise/*``: wrong=0/512, including
                                        # the dst==src0 aliasing used here).
                                        if op_kind == "logical_and":
                                            T.tile.bitwise_and(mask, mask, mask2)
                                        else:
                                            T.tile.bitwise_or(mask, mask, mask2)
                                    # 1 bit -> 1 byte, vectorised.  A set mask bit
                                    # takes src0 (the polarity ``_compile_where``
                                    # and ``_compile_masked_fill`` rely on), so
                                    # ``one_ub`` must be src0.
                                    T.tile.select(
                                        sel_ub,
                                        mask,
                                        one_ub,
                                        zero_ub,
                                        "VSEL_TENSOR_TENSOR_MODE",
                                    )
                                    T.tile.cast(c_ub, sel_ub, "CAST_RINT", tile)
                                T.barrier_all()
                                if full:
                                    T.copy(c_ub, C[start])
                                else:
                                    if tail:
                                        if start < out_numel:  # noqa: SIM102
                                            if out_vec:
                                                T.copy(
                                                    c_ub[0:out_vec],
                                                    C[start : start + out_vec],
                                                )
                                            for lane in T.serial(out_blk):
                                                if out_vec + lane < tail:
                                                    C[start + out_vec + lane] = c_ub[
                                                        out_vec + lane
                                                    ]

            return main
        return kernel()

    def geometry(cap: int) -> tuple[int, int] | None:
        """``(tile, units_per_tile)`` for one UB budget candidate, or None.

        None means this budget cannot produce a legal suffix-broadcast tile --
        ``units * repeat`` would leave the compare/select operands off their
        256-byte block granularity.  The caller then falls back to the generic
        path rather than emitting a kernel whose operands are ragged.
        """
        if mode == "tail_broadcast":
            units, _ = pack_broadcast_units(repeat, unit_count, cap, in_bytes)
            # One staging cell per slot, so the slot count cannot exceed the
            # staging buffer.
            units = min(units, _STAGE_SLOTS)
            while units > 1 and (units * repeat) % grain:
                units -= 1
            tile = units * repeat
            return None if tile % grain else (tile, units)
        return fit_same_tile(cap, out_numel, hard_cap=TILE_HARD_CAP), 1

    def _generic_fallback():
        return _compile_predicate(
            a_numel, b_numel, out_shape, a_strides, b_strides, in_dtype,
            out_dtype, semantic_dtype, op_kind, "generic",
        )

    # Declared buffers adding up to under budget is not the same as fitting:
    # some intrinsics take an implicit scratch buffer that never appears in the
    # Python source, and the lowering places it in the same arena without
    # checking the total.  Read the emitted offsets back and step down the
    # ladder until they land inside the unified buffer (T195 acceptance 4).
    floor_geometry = geometry(TILE_GRAIN)
    compiled = None
    for candidate_cap in cap_ladder(ub_cap):
        plan = geometry(candidate_cap)
        if plan is None:
            break
        tile, units = plan
        try:
            compiled = build(tile, units)
        except Exception:
            # A suffix-broadcast tile that does not lower (an intrinsic missing
            # at this dtype, say) must not take the operator down: the generic
            # per-lane gather is slower but always correct.
            if mode == "tail_broadcast":
                return _generic_fallback()
            raise
        if ub_fits(compiled.get_kernel_source()):
            return compiled
        if (tile, units) == floor_geometry:
            break
    if mode == "tail_broadcast":
        return _generic_fallback()
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
    a_direct = math.prod(a_shape) == out_numel and _is_contiguous(out_shape, a_strides)
    b_direct = math.prod(b_shape) == out_numel and _is_contiguous(out_shape, b_strides)
    repeat, unit_count, broadcast_side = 1, 1, ""
    if a_direct and b_direct:
        mode = "same"
    else:
        # R346.  Exactly one operand broadcasting over a contiguous output suffix
        # is the manifest's ``cnn-feat-broadcast`` ([256,1,1] against
        # [16,256,56,56]), and it is the shape every one of these eight operators
        # was slowest on.  ``elementwise_binary.py`` has had this geometry since
        # R263 and R334 gave it to ``WhereFwdOp``; this is the same path.
        info = None
        if a_direct and not b_direct:
            info = _tail_broadcast_info(b_shape, out_shape, b_strides)
            broadcast_side = "b"
        elif b_direct and not a_direct:
            info = _tail_broadcast_info(a_shape, out_shape, a_strides)
            broadcast_side = "a"
        if info is None:
            mode, broadcast_side = "generic", ""
        else:
            mode = "tail_broadcast"
            repeat, unit_count = info
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
        repeat,
        unit_count,
        broadcast_side,
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

    invoke.compiled = compiled
    return invoke

# ``T.tile.select`` has no bfloat16 lowering on dav-2201 (measured,
# R195-data/probe_mixed_intrinsics.log, recorded in elementwise_mixed.py's
# ``_SELECT_DTYPES``).  ``torch.where`` picks between two bit patterns and
# performs no arithmetic, so bfloat16 rides in a float16 buffer: same width,
# and moving bits never inspects them.  That also makes the result BIT-EXACT,
# which the shipped fp32 round trip was not -- ``CAST_RINT`` turns a bf16 NaN
# payload 0x7fc0 into 0x7fff (same measurement).  WhereFwdOp's tolerance is
# atol=rtol=0 and its gate is ``torch.equal``, so bit-exactness is the point.
_WHERE_STORAGE = {"float16": "float16", "bfloat16": "float16", "float32": "float32"}

#: Element granularity every where tile has to respect.  128 float16 lanes is
#: 256 bytes, which is what ``T.tile.compare`` needs from its (float16) source.
_WHERE_GRAIN = 128


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
    repeat: int = 1,
    unit_count: int = 1,
    broadcast_side: str = "",
):
    """Compile a three-input where kernel whose select is a VECTOR select.

    R334.  The shipped kernel promoted both operands to fp32 and then ran::

        for lane in T.serial(tile):
            output32[lane] = T.if_then_else(condition_ub[lane] != 0, ...)

    on EVERY path -- contiguous, full tile, every shape.  That is one scalar
    iteration per output element for the whole tensor, and it is why WhereFwdOp
    was 60-170x slower than ``aclnnSWhere`` on every large case rather than only
    on the broadcast ones (coverage: 8M 5036 us, 16M 9835 us, 256M 156016 us,
    broadcast 13621 us -- a constant ~0.58 ns/element, i.e. shape-independent,
    which is what rules out a broadcast-materialisation explanation).

    The replacement is ``T.tile.compare`` + ``T.tile.select``, the same pair the
    masked-fill vector path already uses (elementwise_mixed.py).  A select moves
    bits, so there is no fp32 promotion at all and the result is bit-exact --
    see ``_WHERE_STORAGE`` for why bfloat16 rides in a float16 buffer.
    """
    out_numel = math.prod(out_shape)
    storage = _WHERE_STORAGE[dtype]
    value_bytes = _ELEMENT_BYTES[storage]
    all_contiguous = mode == "same"
    # --- Geometry ------------------------------------------------------
    # UB census, every buffer live on every path:
    #   condition_ub int8 | cond16 float16 (compare needs a float source; the
    #   int8 -> float16 cast is exact over the whole int8 range and int8 -> fp32
    #   does not lower on dav-2201) | input_ub, other_ub, output_ub in the
    #   select storage | packed compare mask at 1/8 byte per element.
    live_eighths = 8 * (1 + 2 + 3 * value_bytes) + 1
    # One extra element of staging for the suffix-broadcast source scalar; it is
    # a single element, not a per-element cost, so it does not enter live_eighths.
    ub_cap = (UB_BUDGET_BYTES * 8) // live_eighths

    def build(tile: int, units_per_tile: int = 1):
        # ``T.tile.compare`` wants operands that are a whole number of 256-byte
        # blocks (PROJECT_STATE 13.55 trap 3).  The compare source is ``cond16``
        # (float16), so 128 elements -- not TILE_GRAIN's 256 -- is the real
        # granularity, and it also keeps the packed mask byte-aligned and the
        # select operands whole-block.  Asserting the three byte counts rather
        # than a single element count: the suffix-broadcast geometry's tile is
        # ``units * repeat``, which has no reason to be a multiple of 256
        # (measured: the fp32 ``broadcast`` case picks 6272 = 49 * 128).
        assert tile % _WHERE_GRAIN == 0, tile
        assert (tile * 2) % 256 == 0, tile
        assert (tile * value_bytes) % 256 == 0, (tile, value_bytes)
        block_total = tile * 2
        if mode == "tail_broadcast":
            total_tiles = max(1, math.ceil(unit_count / units_per_tile))
            logical_blocks = max(1, math.ceil(total_tiles / 2))
        else:
            logical_blocks = max(1, math.ceil(out_numel / block_total))
        # Without the cap this was ``launch_blocks == logical_blocks`` and
        # ``grid_repeats == 1``: the loop below existed but never strided.
        launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
        grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)
        # --- Ragged tail (R334, same rule as elementwise_binary.py) --------
        # Every context start is a multiple of ``tile``, so the ONE ragged tile
        # is at ``(out_numel // tile) * tile`` and ``tail`` is compile-time.
        # A partial-extent copy moves WHOLE 32-byte blocks, so only the
        # ``*_vec`` prefix may ride ``T.copy`` -- for the STORE that is not an
        # over-read but an over-WRITE past the end of OUTPUT, which is why the
        # store uses ``val_vec`` too and never ``tail``.
        tail = out_numel % tile
        cond_blk = 32
        val_blk = 32 // value_bytes
        cond_vec = tail - tail % cond_blk
        val_vec = tail - tail % val_blk

        @tilelang.jit(out_idx=[-1])
        def kernel():
            @T.prim_func
            def main(
                # Signed int8 rather than uint8: TVM's block-bound analysis can
                # otherwise form a negative sentinel while indexing a uint8
                # buffer from the unsigned Kernel context (same reason
                # ``_compile_masked_fill`` takes the mask as int8).
                CONDITION: T.Tensor((condition_numel,), "int8"),
                INPUT: T.Tensor((input_numel,), storage),
                OTHER: T.Tensor((other_numel,), storage),
                OUTPUT: T.Tensor((out_numel,), storage),
            ):
                with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                    condition_ub = T.alloc_ub((tile,), "int8")
                    cond16 = T.alloc_ub((tile,), "float16")
                    input_ub = T.alloc_ub((tile,), storage)
                    other_ub = T.alloc_ub((tile,), storage)
                    output_ub = T.alloc_ub((tile,), storage)
                    packed = T.alloc_ub((tile // 8,), "uint8")
                    # One-element staging for the suffix-broadcast source
                    # scalar, kept allocated on every compile-time path so the
                    # frontend does not treat it as a branch-local name.
                    scalar_ub = T.alloc_ub((1,), storage)
                    with T.Scope("V"):
                        for grid_repeat in T.serial(grid_repeats):
                            logical_cid = cid + grid_repeat * launch_blocks
                            if logical_cid < logical_blocks:
                                start = logical_cid * block_total + vid * tile
                                # ``block_total`` is two tiles wide, so this is
                                # the flat index of the tile this context owns.
                                unit = (logical_cid * 2 + vid) * units_per_tile
                                full = start + tile <= out_numel
                                if mode == "tail_broadcast":
                                    # Exactly the geometry ``elementwise_binary``
                                    # already uses for ``[256,1,1]`` against
                                    # ``[16,256,56,56]``: the broadcast operand
                                    # contributes ONE scalar per packed unit,
                                    # blown up by ``T.tile.broadcast``, while the
                                    # direct operands ride a full-width T.copy.
                                    # The generic path this replaces re-read all
                                    # three operands one lane at a time.
                                    if start < out_numel:
                                        if broadcast_side != "condition":
                                            if full:
                                                T.copy(CONDITION[start], condition_ub)
                                            else:
                                                if cond_vec:
                                                    T.copy(
                                                        CONDITION[start : start + cond_vec],
                                                        condition_ub[0:cond_vec],
                                                    )
                                                for lane in T.serial(cond_blk):
                                                    if cond_vec + lane < tail:
                                                        condition_ub[cond_vec + lane] = CONDITION[
                                                            start + cond_vec + lane
                                                        ]
                                        if broadcast_side != "input":
                                            if full:
                                                T.copy(INPUT[start], input_ub)
                                            else:
                                                if val_vec:
                                                    T.copy(
                                                        INPUT[start : start + val_vec],
                                                        input_ub[0:val_vec],
                                                    )
                                                for lane in T.serial(val_blk):
                                                    if val_vec + lane < tail:
                                                        input_ub[val_vec + lane] = INPUT[
                                                            start + val_vec + lane
                                                        ]
                                        else:
                                            for slot in range(units_per_tile):
                                                scalar_ub[0] = INPUT[
                                                    (unit + slot) % input_numel
                                                ]
                                                T.tile.broadcast(
                                                    input_ub[
                                                        slot * repeat : (slot + 1) * repeat
                                                    ]
                                                    if units_per_tile > 1
                                                    else input_ub,
                                                    scalar_ub,
                                                )
                                        if broadcast_side != "other":
                                            if full:
                                                T.copy(OTHER[start], other_ub)
                                            else:
                                                if val_vec:
                                                    T.copy(
                                                        OTHER[start : start + val_vec],
                                                        other_ub[0:val_vec],
                                                    )
                                                for lane in T.serial(val_blk):
                                                    if val_vec + lane < tail:
                                                        other_ub[val_vec + lane] = OTHER[
                                                            start + val_vec + lane
                                                        ]
                                        else:
                                            for slot in range(units_per_tile):
                                                scalar_ub[0] = OTHER[
                                                    (unit + slot) % other_numel
                                                ]
                                                T.tile.broadcast(
                                                    other_ub[
                                                        slot * repeat : (slot + 1) * repeat
                                                    ]
                                                    if units_per_tile > 1
                                                    else other_ub,
                                                    scalar_ub,
                                                )
                                elif all_contiguous:
                                    if full:
                                        T.copy(CONDITION[start], condition_ub)
                                        T.copy(INPUT[start], input_ub)
                                        T.copy(OTHER[start], other_ub)
                                    else:
                                        # Lanes at or past ``tail`` are never
                                        # stored, so they are left as they are
                                        # rather than zeroed: ``T.tile.fill``
                                        # rejects int8 and a per-lane zeroing
                                        # loop is exactly the cost this change
                                        # removes.  A select reads bits, so
                                        # stale lanes cannot trap.
                                        if tail:
                                            # A Python int and a PrimExpr may
                                            # not share an ``and``; keep the
                                            # compile-time test on the outside.
                                            if start < out_numel:
                                                if cond_vec:
                                                    T.copy(
                                                        CONDITION[start : start + cond_vec],
                                                        condition_ub[0:cond_vec],
                                                    )
                                                for lane in T.serial(cond_blk):
                                                    if cond_vec + lane < tail:
                                                        condition_ub[cond_vec + lane] = CONDITION[
                                                            start + cond_vec + lane
                                                        ]
                                                if val_vec:
                                                    T.copy(
                                                        INPUT[start : start + val_vec],
                                                        input_ub[0:val_vec],
                                                    )
                                                    T.copy(
                                                        OTHER[start : start + val_vec],
                                                        other_ub[0:val_vec],
                                                    )
                                                for lane in T.serial(val_blk):
                                                    if val_vec + lane < tail:
                                                        input_ub[val_vec + lane] = INPUT[
                                                            start + val_vec + lane
                                                        ]
                                                        other_ub[val_vec + lane] = OTHER[
                                                            start + val_vec + lane
                                                        ]
                                else:
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
                                T.tile.cast(cond16, condition_ub, "CAST_NONE", tile)
                                # Mask bit set where the condition is true; a
                                # select takes src0 on a set bit (the polarity
                                # ``_compile_masked_fill`` relies on).
                                T.tile.compare(packed, cond16, 0.0, "NE")
                                T.tile.select(
                                    output_ub,
                                    packed,
                                    input_ub,
                                    other_ub,
                                    "VSEL_TENSOR_TENSOR_MODE",
                                )
                                T.barrier_all()
                                if full:
                                    T.copy(output_ub, OUTPUT[start])
                                else:
                                    if tail:
                                        if start < out_numel:  # noqa: SIM102
                                            if val_vec:
                                                T.copy(
                                                    output_ub[0:val_vec],
                                                    OUTPUT[start : start + val_vec],
                                                )
                                            for lane in T.serial(val_blk):
                                                if val_vec + lane < tail:
                                                    OUTPUT[start + val_vec + lane] = output_ub[
                                                        val_vec + lane
                                                    ]

            return main

        return kernel()

    # Declared buffers summing under budget is not the same as fitting: some
    # intrinsics carry implicit scratch the lowering places in the same arena
    # without checking.  Read the emitted offsets back and step down.
    def geometry(cap: int) -> tuple[int, int] | None:
        """``(tile, units_per_tile)`` for one UB budget candidate, or None.

        None means this budget cannot produce a legal suffix-broadcast tile --
        the packed unit count would leave ``units * repeat`` off the 128-lane
        compare granularity.  The caller then falls back to the generic path
        rather than emitting a kernel whose compare operand is ragged.
        """
        if mode == "tail_broadcast":
            units, _ = pack_broadcast_units(repeat, unit_count, cap, value_bytes)
            while units > 1 and (units * repeat) % _WHERE_GRAIN:
                units -= 1
            tile = units * repeat
            return None if tile % _WHERE_GRAIN else (tile, units)
        return fit_same_tile(cap, out_numel, hard_cap=TILE_HARD_CAP), 1

    floor_geometry = geometry(TILE_GRAIN)
    compiled = None
    for candidate_cap in cap_ladder(ub_cap):
        plan = geometry(candidate_cap)
        if plan is None:
            break
        tile, units = plan
        compiled = build(tile, units)
        if ub_fits(compiled.get_kernel_source()):
            return compiled
        if (tile, units) == floor_geometry:
            break
    if mode == "tail_broadcast":
        # Either no legal broadcast tile exists or none of them fit: the generic
        # per-lane gather is slow but always correct, so take it rather than
        # ship a kernel that does not lower.
        return _compile_where(
            condition_numel, input_numel, other_numel, out_shape,
            condition_strides, input_strides, other_strides, dtype, "generic",
        )
    usage = ub_usage(compiled.get_kernel_source()) if compiled else None
    raise ValueError(
        f"where/{dtype} mode={mode} needs {usage[0] if usage else '?'} UB bytes, "
        f"over the {usage[1] if usage else '?'} available; no smaller tile fits"
    )


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
    operands = (
        ("condition", condition_shape, condition_strides),
        ("input", input_shape, input_strides),
        ("other", other_shape, other_strides),
    )
    direct = {
        name: math.prod(shape) == out_numel and _is_contiguous(out_shape, strides)
        for name, shape, strides in operands
    }
    mode, repeat, unit_count, side = "generic", 1, 1, ""
    if all(direct.values()):
        mode = "same"
    else:
        # R334.  One suffix-broadcast operand against two direct ones is the
        # ``broadcast`` workload ([16,256,56,56] x [256,1,1] x [16,256,56,56]),
        # and it is the only shape left on the generic per-lane gather.  The
        # condition is excluded on purpose: ``T.tile.broadcast`` is driven from a
        # UB buffer of the operand's own storage type and the condition's is
        # int8, which this file has no verified broadcast lowering for.
        broadcast_names = [name for name, ok in direct.items() if not ok]
        if broadcast_names == ["input"] or broadcast_names == ["other"]:
            name = broadcast_names[0]
            shape, strides = next(
                (shp, st) for nm, shp, st in operands if nm == name
            )
            info = _tail_broadcast_info(shape, out_shape, strides)
            if info is not None:
                mode, (repeat, unit_count), side = "tail_broadcast", info, name
    dtype_name = str(dtype).replace("torch.", "")
    compiled = _compile_where(
        math.prod(condition_shape),
        math.prod(input_shape),
        math.prod(other_shape),
        out_shape,
        condition_strides,
        input_strides,
        other_strides,
        dtype_name,
        mode,
        repeat,
        unit_count,
        side,
    )
    # bfloat16 rides in a float16 buffer (see ``_WHERE_STORAGE``); reinterpret
    # on the way in and back on the way out.  Nothing in between reads a value.
    reinterpret = torch.float16 if dtype == torch.bfloat16 else None

    def invoke(condition, input, other):
        if (
            tuple(condition.shape) != condition_shape
            or tuple(input.shape) != input_shape
            or tuple(other.shape) != other_shape
        ):
            raise ValueError("WhereFwdOp kernel shape mismatch")
        condition_flat = condition.reshape(-1)
        condition_flat = (
            condition_flat.view(torch.int8)
            if condition_flat.dtype in (torch.bool, torch.uint8)
            else condition_flat
        )
        input_flat = input.reshape(-1)
        other_flat = other.reshape(-1)
        if reinterpret is not None:
            input_flat = input_flat.view(reinterpret)
            other_flat = other_flat.view(reinterpret)
        result = compiled(condition_flat, input_flat, other_flat)
        if reinterpret is not None:
            result = result.view(dtype)
        return result if tuple(out_shape) == (result.numel(),) else result.reshape(out_shape)

    invoke.compiled = compiled
    return invoke
