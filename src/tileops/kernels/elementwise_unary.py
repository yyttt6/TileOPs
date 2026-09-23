"""Reusable Ascend vector template for unary elementwise operators.

The kernel is deliberately shape-specialized, just like the binary template:
the input is flattened only as a metadata view and every block owns a bounded
tile.  Both vector contexts participate in the address calculation.
"""

import math
import os
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count


FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
PREDICATE_DTYPES = (torch.uint8,)
INT_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
MAX_BLOCK_COUNT = 65535

# Same VECCALC watermark the binary template reuses: the lowering reserves
# 196352 bytes and the shipped geometry there proved 180224 of them safe.
UB_BUDGET_BYTES = 180224

# ``T.tile.compare`` emits a packed one-bit mask and dav-2201 has a 128-lane
# native repeat, so every tile must stay a whole number of mask blocks.  This
# is the *granularity*, not the size -- reading it as the size is what pinned
# the whole family at 128 elements per block (R194 / PROJECT_STATE 13.49).
MASK_GRAIN = 128

# R194 probe (exp, 16M elements, card 2): bandwidth stops improving past a
# 16384-element tile, and the 22528 point measured slightly slower.
TILE_HARD_CAP = 16384

#: R355 (T355 #58).  ``logical_not`` widened the packed one-bit compare mask to
#: the byte-backed bool output with ONE SCALAR ITERATION PER OUTPUT ELEMENT, on
#: every path -- exactly the defect R346 found in ``elementwise_predicate.py`` and
#: fixed there.  ``elementwise-256M/bool`` measured 75870 us against a 450 us
#: vendor (ratio 0.0059, R355-data/logs/lnot-before-p1-cases.txt).
#:
#: The replacement is R346's, unchanged: ``T.tile.select`` between two constant
#: float16 tiles followed by ``T.tile.cast`` down to the output byte.  float16 is
#: forced by measurement, not taste -- on dav-2201 ``cast float16 -> uint8`` lowers
#: in all four rounding modes while ``cast float32 -> uint8`` does not lower at all
#: (R346-data/probe/p1_intrinsics.log).  A compare mask is a packed bitmask with no
#: dtype of its own, so the expansion width is free to be float16, and uint8's whole
#: value range is exact there (R346 ``_F16_EXACT``).
_EXPAND_DTYPE = "float16"

#: Which op_kinds take the vector expansion.  Deliberately ONE entry: adding an
#: op here changes its ``live_eighths`` and therefore its tile, its block count and
#: its measured number.  ``isnan`` / ``isinf`` / ``isfinite`` carry the identical
#: scalar loop and would be fixed by adding their names -- R355 left them out
#: because they are not in the 150-row scope and moving their numbers was not
#: this round's mandate.  See R355.md section 4.4.
_NEEDS_EXPAND = frozenset({"logical_not"})

#: R355 control arm.  ``before`` compiles the shipped scalar expansion so the same
#: tree can run both arms on the same card (``_tuning_common.md`` section 4).
_LOGICAL_NOT_ARM = os.environ.get("TILEOPS_LOGICAL_NOT_ARM", "t355").strip().lower()
if _LOGICAL_NOT_ARM not in {"t355", "before"}:
    raise ValueError(
        f"TILEOPS_LOGICAL_NOT_ARM must be 't355' or 'before', got {_LOGICAL_NOT_ARM!r}"
    )

#: R360 (T360) control arm.  ``before`` compiles the shipped expressions for
#: ``rsqrt`` / ``reciprocal`` / ``erf`` / the rounding four, so the same tree can run
#: both arms on the same card (``_tuning_common.md`` section 4).  Every op_kind NOT
#: named in the four ``_T360_*`` sets below emits byte-for-byte the same kernel on
#: both arms -- that invariant is what keeps the 18 already-passing operators out of
#: this change, and ``R360-data/probe/p3_tile_audit.py`` checks it mechanically.
_T360_ARM = os.environ.get("TILEOPS_T360_ARM", "t360").strip().lower()
if _T360_ARM not in {"t360", "before"}:
    raise ValueError(
        f"TILEOPS_T360_ARM must be 't360' or 'before', got {_T360_ARM!r}"
    )

#: R360.  The op_kinds whose expression this round replaces.  Membership is what
#: decides both the emitted code AND the scratch budget, so the two must be read
#: from the same place; see ``_NEEDS_*`` below.
_T360_ROUNDING = frozenset({"floor", "ceil", "round", "trunc"})
#: Which of them need a whole tile of 1.0 for ``T.tile.div`` (which, unlike ``mul``
#: and ``add``, has no scalar-operand form -- ``ascend_tile.py:1059`` takes a Buffer).
_T360_NEEDS_ONE = frozenset({"rsqrt", "reciprocal", "erf"})

# R194 probe: launch-block count is the dominant cost -- one block per tile
# (the shipped behaviour) spends ~0.11 us per block and caps exp at 15.7 GB/s,
# while a grid-stride loop over 48-192 blocks reaches 200-696 GB/s at the same
# tile.  48-192 is a plateau; 384 starts to fall off and 1536 collapses.
LAUNCH_BLOCK_CAP = 48

_ELEMENT_BYTES = {
    "float16": 2, "bfloat16": 2, "float32": 4,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "uint8": 1, "bool": 1,
}

# Which scratch buffers each expression actually reads or writes.  Anything not
# listed here keeps a one-mask-block allocation so it does not consume tile
# budget; the branch that would touch it is not emitted for that op.  Derived
# by reading every ``op_kind`` branch in ``_compile_unary`` -- keep in sync.
# T263: the inverse-trig trio evaluates a Horner polynomial with a range
# reduction, so it needs the same four fp32 tiles and two packed masks the
# rounding four already budget for.  ``tan`` only needs a second and third
# tile (sin, cos, then a divide).
_NEEDS_TMP2 = frozenset({
    "rsqrt", "floor", "ceil", "round", "trunc", "erf", "sign",
    "tan", "asin", "acos", "atan",
})
_NEEDS_TMP3 = frozenset({
    "rsqrt", "floor", "ceil", "round", "trunc", "erf", "sign",
    "tan", "asin", "acos", "atan",
})
_NEEDS_TMP4 = frozenset({
    "floor", "ceil", "round", "trunc", "erf",
    "asin", "acos", "atan",
})
_NEEDS_I32 = frozenset({"floor", "ceil", "round", "trunc"})
_NEEDS_MASK = frozenset({
    "floor", "ceil", "round", "trunc", "erf", "sign",
    "isnan", "isinf", "isfinite", "logical_not",
    "asin", "acos", "atan",
})
# R360.  The ``t360`` arm stops using some of the above and starts using ``one``.
# Expressed as a diff against the shipped sets rather than as new literals, so that
# an op_kind this round does not touch cannot change budget by accident:
#   rsqrt      Sqrt+Div needs one extra tile (tmp2) and ``one``; the Newton scratch
#              (tmp3) and nothing else goes away.
#   reciprocal gains ``one`` only.
#   erf        gains ``one``; its tmp3/tmp4 are still the Horner scratch.
#   rounding4  the int32 buffer and the |x|<threshold mask both go away with the
#              overflow guard (R360-data/probe/p1_semantics.txt q1); tmp2 goes with
#              the redundant inner select.  tmp3/tmp4/mask3 stay for the signed-zero
#              repair, which q1 proved is still required.
if _T360_ARM == "t360":
    _NEEDS_TMP2 = (_NEEDS_TMP2 - _T360_ROUNDING)
    _NEEDS_TMP3 = (_NEEDS_TMP3 - {"rsqrt"})
    _NEEDS_I32 = _NEEDS_I32 - _T360_ROUNDING
    _NEEDS_MASK = _NEEDS_MASK - _T360_ROUNDING
_NEEDS_ONE = _T360_NEEDS_ONE if _T360_ARM == "t360" else frozenset()
_NEEDS_ZERO = frozenset({"neg", "erf"})
if _T360_ARM == "t360":
    # R360: the ``t360`` erf writes ``v * -1`` where the shipped one wrote ``zero - v``,
    # so it no longer reads ``zero`` at all -- which also removes one vector fill per
    # tile.  ``neg`` keeps it (touching ``neg`` would move an operator that passes).
    _NEEDS_ZERO = _NEEDS_ZERO - {"erf"}
# T263: ``T.tile.sin`` / ``T.tile.cos`` take an *optional* public ``tmp``
# arena, and when it is not passed the lowering reserves one itself -- one
# compute-dtype tile per call, which this template's budget never counted.
# ``sin`` and ``cos`` alone happen to fit anyway (two live tiles), but ``tan``
# issues both and holds four, and bf16 tan overran the 196352-byte VECCALC
# reservation and died with a hard aicore exception (507015) while fp16 and
# fp32 tan survived -- exactly the three-way split the arithmetic predicts:
#     bf16 tan   8192 x 20 B = 163840 + 2 x 8192 x 4 B = 229376  > 196352  X
#     fp16 tan   8192 x 12 B =  98304 + 2 x 8192 x 2 B = 131072  < 180224  ok
#     fp32 tan   4096 x 24 B =  98304 + 2 x 4096 x 4 B = 131072  < 180224  ok
# Charging the two arenas here shrinks bf16 tan's tile to 4096 and fixes it.
# R263-data/06-tan-bf16-ub-overflow.txt has the raw before/after.
# NOTE deliberately scoped to ``tan``: adding the charge to the shipped
# ``sin`` / ``cos`` would move their tile and therefore their measured number,
# which T263 forbids.
_NEEDS_TRIG_ARENA = frozenset({"tan"})
# A third packed mask, only for the rounding four: it holds "the rounded value is
# zero", which is what the signed-zero repair selects on.
#
# 🚨 R360: this is deliberately NOT ``= _NEEDS_I32``.  It used to be, and the ``t360``
# arm removes the rounding four from ``_NEEDS_I32`` (no int32 round trip any more) --
# which silently shrank ``mask3`` from ``tile`` to one MASK_GRAIN while
# ``T.tile.compare`` still wrote ``tile/8`` bytes into it.  bench1.py caught it as
# floor max_abs_err=6.0 / max_rel_err=1.0 on all five cases
# (R360-data/logs/floor-mask3-bug/).  The signed-zero repair survives the rewrite,
# so mask3's membership must be stated from the rounding set itself.
_NEEDS_MASK3 = _T360_ROUNDING


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _supported_message(
    op_name: str, dtype: torch.dtype, supported: tuple[torch.dtype, ...]
) -> str:
    names = ", ".join(str(item) for item in supported)
    return f"{op_name} does not support dtype {dtype}; supported dtypes are [{names}], but received input dtype {dtype}"


@lru_cache(maxsize=128)
def _compile_unary(n_total: int, input_dtype: str, output_dtype: str, op_kind: str):
    # Ascend vector transcendental instructions operate on fp32 reliably for
    # BF16.  Keep the source/destination buffers in their public dtypes and use
    # fp32 scratch for composed expressions and predicates.
    bool_logical = op_kind == "logical_not" and input_dtype == "uint8"
    # T263: asin/acos/atan evaluate a 5-term Horner polynomial after a range
    # reduction.  In fp16 the reduction alone loses more than the manifest
    # tolerance, so these compute in fp32 scratch exactly like erf does and
    # cast once on writeback.  ``log2`` and ``tan`` stay on the native dtype
    # (they are one intrinsic plus one arithmetic op, the same shape as
    # ``log`` / ``sin``), which keeps their tile as wide as ``log``'s.
    fp32_unary = op_kind in {
        "floor", "ceil", "round", "trunc", "erf", "rsqrt",
        "asin", "acos", "atan",
    }
    compute_dtype = (
        "float32"
        if input_dtype == "bfloat16"
        or op_kind == "reciprocal"
        and input_dtype not in {"float16", "float32"}
        or bool_logical
        or fp32_unary
        else input_dtype
    )
    # R355.  Which path this op_kind's bit->byte expansion takes.  Read the arm
    # here rather than at the call site so the shipped scalar template stays
    # reachable byte-for-byte for the control arm.
    vector_expand = op_kind in _NEEDS_EXPAND and _LOGICAL_NOT_ARM != "before"
    if vector_expand and bool_logical:
        # A bool/uint8 input has to reach ``T.tile.compare`` in a dtype that has a
        # compare lowering.  R346 measured bool/uint8/int8 as exact in float16
        # (``_F16_EXACT``), and float16 is already the expansion width, so the
        # whole chain stays at one width instead of the shipped float32 scratch
        # (which has no ``cast -> uint8`` lowering at all).
        compute_dtype = _EXPAND_DTYPE
    needs_compute_cast = compute_dtype != input_dtype
    is_predicate = output_dtype == "uint8"
    integral_threshold = {
        "float16": 2048.0,
        "bfloat16": 256.0,
        "float32": 8388608.0,
    }.get(input_dtype, 0.0)

    # --- Geometry (R194) -------------------------------------------------
    # Two things decide throughput here and the shipped template got both
    # wrong: the tile was pinned to the mask granularity instead of the UB
    # budget, and every block handled exactly one tile, so the block count
    # grew with the tensor and ~0.11 us of per-block dispatch dominated
    # everything.  Allocate the tile from the buffers this expression really
    # uses, then cover the tensor with a bounded grid-stride loop.
    needs_tmp2 = op_kind in _NEEDS_TMP2
    needs_tmp3 = op_kind in _NEEDS_TMP3
    needs_tmp4 = op_kind in _NEEDS_TMP4
    needs_i32 = op_kind in _NEEDS_I32
    needs_mask = op_kind in _NEEDS_MASK
    needs_zero = op_kind in _NEEDS_ZERO
    needs_mask3 = op_kind in _NEEDS_MASK3
    needs_trig_arena = op_kind in _NEEDS_TRIG_ARENA
    needs_one = op_kind in _NEEDS_ONE
    _compute_bytes = _ELEMENT_BYTES[compute_dtype]
    # Eighths of a byte per element, because the two packed masks cost 1/8
    # of a byte each and integer arithmetic must stay exact.
    live_eighths = 8 * (
        _ELEMENT_BYTES[input_dtype]
        + _ELEMENT_BYTES[output_dtype]
        + _compute_bytes                                  # x_calc
        + _compute_bytes                                  # tmp
        + (_compute_bytes if needs_tmp2 else 0)
        + (_compute_bytes if needs_tmp3 else 0)
        + (_compute_bytes if needs_tmp4 else 0)
        + (4 if needs_i32 else 0)
        + (_compute_bytes if needs_zero else 0)
        + (_compute_bytes if needs_one else 0)          # R360: T.tile.div's 1.0 operand
        + (2 * _compute_bytes if needs_trig_arena else 0)   # sin + cos arenas
        # R355: expand_sel + expand_one + expand_zero, float16, and only for the
        # op_kinds in ``_NEEDS_EXPAND``.  Every other op_kind's total is unchanged.
        + (3 * _ELEMENT_BYTES[_EXPAND_DTYPE] if vector_expand else 0)
    ) + (2 if needs_mask else 0) + (1 if needs_mask3 else 0)
    ub_cap = (UB_BUDGET_BYTES * 8) // live_eighths
    # Never widen past what the tensor needs: a tile larger than half the
    # tensor only lengthens the scalar tail loop, and small shapes are
    # already at parity with the vendor because both sides are launch-bound.
    needed = max(MASK_GRAIN, math.ceil(n_total / 2))
    ceiling = min(ub_cap, TILE_HARD_CAP, needed)
    ceiling = max(MASK_GRAIN, ceiling - ceiling % MASK_GRAIN)
    # Prefer a tile that divides the tensor.  A single ragged tile falls into
    # the per-lane scalar tail below, and that one tile is on the critical
    # path of the whole launch: measured, fp32 exp at 16M went 2.65x slower
    # (113 us -> 300 us) purely because 745 blocks x 22528 overshot 2**24 by
    # one partial tile, while the fp16 geometry divided exactly and paid
    # nothing (R194 / PROJECT_STATE 13.50).  Give up at most half the tile
    # width chasing a divisor -- below that the narrower DMA costs more than
    # the tail.
    tile = ceiling
    candidate = ceiling
    while candidate >= max(MASK_GRAIN, ceiling // 2):
        if n_total % (2 * candidate) == 0:
            tile = candidate
            break
        candidate -= MASK_GRAIN
    scratch2 = tile if needs_tmp2 else MASK_GRAIN
    scratch3 = tile if needs_tmp3 else MASK_GRAIN
    scratch4 = tile if needs_tmp4 else MASK_GRAIN
    scratch_i32 = tile if needs_i32 else MASK_GRAIN
    scratch_mask = tile if needs_mask else MASK_GRAIN
    scratch_mask3 = tile if needs_mask3 else MASK_GRAIN
    scratch_zero = tile if needs_zero else MASK_GRAIN
    scratch_one = tile if needs_one else MASK_GRAIN
    scratch_expand = tile if vector_expand else MASK_GRAIN
    block_total = tile * 2
    logical_blocks = max(1, math.ceil(n_total / block_total))
    launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
    grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)

    @tilelang.jit(out_idx=[-1])
    def kernel():
        @T.prim_func
        def main(
            X: T.Tensor((n_total,), input_dtype),
            Y: T.Tensor((n_total,), output_dtype),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                x_ub = T.alloc_ub((tile,), input_dtype)
                x_calc = T.alloc_ub((tile,), compute_dtype)
                y_ub = T.alloc_ub((tile,), output_dtype)
                tmp = T.alloc_ub((tile,), compute_dtype)
                tmp2 = T.alloc_ub((scratch2,), compute_dtype)
                tmp3 = T.alloc_ub((scratch3,), compute_dtype)
                tmp4 = T.alloc_ub((scratch4,), compute_dtype)
                rounded_i32 = T.alloc_ub((scratch_i32,), "int32")
                # Compare emits a packed one-bit mask (8 lanes per byte).
                mask = T.alloc_ub((scratch_mask // 8,), "uint8")
                mask2 = T.alloc_ub((scratch_mask // 8,), "uint8")
                mask3 = T.alloc_ub((scratch_mask3 // 8,), "uint8")
                zero = T.alloc_ub((scratch_zero,), compute_dtype)
                # R360.  ``T.tile.div`` has no scalar-operand form, so 1/x needs a
                # whole tile of ones.  Allocated on every path at one mask block so
                # the frontend never sees a branch-local name, exactly like R355's
                # expansion tiles below.
                one = T.alloc_ub((scratch_one,), compute_dtype)
                # R355.  The vector bit->byte expansion's three float16 tiles.
                # Allocated on every compile-time path at one mask block so the
                # frontend never sees a branch-local name.
                expand_sel = T.alloc_ub((scratch_expand,), _EXPAND_DTYPE)
                expand_one = T.alloc_ub((scratch_expand,), _EXPAND_DTYPE)
                expand_zero = T.alloc_ub((scratch_expand,), _EXPAND_DTYPE)

                with T.Scope("V"):
                    if vector_expand:
                        # Hoisted out of the grid-stride loop on purpose: the two
                        # select operands are loop invariants, so this is two vector
                        # fills per BLOCK, not per tile (R346's wording and reason).
                        T.tile.fill(expand_one, 1.0)
                        T.tile.fill(expand_zero, 0.0)
                        T.barrier_all()
                    if needs_one:
                        # R360.  Same reason as R346's two fills above: the divisor's
                        # numerator is a loop invariant, so this is ONE vector fill per
                        # BLOCK, not one per tile.  The shipped ``rsqrt`` paid the
                        # opposite price -- its ``T.tile.fill(tmp3, 1.5)`` sat inside
                        # the Newton loop and so ran twice per tile
                        # (R360-data/p0-ir/src-rsqrt-float16.cpp:59).
                        T.tile.fill(one, 1.0)
                        T.barrier_all()
                    # A bounded grid-stride loop: ``launch_blocks`` blocks walk the
                    # tensor instead of one block per tile.  Per-block dispatch is
                    # ~0.11 us and used to be the whole cost (R194 / 13.49); a loop
                    # iteration is nearly free by comparison.
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            start = logical_cid * block_total + vid * tile
                            full = start + tile <= n_total
                            if full:
                                T.copy(X[start], x_ub)
                            else:
                                if input_dtype in {"int64", "uint8"}:
                                    for lane in T.serial(tile):
                                        idx = start + lane
                                        if idx < n_total:
                                            x_ub[lane] = X[idx]
                                        else:
                                            x_ub[lane] = 0
                                else:
                                    T.tile.fill(x_ub, 0)
                                    for lane in T.serial(tile):
                                        idx = start + lane
                                        if idx < n_total:
                                            x_ub[lane] = X[idx]
                            T.barrier_all()

                            if op_kind == "bitwise_not" or (
                                bool_logical and not vector_expand
                            ):
                                pass
                            elif needs_compute_cast:
                                T.tile.cast(x_calc, x_ub, "CAST_NONE", tile)
                            else:
                                # A UB assignment is a metadata-level alias in the
                                # frontend; the following operations overwrite only tmp
                                # buffers and never mutate the input tensor.
                                T.copy(x_ub, x_calc)

                            if needs_zero:
                                T.tile.fill(zero, 0.0)
                            if op_kind == "abs":
                                T.tile.abs(tmp, x_calc)
                            elif op_kind == "neg":
                                T.tile.sub(tmp, zero, x_calc)
                            elif op_kind == "exp":
                                T.tile.exp(tmp, x_calc)
                            elif op_kind == "log":
                                T.tile.ln(tmp, x_calc)
                            elif op_kind == "log2":
                                # log2(x) = ln(x) * log2(e).  One extra vector
                                # multiply on top of the ``log`` expression, so
                                # this stays on ``log``'s native-dtype path and
                                # inherits its tile width.
                                T.tile.ln(tmp, x_calc)
                                T.tile.mul(tmp, tmp, 1.4426950408889634)
                            elif op_kind == "tan":
                                # dav-2201 has sin and cos intrinsics but no tan,
                                # so the quotient is formed explicitly.  Both
                                # sources live in their own scratch tiles because
                                # ``T.tile.div`` is only exercised elsewhere with
                                # a destination distinct from both operands.
                                T.tile.sin(tmp2, x_calc)
                                T.tile.cos(tmp3, x_calc)
                                T.tile.div(tmp, tmp2, tmp3)
                            elif op_kind == "atan":
                                # Range-reduce to [0, 1] with
                                # atan(a) = pi/2 - atan(1/a) for a > 1, evaluate
                                # the odd Hastings/Cephes minimax polynomial
                                # there (|err| <= ~1e-7 in fp32), then restore
                                # the sign.  Every step is a whole-tile vector
                                # op: a per-lane loop here is what made the
                                # rounding four scalar and cost 40x (R194).
                                T.tile.abs(tmp2, x_calc)
                                T.tile.compare(mask, tmp2, 1.0, "GT")
                                T.tile.fill(tmp3, 1.0)
                                T.tile.div(tmp4, tmp3, tmp2)
                                T.tile.select(
                                    tmp3, mask, tmp4, tmp2, "VSEL_TENSOR_TENSOR_MODE"
                                )
                                T.tile.mul(tmp4, tmp3, tmp3)
                                # Degree-6-in-z^2 minimax fit of atan(z)/z on
                                # [0, 1]: |err| <= 5.8e-7, 25x under the fp32
                                # manifest tolerance.  The five-term
                                # Abramowitz-Stegun 4.4.49 set that this
                                # replaced is only good to 1.16e-5 and measured
                                # 1.15e-5 on device -- inside tolerance but with
                                # no margin (R263-data/03-atan-poly-before.txt).
                                # Each extra term is two vector ops; seven is
                                # where the accuracy stops being the binding
                                # constraint.
                                T.tile.fill(tmp2, 0.008006898229)
                                T.tile.mul(tmp2, tmp2, tmp4)
                                T.tile.add(tmp2, tmp2, -0.03744297094)
                                T.tile.mul(tmp2, tmp2, tmp4)
                                T.tile.add(tmp2, tmp2, 0.08435407574)
                                T.tile.mul(tmp2, tmp2, tmp4)
                                T.tile.add(tmp2, tmp2, -0.1351217486)
                                T.tile.mul(tmp2, tmp2, tmp4)
                                T.tile.add(tmp2, tmp2, 0.198873209)
                                T.tile.mul(tmp2, tmp2, tmp4)
                                T.tile.add(tmp2, tmp2, -0.3332701333)
                                T.tile.mul(tmp2, tmp2, tmp4)
                                T.tile.add(tmp2, tmp2, 0.9999994166)
                                T.tile.mul(tmp2, tmp2, tmp3)
                                T.tile.fill(tmp4, 1.5707963267948966)
                                T.tile.sub(tmp4, tmp4, tmp2)
                                T.tile.select(
                                    tmp3, mask, tmp4, tmp2, "VSEL_TENSOR_TENSOR_MODE"
                                )
                                T.tile.compare(mask2, x_calc, 0.0, "LT")
                                T.tile.mul(tmp4, tmp3, -1.0)
                                T.tile.select(
                                    tmp, mask2, tmp4, tmp3, "VSEL_TENSOR_TENSOR_MODE"
                                )
                            elif op_kind in {"asin", "acos"}:
                                # Cephes single-precision asinf: reduce with
                                # asin(a) = pi/2 - 2*asin(sqrt((1-a)/2)) above
                                # 0.5 so the polynomial argument stays where the
                                # minimax fit is valid, then restore the sign.
                                # acos is pi/2 - asin; the subtraction cancels
                                # to ~1e-7 absolute, which the manifest fp32
                                # tolerance (atol 1e-5) absorbs.
                                #
                                # T263: measured 3.5e-7 max absolute error
                                # against a float64 reference over |x| <= 1.
                                # The 5.6e-5 disagreement with ``torch.asin``
                                # on NPU is the *vendor* op being wrong -- it
                                # peaks at 1/sqrt(2), where aclnnAsin evidently
                                # switches branches (R263-data/05-asin-fp64-
                                # cross-check.txt: ours 3.5e-7, torch_npu
                                # 5.6e-5, torch CPU 2e-8).  Holding the branch
                                # mask live across the sqrt/Horner chain was
                                # ruled out as a cause: recomputing it before
                                # each select left the number bit-identical.
                                T.tile.abs(tmp2, x_calc)
                                T.tile.compare(mask, tmp2, 0.5, "GT")
                                T.tile.mul(tmp3, tmp2, tmp2)
                                T.tile.fill(tmp4, 1.0)
                                T.tile.sub(tmp4, tmp4, tmp2)
                                T.tile.mul(tmp4, tmp4, 0.5)
                                T.tile.select(
                                    tmp, mask, tmp4, tmp3, "VSEL_TENSOR_TENSOR_MODE"
                                )
                                T.tile.sqrt(tmp4, tmp)
                                T.tile.select(
                                    tmp3, mask, tmp4, tmp2, "VSEL_TENSOR_TENSOR_MODE"
                                )
                                T.tile.fill(tmp4, 0.042163199048)
                                T.tile.mul(tmp4, tmp4, tmp)
                                T.tile.add(tmp4, tmp4, 0.024181311049)
                                T.tile.mul(tmp4, tmp4, tmp)
                                T.tile.add(tmp4, tmp4, 0.045470025998)
                                T.tile.mul(tmp4, tmp4, tmp)
                                T.tile.add(tmp4, tmp4, 0.074953002686)
                                T.tile.mul(tmp4, tmp4, tmp)
                                T.tile.add(tmp4, tmp4, 0.16666752422)
                                T.tile.mul(tmp4, tmp4, tmp)
                                T.tile.mul(tmp4, tmp4, tmp3)
                                T.tile.add(tmp4, tmp4, tmp3)
                                T.tile.fill(tmp2, 1.5707963267948966)
                                T.tile.mul(tmp3, tmp4, 2.0)
                                T.tile.sub(tmp2, tmp2, tmp3)
                                T.tile.select(
                                    tmp3, mask, tmp2, tmp4, "VSEL_TENSOR_TENSOR_MODE"
                                )
                                T.tile.compare(mask2, x_calc, 0.0, "LT")
                                T.tile.mul(tmp4, tmp3, -1.0)
                                T.tile.select(
                                    tmp, mask2, tmp4, tmp3, "VSEL_TENSOR_TENSOR_MODE"
                                )
                                if op_kind == "acos":
                                    T.tile.mul(tmp2, tmp, -1.0)
                                    T.tile.add(tmp, tmp2, 1.5707963267948966)
                            elif op_kind == "log1p":
                                T.tile.add(tmp, x_calc, 1.0)
                                T.tile.ln(tmp, tmp)
                            elif op_kind == "expm1":
                                T.tile.exp(tmp, x_calc)
                                T.tile.sub(tmp, tmp, 1.0)
                            elif op_kind == "sqrt":
                                T.tile.sqrt(tmp, x_calc)
                            elif op_kind == "rsqrt":
                                if _T360_ARM == "before":
                                    T.tile.rsqrt(tmp, x_calc)
                                    # Ascend's vector rsqrt is a fast approximation.  Keep
                                    # both Newton corrections in fp32 scratch before the
                                    # public dtype cast.
                                    for _ in T.serial(2):
                                        T.tile.mul(tmp2, tmp, tmp)
                                        T.tile.mul(tmp2, tmp2, x_calc)
                                        T.tile.mul(tmp2, tmp2, 0.5)
                                        T.tile.fill(tmp3, 1.5)
                                        T.tile.sub(tmp3, tmp3, tmp2)
                                        T.tile.mul(tmp, tmp, tmp3)
                                else:
                                    # R360.  The comment above is right that ``T.tile.rsqrt``
                                    # is an estimate -- measured 3.257e-03 relative, needing
                                    # BOTH Newton steps to reach 1.72 ulp, i.e. 13 vector
                                    # passes over every tile.  But ``T.tile.div`` is a single
                                    # pass at 0.76 ulp, so 1/sqrt(x) built from the exact
                                    # ``Sqrt`` and one ``Div`` is both 11 passes cheaper and
                                    # MORE accurate than the Newton chain it replaces.
                                    # Measured three ways in R360-data/probe/p1_semantics.txt
                                    # (q2 and q4).  ``tmp2`` holds sqrt(x) because ``Div``'s
                                    # destination is kept distinct from both operands, the
                                    # same rule ``tan`` follows below.
                                    T.tile.sqrt(tmp2, x_calc)
                                    T.tile.div(tmp, one, tmp2)
                            elif op_kind == "reciprocal":
                                if _T360_ARM == "before":
                                    T.tile.reciprocal(tmp, x_calc)
                                else:
                                    # R360.  ``T.tile.reciprocal`` is the same estimate
                                    # instruction as ``rsqrt`` and carries 2.846e-03 relative
                                    # error with no correction at all, against an fp32
                                    # tolerance of 1e-5 -- which is why ``ReciprocalFwdOp``
                                    # is recorded `blocked / correctness failed` while its
                                    # ratios (0.937/0.948) would otherwise pass.  ``Div`` is
                                    # one pass, 5.763e-08, and fixes it.
                                    T.tile.div(tmp, one, x_calc)
                            elif op_kind == "sin":
                                T.tile.sin(tmp, x_calc)
                            elif op_kind == "cos":
                                T.tile.cos(tmp, x_calc)
                            elif op_kind == "sigmoid":
                                T.tile.sigmoid(tmp, x_calc)
                            elif op_kind == "tanh":
                                T.tile.mul(tmp, x_calc, 2.0)
                                T.tile.sigmoid(tmp, tmp)
                                T.tile.mul(tmp, tmp, 2.0)
                                T.tile.sub(tmp, tmp, 1.0)
                            elif op_kind in {"floor", "ceil", "round", "trunc"}:
                                # The rounding mode is effective on float-to-int casts.
                                # Values outside this dtype-specific interval are already
                                # integral in the source format, so preserve them instead
                                # of risking integer overflow during the conversion.
                                #
                                # R194: the two selects below used to be
                                # ``T.serial(tile)`` lane loops over the packed
                                # masks, which made the whole expression scalar --
                                # 3.0 GB/s against a vendor baseline sitting on the
                                # HBM plateau.  That cost is proportional to the
                                # element count, so no tile or grid change moves it;
                                # only vectorizing the selects does.  Semantics are
                                # preserved exactly, signed-zero repair included.
                                if _T360_ARM == "t360":
                                    # R360.  ``compute_dtype`` is float32 on every path
                                    # that reaches this branch (bfloat16 via the input
                                    # dtype, float16 via ``fp32_unary``, float32 natively;
                                    # integral dtypes are rewritten to ``identity`` in
                                    # families/elementwise_unary_math.py:19-21).  And
                                    # AscendC lowers a float32->float32 ``Cast`` with a
                                    # round mode to ONE instruction, which R360 measured
                                    # bitwise-identical to a float64 reference on 2048
                                    # inputs including +-inf, NaN, 1e20 and +-2^23 --
                                    # with exactly one class of exception: it never emits
                                    # -0.0 (R360-data/probe/p1_semantics.txt, q1).
                                    #
                                    # So the int32 round trip disappears, and with it the
                                    # |x| < threshold guard whose only job was to keep that
                                    # conversion from overflowing.  The signed-zero repair
                                    # is the one part q1 proved is still needed, so it stays
                                    # verbatim.  10 vector passes -> 6 (float16/bfloat16),
                                    # 8 -> 4 (float32).
                                    T.tile.cast(
                                        tmp3,
                                        x_calc,
                                        {
                                            "floor": "CAST_FLOOR",
                                            "ceil": "CAST_CEIL",
                                            "round": "CAST_RINT",
                                            "trunc": "CAST_TRUNC",
                                        }[op_kind],
                                        tile,
                                    )
                                    T.tile.mul(tmp4, x_calc, 0.0)
                                    T.tile.compare(mask3, tmp3, 0.0, "EQ")
                                    T.tile.select(
                                        tmp, mask3, tmp4, tmp3, "VSEL_TENSOR_TENSOR_MODE"
                                    )
                                else:
                                    T.tile.abs(tmp, x_calc)
                                    T.tile.compare(mask, tmp, integral_threshold, "LT")
                                    # Cast ``x_calc`` straight through.  The shipped code
                                    # first parked out-of-range lanes at 0.0 to keep the
                                    # int32 conversion from overflowing, but whatever the
                                    # conversion produces for those lanes is discarded by
                                    # the final select, so the clamp only cost a vector op.
                                    # These expressions are vector-bound (msprof: floor
                                    # aiv_vec_ratio 0.729), so op count is the lever.
                                    if op_kind == "floor":
                                        T.tile.cast(rounded_i32, x_calc, "CAST_FLOOR", tile)
                                    elif op_kind == "ceil":
                                        T.tile.cast(rounded_i32, x_calc, "CAST_CEIL", tile)
                                    elif op_kind == "round":
                                        T.tile.cast(rounded_i32, x_calc, "CAST_RINT", tile)
                                    else:
                                        T.tile.cast(rounded_i32, x_calc, "CAST_TRUNC", tile)
                                    T.tile.cast(tmp3, rounded_i32, "CAST_NONE", tile)
                                    # ``x * 0`` carries the sign of x, so a rounded
                                    # result of zero recovers -0.0 where the int32
                                    # round trip dropped the sign.
                                    T.tile.mul(tmp4, x_calc, 0.0)
                                    T.tile.compare(mask3, tmp3, 0.0, "EQ")
                                    T.tile.select(
                                        tmp2, mask3, tmp4, tmp3, "VSEL_TENSOR_TENSOR_MODE"
                                    )
                                    # The shipped code also excluded ``|x| == 0`` from the
                                    # rounded branch.  That exclusion is redundant given the
                                    # signed-zero repair above: for x = +-0.0 the repair
                                    # already yields +-0.0, so both branches agree.  Dropping
                                    # it removes a compare and a mask AND (verified bitwise
                                    # against the pre-change implementation).
                                    T.tile.select(
                                        tmp, mask, tmp2, x_calc, "VSEL_TENSOR_TENSOR_MODE"
                                    )
                            elif op_kind == "erf":
                                # Abramowitz and Stegun 7.1.26, evaluated in fp32:
                                # max approximation error <= 1.5e-7.
                                T.tile.abs(tmp, x_calc)
                                T.tile.mul(tmp2, tmp, 0.3275911)
                                T.tile.add(tmp2, tmp2, 1.0)
                                if _T360_ARM == "before":
                                    T.tile.reciprocal(tmp3, tmp2)
                                    # The hardware reciprocal is an estimate. Two Newton
                                    # steps recover fp32 accuracy before Horner evaluation.
                                    T.tile.mul(tmp4, tmp2, tmp3)
                                    T.tile.mul(tmp4, tmp4, -1.0)
                                    T.tile.add(tmp4, tmp4, 2.0)
                                    T.tile.mul(tmp3, tmp3, tmp4)
                                    T.tile.mul(tmp4, tmp2, tmp3)
                                    T.tile.mul(tmp4, tmp4, -1.0)
                                    T.tile.add(tmp4, tmp4, 2.0)
                                    T.tile.mul(tmp3, tmp3, tmp4)
                                else:
                                    # R360.  The estimate-plus-two-Newton chain above is
                                    # NINE vector passes to reach 1.295e-07 on exactly this
                                    # t-domain.  One ``T.tile.div`` reaches 5.763e-08 there
                                    # -- cheaper AND more accurate
                                    # (R360-data/probe/p1_semantics.txt, q3 vs q4).
                                    T.tile.div(tmp3, one, tmp2)
                                if _T360_ARM == "before":
                                    T.tile.fill(tmp2, 1.061405429)
                                    T.tile.mul(tmp2, tmp2, tmp3)
                                else:
                                    # R360.  A vector fill of a constant followed by a
                                    # tensor-tensor multiply by that constant tile is the
                                    # same arithmetic as one scalar multiply, for one pass
                                    # instead of two.  Constant unchanged (contract five).
                                    T.tile.mul(tmp2, tmp3, 1.061405429)
                                T.tile.add(tmp2, tmp2, -1.453152027)
                                T.tile.mul(tmp2, tmp2, tmp3)
                                T.tile.add(tmp2, tmp2, 1.421413741)
                                T.tile.mul(tmp2, tmp2, tmp3)
                                T.tile.add(tmp2, tmp2, -0.284496736)
                                T.tile.mul(tmp2, tmp2, tmp3)
                                T.tile.add(tmp2, tmp2, 0.254829592)
                                T.tile.mul(tmp2, tmp2, tmp3)
                                T.tile.mul(tmp4, tmp, tmp)
                                if _T360_ARM == "before":
                                    T.tile.sub(tmp4, zero, tmp4)
                                    T.tile.exp(tmp4, tmp4)
                                    T.tile.mul(tmp2, tmp2, tmp4)
                                    T.tile.fill(tmp4, 1.0)
                                    T.tile.sub(tmp2, tmp4, tmp2)
                                    T.tile.sub(tmp3, zero, tmp2)
                                else:
                                    # R360.  Three exact rewrites, no constant touched:
                                    #  * ``0 - v`` -> ``v * -1``, which drops erf's whole
                                    #    dependence on the ``zero`` tile and therefore the
                                    #    per-tile ``T.tile.fill(zero, 0.0)`` above.  The two
                                    #    differ only in the sign of a zero result, and both
                                    #    consumers here discard that: ``exp(-0.0) ==
                                    #    exp(+0.0) == 1.0``, and a zero ``tmp2`` can only
                                    #    arise at x == 0, which the ``mask2`` select below
                                    #    overrides with ``x_calc`` regardless.
                                    #  * the 1.0 fill is now the hoisted ``one`` tile, so
                                    #    ``1 - v`` is one pass instead of a fill plus a sub.
                                    T.tile.mul(tmp4, tmp4, -1.0)
                                    T.tile.exp(tmp4, tmp4)
                                    T.tile.mul(tmp2, tmp2, tmp4)
                                    T.tile.sub(tmp2, one, tmp2)
                                    T.tile.mul(tmp3, tmp2, -1.0)
                                # R194: was two nested per-lane selects.  ``tmp4`` is
                                # dead here (it last held the 1.0 fill), so it can
                                # carry the sign-resolved value.
                                T.tile.compare(mask, x_calc, 0.0, "LT")
                                T.tile.select(
                                    tmp4, mask, tmp3, tmp2, "VSEL_TENSOR_TENSOR_MODE"
                                )
                                T.tile.compare(mask2, x_calc, 0.0, "EQ")
                                T.tile.select(
                                    tmp, mask2, x_calc, tmp4, "VSEL_TENSOR_TENSOR_MODE"
                                )
                            elif op_kind == "identity":
                                T.copy(x_ub, y_ub)
                            elif op_kind == "bitwise_not":
                                # AscendC's vector Not accepts only narrow integer storage
                                # on dav-2201.  Keep int32/int64 semantics in UB-local
                                # scalar expressions and still use a single GM copy per
                                # full tile.
                                if input_dtype in {"int32", "int64"}:
                                    for lane in T.serial(tile):
                                        y_ub[lane] = ~x_ub[lane]
                                else:
                                    T.tile.bitwise_not(y_ub, x_ub)
                            elif bool_logical and not vector_expand:
                                for lane in T.serial(tile):
                                    y_ub[lane] = T.if_then_else(x_ub[lane] == 0, 1, 0)
                            elif op_kind == "logical_not":
                                T.tile.compare(mask, x_calc, 0.0, "EQ")
                                if vector_expand:
                                    # 1 bit -> 1 byte, vectorised.  A set mask bit
                                    # takes src0, so ``expand_one`` must be src0 --
                                    # the polarity ``_compile_where`` and
                                    # ``_compile_masked_fill`` already rely on.
                                    T.tile.select(
                                        expand_sel,
                                        mask,
                                        expand_one,
                                        expand_zero,
                                        "VSEL_TENSOR_TENSOR_MODE",
                                    )
                                    T.tile.cast(y_ub, expand_sel, "CAST_RINT", tile)
                                else:
                                    for lane in T.serial(tile):
                                        y_ub[lane] = (mask[lane // 8] >> (lane % 8)) & 1
                            elif op_kind == "isnan":
                                # NaN does not satisfy AscendC's ordinary NE predicate;
                                # derive it as ``not finite and not infinite``.
                                T.tile.abs(tmp, x_calc)
                                T.tile.compare(mask, tmp, T.infinity(compute_dtype), "LT")
                                T.tile.compare(mask2, tmp, T.infinity(compute_dtype), "EQ")
                                for lane in T.serial(tile):
                                    finite = (mask[lane // 8] >> (lane % 8)) & 1
                                    infinite = (mask2[lane // 8] >> (lane % 8)) & 1
                                    y_ub[lane] = T.if_then_else(
                                        finite == 0, T.if_then_else(infinite == 0, 1, 0), 0
                                    )
                            elif op_kind in {"isinf", "isfinite"}:
                                T.tile.abs(tmp, x_calc)
                                T.tile.compare(
                                    mask,
                                    tmp,
                                    T.infinity(compute_dtype),
                                    "EQ" if op_kind == "isinf" else "LT",
                                )
                                if op_kind == "isinf":
                                    for lane in T.serial(tile):
                                        y_ub[lane] = (mask[lane // 8] >> (lane % 8)) & 1
                                else:
                                    for lane in T.serial(tile):
                                        y_ub[lane] = (mask[lane // 8] >> (lane % 8)) & 1
                            elif op_kind == "sign":
                                # R194: was a per-lane loop building two 0/1 tiles.
                                # ``VSEL_TENSOR_SCALAR_MODE`` needs a Buffer for src0,
                                # so keep a ones tile in tmp3 and select against 0.0.
                                T.tile.compare(mask, x_calc, 0.0, "GT")
                                T.tile.compare(mask2, x_calc, 0.0, "LT")
                                T.tile.fill(tmp3, 1.0)
                                T.tile.select(
                                    tmp, mask, tmp3, 0.0, "VSEL_TENSOR_SCALAR_MODE"
                                )
                                T.tile.select(
                                    tmp2, mask2, tmp3, 0.0, "VSEL_TENSOR_SCALAR_MODE"
                                )
                                T.tile.sub(tmp, tmp, tmp2)
                            else:
                                raise ValueError(
                                    f"unsupported unary kernel operation {op_kind!r}"
                                )

                            if op_kind not in {
                                "identity",
                                "bitwise_not",
                                "logical_not",
                                "isnan",
                                "isinf",
                                "isfinite",
                            }:
                                if is_predicate:
                                    T.tile.cast(y_ub, tmp, "CAST_RINT", tile)
                                elif needs_compute_cast:
                                    T.tile.cast(y_ub, tmp, "CAST_RINT", tile)
                                else:
                                    T.copy(tmp, y_ub)
                            T.barrier_all()
                            if full:
                                T.copy(y_ub, Y[start])
                            else:
                                for lane in T.serial(tile):
                                    idx = start + lane
                                    if idx < n_total:
                                        Y[idx] = y_ub[lane]

        return main

    return kernel()


def build_unary_kernel(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    op_kind: str,
    supported_dtypes: tuple[torch.dtype, ...],
    output_dtype: torch.dtype | None = None,
    op_name: str,
):
    if dtype not in supported_dtypes:
        raise TypeError(_supported_message(op_name, dtype, supported_dtypes))
    if output_dtype is None:
        output_dtype = dtype
    if op_kind == "reciprocal" and dtype in INT_DTYPES:
        output_dtype = torch.float32
    n_total = math.prod(tuple(shape))
    kernel_input_dtype = torch.uint8 if dtype == torch.bool else dtype
    kernel_output_dtype = torch.uint8 if output_dtype == torch.bool else output_dtype
    compiled = _compile_unary(
        n_total,
        _dtype_name(kernel_input_dtype),
        _dtype_name(kernel_output_dtype),
        op_kind,
    )

    def invoke(input_tensor: torch.Tensor):
        if tuple(input_tensor.shape) != tuple(shape):
            raise ValueError(
                f"{op_name} kernel shape mismatch: expected input={tuple(shape)}, received input={tuple(input_tensor.shape)}"
            )
        flat = input_tensor if input_tensor.ndim == 1 else input_tensor.reshape(-1)
        if dtype == torch.bool:
            flat = flat.view(torch.uint8)
        # The kernel is declared with out_idx=[-1], so TileLang allocates and
        # returns the output. Passing an explicit tensor here would make it an
        # extra input, as documented by the out_idx contract.
        result = compiled(flat)
        if output_dtype == torch.bool:
            result = result.view(torch.bool)
        return result if tuple(shape) == (result.numel(),) else result.reshape(shape)

    return invoke
