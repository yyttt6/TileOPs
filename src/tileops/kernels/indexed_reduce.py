"""Ascend AIV argmax row-reduction kernel.

The input is presented as a flattened ``[M, N]`` view by the builder.  Each
vector context owns a disjoint row range, and the output is padded so the
writeback is always a vector copy (scalar GM stores can be removed by the
auto CV combiner on dav-2201).

R198: the per-lane scalar comparison loop that used to carry the reduction
(``_SUB_M * _BLOCK_N`` = 16384 scalar iterations per tile, on the main path)
is replaced by a vector sequence.  The index of the winning lane is recovered
without leaving vector code:

    tile_max            = reduce_max(x)                       # per row
    is_max              = (x == broadcast(tile_max))          # packed mask
    candidate_index     = select(is_max, ramp, +BIG)           # ramp[r, c] = c
    first_index_in_tile = reduce_min(candidate_index) + base

``reduce_min`` over the candidate columns yields the *lowest* matching column,
and the cross-tile merge uses a strict comparison, so the earliest index wins
every tie -- matching ``torch.arg{max,min}``.

Two correctness constraints are load-bearing here (cf. PROJECT_STATE 13.20 /
R090): the neutral element of ``reduce_max`` is -inf, not 0, so the padded
lanes of a trailing tile must be driven to the *signed-infinity* sentinel
before the reduction, and the same lanes must additionally be cleared out of
the index-candidate mask -- a sentinel alone is not enough, because real input
data may legitimately contain +/-inf and would then tie with the padding.
"""

from functools import lru_cache
import math
import os

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count


_BLOCK_M = 128
_VEC = 2
_SUB_M = _BLOCK_M // _VEC
_PADDED_OUTPUT = 64

# --- Geometry (R198) ---------------------------------------------------------
# ``T.tile.compare`` requires a 256-byte aligned operand (codegen_ascend.cc:1926,
# "Compare alignment error").  The cross-tile merge compares the per-row state
# vectors, so the tile row count is pinned to ``_SUB_M`` = 64 (64 x fp32 = 256 B)
# -- a (32, 512) tile holding the same number of lanes does not compile.
_TILE_ROWS = _SUB_M
# With the rows pinned, UB fits three fp32 tiles of (_TILE_ROWS, _BLOCK_N) plus
# one input-dtype tile.  fp32: 4 * 64 * 128 * 4 = 131072 B; bf16/fp16 needs less.
# The measured sweep (R198-data/probe/probe_argmax_geom.json) shows fp16 and fp32
# take the same time at this geometry, i.e. the kernel is element-rate limited
# rather than byte-rate limited, so widening _BLOCK_N is not the lever -- the
# vector-op count per lane is.  See the trailing-tile guards below.
_BLOCK_N = 128
# fp32 holds integers exactly up to 2**24.  Beyond that the index arithmetic in
# the vector path would start rounding, so those (absent from every manifest
# shape) keep the original scalar comparison loop.
_MAX_VECTOR_INDEX = 1 << 24


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    raise TypeError(f"indexed arg-reduce supports floating dtypes, got {dtype}")


# --- R262: spreading a starved arg-reduce over the reduction axis -------------
#
# `launch_blocks = launch_block_count(m_tiles)` with `_BLOCK_M = 128` makes the
# grid a function of `m` alone.  `lm-head-argmax` is m = 4: ONE launch block out
# of 48, and worse, `_TILE_ROWS = 64` with 4 valid rows means 94% of every
# vector instruction is spent on rows that do not exist.  Measured cost (R262
# §3): 1699.1 us against a 10.5 us vendor baseline, ratio 0.0062.
#
# The cure is R198's `reduction._axis_split_factor` idea applied to an INDEXED
# reduction: view `(m, n)` as `(m, k, n/k)`, i.e. the contiguous 2-D view
# `(m*k, n/k)`, and let stage 1 be the ordinary row kernel over `m*k` rows.
# For lm-head-argmax that is `(3200, 128)`: 25 launch blocks, every one of the
# 64 rows per lane real, exactly one n-tile, no m-tail and no n-tail -- the
# total vector work drops from 800 x (64, 128) tiles to 50, a 16x reduction
# ON TOP OF the 25x more cores.
#
# ⚠️ The tie-break has to survive the split, and it does, EXACTLY:
#
#   * stage 1 already returns, per chunk, the LOWEST local column attaining
#     that chunk's extremum (`reduce_min` over the candidate columns);
#   * chunk `j` covers global columns `[j*w, (j+1)*w)` with `w = n/k`, so the
#     chunks are ordered and disjoint and chunk `j`'s winner has global index
#     `j*w + local`;
#   * stage 2 takes the row extremum over the k chunk values and then
#     `reduce_min` over `j*w + local` restricted to the chunks that ATTAIN it.
#
# The minimum of the global indices of the per-chunk-lowest winners over all
# winning chunks is the global lowest index attaining the global extremum.  So
# the split is bit-identical to the unsplit kernel, not merely equivalent up to
# ties -- which is why R262 could check it against `torch.arg{max,min}` index
# for index rather than value for value.  (Verified: R262 §5, 0 index
# mismatches over 27 shape x dtype x value-class combinations including an
# all-ties class.)
_ARG_AXIS_SPLIT = os.environ.get("TILEOPS_ARG_AXIS_SPLIT", "1") != "0"


def _t362_arm() -> str:
    """R362's control-arm switch.  "base" = the tree as T362 found it."""
    return os.environ.get("TILEOPS_T362_ARM", "r362")

#: Split until stage 1 has at least this many launch blocks.
_ARG_SPLIT_TARGET_BLOCKS = 48
#: ... and leave a case alone once it already has this many.
#:
#: ⚠️ MEASURED: 8, not 32.  16 launch blocks is only a third of the machine and
#: `hidden-state-argmax` (m = 2048, 16 blocks) sits at 0.7064, so raising this
#: to 32 to "rescue" it looked obviously right.  It is not -- R262 §4.2 ran the
#: A/B and splitting that case made it WORSE:
#:
#:     hidden-state-argmax fp16   0.7064 -> 0.5134   (54.5 us -> 74.8 us)
#:     hidden-state-argmax bf16   0.8902 -> 0.6871   (53.5 us -> 71.5 us)
#:
#: This reproduces R198's own negative result for the plain reductions
#: (`reduction._SPLIT_MIN_BLOCKS`, "a case that already has >= 8 row blocks is
#: left alone"): once the machine is reasonably busy the extra kernel launch
#: and the GM round trip cost more than the added parallelism buys.  The two
#: thresholds are deliberately the same number.
_ARG_SPLIT_MIN_BLOCKS = int(os.environ.get("TILEOPS_ARG_SPLIT_MIN_BLOCKS", "8"))


def _arg_axis_split_factor(m: int, n: int) -> int | None:
    """Chunks to cut an indexed reduction's axis into, or None to leave it.

    Requirements, in order of how much they cost when violated:

    * ``k`` divides ``n`` -- otherwise ``(m, n) -> (m*k, n/k)`` is not a view
      and the chunks are not equal width, which breaks the ``j*w + local``
      index reconstruction in stage 2.
    * ``n/k`` is a multiple of ``_BLOCK_N`` -- otherwise stage 1 gets a ragged
      trailing tile whose guard ops it then pays on EVERY tile.
    * ``m*k`` is a multiple of ``_BLOCK_M`` -- otherwise stage 1 takes the
      per-row guarded copy loop, which is the disease being cured.
    """
    if not _ARG_AXIS_SPLIT or m <= 0 or n <= 0:
        return None
    if (m + _BLOCK_M - 1) // _BLOCK_M >= _ARG_SPLIT_MIN_BLOCKS:
        return None                       # not starved; leave it alone
    best = None
    best_blocks = (m + _BLOCK_M - 1) // _BLOCK_M
    k = 2
    while k <= n // _BLOCK_N:
        w = n // k
        if n % k == 0 and w % _BLOCK_N == 0 and (m * k) % _BLOCK_M == 0:
            blocks = (m * k) // _BLOCK_M
            if blocks > best_blocks:
                best_blocks = blocks
                best = k
            if best_blocks >= _ARG_SPLIT_TARGET_BLOCKS:
                break
        k += 1
    return best


#: --- R362: stage 2's width is a FREE PARAMETER, and it was costing 44% -----
#:
#: Measured (R362 `probe/base/p1_ArgmaxFwdOp.stdout`, Level1, reps=20,
#: `lm-head-argmax` fp16, the `ratio_min` case):
#:
#:     stage 1   7.84 us   25 launch blocks, reads 800 KiB   (scalar 3.51 = 45%)
#:     stage 2   6.24 us    1 launch block,  reads 12.8 KiB
#:               -------
#:              14.10 us   against an 8.50 us vendor baseline
#:
#: Stage 2 is one launch block BY CONSTRUCTION (`m` is small, which is why the
#: split exists), so its cost is set by its tile: `(8, 896)` = 7168 element
#: slots to reduce 4 rows of 800.  That width comes from `k`, and `k` was
#: chosen by stage 1's parallelism alone.
#:
#: It does not have to be.  `_compile` already accepts `tile_rows` / `block_n`
#: (they were added for stage 2), and `tile_rows < 64` is legal whenever there
#: is a single n-tile -- which stage 1 ALWAYS has, because its column count IS
#: `n // k` and `block_n` can be set to it.  So for `lm-head` the SAME 25 launch
#: blocks and the SAME 8192 element slots per tile are reachable at four `k`:
#:
#:     k=800   stage-1 tile (64, 128)    stage 2 (4, 800) -> an (8, 896) tile
#:     k=400   stage-1 tile (32, 256)    stage 2 (4, 400) -> an (8, 512) tile
#:     k=200   stage-1 tile (16, 512)    stage 2 (4, 200) -> an (8, 256) tile
#:     k=100   stage-1 tile ( 8, 1024)   stage 2 (4, 100) -> an (8, 128) tile
#:
#: `_arg_split_plan` keeps the block count the old rule picked -- so stage 1 is
#: NOT being retuned here, only reshaped -- and then takes the SMALLEST `k`
#: that still reaches it.  Smaller `k` also shrinks the scalar prologue loop,
#: which runs `tile_rows` times per launch.
#:
#: ⚠️ The tie-break argument in `_arg_axis_split_factor` above is independent of
#: `k` and of the tile shape: it only needs the chunks to be equal width,
#: ordered and disjoint.  So this stays bit-identical to the unsplit kernel.
#:
#: ⚠️ UB is held at `_SUB_M * _BLOCK_N` ELEMENTS (8192), i.e. never more than
#: the shipped geometry already uses, so no new UB waterline is being tested.
_TILE_CELL_BUDGET = _SUB_M * _BLOCK_N

#: Legal stage-1 tile row counts.  8 is `common.ROW_EXTENT_GRAIN` (below it the
#: `(sub_m,)` fp32 state buffers encode a zero-block repeat -> aicore exception
#: 507015, R200 section 8.1); 64 is the shipped value.
_S1_TILE_ROWS = (8, 16, 32, _SUB_M)


def _arg_split_plan(m: int, n: int) -> tuple[int, int, int] | None:
    """``(k, stage1_tile_rows, stage1_block_n)`` for the axis split, or None.

    Returns the plan whose stage-1 launch-block count equals what
    ``_arg_axis_split_factor`` would have produced, with the smallest ``k``.
    """
    k0 = _arg_axis_split_factor(m, n)
    if k0 is None:
        return None
    target_blocks = (m * k0) // _BLOCK_M
    best = None
    for k in range(1, k0 + 1):
        if n % k or k > n:
            continue
        width = n // k
        if width % _BLOCK_N:
            continue
        for tile_rows in _S1_TILE_ROWS:
            if tile_rows * width > _TILE_CELL_BUDGET:
                continue
            block_m = tile_rows * _VEC
            if (m * k) % block_m:
                continue
            if (m * k) // block_m != target_blocks:
                continue
            if best is None:
                best = (k, tile_rows, width)
            break
        if best is not None:
            break
    return best if best is not None else (k0, _SUB_M, _BLOCK_N)


def describe_strided(outer: int, n: int, inner: int, dtype_name: str) -> dict:
    """The strided-axis geometry, or ``{}``.  Diagnostics only (R262 §2)."""
    plan = _strided_plan(outer, n, inner, dtype_name)
    return {} if plan is None else {"strided_" + k: v for k, v in plan.items()}


def describe_plan(m: int, n: int, dtype_name: str, outer: int = 1,
                  inner: int = 1) -> dict:
    """The geometry this builder would use.  Diagnostics only (R262 §2).

    ``outer`` / ``inner`` describe the ORIGINAL axis position so the caller can
    see whether the strided-axis kernel takes the case; pass the defaults for a
    trailing reduction axis.
    """
    strided = _strided_plan(outer, n, inner, dtype_name)
    if strided is not None:
        out = {"path": "strided", "k": None}
        out.update({"strided_" + q: v for q, v in strided.items()})
        return out
    if _t362_arm() == "base":
        plan = _arg_axis_split_factor(m, n)
        plan = None if plan is None else (plan, _SUB_M, _BLOCK_N)
    else:
        plan = _arg_split_plan(m, n)
    if plan is None:
        k, s1_tile_rows, s1_block_n = None, _SUB_M, _BLOCK_N
        rows, cols = m, n
    else:
        k, s1_tile_rows, s1_block_n = plan
        rows, cols = m * k, n // k
    s1_block_m = s1_tile_rows * _VEC
    out = {
        "path": "row_split" if k is not None else "row",
        "k": k,
        "s1_rows": rows,
        "s1_cols": cols,
        "s1_tile_rows": s1_tile_rows,
        "s1_block_n": s1_block_n,
        "s1_n_tiles": (cols + s1_block_n - 1) // s1_block_n,
        "s1_launch_blocks": launch_block_count((rows + s1_block_m - 1) // s1_block_m),
        "s1_grid_repeats": grid_repeat_count((rows + s1_block_m - 1) // s1_block_m,
                                             launch_block_count((rows + s1_block_m - 1) // s1_block_m)),
        "s1_has_m_tail": rows % s1_block_m != 0,
        "s1_has_n_tail": cols % s1_block_n != 0,
        "scalar_fallback": cols > _MAX_VECTOR_INDEX,
    }
    if k is not None:
        c_rows, c_width = _combine_geometry(m, k)
        out.update({
            "s2_rows": m, "s2_cols": k,
            "s2_tile_rows": c_rows,
            "s2_block_n": c_width,
            "s2_n_tiles": (k + c_width - 1) // c_width,
            "s2_launch_blocks": launch_block_count(
                (m + c_rows * _VEC - 1) // (c_rows * _VEC)),
        })
    return out


def _combine_geometry(m: int, k: int) -> tuple[int, int]:
    """``(tile_rows, block_n)`` for stage 2 of the axis split.

    🚨 THIS IS WHERE THE SPLIT'S TIME GOES, and the reason is not obvious.
    Measured per stage with the harness timer
    (`R262-data/39_stage_timing.json`), `lm-head-argmax` fp16 at k = 800 with
    stage 2 on the default (64, 128) tile:

        stage 1   7.75 us   25 launch blocks, reads the whole 800 KiB input
        stage 2  26.50 us    1 launch block,  reads 6.4 KiB
                 ------
                 34.25 us   -> stage 2 is 77% of the op

    Stage 2 is one launch block *by construction* -- `m` is small, which is why
    the split exists at all -- so nothing amortises its per-launch cost, and on
    a (64, 128) tile that cost is: an 8192-element `arith_progression`, a
    64-iteration SCALAR loop building `row_offset`, a broadcast and a subtract,
    7 n-tiles' worth of cross-tile merge, and a 64-iteration scalar int64 cast
    loop -- all to reduce 4 rows of 800 fp32.  56 of every 64 tile rows are
    padding.

    So stage 2 gets its own geometry: `tile_rows` down to the `(sub_m,)` vector
    floor (`common.ROW_EXTENT_GRAIN` = 8) and `block_n` wide enough to cover
    `k` in ONE tile, which is what makes `tile_rows < 64` legal (no cross-tile
    merge, hence no 256-byte `T.tile.compare` on the state vectors).  The
    prologue then runs on 8 rows instead of 64 and the merge disappears.

    Falls back to the default geometry when one tile of `k` columns will not
    fit in UB.
    """
    width = ((k + _STRIDED_GRAIN - 1) // _STRIDED_GRAIN) * _STRIDED_GRAIN
    rows = _COMBINE_TILE_ROWS
    # 5 fp32 tiles: values, values_fp32, ramp, work, idx_tile.
    if rows * width * 5 * 4 > _STRIDED_UB_BUDGET:
        return _SUB_M, _COMBINE_BLOCK_N
    if m > rows * _VEC:
        # More rows than one launch block of the narrow geometry can hold: the
        # narrow tile would multiply the launch count instead of shrinking the
        # prologue.  Keep the default.
        return _SUB_M, _COMBINE_BLOCK_N
    return rows, width


#: `common.ROW_EXTENT_GRAIN`: below 8 lanes the vector ops on the `(sub_m,)`
#: fp32 state buffers encode a zero-block repeat and the core raises aicore
#: exception 507015 (R200 §8.1 measured it at sub_m in {1, 2, 4}).
_COMBINE_TILE_ROWS = 8

#: Stage 2's FALLBACK tile width.  It holds FOUR fp32 tiles rather than the main path's
#: three-plus-one-dtype (it carries the incoming local indices as well), so at
#: `_TILE_ROWS = 64` a 128-wide tile is 4 * 64 * 128 * 4 = 131072 B -- still
#: inside `common.UB_BUDGET_NORM_BYTES` (147456), which is why this is 128 and
#: not smaller.  Kept as its own name so the two stages can be retuned apart.
_COMBINE_BLOCK_N = 128


@lru_cache(maxsize=64)
def _compile(m: int, n: int, dtype_name: str, op_kind: str,
             out0_dtype: str = "int64", chunk: int = 0,
             tile_rows: int = _SUB_M, block_n: int = 0):
    """Compile one static row shape.

    Ties resolve to the lowest index: ``reduce_min`` picks the first matching
    column inside a tile and the cross-tile merge only updates on a strict
    comparison.  This matches PyTorch arg{max,min}.

    ⚠️ ONE signature serves all three configurations -- two inputs
    ``(A, Q)`` and two outputs ``(OUT0, OUT1)`` -- on purpose.  The tile body
    below is ~110 lines and the three variants differ only in their signature,
    so the obvious factorings were tried first and both fail:

    * a nested plain Python function is evaluated by the TVMScript parser as an
      ordinary call, and ``T.serial`` then comes back as a bare ``ForFrame``
      ("'ForFrame' object is not iterable");
    * ``T.macro``, the supported way to inline an AST, **segfaults** this
      codegen (measured: R262-data/34_macro_segv.log, a plain no-split
      ``ArgmaxFwdOp`` build crashes the process during compilation).

    So the unused slots are wired to a tensor that is already there:
    ``Q = A`` when there is no incoming index tensor (never read, so no extra
    GM traffic), and ``OUT1`` carries the winning *value* in the single-stage
    and combine configurations, where it is simply ignored by the caller.  The
    cost is one extra ``(m_pad,)`` fp32 allocation and one 256-byte vector
    store per row block.

    Configurations:

    ``out0_dtype="int64", chunk=0``
        The original single-stage kernel.  ``(m, n) -> (m,) int64``.
    ``out0_dtype="float32", chunk=0``
        Stage 1 of the axis split: ``OUT0`` is the per-chunk extremum,
        ``OUT1`` the lowest local column attaining it.
    ``out0_dtype="int64", chunk=w``
        Stage 2 of the axis split.  ``A`` is stage 1's ``(m, k)`` values, ``Q``
        its ``(m, k)`` local columns, and the global index is reconstructed as
        ``column * w + local``.  ``dtype_name`` must be ``"float32"``.
    """
    if out0_dtype not in ("int64", "float32"):
        raise ValueError(f"unsupported out0 dtype {out0_dtype!r}")
    if chunk and (out0_dtype != "int64" or dtype_name != "float32"):
        raise ValueError("the combine stage takes fp32 values and emits int64")
    partial = out0_dtype == "float32"
    # ⚠️ MUST be a Python bool, not the int itself: inside a prim_func body the
    # TVMScript parser turns ``if chunk:`` into a TIR predicate (an int lowers
    # to IntImm) and then walks BOTH branches, which failed with "Undefined
    # variable: idx_tile" on the no-split path.
    combine = bool(chunk)

    if block_n <= 0:
        block_n = _COMBINE_BLOCK_N if chunk else _BLOCK_N
    # ⚠️ `tile_rows` below 64 is legal ONLY with a single n-tile.  The 64 is
    # not a tile-shape choice, it is `T.tile.compare`'s 256-byte operand
    # requirement (R198, codegen_ascend.cc:1926) applied to the CROSS-TILE
    # MERGE, which compares the `(tile_rows,)` fp32 state vectors: 64 fp32 =
    # 256 B, 8 fp32 = 32 B and does not compile.  With `n_tiles == 1` there is
    # no merge to emit, so the constraint lifts -- and that is exactly the
    # configuration stage 2 of the axis split wants.
    block_m = tile_rows * _VEC
    m_tiles = (m + block_m - 1) // block_m
    n_tiles = (n + block_n - 1) // block_n
    single_tile = n_tiles == 1
    if tile_rows != _SUB_M and not single_tile:
        raise ValueError(
            f"_compile(tile_rows={tile_rows}) needs one n-tile, got {n_tiles}"
        )
    has_m_tail = m % block_m != 0
    has_n_tail = n % block_n != 0
    launch_blocks = launch_block_count(m_tiles)
    grid_repeats = grid_repeat_count(m_tiles, launch_blocks)
    need_cast = dtype_name != "float32"
    # ⚠️ MUST be a Python bool for the same reason `combine` above is: inside a
    # prim_func body the TVMScript parser turns a non-bool `if` into a TIR
    # predicate and walks both branches.
    vector_row_offset = _t362_arm() != "base"
    # 2**30: exactly representable in fp32 and far above any supported index, so
    # a tile with no matching lane always loses the cross-tile min.
    big_index = float(1 << 30)

    @tilelang.jit(
        out_idx=[2, 3],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((m, n), dtype_name),
            Q: T.Tensor((m, n), dtype_name),
            OUT0: T.Tensor((m_tiles * block_m,), out0_dtype),
            OUT1: T.Tensor((m_tiles * block_m,), "float32"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                values = T.alloc_ub((tile_rows, block_n), dtype_name)
                values_fp32 = T.alloc_ub((tile_rows, block_n), "float32")
                ramp = T.alloc_ub((tile_rows, block_n), "float32")
                work = T.alloc_ub((tile_rows, block_n), "float32")
                # Stage 2 only: the local columns stage 1 found.  A fifth
                # (64, 128) fp32 tile -- 5 * 32768 = 163840 B, over
                # `common.UB_BUDGET_NORM_BYTES` (147456) but under the
                # elementwise waterline (180224).  It compiles and runs; if a
                # future change pushes it over, shrink `_COMBINE_BLOCK_N`
                # rather than dropping a tile, because `T.tile.compare` wants
                # 128-lane native repeats.
                #
                # ⚠️ Allocated UNCONDITIONALLY (degenerate on the other two
                # paths) because a `T.alloc_ub` inside a Python-level `if`
                # block is scoped to that block: the parser accepts the
                # allocation and then reports "Undefined variable: idx_tile" at
                # the first use after the block.  32 wasted bytes is the price.
                idx_tile = T.alloc_ub(
                    (tile_rows, block_n) if combine else (8,), "float32"
                )
                row_offset = T.alloc_ub((tile_rows,), "float32")
                best_values = T.alloc_ub((tile_rows,), "float32")
                best_index = T.alloc_ub((tile_rows,), "float32")
                tile_values = T.alloc_ub((tile_rows,), "float32")
                tile_index = T.alloc_ub((tile_rows,), "float32")
                best_indices = T.alloc_ub((tile_rows,), "int64")
                mask = T.alloc_ub((tile_rows * block_n // 8,), "uint8")
                valid = T.alloc_ub((tile_rows * block_n // 8,), "uint8")
                row_mask = T.alloc_ub((max(tile_rows, 8) // 8,), "uint8")

                with T.Scope("V"):
                    # ``-inf`` for argmax: the neutral element of a max reduction.
                    # Built from T.infinity because a Python float("inf") lowers
                    # to CUDART_INF, which does not exist on dav-2201.
                    sentinel = (
                        -T.infinity("float32")
                        if op_kind == "argmax"
                        else T.infinity("float32")
                    )
                    # ramp[r, c] = c, built once per launch.  arith_progression
                    # fills flattened storage (verified: probe_intrinsics.json
                    # q1, row 1 starts at block_n), so the per-row base is
                    # subtracted back off with a broadcast.
                    T.tile.arith_progression(ramp, 0.0, 1.0, tile_rows * block_n)
                    if vector_row_offset:
                        # R362: `row_offset[r] = r * block_n` IS an arithmetic
                        # progression -- first 0, step block_n, count tile_rows.
                        # One vector intrinsic instead of `tile_rows` scalar
                        # stores, and the values are identical (every r*block_n
                        # below 2**24 is exact in fp32; block_n <= 4096 and
                        # tile_rows <= 64, so the largest is 258048).
                        #
                        # Measured cost of the loop it replaces (R362
                        # probe/base/p1_ArgmaxFwdOp.stdout, lm-head fp16 stage 1,
                        # tile_rows = 64): aiv_scalar_time 3.51 us out of a
                        # 7.84 us launch, i.e. ~55 ns per scalar iteration.
                        T.tile.arith_progression(row_offset, 0.0,
                                                 float(block_n), tile_rows)
                    else:
                        for r in T.serial(tile_rows):
                            row_offset[r] = T.Cast("float32", r * block_n)
                    T.tile.broadcast(work, row_offset)
                    T.tile.sub(ramp, ramp, work)

                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * block_m + vid * tile_rows
                            T.tile.fill(best_values, sentinel)
                            T.tile.fill(best_index, 0.0)

                            for nt in T.serial(n_tiles):
                                base = nt * block_n
                                if not has_m_tail or logical_cid < m_tiles - 1:
                                    # Whole row block is in range: one strided
                                    # 2D transfer instead of tile_rows of them.
                                    T.copy(A[row_base, base], values, pad_value=0)
                                    if combine:
                                        T.copy(Q[row_base, base], idx_tile, pad_value=0)
                                else:
                                    for r in T.serial(tile_rows):
                                        if row_base + r < m:
                                            T.copy(
                                                A[row_base + r, base],
                                                values[r, :],
                                                pad_value=0,
                                            )
                                            if combine:
                                                T.copy(
                                                    Q[row_base + r, base],
                                                    idx_tile[r, :],
                                                    pad_value=0,
                                                )
                                T.barrier_all()   # MTE2 -> V (13.55 #1/#2)
                                if need_cast:
                                    T.tile.cast(
                                        values_fp32,
                                        values,
                                        "CAST_NONE",
                                        tile_rows * block_n,
                                    )
                                else:
                                    T.copy(values, values_fp32)

                                if has_n_tail and nt == n_tiles - 1:
                                    # Trailing tile only.  Emitting these three
                                    # vector ops unconditionally would tax every
                                    # full tile, and the kernel is element-rate
                                    # limited, so they are guarded.
                                    T.tile.compare(
                                        valid, ramp, T.Cast("float32", n - base), "LT"
                                    )
                                    T.tile.select(
                                        values_fp32,
                                        valid,
                                        values_fp32,
                                        sentinel,
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
                                if op_kind == "argmax":
                                    T.reduce_max(
                                        values_fp32, tile_values, dim=-1, clear=True
                                    )
                                else:
                                    T.reduce_min(
                                        values_fp32, tile_values, dim=-1, clear=True
                                    )
                                T.tile.broadcast(work, tile_values)
                                T.tile.compare(mask, values_fp32, work, "EQ")
                                if has_n_tail and nt == n_tiles - 1:
                                    # A sentinel-valued padding lane ties with
                                    # real +/-inf data, so drop the pad lanes
                                    # from the index candidates as well.
                                    T.tile.bitwise_and(mask, mask, valid)
                                if combine:
                                    # Reconstruct the global index of every lane:
                                    # (this tile's column) * chunk + the local
                                    # column stage 1 found inside that chunk.
                                    # ``values_fp32`` is dead after the compare
                                    # above, so it doubles as the scratch tile --
                                    # a fifth fp32 tile would not fit in UB.
                                    T.tile.mul(values_fp32, ramp,
                                               T.Cast("float32", chunk))
                                    T.tile.add(values_fp32, values_fp32, idx_tile)
                                    T.tile.select(
                                        work, mask, values_fp32, big_index,
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
                                    T.reduce_min(work, tile_index, dim=-1, clear=True)
                                    T.tile.add(
                                        tile_index, tile_index,
                                        T.Cast("float32", base * chunk)
                                    )
                                else:
                                    T.tile.select(
                                        work, mask, ramp, big_index,
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
                                    T.reduce_min(work, tile_index, dim=-1, clear=True)
                                    T.tile.add(
                                        tile_index, tile_index, T.Cast("float32", base)
                                    )
                                if single_tile:
                                    # No merge to do: this tile IS the row.  Worth
                                    # special-casing rather than letting the merge
                                    # run once against the sentinel, because the
                                    # merge is the only thing that needs
                                    # `tile_rows == 64` (see the note above) and
                                    # because it is 4 more vector ops on a path
                                    # whose whole budget is the launch overhead.
                                    T.copy(tile_values, best_values)
                                    T.copy(tile_index, best_index)
                                else:
                                    # Strict comparison: the earliest tile keeps a tie.
                                    if op_kind == "argmax":
                                        T.tile.compare(
                                            row_mask, tile_values, best_values, "GT"
                                        )
                                    else:
                                        T.tile.compare(
                                            row_mask, tile_values, best_values, "LT"
                                        )
                                    T.tile.select(
                                        best_index, row_mask, tile_index, best_index,
                                        "VSEL_TENSOR_TENSOR_MODE",
                                    )
                                    if op_kind == "argmax":
                                        T.tile.max(best_values, best_values, tile_values)
                                    else:
                                        T.tile.min(best_values, best_values, tile_values)

                            # V -> scalar -> MTE3.  ``best_index`` is written
                            # by T.tile.select now; it used to be written by the
                            # scalar comparison loop, which ordered the UB write
                            # against this read implicitly (13.55 #2).
                            T.barrier_all()
                            if partial:
                                T.copy(best_values, OUT0[row_base : row_base + tile_rows])
                            else:
                                for r in T.serial(tile_rows):
                                    best_indices[r] = T.Cast("int64", best_index[r])
                                T.copy(best_indices, OUT0[row_base : row_base + tile_rows])
                            # OUT1 is the local column in stage 1 and the winning
                            # value everywhere else; the caller ignores it there.
                            if partial:
                                T.copy(best_index, OUT1[row_base : row_base + tile_rows])
                            else:
                                T.copy(best_values, OUT1[row_base : row_base + tile_rows])

        return main

    return factory()


# --- R262: the OTHER failure mode -- a non-last reduction axis ---------------
#
# `3d-non-last-axis-argmax` ((4, 128, 4096), dim=0) measured 2061.4 us against
# a 31.2 us baseline, ratio 0.0152.  It is NOT the starvation above: m = 524288
# gives 4096 row blocks, the whole machine.  Two other things cost it:
#
#   1. The builder reaches the row kernel by MATERIALISING a transpose:
#      `x.permute(1, 2, 0).reshape(m, n).contiguous()`, a 2 M-element strided
#      copy that shows up as the second kernel in `kernels_per_call = [2]`.
#   2. `n = 4` against `_BLOCK_N = 128`: 96.9% of every vector instruction is
#      spent on columns that do not exist.  4096 tiles of (64, 128) = 33.5 M
#      element slots for 2 M real elements.
#
# Both disappear if the reduction is done in the layout the tensor already has.
# View the input as `(outer, n, inner)` -- for a single `dim` that is always a
# view, no copy -- and reduce along `n`, which is now the SLOW axis of a
# `(n, W)` tile.  Reducing along tile rows is `n-1` full-width vector `max`
# ops; recovering the index is a downward scan over the same rows, so the
# LOWEST attaining row is the last one written and the torch tie-break is
# reproduced exactly:
#
#     acc = max_i row_i
#     for i in n-1 .. 0:  idx = select(row_i == acc, i, idx)
#
# ⚠️ Applies only when `inner >= _BLOCK_N` (a trailing reduction axis has
# `inner == 1`, where this degenerates to a one-column tile) and when some
# tile width divides `inner` exactly -- a ragged trailing width would need a
# partial-width store, which is not worth the surface area for the shapes that
# reach here.  Everything else keeps the row kernel.
#
# ⚠️ The kernel emits fp32 indices and the wrapper casts to int64 on the host.
# There is no vector fp32 -> int64 cast, and the row kernel's scalar cast loop
# (`for r in T.serial(_SUB_M)`) would be `W` = 2048 scalar iterations per tile
# here rather than 64.  fp32 represents every integer below 2**24 exactly and
# the index is bounded by `n`, so the cast is lossless; it costs one extra
# torch kernel, which is why `kernels_per_call` stays at 2 rather than dropping
# to 1.
_STRIDED_AXIS = os.environ.get("TILEOPS_ARG_STRIDED_AXIS", "1") != "0"

#: Widest tile column count for the strided-axis kernel.  R200's one-process-
#: per-width probes cleared copy/fill/max/compare/select to 4096, but the UB
#: budget binds first here (`n` rows of dtype + fp32 plus three fp32 row
#: vectors), so this is a cap, not the operating point.
_STRIDED_WIDTH_CAP = 4096
#: Column granularity.  128 rather than 16 because `T.tile.compare` emits
#: packed masks and wants 128-lane native repeats (`common.TILE_GRAIN`).
_STRIDED_GRAIN = 128
#: ⚠️ `launch_block_count` caps at `MAX_BLOCK_COUNT = 65535`, NOT at 48, so a
#: wide `inner` would ask for hundreds of blocks (256 tiles / 2 lanes = 128 for
#: the 3d manifest shape).  R194 measured 48-192 as a plateau for the
#: elementwise templates and `common.LAUNCH_BLOCK_CAP` is 48; A/B'd in R262
#: §4.3.  Override with TILEOPS_ARG_STRIDED_BLOCK_CAP.
_STRIDED_BLOCK_CAP = int(os.environ.get("TILEOPS_ARG_STRIDED_BLOCK_CAP", "48"))
#: Same UB waterline the normalization templates use: the declared buffers
#: summing under budget is not the same as fitting, because `T.tile.compare`
#: and friends carry scratch that never appears in the allocation list
#: (`common.UB_BUDGET_NORM_BYTES`, R200 §8.1).
_STRIDED_UB_BUDGET = 147456


def _strided_plan(outer: int, n: int, inner: int, dtype_name: str):
    """Tile width for the strided-axis kernel, or None to keep the row kernel."""
    if not _STRIDED_AXIS:
        return None
    if outer <= 0 or n <= 0 or inner < _STRIDED_GRAIN:
        return None
    itemsize = 2 if dtype_name in ("float16", "bfloat16") else 4
    # (n, W) dtype + (n, W) fp32 + 3 x (W,) fp32 + (W/8) mask
    per_col = n * (itemsize + 4) + 3 * 4 + 1
    hi = min(_STRIDED_WIDTH_CAP, (inner // _STRIDED_GRAIN) * _STRIDED_GRAIN)
    while hi >= _STRIDED_GRAIN and hi * per_col > _STRIDED_UB_BUDGET:
        hi -= _STRIDED_GRAIN
    if hi < _STRIDED_GRAIN:
        return None                      # a single row does not fit; give up
    width = 0
    w = hi
    while w >= _STRIDED_GRAIN:
        if inner % w == 0:
            width = w
            break
        w -= _STRIDED_GRAIN
    if width == 0:
        return None                      # no exact divisor; keep the row kernel
    col_tiles = inner // width
    total_tiles = outer * col_tiles
    lanes = launch_block_count(min((total_tiles + _VEC - 1) // _VEC,
                                   _STRIDED_BLOCK_CAP))
    return {
        "width": width,
        "col_tiles": col_tiles,
        "total_tiles": total_tiles,
        "launch_blocks": lanes,
        "grid_repeats": grid_repeat_count(total_tiles, lanes * _VEC),
        "ub_bytes": width * per_col,
    }


@lru_cache(maxsize=64)
def _compile_strided(outer: int, n: int, inner: int, dtype_name: str,
                     op_kind: str, width: int):
    """Reduce ``(outer, n, inner)`` along ``n``; emit fp32 indices."""
    col_tiles = inner // width
    total_tiles = outer * col_tiles
    launch_blocks = launch_block_count(min((total_tiles + _VEC - 1) // _VEC,
                                           _STRIDED_BLOCK_CAP))
    grid_repeats = grid_repeat_count(total_tiles, launch_blocks * _VEC)
    need_cast = dtype_name != "float32"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((outer * n, inner), dtype_name),
            C: T.Tensor((outer * inner,), "float32"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                tile = T.alloc_ub((n, width), dtype_name)
                tile32 = T.alloc_ub((n, width), "float32")
                acc = T.alloc_ub((width,), "float32")
                idx = T.alloc_ub((width,), "float32")
                cand = T.alloc_ub((width,), "float32")
                mask = T.alloc_ub((width // 8,), "uint8")

                with T.Scope("V"):
                    for rep in T.serial(grid_repeats):
                        tid = (cid + rep * launch_blocks) * _VEC + vid
                        if tid < total_tiles:
                            o = tid // col_tiles
                            t = tid % col_tiles
                            # One (n, width) strided 2D transfer: n rows of
                            # `width` CONTIGUOUS elements, `inner` apart.  This
                            # is the whole point -- the row kernel had to
                            # materialise a transpose to get here.
                            T.copy(A[o * n, t * width], tile)
                            T.barrier_all()          # MTE2 -> V (13.55 #1/#2)
                            if need_cast:
                                T.tile.cast(tile32, tile, "CAST_NONE", n * width)
                            else:
                                T.copy(tile, tile32)
                            T.copy(tile32[0, :], acc)
                            for i in T.serial(n - 1):
                                if op_kind == "argmax":
                                    T.tile.max(acc, acc, tile32[i + 1, :])
                                else:
                                    T.tile.min(acc, acc, tile32[i + 1, :])
                            # Downward scan: every row that attains `acc`
                            # overwrites `idx`, so the LAST write -- the
                            # smallest i -- survives.  Matches torch.
                            T.tile.fill(idx, 0.0)
                            for i in T.serial(n):
                                row = n - 1 - i
                                T.tile.fill(cand, 0.0)
                                T.tile.add(cand, cand, T.Cast("float32", row))
                                T.tile.compare(mask, tile32[row, :], acc, "EQ")
                                T.tile.select(
                                    idx, mask, cand, idx,
                                    "VSEL_TENSOR_TENSOR_MODE",
                                )
                            T.barrier_all()
                            T.copy(idx, C[o * inner + t * width])
        return main
    return factory()


@lru_cache(maxsize=64)
def _compile_scalar_fallback(m: int, n: int, dtype_name: str, op_kind: str):
    """Original per-lane comparison loop.

    Retained for ``n > 2**24``, where fp32 can no longer hold a column index
    exactly and the vector path above would round.  No manifest shape reaches
    this, so it is correctness insurance rather than a performance path.
    """

    _BLOCK_N_SCALAR = 256
    m_tiles = (m + _BLOCK_M - 1) // _BLOCK_M
    n_tiles = (n + _BLOCK_N_SCALAR - 1) // _BLOCK_N_SCALAR
    launch_blocks = launch_block_count(m_tiles)
    grid_repeats = grid_repeat_count(m_tiles, launch_blocks)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((m, n), dtype_name),
            C: T.Tensor((m_tiles * _BLOCK_M,), "int64"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                values = T.alloc_ub((_SUB_M, _BLOCK_N_SCALAR), dtype_name)
                values_fp32 = T.alloc_ub((_SUB_M, _BLOCK_N_SCALAR), "float32")
                best_values = T.alloc_ub((_SUB_M,), "float32")
                best_indices = T.alloc_ub((_SUB_M,), "int64")

                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * _BLOCK_M + vid * _SUB_M
                            for r in T.serial(_SUB_M):
                                if row_base + r < m:
                                    best_values[r] = (
                                        -T.infinity("float32")
                                        if op_kind == "argmax"
                                        else T.infinity("float32")
                                    )
                                    best_indices[r] = 0

                            for nt in T.serial(n_tiles):
                                for r in T.serial(_SUB_M):
                                    if row_base + r < m:
                                        T.copy(
                                            A[
                                                row_base + r,
                                                nt * _BLOCK_N_SCALAR : nt
                                                * _BLOCK_N_SCALAR
                                                + _BLOCK_N_SCALAR,
                                            ],
                                            values[r, :],
                                            pad_value=0,
                                        )
                                if dtype_name == "float32":
                                    T.copy(values, values_fp32)
                                else:
                                    T.tile.cast(
                                        values_fp32,
                                        values,
                                        "CAST_NONE",
                                        _SUB_M * _BLOCK_N_SCALAR,
                                    )

                                for r in T.serial(_SUB_M):
                                    if row_base + r < m:
                                        for j in T.serial(_BLOCK_N_SCALAR):
                                            global_index = nt * _BLOCK_N_SCALAR + j
                                            if global_index < n and (
                                                values_fp32[r, j] > best_values[r]
                                                if op_kind == "argmax"
                                                else values_fp32[r, j] < best_values[r]
                                            ):
                                                best_values[r] = values_fp32[r, j]
                                                best_indices[r] = global_index

                            T.copy(best_indices, C[row_base : row_base + _SUB_M])

        return main

    return factory()


def _compile_dispatch(m: int, n: int, dtype_name: str, op_kind: str):
    """Return ``(compiled, split)``; ``split`` is ``None`` or ``(k, s1, s2)``."""
    if n > _MAX_VECTOR_INDEX:
        return _compile_scalar_fallback(m, n, dtype_name, op_kind), None
    if _t362_arm() == "base":
        k = _arg_axis_split_factor(m, n)
        plan = None if k is None else (k, _SUB_M, _BLOCK_N)
    else:
        plan = _arg_split_plan(m, n)
    if plan is None:
        return _compile(m, n, dtype_name, op_kind), None
    k, s1_tile_rows, s1_block_n = plan
    stage1 = _compile(m * k, n // k, dtype_name, op_kind, out0_dtype="float32",
                      tile_rows=s1_tile_rows, block_n=s1_block_n)
    rows, width = _combine_geometry(m, k)
    stage2 = _compile(m, k, "float32", op_kind, out0_dtype="int64",
                      chunk=n // k, tile_rows=rows, block_n=width)
    return stage1, (k, stage1, stage2)


def _axes(dim, ndim: int) -> tuple[int, ...]:
    values = tuple(range(ndim)) if dim is None else (dim,)
    raw = tuple(int(axis) for axis in values)
    if any(axis < -ndim or axis >= ndim for axis in raw):
        raise ValueError(f"reduction dimension out of range for rank {ndim}: {dim!r}")
    axes = tuple(sorted(axis + ndim if axis < 0 else axis for axis in raw))
    if len(set(axes)) != len(axes):
        raise ValueError(f"reduction dimensions must be unique, got {dim!r}")
    return axes


def _build(input_shape, dtype, dim, keepdim, op_kind: str, op_name: str):
    axes = _axes(dim, len(input_shape))
    reduced = set(axes)
    output_shape = (
        tuple(1 if i in reduced else input_shape[i] for i in range(len(input_shape)))
        if keepdim
        else tuple(input_shape[i] for i in range(len(input_shape)) if i not in reduced)
    )
    m = math.prod(output_shape) if output_shape else 1
    n = math.prod(input_shape[i] for i in axes)
    if n <= 0:
        raise ValueError(f"{op_name} does not support an empty reduction dimension")
    order = tuple(i for i in range(len(input_shape)) if i not in reduced) + axes
    dtype_name = _dtype_name(dtype)
    if dim is not None and len(axes) != 1:
        raise ValueError(f"{op_name} accepts one reduction dimension or None")
    # --- strided-axis path: no transpose, tile along the axes we keep --------
    strided = None
    if dim is not None and len(axes) == 1:
        ax = axes[0]
        outer = math.prod(input_shape[:ax])
        inner = math.prod(input_shape[ax + 1:])
        plan = _strided_plan(outer, n, inner, dtype_name)
        if plan is not None:
            strided = (outer, inner, plan)

    if strided is not None:
        outer, inner, plan = strided
        compiled_strided = _compile_strided(outer, n, inner, dtype_name,
                                            op_kind, plan["width"])

        def launch(x):
            flat = x if x.is_contiguous() else x.contiguous()
            out = compiled_strided(flat.reshape(outer * n, inner))
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out.to(torch.int64).reshape(output_shape)

        launch.compiled = compiled_strided
        launch.plan = plan
        return launch

    compiled, split = _compile_dispatch(m, n, dtype_name, op_kind)

    def launch(x):
        view = x if order == tuple(range(len(input_shape))) else x.permute(order)
        # A non-last reduction axis cannot be represented as a contiguous
        # [M, N] view on arbitrary ranks.  This mirrors the existing
        # TileOPs reduction template's fallback; the kernel itself still owns
        # the reduction and all output writes.
        flat = view.reshape(m, n)
        if not flat.is_contiguous():
            flat = flat.contiguous()
        if split is None:
            # OUT1 (the winning value) is ignored on this path; see _compile.
            return compiled(flat, flat)[0][:m].reshape(output_shape)
        k, stage1, stage2 = split
        # (m, n) -> (m, k, n//k) -> (m*k, n//k); contiguous, so this is a view.
        chunked = flat.reshape(m * k, n // k)
        vals, locs = stage1(chunked, chunked)
        result = stage2(vals[: m * k].reshape(m, k), locs[: m * k].reshape(m, k))[0]
        return result[:m].reshape(output_shape)

    return launch


def build_argmax_kernel(input_shape, dtype, dim, keepdim):
    """Return a callable implementing the TileOPs ArgmaxFwdOp contract."""

    return _build(input_shape, dtype, dim, keepdim, "argmax", "ArgmaxFwdOp")


def build_argmin_kernel(input_shape, dtype, dim, keepdim):
    """Return a callable implementing the TileOPs ArgminFwdOp contract."""

    return _build(input_shape, dtype, dim, keepdim, "argmin", "ArgminFwdOp")


__all__ = ["build_argmax_kernel", "build_argmin_kernel"]
