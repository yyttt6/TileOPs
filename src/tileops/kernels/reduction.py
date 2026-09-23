"""Ascend AIV single-pass row-reduction kernels."""

import math
import os
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch

from .common import (
    grid_repeat_count,
    launch_block_count,
    plan_rowwise_norm,
)


_BLOCK_M = 128
_VEC = 2
_SUB_M = _BLOCK_M // _VEC
_BLOCK_N = 256
_SCALAR_BLOCK_N = 16384
_PADDED_OUTPUT = 64
_FP32_MAX = 3.402823466e38

_FLOAT_OPS = frozenset({"sum", "mean", "amax", "amin", "l1", "l2", "inf", "prod", "var", "std", "var_mean", "logsumexp"})
_WELFORD_OPS = frozenset({"var", "std", "var_mean"})


@lru_cache(maxsize=128)
def _compile_logsumexp(m: int, n: int, dtype_name: str, keepdim: bool):
    block_n = 128
    block_m = 32
    sub_m = _SUB_M
    m_tiles = (m + block_m - 1) // block_m
    n_tiles = (n + block_n - 1) // block_n
    has_m_tail = m % block_m != 0
    has_n_tail = n % block_n != 0
    tail_n = n % block_n
    launch_blocks = launch_block_count(m_tiles)
    grid_repeats = grid_repeat_count(m_tiles, launch_blocks)
    need_cast = dtype_name != "float32"
    @tilelang.jit(out_idx=[1], pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    })
    def factory():
        @T.prim_func
        def main(A: T.Tensor((m, n), dtype_name), C: T.Tensor((m_tiles * block_m,), dtype_name)):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((sub_m, block_n), dtype_name)
                x32 = T.alloc_ub((sub_m, block_n), "float32")
                work = T.alloc_ub((sub_m, block_n), "float32")
                tile = T.alloc_ub((sub_m,), "float32")
                row_max = T.alloc_ub((sub_m,), "float32")
                row_sum = T.alloc_ub((sub_m,), "float32")
                out = T.alloc_ub((sub_m,), dtype_name)
                with T.Scope("V"):
                    for rep in T.serial(grid_repeats):
                        logical_cid = cid + rep * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * block_m + vid * sub_m
                            T.tile.fill(row_max, -T.infinity("float32"))
                            T.tile.fill(row_sum, 0.0)
                            for nt in T.serial(n_tiles):
                                T.tile.fill(src, 0.0)
                                if not has_n_tail or nt < n_tiles - 1:
                                    if not has_m_tail:
                                        T.copy(A[row_base, nt * block_n], src)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(A[row_base + r, nt * block_n], src[r, :])
                                else:
                                    for r in T.serial(sub_m):
                                        if row_base + r < m:
                                            for c in T.serial(block_n):
                                                if c < tail_n:
                                                    src[r, c] = A[row_base + r, nt * block_n + c]
                                if need_cast:
                                    T.tile.cast(x32, src, "CAST_NONE", sub_m * block_n)
                                else:
                                    T.copy(src, x32)
                                T.reduce_max(x32, tile, dim=-1, clear=True)
                                T.tile.max(row_max, row_max, tile)
                            for nt in T.serial(n_tiles):
                                T.tile.fill(src, 0.0)
                                if not has_n_tail or nt < n_tiles - 1:
                                    if not has_m_tail:
                                        T.copy(A[row_base, nt * block_n], src)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(A[row_base + r, nt * block_n], src[r, :])
                                else:
                                    for r in T.serial(sub_m):
                                        if row_base + r < m:
                                            for c in T.serial(block_n):
                                                if c < tail_n:
                                                    src[r, c] = A[row_base + r, nt * block_n + c]
                                if need_cast:
                                    T.tile.cast(x32, src, "CAST_NONE", sub_m * block_n)
                                else:
                                    T.copy(src, x32)
                                T.tile.broadcast(work, row_max)
                                T.tile.sub(x32, x32, work)
                                T.tile.exp(x32, x32)
                                if has_n_tail and nt == n_tiles - 1:
                                    for r in T.serial(sub_m):
                                        for c in T.serial(block_n):
                                            if c >= tail_n:
                                                x32[r, c] = 0.0
                                T.reduce_sum(x32, tile, dim=-1, clear=True)
                                T.tile.add(row_sum, row_sum, tile)
                            T.tile.ln(row_sum, row_sum)
                            T.tile.add(row_sum, row_sum, row_max)
                            T.tile.cast(out, row_sum, "CAST_RINT", sub_m)
                            T.copy(out, C[row_base : row_base + sub_m])
        return main
    return factory()
# --- R262: the single-pass online logsumexp -----------------------------------
#
# `_compile_logsumexp` above is the ORIGINAL kernel, kept only so R262's A/B can
# be re-run (`TILEOPS_LSE_ONLINE=0`).  It has two independent defects, both
# measured in R262 §3:
#
#   1. `block_n = 128`, hardcoded.  Every GM->UB transfer is a `sub_m`-way
#      strided read of 128 * itemsize = 256 bytes, and the per-lane vector-op
#      count is proportional to `n / 128`.  For `lm-head-logits` (n = 102400)
#      that is 800 n-tiles.  This is exactly the defect R200 removed from the
#      normalization templates by introducing `common.plan_rowwise_norm`;
#      logsumexp was never converted.
#   2. TWO full passes over N (one for the max, one for the shifted exp-sum),
#      i.e. 2x the GM traffic and 2x the tile count of the one-pass form.
#
# Together: 1600 tile iterations for lm-head-logits where 80 suffice.  The
# natural experiment is in our own tree -- `families/two_pass.py::_compile`
# (SoftmaxFwdOp) already has BOTH fixes (planner + online recurrence) and
# measured 354.9 us on the identical (4, 102400) fp16 shape against this
# kernel's 2601.0 us, while additionally having to WRITE an (m, n) output that
# logsumexp does not.  7.3x of the gap is geometry, before any change to how
# the work is spread over cores.
#
# The recurrence is the standard online-softmax one and is taken from our own
# `families/two_pass.py` pass 1 (R200), not from either tilelang repo:
#
#     m_new = max(m_run, max(tile))
#     s_run = s_run * exp(m_run - m_new) + sum(exp(tile - m_new))
#     m_run = m_new
#   epilogue: y = m_run + ln(s_run)
#
# ⚠️ NOT bit-identical to the two-pass form: the two-pass version shifts every
# element by the FINAL row max, the online version shifts tile t by the running
# max after tile t and rescales.  Both are exact in exact arithmetic; in fp32
# they differ in the last bits.  R262 §5 has the fp64-reference comparison.
#
# ⚠️ An all-`-inf` row yields NaN here (`-inf - -inf`), where torch yields
# `-inf`.  This is NOT a regression: the two-pass kernel above does the same
# (`exp(-inf - -inf)`).  Recorded in R262's assumption list.
_LSE_ONLINE = os.environ.get("TILEOPS_LSE_ONLINE", "1") != "0"

#: Most-negative FINITE value per dtype, used to pad a ragged trailing n-tile.
#: `-inf` cannot be used: `exp(-inf - -inf)` is NaN, and an all-pad tile is
#: reachable whenever the m-tail leaves a lane with no copied data.
_LSE_PAD_VALUE = {
    "float16": -65504.0,
    "bfloat16": -3.38953139e38,
    "float32": -_FP32_MAX,
}


def _itemsize(dtype_name: str) -> int:
    return 2 if dtype_name in ("float16", "bfloat16") else 4


def _plan_logsumexp(m: int, n: int, dtype_name: str, out_dtype: str):
    """Geometry for the online logsumexp tile.

    UB census, read off the allocation list in `_compile_logsumexp_online`:
      (sub_m, block_n) -> src[itemsize] + x32[4] + work[4]
      (sub_m,)         -> row_max, row_sum, tile_max, tile_sum, tmp [5 x fp32]
                          + out[out_itemsize]
    A guessed census is a silent UB overflow (R194 2.2), so one spare fp32 row
    slot is budgeted on top of the five that are actually allocated.
    """
    itemsize = _itemsize(dtype_name)
    return plan_rowwise_norm(
        m, n, itemsize,
        bytes_per_cell=itemsize + 8,
        bytes_per_col=0,
        bytes_per_row=6 * 4 + _itemsize(out_dtype),
        grain=32 if n <= 512 else None,
    )


@lru_cache(maxsize=128)
def _compile_logsumexp_online(m: int, n: int, dtype_name: str, out_dtype: str):
    """One pass over N, planner-chosen tile, optional fp32 partial output.

    `out_dtype == "float32"` makes this a *stage 1*: the value it writes is the
    logsumexp of the row it was handed, which is exactly the quantity the
    two-stage axis split needs (see `_lse_axis_split_factor`).
    """
    plan = _plan_logsumexp(m, n, dtype_name, out_dtype)
    sub_m = plan["sub_m"]
    block_m = plan["block_m"]
    block_n = plan["block_n"]
    m_tiles = plan["m_tiles"]
    n_tiles = plan["n_tiles"]
    m_pad = plan["m_pad"]
    has_m_tail = plan["has_m_tail"]
    has_n_tail = plan["has_n_tail"]
    launch_blocks = plan["launch_blocks"]
    grid_repeats = plan["grid_repeats"]
    need_cast = dtype_name != "float32"
    out_need_cast = out_dtype != "float32"
    pad_value = _LSE_PAD_VALUE[dtype_name]

    @tilelang.jit(out_idx=[1], pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    })
    def factory():
        @T.prim_func
        def main(A: T.Tensor((m, n), dtype_name), C: T.Tensor((m_pad,), out_dtype)):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((sub_m, block_n), dtype_name)
                x32 = T.alloc_ub((sub_m, block_n), "float32")
                work = T.alloc_ub((sub_m, block_n), "float32")
                row_max = T.alloc_ub((sub_m,), "float32")
                row_sum = T.alloc_ub((sub_m,), "float32")
                tile_max = T.alloc_ub((sub_m,), "float32")
                tile_sum = T.alloc_ub((sub_m,), "float32")
                tmp = T.alloc_ub((sub_m,), "float32")
                out = T.alloc_ub((sub_m,), out_dtype)
                with T.Scope("V"):
                    for rep in T.serial(grid_repeats):
                        logical_cid = cid + rep * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * block_m + vid * sub_m
                            T.tile.fill(row_max, -T.infinity("float32"))
                            T.tile.fill(row_sum, 0.0)
                            for nt in T.serial(n_tiles):
                                # A lane with no copied data (m-tail) or a
                                # ragged trailing tile must read as "no
                                # contribution", so pre-fill with the most
                                # negative finite value of the dtype.
                                if has_n_tail or has_m_tail:
                                    T.tile.fill(src, pad_value)
                                if not has_n_tail or nt < n_tiles - 1:
                                    if not has_m_tail:
                                        T.copy(A[row_base, nt * block_n], src)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(A[row_base + r, nt * block_n], src[r, :])
                                else:
                                    for r in T.serial(sub_m):
                                        if row_base + r < m:
                                            for c in T.serial(block_n):
                                                if c < n - nt * block_n:
                                                    src[r, c] = A[row_base + r, nt * block_n + c]
                                if need_cast:
                                    T.tile.cast(x32, src, "CAST_NONE", sub_m * block_n)
                                else:
                                    T.copy(src, x32)
                                # --- online recurrence, one pass over N ---
                                T.reduce_max(x32, tile_max, dim=-1, clear=True)
                                T.tile.max(tile_max, row_max, tile_max)   # m_new
                                T.tile.sub(tmp, row_max, tile_max)        # m_old-m_new
                                T.tile.exp(tmp, tmp)                      # <= 1
                                T.tile.mul(tmp, row_sum, tmp)             # rescaled s
                                T.tile.broadcast(work, tile_max)
                                T.tile.sub(x32, x32, work)
                                T.tile.exp(x32, x32)
                                T.reduce_sum(x32, tile_sum, dim=-1, clear=True)
                                T.tile.add(row_sum, tile_sum, tmp)
                                T.copy(tile_max, row_max)
                            T.tile.ln(row_sum, row_sum)
                            T.tile.add(row_sum, row_sum, row_max)
                            if out_need_cast:
                                T.tile.cast(out, row_sum, "CAST_RINT", sub_m)
                                T.copy(out, C[row_base : row_base + sub_m])
                            else:
                                T.copy(row_sum, C[row_base : row_base + sub_m])
        return main
    return factory()


# --- R262: spreading a starved logsumexp over the reduction axis --------------
#
# The launch grid of every kernel in this file is a function of `m` alone, so a
# small `m` uses a small fraction of the machine no matter how large `n` is.
# `lm-head-logits` is m = 4: one row block, 1 of 48.  R198 already built the
# cure for the plain associative reductions (`_axis_split_factor` +
# `_FULL_STAGES` below); logsumexp short-circuited above it and never got it.
#
# logsumexp splits exactly as cleanly, because logsumexp of a concatenation is
# the logsumexp of the per-chunk logsumexps:
#
#     lse(x) = lse_j( lse(chunk_j) )
#
# so BOTH stages are the same kernel: stage 1 is `(m*k, n/k) -> (m*k,)` fp32
# partials, stage 2 is `(m, k) -> (m,)` in the output dtype.  `k` must divide
# `n` only so that the reshape is a view; unlike `mean` the identity does not
# require equal chunk sizes.
#
# ⚠️ This is a REASSOCIATION, not a bit-identical rewrite (see the online
# recurrence note above) -- and stage 1 rounds its partial to fp32, which the
# unsplit kernel never does.  fp32 holds a logsumexp of this magnitude to ~1e-7
# relative; R262 §5 measures it against an fp64 reference.
_LSE_AXIS_SPLIT = os.environ.get("TILEOPS_LSE_AXIS_SPLIT", "1") != "0"

#: Stage 1 wants at least this many launch blocks before the extra kernel and
#: GM round trip pay for themselves.  48 = `common.VECTOR_CORE_COUNT`.
_LSE_SPLIT_TARGET_BLOCKS = 48
#: ... and a case that already reaches this many is left alone.  8 is R198's
#: `_SPLIT_MIN_BLOCKS`, kept identical so the two splits agree on "starved".
_LSE_SPLIT_MIN_BLOCKS = 8
#: Stage 1 must keep a wide tile or the split trades starvation for a narrow
#: `block_n`, which is defect (1) again.  `common.TILE_WIDTH_CAP` is 1280 and
#: the planner rounds down to a divisor, so require a chunk of at least this.
#:
#: ⚠️ 256, not 512.  512 was the first value and it forced `k = 200` for
#: `lm-head-logits`, which is the STAGE 2 TRAP below: 200 is not a multiple of
#: the planner's small-n grain (32), so `plan_rowwise_norm` rounds the tile up
#: to 224, `has_n_tail` becomes True, and with `n_tiles == 1` that single tile
#: is the ragged one -- so ALL of stage 2's load goes through the
#: `for r: for c:` double scalar loop, 8 * 224 = 1792 scalar UB writes.
#: Measured (R262-data/39_stage_timing.json): stage 1 = 5.75 us on 48 blocks
#: reading the whole 800 KiB input, stage 2 = 36.50 us on ONE block reading
#: 3.2 KiB.  Stage 2 was 86% of the op.
_LSE_SPLIT_MIN_CHUNK = 256


def _lse_axis_split_factor(m: int, n: int, dtype_name: str, out_dtype: str):
    """Chunks to cut logsumexp's reduction axis into, or None to leave it alone.

    Chosen against the ACTUAL planner output rather than a nominal block height:
    `plan_rowwise_norm` picks `sub_m` (hence `block_m`) from the UB budget, so
    the number of launch blocks a given `k` buys is not knowable without asking
    it.  Candidates are scored on (blocks reached, chunk width kept).
    """
    if not _LSE_AXIS_SPLIT or m <= 0 or n <= 0:
        return None
    try:
        base = _plan_logsumexp(m, n, dtype_name, out_dtype)
    except NotImplementedError:
        return None
    if base["launch_blocks"] >= _LSE_SPLIT_MIN_BLOCKS:
        return None                      # not starved; leave it alone
    best = None
    best_blocks = base["launch_blocks"]
    k = 2
    while k <= n // _LSE_SPLIT_MIN_CHUNK:
        if n % k == 0:
            try:
                plan = _plan_logsumexp(m * k, n // k, dtype_name, "float32")
            except NotImplementedError:
                k += 1
                continue
            blocks = plan["launch_blocks"]
            # 🚨 REJECT any k that leaves EITHER stage on a ragged tile.  Stage 2 is
            # a single launch block by construction (m is small -- that is why
            # we are splitting), so it has nothing to hide a scalar gather
            # behind, and `n_tiles == 1` means the ragged tile is the only
            # tile.  This one predicate is worth 2.4x of the whole op; see
            # `_LSE_SPLIT_MIN_CHUNK`.
            try:
                stage2 = _plan_logsumexp(m, k, "float32", out_dtype)
            except NotImplementedError:
                k += 1
                continue
            if stage2["has_n_tail"] or plan["has_n_tail"]:
                # Stage 1 matters for the same reason once the split has done
                # its job: `k` is chosen so stage 1 has ONE n-tile per lane, so
                # a ragged tile there is not amortised either.  Dropping only
                # the stage-2 predicate picked k=256 for `lm-head-logits`,
                # whose stage 1 is a ragged 416-wide tile -- it moves the
                # scalar gather from stage 2 to stage 1 rather than removing
                # it.  Requiring both gives k=320: 48 blocks, one 320-wide
                # tile per lane in stage 1, one 320-wide tile in stage 2, no
                # scalar path anywhere.
                k += 1
                continue
            # Prefer more blocks; on a tie prefer the SMALLER k, i.e. the wider
            # stage-1 chunk and the cheaper stage 2.
            if blocks > best_blocks:
                best_blocks = blocks
                best = k
            if best_blocks >= _LSE_SPLIT_TARGET_BLOCKS:
                break
        k += 1
    return best


_LOGICAL_OPS = frozenset({"all", "any"})


def _dtype_name(dtype: torch.dtype, op_name: str) -> str:
    names = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise TypeError(
            f"{op_name} supports float16, bfloat16, and float32, got {dtype}"
        ) from exc


def _axes(dim, ndim: int, *, empty_is_full: bool = True) -> tuple[int, ...]:
    if dim is None:
        values = tuple(range(ndim))
    elif isinstance(dim, int):
        values = (dim,)
    else:
        values = tuple(dim)
        if not values and empty_is_full:
            values = tuple(range(ndim))
    raw_axes = tuple(int(axis) for axis in values)
    if any(axis < -ndim or axis >= ndim for axis in raw_axes):
        raise ValueError(f"reduction dimension out of range for rank {ndim}: {dim!r}")
    axes = tuple(sorted(axis + ndim if axis < 0 else axis for axis in raw_axes))
    if len(set(axes)) != len(axes):
        raise ValueError(f"reduction dimensions must be unique, got {dim!r}")
    return axes


def _identity(op_kind: str) -> float:
    if op_kind == "amax":
        return -_FP32_MAX
    if op_kind == "amin":
        return _FP32_MAX
    if op_kind == "all":
        return 1.0
    return 0.0


@lru_cache(maxsize=128)
def _compile_extra(m: int, n: int, dtype_name: str, op_kind: str, correction: int,
                   out_dtype: str | None = None, groups: int = 1):
    """Shared fp32 Welford/product reduction kernel.

    The Welford state follows the Chan merge used by the normalization family:
    every input tile produces ``(count, mean, M2)`` in fp32 and is merged into
    the running state.  ``correction`` is applied only to the final denominator.

    R198: the merge granularity changed from one *element* to one *tile*.  The
    state is still ``(count, mean, M2)`` in fp32 and the merge is still the Chan
    formula, but ``(count, mean, M2)`` for a whole ``(sub_m, block_n)`` tile is
    now produced by two vector ``reduce_sum`` passes instead of ``sub_m *
    block_n`` scalar Welford updates, and the merge itself runs on the
    ``(sub_m,)`` state vectors.  This is a strictly coarser -- and in the usual
    error analysis no worse -- summation order, but it is *not* bit-identical to
    the per-element version: floating point addition is not associative.  See
    R198 for the fp64-reference error comparison.

    Product is likewise accumulated into a full tile of running products and
    collapsed once at the end, so its multiplication order changes too.  fp
    multiplication is not associative either, and the reassociation also moves
    where an intermediate may overflow to inf.
    """
    # R198 r3: ``welford_partial`` emits the raw (M2, mean) pair in fp32 instead
    # of the finished variance, so a second stage can combine the partials.
    out_dtype = dtype_name if out_dtype is None else out_dtype
    # NB: every writeback below tests ``out_dtype``, never ``dtype_name``.  They
    # agree whenever the output dtype follows the input, which is why the bug
    # below stayed hidden for so long -- but the split asks this factory for
    # fp32 partials from a bf16 input, and the old ``dtype_name == "float32"``
    # test then took the *cast* branch and applied CAST_RINT to fp32 -> fp32,
    # i.e. rounded the partial products to integers.  Measured damage before the
    # fix (probe: /tmp/r198_prod_ab.py, recorded in R4.3): bf16 relative error
    # 3.4e-03 -> 3.8e+04, while fp32 was unaffected because it took T.copy.
    # This is the same defect class already fixed in _compile and
    # _compile_scalar in round 2.
    partial_mode = op_kind == "welford_partial"
    if partial_mode:
        op_kind = "var"          # same accumulation, different writeback
    block_n = 128
    block_m = _BLOCK_M
    sub_m = _SUB_M
    m_tiles = (m + block_m - 1) // block_m
    n_tiles = (n + block_n - 1) // block_n
    has_m_tail = m % block_m != 0
    has_n_tail = n % block_n != 0
    tail_n = n % block_n
    launch_blocks = launch_block_count(m_tiles)
    grid_repeats = grid_repeat_count(m_tiles, launch_blocks)
    need_cast = dtype_name != "float32"
    # Keep one static two-output ABI for this shared factory.  Single-output
    # callers discard the second allocation in the Python launch wrapper;
    # VarMean exposes both.  This avoids conditional prim_func signatures.
    outputs = [1, 2]

    @tilelang.jit(out_idx=outputs, pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    })
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((groups * m, n), dtype_name),
            C: T.Tensor((m_tiles * block_m,), out_dtype),
            D: T.Tensor((m_tiles * block_m,), out_dtype),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((sub_m, block_n), dtype_name)
                x32 = T.alloc_ub((sub_m, block_n), "float32")
                work = T.alloc_ub((sub_m, block_n), "float32")
                partial = T.alloc_ub((sub_m,), "float32")
                count = T.alloc_ub((sub_m,), "float32")
                mean = T.alloc_ub((sub_m,), "float32")
                m2 = T.alloc_ub((sub_m,), "float32")
                tile_mean = T.alloc_ub((sub_m,), "float32")
                tile_count = T.alloc_ub((sub_m,), "float32")
                row_m2 = T.alloc_ub((sub_m,), "float32")
                delta = T.alloc_ub((sub_m,), "float32")
                merged = T.alloc_ub((sub_m,), "float32")
                ratio = T.alloc_ub((sub_m,), "float32")
                cross = T.alloc_ub((sub_m,), "float32")
                out = T.alloc_ub((sub_m,), out_dtype)
                # R198: packed per-column validity mask for the trailing tile,
                # built once per launch.  The deviations of the padded lanes must
                # be forced to 0 before they reach the M2 reduction, and the
                # padded lanes must be forced to the *multiplicative* identity
                # 1.0 before they reach the product accumulator -- neither is 0
                # and the input fill is 0, so getting this wrong is a silent
                # wrong-value bug of exactly the class recorded in R090.
                valid = T.alloc_ub((sub_m * block_n // 8,), "uint8")
                with T.Scope("V"):
                    if has_n_tail:
                        # valid[r, c] = (c < tail_n).  Built from a column ramp
                        # laid down once per launch; x32/work are still dead here
                        # so they double as scratch.  arith_progression fills
                        # flattened storage, hence the per-row base subtraction
                        # (verified in R198-data/probe/probe_intrinsics.json q1).
                        T.tile.arith_progression(work, 0.0, 1.0, sub_m * block_n)
                        for r in T.serial(sub_m):
                            partial[r] = T.Cast("float32", r * block_n)
                        T.tile.broadcast(x32, partial)
                        T.tile.sub(work, work, x32)
                        T.tile.compare(valid, work, float(tail_n), "LT")
                    for rep in T.serial(grid_repeats):
                        logical_cid = cid + rep * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * block_m + vid * sub_m
                            T.tile.fill(count, 0.0)
                            T.tile.fill(mean, 0.0)
                            T.tile.fill(m2, 0.0)
                            # Product is deliberately accumulated in fp32.  It is
                            # mathematically exact until IEEE fp32 overflow/underflow.
                            if op_kind == "prod":
                                T.tile.fill(mean, 1.0)
                                # R198: block_n independent running products.
                                T.tile.fill(work, 1.0)
                            for g in T.serial(groups):
                                for nt in T.serial(n_tiles):
                                    T.tile.fill(src, 0.0)
                                    if not has_n_tail or nt < n_tiles - 1:
                                        if not has_m_tail:
                                            T.copy(A[g * m + row_base, nt * block_n], src)
                                        else:
                                            for r in T.serial(sub_m):
                                                if row_base + r < m:
                                                    T.copy(A[g * m + row_base + r, nt * block_n], src[r, :])
                                    else:
                                        # R198: the whole trailing tile is copied and
                                        # the invalid columns are masked below, which
                                        # replaces a per-element scalar gather.
                                        if not has_m_tail:
                                            T.copy(A[g * m + row_base, nt * block_n], src, pad_value=0)
                                        else:
                                            for r in T.serial(sub_m):
                                                if row_base + r < m:
                                                    T.copy(A[g * m + row_base + r, nt * block_n],
                                                           src[r, :], pad_value=0)
                                    # R198 r3: MTE2 -> V.  ``src`` was just filled
                                    # by a GM copy and is about to be read by a
                                    # vector op.  _compile and count_nonzero both
                                    # barrier at this exact point; _compile_extra
                                    # did not, and its main path used to be a scalar
                                    # Welford loop that ordered things implicitly.
                                    # Now that it is vector, the implicit guarantee
                                    # is gone (§13.55 #1/#2).
                                    T.barrier_all()
                                    if need_cast:
                                        T.tile.cast(x32, src, "CAST_NONE", sub_m * block_n)
                                    else:
                                        T.copy(src, x32)
                                    if has_n_tail and nt == n_tiles - 1:
                                        # Drive the padded columns of the trailing
                                        # tile to this reduction's identity BEFORE
                                        # any reduce touches them.  The identity is
                                        # 1.0 for a product and 0.0 for a sum -- not
                                        # the 0 that the input fill happens to leave
                                        # behind (R090 / PROJECT_STATE 13.20).  Doing
                                        # this here rather than relying on T.copy's
                                        # pad_value keeps the tile sum correct even
                                        # if the padded read returns neighbouring row
                                        # data instead of zeros.
                                        T.tile.select(
                                            x32, valid, x32,
                                            1.0 if op_kind == "prod" else 0.0,
                                            "VSEL_TENSOR_SCALAR_MODE",
                                        )
                                    if op_kind == "prod":
                                        # R198: was sub_m * block_n scalar multiplies
                                        # per tile on the main path.  The running
                                        # products live in a full tile and are
                                        # collapsed once, after the n loop.
                                        T.tile.mul(work, work, x32)
                                    else:
                                        # R198: Chan merge, one tile at a time.
                                        # tile_sum -> tile_mean; deviations from
                                        # tile_mean -> tile M2.  Same state and same
                                        # merge formula as the scalar version, but the
                                        # per-tile statistics come from two vector
                                        # reduce_sum passes.
                                        T.reduce_sum(x32, partial, dim=-1, clear=True)
                                        if has_n_tail and nt == n_tiles - 1:
                                            T.tile.fill(tile_count, float(tail_n))
                                        else:
                                            T.tile.fill(tile_count, float(block_n))
                                        T.tile.div(tile_mean, partial, tile_count)
                                        T.tile.broadcast(work, tile_mean)
                                        T.tile.sub(work, x32, work)
                                        if has_n_tail and nt == n_tiles - 1:
                                            # A padded lane holds (0 - tile_mean),
                                            # which would otherwise be counted as a
                                            # real deviation.  Force it to 0.
                                            T.tile.select(work, valid, work, 0.0,
                                                          "VSEL_TENSOR_SCALAR_MODE")
                                        T.tile.mul(work, work, work)
                                        T.reduce_sum(work, row_m2, dim=-1, clear=True)
                                        # merged = count + tile_count
                                        T.tile.add(merged, count, tile_count)
                                        # delta  = tile_mean - mean
                                        T.tile.sub(delta, tile_mean, mean)
                                        # mean  += delta * tile_count / merged
                                        T.tile.mul(ratio, delta, tile_count)
                                        T.tile.div(ratio, ratio, merged)
                                        T.tile.add(mean, mean, ratio)
                                        # m2 += tile_m2 + delta^2 * count*tile_count/merged
                                        T.tile.mul(cross, count, tile_count)
                                        T.tile.div(cross, cross, merged)
                                        T.tile.mul(ratio, delta, delta)
                                        T.tile.mul(cross, cross, ratio)
                                        T.tile.add(m2, m2, row_m2)
                                        T.tile.add(m2, m2, cross)
                                        T.copy(merged, count)
                            if op_kind == "prod":
                                # R198: collapse the block_n running products.
                                # This is the only scalar loop left on the prod
                                # path and it now runs once per row block rather
                                # than once per input tile, i.e. its cost is
                                # O(sub_m * block_n) instead of O(sub_m * n).
                                # V -> scalar: ``work`` holds vector-written
                                # running products (§13.55 #2).
                                T.barrier_all()
                                for r in T.serial(sub_m):
                                    for c in T.serial(block_n):
                                        mean[r] = mean[r] * work[r, c]
                                if out_dtype == "float32":
                                    T.copy(mean, out)
                                else:
                                    T.tile.cast(out, mean, "CAST_RINT", sub_m)
                                # V -> MTE3 (§13.55 #2)
                                T.barrier_all()
                                T.copy(out, C[row_base : row_base + sub_m])
                            elif partial_mode:
                                # Raw partials, fp32, no division and no cast:
                                # stage 2 needs M2 and mean exactly as computed.
                                T.barrier_all()          # V -> MTE3 (13.55 #2)
                                T.copy(m2, C[row_base : row_base + sub_m])
                                T.barrier_all()
                                T.copy(mean, D[row_base : row_base + sub_m])
                            else:
                                T.tile.div(partial, m2,
                                           T.cast(float(groups * n - correction), "float32"))
                                if op_kind == "std":
                                    T.tile.sqrt(partial, partial)
                                if op_kind == "var_mean":
                                    if out_dtype == "float32":
                                        T.copy(partial, out)
                                    else:
                                        T.tile.cast(out, partial, "CAST_RINT", sub_m)
                                    T.barrier_all()          # V -> MTE3 (§13.55 #2)
                                    T.copy(out, C[row_base : row_base + sub_m])
                                    if out_dtype == "float32":
                                        T.copy(mean, out)
                                    else:
                                        T.tile.cast(out, mean, "CAST_RINT", sub_m)
                                    T.barrier_all()          # V -> MTE3 (§13.55 #2)
                                    T.copy(out, D[row_base : row_base + sub_m])
                                else:
                                    if out_dtype == "float32":
                                        T.copy(partial, out)
                                    else:
                                        T.tile.cast(out, partial, "CAST_RINT", sub_m)
                                    T.barrier_all()          # V -> MTE3 (§13.55 #2)
                                    T.copy(out, C[row_base : row_base + sub_m])
        return main
    return factory()


@lru_cache(maxsize=128)
def _compile_welford_combine(k: int, n: int, out_dtype: str, op_kind: str,
                             correction: int):
    """Stage 2 of the split variance: combine k equal-sized Welford partials.

    Every group holds exactly ``c = n // k`` elements, and for *equal* group
    sizes the parallel-variance combination collapses to two ordinary sums:

        mean_total = (1 / k) * sum_i mean_i
        M2_total   = sum_i M2_i  +  c * sum_i (mean_i - mean_total)^2

    So no Chan merge tree is needed here -- both terms are plain reductions over
    a k-element vector.  This is the standard equal-group form and is exact in
    exact arithmetic, not an approximation; it is the same identity that makes
    ``mean -> mean`` valid in _FULL_STAGES, which is why k must divide n.

    k <= _FULL_ROWS_CAP = 12288, so three fp32 tiles of k lanes is at most
    147456 bytes of UB.
    """
    c = n // k

    @tilelang.jit(out_idx=[2, 3], pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    })
    def factory():
        @T.prim_func
        def main(
            MEANS: T.Tensor((1, k), "float32"),
            M2S: T.Tensor((1, k), "float32"),
            C: T.Tensor((_PADDED_OUTPUT,), out_dtype),
            D: T.Tensor((_PADDED_OUTPUT,), out_dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                mu = T.alloc_ub((1, k), "float32")
                m2 = T.alloc_ub((1, k), "float32")
                work = T.alloc_ub((1, k), "float32")
                acc = T.alloc_ub((1,), "float32")
                mean_tot = T.alloc_ub((1,), "float32")
                m2_tot = T.alloc_ub((1,), "float32")
                dev = T.alloc_ub((1,), "float32")
                padded = T.alloc_ub((_PADDED_OUTPUT,), "float32")
                out = T.alloc_ub((_PADDED_OUTPUT,), out_dtype)
                with T.Scope("V"):
                    if vid == 0:
                        T.copy(MEANS, mu)
                        T.copy(M2S, m2)
                        T.barrier_all()          # MTE2 -> V (13.55 #1/#2)
                        T.reduce_sum(mu, acc, dim=-1, clear=True)
                        T.tile.mul(mean_tot, acc, 1.0 / k)
                        T.reduce_sum(m2, m2_tot, dim=-1, clear=True)
                        T.tile.broadcast(work, mean_tot)
                        T.tile.sub(work, mu, work)
                        T.tile.mul(work, work, work)
                        T.reduce_sum(work, dev, dim=-1, clear=True)
                        T.tile.mul(dev, dev, float(c))
                        T.tile.add(m2_tot, m2_tot, dev)
                        T.tile.div(
                            m2_tot, m2_tot,
                            T.cast(float(n - correction), "float32"),
                        )
                        if op_kind == "std":
                            T.tile.sqrt(m2_tot, m2_tot)
                        T.tile.fill(padded, 0.0)
                        T.tile.broadcast(padded, m2_tot)
                        T.barrier_all()          # V -> MTE3 (13.55 #2)
                        if out_dtype == "float32":
                            T.copy(padded, C)
                        else:
                            T.tile.cast(out, padded, "CAST_RINT", _PADDED_OUTPUT)
                            T.barrier_all()
                            T.copy(out, C)
                        T.tile.fill(padded, 0.0)
                        T.tile.broadcast(padded, mean_tot)
                        T.barrier_all()
                        if out_dtype == "float32":
                            T.copy(padded, D)
                        else:
                            T.tile.cast(out, padded, "CAST_RINT", _PADDED_OUTPUT)
                            T.barrier_all()
                            T.copy(out, D)
        return main

    return factory()


@lru_cache(maxsize=128)
def _compile(
    m: int,
    n: int,
    dtype_name: str,
    out_dtype: str,
    op_kind: str,
    diagnostic_sentinel: bool,
    block_n: int = _BLOCK_N,
    groups: int = 1,
):
    # R340: ``groups`` is the leading-reduced-axis extent.  The GM tensor is
    # declared ``(groups * m, n)`` and row ``i`` of the logical reduction is the
    # concatenation of rows ``g * m + i`` for ``g in [0, groups)``.  That is
    # exactly the memory layout of a contiguous ``(groups, m, n)`` tensor whose
    # first and last axis are reduced and whose middle axis is kept -- i.e.
    # ``[4,128,4096] dim=[0,2]``, the case that sets four of this round's five
    # ``ratio_min`` values.  Before R340 the launch wrapper handed that case a
    # ``permute(...).contiguous()`` copy of the WHOLE tensor
    # (``aclnnInplaceCopy_TransposeAiCore_Transpose`` in the coverage kernel
    # names); with ``groups`` the same reduction reads each element once,
    # straight out of the caller's buffer, in the same contiguous
    # ``(_SUB_M, block_n)`` tiles as before.
    # R198 r3: block_n is a parameter (and part of the lru_cache key).  Stage 2
    # of the axis split reduces only k columns, and k is small by construction
    # (64 for the [4,128,4096] dim=[0,2] case).  With the fixed 256-wide tile
    # that left n % block_n != 0, and this factory's trailing-tile path is a
    # per-lane scalar gather -- _SUB_M * block_n = 16384 scalar iterations to
    # reduce 64 columns.  Measured cost of getting it wrong: that one case went
    # 129.7 us -> 793.6 us, i.e. the split was *correct but 6x slower* (9.8x on
    # the uint8 path, which takes the scalar loop twice).
    block_n = int(block_n)
    m_tiles = (m + _BLOCK_M - 1) // _BLOCK_M
    n_tiles = (n + block_n - 1) // block_n
    has_m_tail = m % _BLOCK_M != 0
    has_n_tail = n % block_n != 0
    tail_n = n % block_n
    launch_blocks = launch_block_count(m_tiles)
    grid_repeats = grid_repeat_count(m_tiles, launch_blocks)
    _narrow_shape = (_SUB_M, block_n) if dtype_name == "uint8" else (1,)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((groups * m, n), dtype_name),
            C: T.Tensor((m_tiles * _BLOCK_M,), out_dtype),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((_SUB_M, block_n), dtype_name)
                calc = T.alloc_ub((_SUB_M, block_n), "float32")
                # R198: fp16 landing pad for the bool/uint8 -> fp32 conversion.
                # dav-2201 has no direct uint8 -> fp32 Cast overload, but the
                # uint8 -> fp16 -> fp32 pair is exact (probe_uint8_ops.json).
                # fp16 landing pad for the bool/uint8 -> fp32 conversion; a
                # 2-byte stub on every other path.  Three constraints forced
                # this exact shape, each one a compile cycle:
                #   - it must not be called ``half``: that is AscendC's fp16
                #     type name and the emitted ``auto half = ..`` will not
                #     compile;
                #   - it must not be a conditional *expression*
                #     (``x if c else None``): that segfaults the TVMScript
                #     parser, which tries to build TIR from both arms;
                #   - it must not be allocated inside ``if dtype_name ==
                #     "uint8":`` either: the name is then scoped to that block
                #     and the later use reports "Undefined variable".
                narrow = T.alloc_ub(_narrow_shape, "float16")
                partial = T.alloc_ub((_SUB_M,), "float32")
                accum = T.alloc_ub((_SUB_M,), "float32")
                mask = T.alloc_ub((_SUB_M // 8,), "uint8")
                out = T.alloc_ub((_SUB_M,), out_dtype)
                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * _BLOCK_M + vid * _SUB_M
                            if diagnostic_sentinel:
                                if out_dtype == "uint8":
                                    for r in T.serial(_SUB_M):
                                        out[r] = 0
                                else:
                                    T.tile.fill(out, 123)
                                T.copy(out, C[row_base : row_base + _SUB_M])
                                T.barrier_all()
                            T.tile.fill(accum, _identity(op_kind))
                            for g in T.serial(groups):
                                for nt in T.serial(n_tiles):
                                    if dtype_name == "uint8":
                                        # T.tile.fill has no uint8 form: AscendC
                                        # Duplicate rejects uint8 (R198 probe
                                        # probe_uint8_ops.json), which is why this
                                        # was a scalar loop.  It is only *needed*
                                        # where some lane of src is not overwritten
                                        # by the copy below, i.e. on a partial tile.
                                        # On a full tile it was 16384 dead scalar
                                        # stores sitting on the main path.
                                        if has_m_tail or has_n_tail:
                                            for r in T.serial(_SUB_M):
                                                for c in T.serial(block_n):
                                                    src[r, c] = 0
                                    else:
                                        T.tile.fill(src, 0)
                                    if not has_m_tail:
                                        if not has_n_tail or nt < n_tiles - 1:
                                            T.copy(A[g * m + row_base, nt * block_n], src)
                                        else:
                                            for r in T.serial(_SUB_M):
                                                for c in T.serial(block_n):
                                                    if c < tail_n:
                                                        src[r, c] = A[
                                                            g * m + row_base + r, nt * block_n + c
                                                        ]
                                    elif logical_cid < m_tiles - 1:
                                        if not has_n_tail or nt < n_tiles - 1:
                                            T.copy(A[g * m + row_base, nt * block_n], src)
                                        else:
                                            for r in T.serial(_SUB_M):
                                                for c in T.serial(block_n):
                                                    if c < tail_n:
                                                        src[r, c] = A[
                                                            g * m + row_base + r, nt * block_n + c
                                                        ]
                                    else:
                                        for r in T.serial(_SUB_M):
                                            if row_base + r < m:
                                                if not has_n_tail or nt < n_tiles - 1:
                                                    T.copy(
                                                        A[g * m + row_base + r, nt * block_n],
                                                        src[r, :],
                                                    )
                                                else:
                                                    for c in T.serial(block_n):
                                                        if c < tail_n:
                                                            src[r, c] = A[
                                                                g * m + row_base + r,
                                                                nt * block_n + c,
                                                            ]
                                    T.barrier_all()
                                    if dtype_name == "float32":
                                        T.copy(src, calc)
                                    elif dtype_name == "uint8":
                                        # R198: was 16384 scalar if_then_else per
                                        # tile on the main path.  ``min(v, 1.0)``
                                        # reproduces ``v != 0 ? 1.0 : 0.0`` exactly
                                        # for every uint8 value (uint8 is >= 0, so
                                        # 0 -> 0.0 and anything >= 1 -> 1.0), which
                                        # keeps this bit-identical even for a
                                        # non-canonical bool byte such as 5.
                                        T.tile.cast(
                                            narrow, src, "CAST_NONE", _SUB_M * block_n
                                        )
                                        T.tile.cast(
                                            calc, narrow, "CAST_NONE", _SUB_M * block_n
                                        )
                                        T.tile.min(calc, calc, 1.0)
                                    else:
                                        T.tile.cast(
                                            calc, src, "CAST_NONE", _SUB_M * block_n
                                        )
                                    if op_kind in {"l1", "inf"}:
                                        T.tile.abs(calc, calc)
                                    elif op_kind == "l2":
                                        T.tile.mul(calc, calc, calc)
                                    elif op_kind in _LOGICAL_OPS:
                                        # R198: dropped a third 16384-iteration
                                        # scalar loop that re-clamped calc to 0/1.
                                        # Provably dead: logical ops are the only
                                        # ops with dtype_name == "uint8" (the
                                        # builder ties them together), so calc is
                                        # already exactly 0.0 or 1.0 here.
                                        pass
                                    if has_n_tail and nt == n_tiles - 1:
                                        for r in T.serial(_SUB_M):
                                            for c in T.serial(block_n):
                                                if c >= tail_n:
                                                    calc[r, c] = _identity(op_kind)
                                    if op_kind in {"sum", "mean", "l1", "l2"}:
                                        T.reduce_sum(calc, partial, dim=-1, clear=True)
                                    elif op_kind in {"amax", "inf", "any"}:
                                        T.reduce_max(calc, partial, dim=-1, clear=True)
                                    else:
                                        T.reduce_min(calc, partial, dim=-1, clear=True)
                                    T.barrier_all()
                                    if op_kind in {"sum", "mean", "l1", "l2"}:
                                        T.tile.add(accum, accum, partial)
                                    elif op_kind in {"amax", "inf", "any"}:
                                        T.tile.max(accum, accum, partial)
                                    else:
                                        T.tile.min(accum, accum, partial)
                            if op_kind == "mean":
                                T.tile.mul(accum, accum, 1.0 / (groups * n))
                            elif op_kind == "l2":
                                T.tile.sqrt(accum, accum)
                            if out_dtype == "uint8":
                                T.tile.compare(mask, accum, 0.0, "NE")
                                T.barrier_all()      # V -> scalar (§13.55 #2)
                                for r in T.serial(_SUB_M):
                                    out[r] = (mask[r // 8] >> (r % 8)) & 1
                                T.copy(out, C[row_base : row_base + _SUB_M])
                            elif out_dtype == "float32":
                                # R198: was ``dtype_name == "float32"``.  These
                                # agree for every op whose output dtype follows
                                # its input, but the two-stage full reduction
                                # asks for fp32 partials from an fp16 input, and
                                # taking the cast branch would then apply
                                # CAST_RINT to fp32->fp32 and round to integer.
                                # V -> MTE3.  The fp16/bf16 branch below always
                                # had this barrier; the fp32 branch did not, and
                                # the two-stage split routes far more traffic
                                # through fp32 partials (§13.55 #2).
                                T.barrier_all()
                                T.copy(accum, C[row_base : row_base + _SUB_M])
                            else:
                                T.tile.cast(out, accum, "CAST_RINT", _SUB_M)
                                T.barrier_all()
                                T.copy(out, C[row_base : row_base + _SUB_M])

        return main

    return factory()


@lru_cache(maxsize=128)
def _compile_scalar(
    n: int,
    dtype_name: str,
    out_dtype: str,
    op_kind: str,
    diagnostic_sentinel: bool,
    block_n: int = _SCALAR_BLOCK_N,
):
    # R198: block_n is a parameter (and part of the lru_cache key) so that stage 2
    # of the two-stage full reduction can size its tile to the partial count.  A
    # fixed 16384 leaves n % block_n != 0 for every realistic partial count, and
    # the trailing-tile path here is a per-lane scalar gather -- 16384 scalar
    # iterations to reduce a few thousand values.
    block_n = int(block_n)
    n_tiles = (n + block_n - 1) // block_n
    has_n_tail = n % block_n != 0
    tail_n = n % block_n
    # See _compile: allocated unconditionally, stubbed when unused.
    _narrow_shape = (1, block_n) if dtype_name == "uint8" else (1,)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((1, n), dtype_name),
            C: T.Tensor((_PADDED_OUTPUT,), out_dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                src = T.alloc_ub((1, block_n), dtype_name)
                calc = T.alloc_ub((1, block_n), "float32")
                narrow = T.alloc_ub(_narrow_shape, "float16")
                partial = T.alloc_ub((1,), "float32")
                accum = T.alloc_ub((1,), "float32")
                padded = T.alloc_ub((_PADDED_OUTPUT,), "float32")
                mask = T.alloc_ub((_PADDED_OUTPUT // 8,), "uint8")
                out = T.alloc_ub((_PADDED_OUTPUT,), out_dtype)
                with T.Scope("V"):
                    if diagnostic_sentinel and vid == 0:
                        if out_dtype == "uint8":
                            for r in T.serial(_PADDED_OUTPUT):
                                out[r] = 0
                        else:
                            T.tile.fill(out, 123)
                        T.copy(out, C)
                        T.barrier_all()
                    T.tile.fill(accum, _identity(op_kind))
                    if vid == 0:
                        for nt in T.serial(n_tiles):
                            if dtype_name == "uint8":
                                # See _compile: no uint8 T.tile.fill exists, and
                                # the fill is only load-bearing on a partial tile.
                                if has_n_tail:
                                    for c in T.serial(block_n):
                                        src[0, c] = 0
                            else:
                                T.tile.fill(src, 0)
                            if not has_n_tail or nt < n_tiles - 1:
                                T.copy(A[0, nt * block_n], src)
                            else:
                                for c in T.serial(block_n):
                                    if c < tail_n:
                                        src[0, c] = A[0, nt * block_n + c]
                            if dtype_name == "float32":
                                T.copy(src, calc)
                            elif dtype_name == "uint8":
                                # R198: was block_n (16384) scalar if_then_else
                                # per tile on the main path.  See _compile.
                                T.tile.cast(narrow, src, "CAST_NONE", block_n)
                                T.tile.cast(calc, narrow, "CAST_NONE", block_n)
                                T.tile.min(calc, calc, 1.0)
                            else:
                                T.tile.cast(calc, src, "CAST_NONE", block_n)
                            if op_kind in {"l1", "inf"}:
                                T.tile.abs(calc, calc)
                            elif op_kind == "l2":
                                T.tile.mul(calc, calc, calc)
                            elif op_kind in _LOGICAL_OPS:
                                # R198: dropped, provably dead.  See _compile.
                                pass
                            if has_n_tail and nt == n_tiles - 1:
                                for c in T.serial(block_n):
                                    if c >= tail_n:
                                        calc[0, c] = _identity(op_kind)
                            if op_kind in {"sum", "mean", "l1", "l2"}:
                                T.reduce_sum(calc, partial, dim=-1, clear=True)
                            elif op_kind in {"amax", "inf", "any"}:
                                T.reduce_max(calc, partial, dim=-1, clear=True)
                            else:
                                T.reduce_min(calc, partial, dim=-1, clear=True)
                            if op_kind in {"sum", "mean", "l1", "l2"}:
                                T.tile.add(accum, accum, partial)
                            elif op_kind in {"amax", "inf", "any"}:
                                T.tile.max(accum, accum, partial)
                            else:
                                T.tile.min(accum, accum, partial)
                        if op_kind == "mean":
                            T.tile.mul(accum, accum, 1.0 / n)
                        elif op_kind == "l2":
                            T.tile.sqrt(accum, accum)
                        if out_dtype == "uint8":
                            T.tile.fill(padded, 0.0)
                            T.tile.broadcast(padded, accum)
                            T.tile.compare(mask, padded, 0.0, "NE")
                            T.barrier_all()      # V -> scalar (§13.55 #2)
                            for r in T.serial(_PADDED_OUTPUT):
                                out[r] = (mask[r // 8] >> (r % 8)) & 1
                            T.copy(out, C)
                        else:
                            T.tile.fill(padded, 0.0)
                            T.tile.broadcast(padded, accum)
                            # R198: was ``dtype_name == "float32"``.  This branch
                            # picks how the fp32 accumulator reaches an
                            # out_dtype tensor, so it must key off out_dtype.
                            # They agree whenever the output dtype follows the
                            # input, but stage 2 of the two-stage full reduction
                            # consumes fp32 partials and may emit fp16/bf16 --
                            # the old test would then have copied an fp32 UB
                            # buffer straight into an fp16 tensor.
                            T.barrier_all()      # V -> MTE3 (§13.55 #2)
                            if out_dtype == "float32":
                                T.copy(padded, C)
                            else:
                                T.tile.cast(out, padded, "CAST_RINT", _PADDED_OUTPUT)
                                T.barrier_all()
                                T.copy(out, C)

        return main

    return factory()


# --- Full-reduction decomposition (R198) -------------------------------------
# ``dim=None`` collapses to m == 1, which ``_compile_scalar`` served with a
# single block and a ``vid == 0`` guard: ONE vector context out of 96.  Measured
# on card 6 (R198-data/step0_bandwidth.csv): 22-28 GB/s for the sum-like ops and
# 7.4-8.4 GB/s for the max-like ones, against 156-231 GB/s for ``_compile`` on
# the very same file and the very same tensor reduced along dim=-1 instead.
#
# So a full reduction is now folded into the *row* factory, which already
# parallelises over rows: the flat run of n elements is viewed as
# ``(rows, n // rows)``, ``_compile`` produces one partial per row, and
# ``_compile_scalar`` collapses those. Every stage pair below is an exact
# identity given that all groups hold the same number of elements -- which is
# why ``rows`` must divide ``n`` exactly and no remainder path exists:
#
#   sum   Sum of row sums.
#   mean  Mean of equal-sized group means is the overall mean.
#   l1    Row sums of |x| are non-negative, so stage 2 is a plain sum.
#   l2    sqrt(sum_i r_i^2) with r_i = sqrt(sum_j x_ij^2); i.e. "l2 of the l2s".
#         Exact in exact arithmetic; in fp it adds one sqrt/square round trip.
#   amax/amin/inf   max/min are associative and exact, so this is bit-identical.
#   all/any         boolean, associative and exact -- bit-identical.
_FULL_STAGES = {
    "sum": ("sum", "sum"),
    "mean": ("mean", "mean"),
    "l1": ("l1", "sum"),
    "l2": ("l2", "l2"),
    "amax": ("amax", "amax"),
    "amin": ("amin", "amin"),
    "inf": ("inf", "inf"),
    "all": ("all", "all"),
    "any": ("any", "any"),
}
# Rows must be a multiple of the row-block height so the stage 1 output is dense,
# and each row must still be at least one _BLOCK_N wide or stage 1 throws away
# most of its vector width.  Capped at 128 * 48 * 2 rows: past roughly two row
# blocks per AI core there is nothing left to win, and the stage 2 input grows.
_FULL_ROWS_CAP = _BLOCK_M * 48 * 2
#: Ascend910B1 has 24 AI cores (each carrying two AIV contexts), so 24 blocks is
#: exactly one launch wave.  R340 measured that the block count -- not the tile
#: width -- is what the "largest rows that fits" rule was getting wrong.
_CORE_BLOCKS = 24


def _full_reduction_rows(n: int) -> int | None:
    """Rows to fold a length-n full reduction into, or None to keep one block."""

    best = None
    rows = _BLOCK_M
    while rows <= _FULL_ROWS_CAP:
        if n % rows == 0 and n // rows >= _BLOCK_N:
            best = rows
        rows += _BLOCK_M
    if best is None:
        return None
    # --- R340: avoid a RAGGED SECOND WAVE ------------------------------------
    # The rule above takes the largest legal `rows`, i.e. the most blocks.  When
    # that lands strictly between one and two full waves the tail wave is nearly
    # empty and costs a whole block-time for a fraction of the work.  Swept on
    # the two manifest shapes that land there (R340-data/probe/p4-fullred32k.json
    # and p5-rows{A,B}.json, D032 instrument, every point checked against torch):
    #
    #   n = 2**20   32 blocks (rows=4096)   16 blocks (rows=2048)
    #     all/bool        15.25 us              12.25 us     -20 %
    #     count_nonzero   40.00 us              25.00 us     -37 %
    #
    # Deliberately NOT applied above two waves: at 64 blocks (n = 2**23 and
    # 2**21) the same sweep says the current choice is right --
    # sum 27.50 us at 64 blocks against 29.75 at 16, var 40.75 against 48.50.
    # So this narrows to exactly the ragged-tail regime and leaves the
    # many-wave regime alone.  (count_nonzero at n = 2**23 would also prefer 16
    # blocks -- 53.00 us against 93.88 -- but sum and var there would not, and
    # they share this function; that headroom is left on the table, see R340 S6.)
    if _CORE_BLOCKS < best // _BLOCK_M <= 2 * _CORE_BLOCKS:
        one_wave = None
        rows = _BLOCK_M
        while rows <= _CORE_BLOCKS * _BLOCK_M:
            if n % rows == 0 and n // rows >= _BLOCK_N:
                one_wave = rows
            rows += _BLOCK_M
        if one_wave is not None:
            best = one_wave
    return best


# --- R198 round 3: the same split, but for m > 1 ------------------------------
# The m == 1 case above is one instance of a general move.  After round 2 every
# op's ratio_min was set by a case with *small m*, not by m == 1:
#
#   SumFwdOp   dim=None [2048,4096] 1.2322   <- fixed in round 2
#              dim=[0,2][4,128,4096] 0.9073  <- m=128 -> ceil(128/128) = 1 block
#   InfNorm    dim=None [2048,4096] 1.0389   <- fixed in round 2
#              dim=[0,2][4,128,4096] 0.5443  <- same one block
#
# m=1, m=4 and m=128 all give a single row block, so all of them use 1/96 of the
# machine.  Splitting the reduction axis fixes every one of them at once: view
# (m, n) as (m, k, n // k).  The buffer is contiguous, so (m * k, n // k) is a
# valid 2D view and stage 1 is the ordinary row factory over m*k rows; stage 2
# then reduces the (m, k) partials along k.  Both stages are existing kernels.
#
# Correctness carries over from the m == 1 argument unchanged, because every
# group still holds exactly n // k elements: max/min/and/or are associative and
# exact, mean-of-equal-groups is the overall mean, and l2-of-l2s is the overall
# l2.  That equal-size property is why k must divide n exactly.
_SPLIT_TARGET_BLOCKS = 48
# Only rescue the genuinely starved shapes.  A case that already has >= 8 row
# blocks is left alone even though 8 is still under the 48-192 launch plateau,
# because those cases are *already beating the vendor baseline* -- Sum dim=-1
# (m=2048, 16 blocks) measured ratio 1.3974 and dim=0 (m=4096, 32 blocks)
# measured 1.1857 in round 2.  Splitting them would add a kernel launch and a
# GM round trip for no headroom worth chasing, and T195's second prohibition is
# explicit that a path already past the baseline should not be touched.
_SPLIT_MIN_BLOCKS = 8
# --- R340: MINIMUM k IS A CORRECTNESS GUARD, NOT A TUNING KNOB ---------------
# Stage 2 of every split in this file reduces a `(_SUB_M, k)` tile.  Swept on
# `[4,128,4096] dim=[0,2]` (R340-data/probe/p3-ksweep1.log, D032 instrument, and
# every point checked against torch on the same input):
#
#     k =  2   count_nonzero returns 32793 for a true count of 16384; `all`
#              returns the wrong bool          <- WRONG ANSWER
#     k =  4   same, count off by 16412        <- WRONG ANSWER
#     k =  8   correct, 21.0 us
#     k = 16   correct, 14.75 us               <- chosen
#     k = 32   correct, 16.25 us
#
# `any` looks clean at k=2/4 only because its true answer is True on every row of
# that input and corrupt lanes cannot flip it -- it is not evidence of safety.
# No manifest shape reaches k <= 4 today (see R340-data/02_dispatch_matrix.txt),
# so this floor changes no shipped dispatch; it closes the hole rather than
# relying on the heuristics never to fall into it.
_MIN_SPLIT_K = 8
# --- MEASURED NEGATIVE RESULT: the axis split is OFF -------------------------
# It is correct (60/60 bitwise identical to the unsplit path, including the
# all-negative and +/-inf value classes that R090's hazard hides in) but it is
# SLOWER, so it is disabled rather than shipped.  On the case it was written for,
# [4,128,4096] dim=[0,2] (m=128, n=16384, k=64), measured on card 6
# (R198-data/probe/perf_axis_split_fix.json):
#
#     unsplit          166.8 us (amax) / 170.8 (l1) / 179.0 (sum)
#     split, 1st try   793.6 us   <- stage 2 fell into the scalar gather
#     split, fixed     255.4 us / 262.1 / 275.4   = 0.65x of unsplit
#
# Parameterising stage 2's tile width (see _compile's block_n) removed the
# scalar gather and bought back 3.1x of the 4.8x regression, but the split is
# still a net loss.  The most likely reason it cannot win here: for a non-last
# reduction axis the launch wrapper already materialises a transposed copy of
# the whole tensor in torch (`permute(...).reshape(m, n).contiguous()`), so both
# variants pay that, and the split only adds a second launch plus a GM round
# trip on top of a kernel that was never the dominant term.
#
# Kept in the tree, disabled by one flag, because the *m == 1* form of the same
# transformation is what made Sum 0.157 -> 0.907 in round 2 -- the idea is right,
# the m > 1 instance of it just is not where this family's remaining time goes.
# To re-test: flip this to True and rerun probe/perf_axis_split_fix.py.
_AXIS_SPLIT_ENABLED = True   # R198 r4: re-testing under harness graph timing


# _compile_extra's tile width is a fixed 128, and its trailing-tile path is a
# per-lane scalar gather (the same trap that cost 6x in round 3), so k is
# required to be a multiple of 128: stage 2 then reduces exactly k columns with
# no trailing tile.
_EXTRA_BLOCK_N = 128


def _prod_split_factor(m: int, n: int) -> int | None:
    """Chunks to cut a product's reduction axis into, or None to leave it alone.

    Product is associative up to floating-point rounding, so the split is
    ``prod of per-group prods``.  Reassociation moves where an intermediate may
    overflow to inf, which is why this is not claimed to be bit-identical.
    """
    if not _AXIS_SPLIT_ENABLED or m <= 0 or n <= 0:
        return None
    if (m + _BLOCK_M - 1) // _BLOCK_M >= _SPLIT_MIN_BLOCKS:
        return None
    best = None
    k = _EXTRA_BLOCK_N
    while k <= n // _EXTRA_BLOCK_N:
        if n % k == 0 and k % _EXTRA_BLOCK_N == 0:
            best = k
            if (m * k + _BLOCK_M - 1) // _BLOCK_M >= _SPLIT_TARGET_BLOCKS:
                break
        k += _EXTRA_BLOCK_N
    return best


def _axis_split_factor(m: int, n: int) -> int | None:
    """Chunks to cut the reduction axis into, or None to leave it alone.

    Returns None when the reduction already has enough row blocks, when no chunk
    count divides n, or when splitting would starve stage 1 of vector width
    (each chunk must still be at least one _BLOCK_N wide).
    """

    if not _AXIS_SPLIT_ENABLED:
        return None                      # measured net loss; see the flag above
    if m <= 0 or n <= 0:
        return None
    if (m + _BLOCK_M - 1) // _BLOCK_M >= _SPLIT_MIN_BLOCKS:
        return None                      # not starved; leave it alone
    best = None
    k = _MIN_SPLIT_K
    while k <= n // _BLOCK_N:
        if n % k == 0:
            blocks = (m * k + _BLOCK_M - 1) // _BLOCK_M
            if blocks >= _SPLIT_TARGET_BLOCKS:
                best = k
                break                    # smallest k that saturates the cores
            best = k                     # otherwise keep the widest split seen
        k += 1
    return best


# --- R340: row-wise variance split (the m > 1 counterpart of the m == 1 one) --
#
# MEASURED, not assumed.  `[4,128,4096] dim=[0,2]` gives m = 128, so
# `m_tiles = 1`, so `launch_block_count(1) = 1`: the whole 2 M-element Welford
# ran on **one block = 2 of 96 vector contexts**.  R340-data/probe/
# p2-pristine-decompose.json measured that kernel alone at 172.1 us against
# 20.25 us for All/Any on the same geometry -- All/Any escape because
# `op_kind in _FULL_STAGES` reaches `_axis_split_factor`, while `var`/`std`/
# `var_mean` are caught by the `_WELFORD_OPS` test before any split is
# considered and `_compile_extra` has no split for m > 1.
#
# The existing `_compile_welford_combine` cannot be reused: it combines k
# partials of ONE row (m == 1) in a single block.  This one combines k partials
# for EVERY row, in the usual (_BLOCK_M rows per block, two _SUB_M halves)
# geometry.
#
# The identity is the same and it is exact, not an approximation, because every
# partial covers exactly the same count c:
#     mean_tot = (1/k) * sum_j mean_j
#     M2_tot   = sum_j M2_j + c * sum_j (mean_j - mean_tot)^2
# That equal-count property is why k must divide the per-group reduction length.
_WELFORD_ROW_MIN_BLOCKS = 8
# Stop growing k once stage 1 has this many row blocks.  SWEPT, not guessed:
# R340-data/probe/p3-welford-k.json.  See the report for the table.
# 16, not 24 or 48.  SWEPT on the shape this exists for (see the k table above):
# k=8 -> 26.25 us, k=16 -> 17.00 us, k=32 -> 17.75 us.  The optimum is INTERIOR --
# growing k past 16 makes it worse, the same shape of result cannbot's
# count_nonzero iteration 5 found for `grid` (grid=16 beat grid=24 by 13.5 %).
# With m = 128 the block count equals k, so a target of 16 lands on k = 16.
_WELFORD_ROW_TARGET_BLOCKS = 16
# UB: three fp32 (_SUB_M, k) tiles = 3 * 64 * k * 4 bytes, so k = 128 is 98304 B,
# comfortably under UB_BUDGET_BYTES; 256 would be 196608 and is the R194 cliff.
_WELFORD_ROW_MAX_K = 128


def _welford_row_split_factor(m: int, n: int) -> int | None:
    """Chunks to cut a row-wise variance's reduction axis into, or None.

    Mirrors `_axis_split_factor`: only rescue the starved shapes (fewer than
    `_WELFORD_ROW_MIN_BLOCKS` row blocks), require k to divide n exactly (equal
    partial counts -- the combine identity depends on it) and require each chunk
    to stay at least one `_EXTRA_BLOCK_N` wide so stage 1 keeps its vector width
    and never reaches `_compile_extra`'s trailing-tile path.
    """
    if m <= 1 or n <= 0:
        return None
    if (m + _BLOCK_M - 1) // _BLOCK_M >= _WELFORD_ROW_MIN_BLOCKS:
        return None                          # not starved; leave it alone
    best = None
    k = _MIN_SPLIT_K
    k_cap = min(_WELFORD_ROW_MAX_K, n // _EXTRA_BLOCK_N)
    while k <= k_cap:
        if n % k == 0:
            blocks = (m * k + _BLOCK_M - 1) // _BLOCK_M
            best = k
            if blocks >= _WELFORD_ROW_TARGET_BLOCKS:
                break                        # smallest k that fills the machine
        k += 1
    return best


@lru_cache(maxsize=128)
def _compile_welford_rows(m: int, k: int, c: int, out_dtype: str, op_kind: str,
                          correction: int):
    """Stage 2 of the row-wise variance split: combine k equal partials per row.

    ``MEANS``/``M2S`` are ``(m, k)`` fp32 exactly as ``_compile_extra``'s
    ``welford_partial`` writeback leaves them.  ``c`` is the element count behind
    one partial (``groups * n // k``), so the total behind one output row is
    ``k * c`` and that -- not ``n`` -- is what the correction is applied to.
    """
    block_m = _BLOCK_M
    sub_m = _SUB_M
    m_tiles = (m + block_m - 1) // block_m
    has_m_tail = m % block_m != 0
    launch_blocks = launch_block_count(m_tiles)
    grid_repeats = grid_repeat_count(m_tiles, launch_blocks)
    total = k * c

    @tilelang.jit(out_idx=[2, 3], pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    })
    def factory():
        @T.prim_func
        def main(
            MEANS: T.Tensor((m, k), "float32"),
            M2S: T.Tensor((m, k), "float32"),
            C: T.Tensor((m_tiles * block_m,), out_dtype),
            D: T.Tensor((m_tiles * block_m,), out_dtype),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                mu = T.alloc_ub((sub_m, k), "float32")
                m2 = T.alloc_ub((sub_m, k), "float32")
                work = T.alloc_ub((sub_m, k), "float32")
                acc = T.alloc_ub((sub_m,), "float32")
                mean_tot = T.alloc_ub((sub_m,), "float32")
                m2_tot = T.alloc_ub((sub_m,), "float32")
                dev = T.alloc_ub((sub_m,), "float32")
                out = T.alloc_ub((sub_m,), out_dtype)
                with T.Scope("V"):
                    for rep in T.serial(grid_repeats):
                        logical_cid = cid + rep * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * block_m + vid * sub_m
                            T.tile.fill(mu, 0.0)
                            T.tile.fill(m2, 0.0)
                            if not has_m_tail:
                                T.copy(MEANS[row_base, 0], mu)
                                T.copy(M2S[row_base, 0], m2)
                            else:
                                for r in T.serial(sub_m):
                                    if row_base + r < m:
                                        T.copy(MEANS[row_base + r, 0], mu[r, :])
                                        T.copy(M2S[row_base + r, 0], m2[r, :])
                            T.barrier_all()          # MTE2 -> V (13.55 #1/#2)
                            T.reduce_sum(mu, acc, dim=-1, clear=True)
                            T.tile.mul(mean_tot, acc, 1.0 / k)
                            T.reduce_sum(m2, m2_tot, dim=-1, clear=True)
                            T.tile.broadcast(work, mean_tot)
                            T.tile.sub(work, mu, work)
                            T.tile.mul(work, work, work)
                            T.reduce_sum(work, dev, dim=-1, clear=True)
                            T.tile.mul(dev, dev, float(c))
                            T.tile.add(m2_tot, m2_tot, dev)
                            T.tile.div(
                                m2_tot, m2_tot,
                                T.cast(float(total - correction), "float32"),
                            )
                            if op_kind == "std":
                                T.tile.sqrt(m2_tot, m2_tot)
                            if out_dtype == "float32":
                                T.copy(m2_tot, out)
                            else:
                                T.tile.cast(out, m2_tot, "CAST_RINT", sub_m)
                            T.barrier_all()          # V -> MTE3 (13.55 #2)
                            T.copy(out, C[row_base : row_base + sub_m])
                            if out_dtype == "float32":
                                T.copy(mean_tot, out)
                            else:
                                T.tile.cast(out, mean_tot, "CAST_RINT", sub_m)
                            T.barrier_all()          # V -> MTE3 (13.55 #2)
                            T.copy(out, D[row_base : row_base + sub_m])
        return main

    return factory()


def _group_split(input_shape, reduced, order, n):
    """Split a non-trailing reduction into (leading groups, per-group length).

    R340.  Returns ``(groups, n_per_group)`` with ``groups * n_per_group == n``.

    ``groups > 1`` is returned only when BOTH hold:

    * today's launch wrapper would have materialised
      ``x.permute(order).reshape(m, n).contiguous()`` -- i.e. the reduction has a
      non-trailing axis -- and
    * the kept axes form ONE CONTIGUOUS BLOCK of the input, so the caller's
      buffer already *is* a ``(groups, m, n_per_group)`` tensor and
      ``x.reshape(groups * m, n_per_group)`` is a pure view.

    Under those two conditions the reduction over ``{leading axes} U {trailing
    axes}`` is exactly "reduce rows ``g * m + i`` for all g" over a
    ``(groups * m, n_per_group)`` matrix, which every factory in this file can
    now do natively (``groups`` argument) without moving a byte.

    Every other shape returns ``(1, n)``, i.e. byte-for-byte today's behaviour.
    The known excluded shape is a kept block that is not contiguous, e.g.
    ``(A,B,C,D) dim=[0,2]``: there the permuted view genuinely is a transpose.
    """
    ndim = len(input_shape)
    if order == tuple(range(ndim)):
        return 1, n                        # already trailing: no copy today
    kept = [i for i in range(ndim) if i not in reduced]
    if not kept:
        return 1, n                        # full reduction: m == 1, no copy
    first, last = kept[0], kept[-1]
    if kept != list(range(first, last + 1)):
        return 1, n                        # kept axes are not one block
    groups = math.prod(input_shape[:first]) if first else 1
    per_group = math.prod(input_shape[last + 1:]) if last + 1 < ndim else 1
    if groups <= 1 or per_group <= 1 or groups * per_group != n:
        return 1, n
    return groups, per_group


def build_reduction_kernel(
    input_shape,
    dtype,
    dim,
    keepdim,
    *,
    op_kind: str,
    op_name: str,
    diagnostic_sentinel: bool = False,
    correction: int = 1,
):
    """Build one static single-pass reduction signature."""

    input_shape = tuple(input_shape)
    if not input_shape or any(int(size) <= 0 for size in input_shape):
        raise ValueError(f"{op_name} requires a non-empty positive input shape")
    if op_kind not in _FLOAT_OPS | _LOGICAL_OPS:
        raise ValueError(f"unsupported reduction kind {op_kind!r}")
    if op_kind in _WELFORD_OPS and int(correction) < 0:
        raise ValueError(f"{op_name} correction must be non-negative, got {correction}")
    logical = op_kind in _LOGICAL_OPS
    if logical:
        if dtype != torch.bool:
            raise TypeError(f"{op_name} currently supports bool input, got {dtype}")
        dtype_name = "uint8"
        out_dtype = "uint8"
    else:
        dtype_name = _dtype_name(dtype, op_name)
        out_dtype = dtype_name
    axes = _axes(dim, len(input_shape), empty_is_full=not logical)
    reduced = set(axes)
    output_shape = (
        tuple(1 if i in reduced else input_shape[i] for i in range(len(input_shape)))
        if keepdim
        else tuple(input_shape[i] for i in range(len(input_shape)) if i not in reduced)
    )
    m = math.prod(output_shape) if output_shape else 1
    n = math.prod(input_shape[i] for i in axes)
    order = tuple(i for i in range(len(input_shape)) if i not in reduced) + axes
    # R340: n becomes the PER-GROUP reduction length; the true reduction length
    # is groups * n.  groups == 1 leaves every downstream expression unchanged.
    # logsumexp is excluded because its two factories (_compile_logsumexp /
    # _compile_logsumexp_online) were not given the groups argument.
    groups, n = (1, n) if op_kind == "logsumexp" else _group_split(
        input_shape, reduced, order, n
    )
    two_stage = None
    welford_split = None
    welford_row_split = None
    prod_split = None
    lse_split = None
    if op_kind == "logsumexp":
        if not _LSE_ONLINE:
            compiled = _compile_logsumexp(m, n, dtype_name, keepdim)
        else:
            k = _lse_axis_split_factor(m, n, dtype_name, out_dtype)
            if k is None:
                compiled = _compile_logsumexp_online(m, n, dtype_name, out_dtype)
            else:
                stage1 = _compile_logsumexp_online(m * k, n // k, dtype_name,
                                                   "float32")
                stage2 = _compile_logsumexp_online(m, k, "float32", out_dtype)
                lse_split = (k, stage1, stage2)
                compiled = stage1
    elif op_kind in _WELFORD_OPS and m == 1 and _full_reduction_rows(n) is not None:
        # Same starvation as everything else: m == 1 gives one row block, so the
        # vectorised Welford runs on 1 of 96 vector contexts.  Round 2 measured
        # VarFwdOp at 0.9145 on its dim=[0,2] case (m=128, many rows) and 0.0014
        # on dim=None -- the accumulation was never the problem, the parallelism
        # was.  This dispatch branch is what those three ops were missing: they
        # were caught by the _WELFORD_OPS test *before* reaching the m == 1 case.
        k = _full_reduction_rows(n)
        stage1 = _compile_extra(k, n // k, dtype_name, "welford_partial",
                                int(correction), out_dtype="float32")
        stage2 = _compile_welford_combine(k, n, out_dtype, op_kind,
                                          int(correction))
        welford_split = (k, stage1, stage2)
        compiled = stage1
    elif op_kind in _WELFORD_OPS and _welford_row_split_factor(m, n) is not None:
        # R340: m > 1 but too few row blocks to fill the machine.  Same move as
        # `_axis_split_factor` makes for `_FULL_STAGES`, with a per-row combine.
        k = _welford_row_split_factor(m, n)
        stage1 = _compile_extra(m * k, n // k, dtype_name, "welford_partial",
                                int(correction), out_dtype="float32",
                                groups=groups)
        stage2 = _compile_welford_rows(m, k, groups * (n // k), out_dtype,
                                       op_kind, int(correction))
        welford_row_split = (k, stage1, stage2)
        compiled = stage1
    elif op_kind == "prod" and _prod_split_factor(m, n) is not None:
        # Same starvation as everything else in this file: ProdFwdOp's ratio_min
        # comes from [64,32768] dim=-1, i.e. m=64 -> one row block.  Both stages
        # are _compile_extra, whose prod path already accumulates into a tile of
        # running products and collapses once.
        k = _prod_split_factor(m, n)
        stage1 = _compile_extra(m * k, n // k, dtype_name, "prod",
                                int(correction), out_dtype="float32",
                                groups=groups)
        stage2 = _compile_extra(m, k, "float32", "prod", int(correction))
        prod_split = (k, stage1, stage2)
        compiled = stage1
    elif op_kind in _WELFORD_OPS or op_kind == "prod":
        compiled = _compile_extra(m, n, dtype_name, op_kind, int(correction),
                                  groups=groups)
    elif m == 1:
        rows = _full_reduction_rows(n)
        if rows is None:
            compiled = _compile_scalar(
                n, dtype_name, out_dtype, op_kind, diagnostic_sentinel
            )
        else:
            kind1, kind2 = _FULL_STAGES[op_kind]
            part_dtype = "uint8" if op_kind in _LOGICAL_OPS else "float32"
            stage1 = _compile(
                rows, n // rows, dtype_name, part_dtype, kind1, diagnostic_sentinel
            )
            # rows is a multiple of _BLOCK_M, so block_n = rows makes stage 2
            # exactly one tile with no trailing-tile scalar path.
            stage2 = _compile_scalar(
                rows, part_dtype, out_dtype, kind2, diagnostic_sentinel,
                block_n=min(rows, _SCALAR_BLOCK_N),
            )
            two_stage = (rows, stage1, stage2, "rows")
            compiled = stage1
    elif op_kind in _FULL_STAGES and _axis_split_factor(m, n) is not None:
        # Small m: too few row blocks to fill the machine.  Split the reduction
        # axis instead of the rows -- see _axis_split_factor.
        k = _axis_split_factor(m, n)
        kind1, kind2 = _FULL_STAGES[op_kind]
        part_dtype = "uint8" if op_kind in _LOGICAL_OPS else "float32"
        stage1 = _compile(
            m * k, n // k, dtype_name, part_dtype, kind1, diagnostic_sentinel,
            groups=groups,
        )
        # Size stage 2's tile so that k divides it exactly: no trailing tile,
        # therefore no scalar gather.
        stage2_block = k if k <= _BLOCK_N else _BLOCK_N
        while stage2_block > 1 and k % stage2_block != 0:
            stage2_block //= 2
        stage2 = _compile(m, k, part_dtype, out_dtype, kind2, diagnostic_sentinel,
                          block_n=stage2_block)
        two_stage = (k, stage1, stage2, "axis")
        compiled = stage1
    else:
        compiled = _compile(m, n, dtype_name, out_dtype, op_kind,
                            diagnostic_sentinel, groups=groups)

    def launch(x):
        if tuple(x.shape) != input_shape:
            raise ValueError(f"{op_name} kernel shape mismatch")
        if logical:
            x = x.view(torch.uint8)
        if groups > 1:
            # R340: pure view of the caller's own buffer.  This is the line the
            # whole round is about -- it replaces a
            # ``permute(...).reshape(m, n).contiguous()`` that copied the ENTIRE
            # tensor through HBM (read + write) before the reduction had read a
            # single byte.  ``reshape`` preserves logical element order, so this
            # is semantically identical whatever the caller's strides; it only
            # avoids the copy when the caller is contiguous, which the op
            # boundary (ops/reduction/reduce.py:398) guarantees.
            flat = x.reshape(groups * m, n)
        else:
            view = x if order == tuple(range(len(input_shape))) else x.permute(order)
            flat = view.reshape(m, n)
            if not flat.is_contiguous():
                flat = flat.contiguous()
        if lse_split is not None:
            k, stage1, stage2 = lse_split
            # (m, n) -> (m, k, n//k) -> (m*k, n//k); contiguous, so a view.
            parts = stage1(flat.reshape(m * k, n // k))
            parts = parts[0] if isinstance(parts, (tuple, list)) else parts
            result = stage2(parts[: m * k].reshape(m, k))
        elif prod_split is not None:
            k, stage1, stage2 = prod_split
            parts = stage1(flat.reshape(groups * m * k, n // k))
            parts = parts[0] if isinstance(parts, (tuple, list)) else parts
            result = stage2(parts[: m * k].reshape(m, k))
        elif welford_row_split is not None:
            k, stage1, stage2 = welford_row_split
            # (groups, m, k, n//k) -> (groups*m*k, n//k); a view.
            m2s, means = stage1(flat.reshape(groups * m * k, n // k))
            result = stage2(means[: m * k].reshape(m, k),
                            m2s[: m * k].reshape(m, k))
        elif welford_split is not None:
            k, stage1, stage2 = welford_split
            # stage 1 emits (M2, mean) per group; stage 2 returns
            # (var-or-std, mean).  Both come back as the usual 2-tuple, so the
            # shared post-processing below slices [:m] and reshapes -- doing it
            # here as well was double-processing and blew up on the padded
            # _PADDED_OUTPUT length.
            m2s, means = stage1(flat.reshape(k, n // k))
            result = stage2(means[:k].reshape(1, k), m2s[:k].reshape(1, k))
        elif two_stage is not None:
            split, stage1, stage2, mode = two_stage
            if mode == "rows":
                partials = stage1(flat.reshape(split, n // split))
                if isinstance(partials, (tuple, list)):
                    partials = partials[0]
                result = stage2(partials[:split].reshape(1, split))
            else:
                # (m, n) -> (m, k, n//k) -> (m*k, n//k); contiguous, so this is
                # a view, not a copy.  With groups > 1 the same view has
                # ``groups`` copies of that row block stacked on top of it, and
                # stage 1 walks them with its own ``g`` loop.
                partials = stage1(flat.reshape(groups * m * split, n // split))
                if isinstance(partials, (tuple, list)):
                    partials = partials[0]
                result = stage2(partials[: m * split].reshape(m, split))
        else:
            result = compiled(flat)
        if op_kind == "logsumexp":
            result = result[:m]
        elif op_kind == "var_mean":
            result = (result[0][:m], result[1][:m])
        else:
            # TileLang returns a bare scalar for the single-output scalar
            # ABI (m == 1), while the row-reduction ABI returns a 1-tuple.
            # T088's shared output unpack assumed the latter and indexed a
            # 0-D tensor, breaking full reductions such as MeanFwdOp(dim=None).
            if isinstance(result, (tuple, list)):
                result = result[0]
            result = result.reshape(1)[:m] if result.ndim == 0 else result[:m]
        if logical:
            result = result.view(torch.bool)
        if op_kind == "var_mean":
            return tuple(v.reshape(output_shape) for v in result)
        return result.reshape(output_shape)

    launch.compiled = compiled

    return launch


def build_mean_kernel(input_shape, dtype, dim, keepdim):
    """Compatibility entry point retained for the existing MeanFwdOp builder."""

    return build_reduction_kernel(
        input_shape,
        dtype,
        dim,
        keepdim,
        op_kind="mean",
        op_name="MeanFwdOp",
    )


__all__ = ["build_mean_kernel", "build_reduction_kernel"]
