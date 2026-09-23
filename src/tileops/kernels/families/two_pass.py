"""Two-pass AIV softmax kernel for the Ascend backend."""

from functools import lru_cache
import math
import os

import tilelang
import tilelang.language as T
import torch

from .._registry import register
from ..common import (
    LAUNCH_BLOCK_CAP,
    TILE_GRAIN,
    UB_BUDGET_NORM_BYTES,
    grid_repeat_count,
    launch_block_count,
    plan_rowwise_norm,
)


_VEC = 2

#: ⚠️ ``_BLOCK_M``/``_BLOCK_N_MAX`` used to be 128 each, i.e. ``sub_m = 64`` rows
#: by 128 columns per vector lane, for every shape.  R200 replaced both with
#: ``kernels.common.plan_rowwise_norm``; see that function and ``_compile_norm``
#: for the measured reasons.  They are kept only as the fallback grain.
_BLOCK_M = 128
_BLOCK_N_MAX = 128


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float32:
        return "float32"
    raise TypeError(f"SoftmaxFwdOp supports floating dtypes, got {dtype}")


def _implicit_dim(ndim: int) -> int:
    # This mirrors TileOPs' _get_softmax_dim rule for dim=None.
    return 0 if ndim in (0, 1, 3) else 1


@lru_cache(maxsize=128)
def _compile_norm(m: int, n: int, dtype_name: str, kind: str, eps: float):
    """Compile the row-wise normalization subset of archetype 3.

    RMSNorm and LayerNorm both reduce a contiguous trailing row, then perform a
    second pass for the affine epilogue.  The output is padded internally so
    every GM write is a vector copy; the public builder slices it back.
    """
    if dtype_name not in {"float16", "bfloat16", "float32"}:
        raise TypeError(f"normalization supports floating dtypes, got {dtype_name}")
    itemsize = 2 if dtype_name in ("float16", "bfloat16") else 4
    # --- Geometry (R200) -------------------------------------------------
    # UB census read off the allocation list in ``factory`` below.  Getting one
    # entry wrong here is a silent UB overflow, so it is spelled out:
    #   (sub_m, block_n) : src[itemsize] + x32[4] + work[4] + bc[4]
    #   (block_n,)       : wub[itemsize] + bub[itemsize] + w32[4] + b32[4]
    #   (sub_m,)         : 18 fp32 scalars (stat stat2 count m2 row_sum row_sq
    #                      row_m2 tile_mean tile_count delta ratio new_count
    #                      correction mean denom refine inv + 1 spare)
    plan = plan_rowwise_norm(
        m, n, itemsize,
        bytes_per_cell=itemsize + 12,
        bytes_per_col=2 * itemsize + 8,
        bytes_per_row=18 * 4,
    )
    sub_m = plan["sub_m"]
    block_m = plan["block_m"]
    block_n = plan["block_n"]
    m_tiles = plan["m_tiles"]
    n_tiles = plan["n_tiles"]
    m_pad = plan["m_pad"]
    n_pad = plan["n_pad"]
    has_m_tail = plan["has_m_tail"]
    has_n_tail = plan["has_n_tail"]
    launch_blocks = plan["launch_blocks"]
    grid_repeats = plan["grid_repeats"]
    need_cast = dtype_name != "float32"
    # The Welford combine across n-tiles is provably the identity when there is
    # exactly one tile: ``count`` starts at 0, so ``ratio = tile_count /
    # tile_count = 1`` and the second correction term is ``0 * delta**2``.
    # Eliding it is bit-exact, not an approximation.
    single_n_tile = n_tiles == 1

    @tilelang.jit(
        out_idx=[1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor([m, n], dtype_name),
            B: T.Tensor([m_pad, n_pad], dtype_name),
            W: T.Tensor([n], dtype_name),
            Bias: T.Tensor([n], dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((sub_m, block_n), dtype_name)
                x32 = T.alloc_ub((sub_m, block_n), "float32")
                work = T.alloc_ub((sub_m, block_n), "float32")
                stat = T.alloc_ub((sub_m,), "float32")
                stat2 = T.alloc_ub((sub_m,), "float32")
                # Welford state: count, mean and M2 are all kept in fp32.
                count = T.alloc_ub((sub_m,), "float32")
                m2 = T.alloc_ub((sub_m,), "float32")
                row_sum = T.alloc_ub((sub_m,), "float32")
                row_sq = T.alloc_ub((sub_m,), "float32")
                row_m2 = T.alloc_ub((sub_m,), "float32")
                tile_mean = T.alloc_ub((sub_m,), "float32")
                tile_count = T.alloc_ub((sub_m,), "float32")
                delta = T.alloc_ub((sub_m,), "float32")
                ratio = T.alloc_ub((sub_m,), "float32")
                new_count = T.alloc_ub((sub_m,), "float32")
                correction = T.alloc_ub((sub_m,), "float32")
                mean = T.alloc_ub((sub_m,), "float32")
                denom = T.alloc_ub((sub_m,), "float32")
                refine = T.alloc_ub((sub_m,), "float32")
                inv = T.alloc_ub((sub_m,), "float32")
                bc = T.alloc_ub((sub_m, block_n), "float32")
                wub = T.alloc_ub((block_n,), dtype_name)
                bub = T.alloc_ub((block_n,), dtype_name)
                w32 = T.alloc_ub((block_n,), "float32")
                b32 = T.alloc_ub((block_n,), "float32")

                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * block_m + vid * sub_m
                            if kind == "rms":
                                T.tile.fill(stat2, 0.0)
                            else:
                                T.tile.fill(stat, 0.0)
                                T.tile.fill(count, 0.0)
                                T.tile.fill(m2, 0.0)
                            for nt in T.serial(n_tiles):
                                if has_n_tail:
                                    T.tile.fill(src, 0.0)
                                if not has_n_tail:
                                    if not has_m_tail:
                                        T.copy(A[row_base, nt * block_n], src)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(A[row_base + r, nt * block_n], src[r, :])
                                elif nt < n_tiles - 1:
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
                                # --- Dead-code elimination by kind (R200 r2) ------------
                                # Both statistic paths used to run for BOTH kinds, but the
                                # ``denom`` selection below consumes only one of them:
                                #     rms   -> stat2 = sum(x^2)
                                #     layer -> m2 / count  (the Welford state)
                                # so each kind paid for the other's reduction in full, on
                                # every n-tile.  Counted in FULL-TILE vector ops
                                # (sub_m x block_n), per n-tile:
                                #     rms  was paying 5 dead  (reduce_sum -> row_sum,
                                #          broadcast, sub, mul, reduce_sum -> row_m2) plus
                                #          the ~14-op Welford chain, and keeping only 2;
                                #     layer was paying 2 dead (mul, reduce_sum -> row_sq),
                                #          and keeping 5.
                                # Safe in both directions: the three rms lines read x32
                                # BEFORE the layer branch re-centres it, and the only
                                # buffer they clobber is ``work``, which the layer branch
                                # overwrites with broadcast(tile_mean) before reading it.
                                # Dropping unreachable work is bit-exact by construction,
                                # so this does NOT re-open the R200 section 5 question.
                                if kind == "rms":
                                    T.tile.mul(work, x32, x32)
                                    T.reduce_sum(work, row_sq, dim=-1)
                                    T.tile.add(stat2, stat2, row_sq)
                                else:
                                    T.reduce_sum(x32, row_sum, dim=-1)
                                    if not has_n_tail or nt < n_tiles - 1:
                                        T.tile.fill(tile_count, T.cast(float(block_n), "float32"))
                                    else:
                                        T.tile.fill(
                                            tile_count,
                                            T.cast(float(n - (n_tiles - 1) * block_n), "float32"),
                                        )
                                    T.tile.div(tile_mean, row_sum, tile_count)
                                    T.tile.broadcast(work, tile_mean)
                                    T.tile.sub(x32, x32, work)
                                    # Padded columns must not contribute to M2.
                                    if has_n_tail and nt == n_tiles - 1:
                                        for r in T.serial(sub_m):
                                            for c in T.serial(block_n):
                                                if c >= n - (n_tiles - 1) * block_n:
                                                    x32[r, c] = 0.0
                                    T.tile.mul(work, x32, x32)
                                    T.reduce_sum(work, row_m2, dim=-1)
                                    if single_n_tile:
                                        # Bit-exact shortcut, see ``single_n_tile``.
                                        T.copy(tile_mean, stat)
                                        T.copy(row_m2, m2)
                                        T.copy(tile_count, count)
                                    else:
                                        T.tile.sub(delta, tile_mean, stat)
                                        T.tile.add(new_count, count, tile_count)
                                        T.tile.div(ratio, tile_count, new_count)
                                        T.tile.mul(correction, delta, ratio)
                                        T.tile.add(stat, stat, correction)
                                        T.tile.mul(correction, delta, delta)
                                        T.tile.mul(ratio, count, tile_count)
                                        T.tile.div(ratio, ratio, new_count)
                                        T.tile.mul(correction, correction, ratio)
                                        T.tile.add(row_m2, row_m2, correction)
                                        T.tile.add(m2, m2, row_m2)
                                        T.tile.add(count, count, tile_count)

                            inv_n = T.cast(1.0 / n, "float32")
                            if kind != "rms":
                                T.copy(stat, mean)
                            if kind == "rms":
                                # RMSNorm keeps its original E[x^2] path.  The
                                # Welford state is only needed by LayerNorm.
                                T.tile.mul(denom, stat2, inv_n)
                                T.tile.add(denom, denom, T.cast(eps, "float32"))
                            else:
                                T.tile.div(denom, m2, count)
                                T.tile.add(denom, denom, T.cast(eps, "float32"))
                            T.tile.rsqrt(inv, denom)
                            # Ascend's vector rsqrt is intentionally fast and
                            # approximate.  One Newton step restores fp32
                            # accuracy before the final dtype cast.
                            T.tile.mul(refine, inv, inv)
                            T.tile.mul(refine, refine, denom)
                            T.tile.mul(refine, refine, 0.5)
                            T.tile.fill(denom, 1.5)
                            T.tile.sub(denom, denom, refine)
                            T.tile.mul(inv, inv, denom)
                            T.tile.broadcast(bc, inv if kind == "rms" else inv)

                            for nt in T.serial(n_tiles):
                                if has_n_tail:
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
                                                if c < n - nt * block_n:
                                                    src[r, c] = A[row_base + r, nt * block_n + c]
                                if need_cast:
                                    T.tile.cast(x32, src, "CAST_NONE", sub_m * block_n)
                                else:
                                    T.copy(src, x32)
                                if kind == "layer":
                                    T.tile.broadcast(work, mean)
                                    T.tile.sub(x32, x32, work)
                                T.tile.mul(x32, x32, bc)
                                # R200: this was a per-ELEMENT scalar GM read of
                                # the whole weight (and bias) vector, re-run by
                                # every task -- ``n * n_tasks`` scalar iterations,
                                # and the single dominant cost of this template.
                                # Differential on llama-405b-decode / n=16384:
                                # RMSNorm 1173.7 -> 542.4 us at unchanged
                                # geometry (2.16x), LayerNorm 1834.4 -> 652.5 us
                                # (2.81x, it loads two vectors).  See
                                # R200-data/probe_vecweight_decode.json.
                                if not has_n_tail or nt < n_tiles - 1:
                                    T.copy(W[nt * block_n], wub)
                                    if kind == "layer":
                                        T.copy(Bias[nt * block_n], bub)
                                else:
                                    # Only the ragged last tile keeps the scalar
                                    # path; a vector copy would read past W.
                                    for c in T.serial(block_n):
                                        if c < n - nt * block_n:
                                            wub[c] = W[nt * block_n + c]
                                            if kind == "layer":
                                                bub[c] = Bias[nt * block_n + c]
                                if need_cast:
                                    T.tile.cast(w32, wub, "CAST_NONE", block_n)
                                    T.tile.broadcast(work, w32)
                                    T.tile.mul(x32, x32, work)
                                    if kind == "layer":
                                        T.tile.cast(b32, bub, "CAST_NONE", block_n)
                                        T.tile.broadcast(work, b32)
                                        T.tile.add(x32, x32, work)
                                    T.tile.cast(src, x32, "CAST_RINT", sub_m * block_n)
                                    T.copy(src, B[row_base, nt * block_n])
                                else:
                                    T.tile.broadcast(work, wub)
                                    T.tile.mul(x32, x32, work)
                                    if kind == "layer":
                                        T.tile.broadcast(work, bub)
                                        T.tile.add(x32, x32, work)
                                    T.copy(x32, B[row_base, nt * block_n])
        return main

    return factory()




# ---------------------------------------------------------------------------
# R261: the resident-row RMSNorm path
# ---------------------------------------------------------------------------
#
# ``_compile_norm`` above is unchanged and still serves LayerNorm and every
# shape this path declines.  What follows replaces it for RMSNorm when the
# shape allows, and the four things it does differently are the four things
# T261 measured one at a time (docs/reports/R261.md section 3):
#
#   1. the ``(sub_m,)`` statistics buffers are padded to ``_STAT_LANES``
#      independently of how many rows a lane owns, so ``sub_m`` may be 1.
#      ``kernels.common.ROW_EXTENT_GRAIN`` documents 8 as a HARD FLOOR on
#      ``sub_m`` and says "fixing THAT needs the (sub_m,) statistics padded to
#      8 lanes independently of how many rows a lane owns; R200 did not do it".
#      This does it.  Every ``T.tile.*`` on a statistics buffer still runs on
#      all 8 lanes (which is what the 32-byte block rule requires); only the
#      ``T.reduce_sum`` destination is narrowed to the ``[0:sub_m]`` region.
#      Measured on llama-405b-decode (m = 1): 38.75 -> 14.75 us.
#   2. the column tile is as WIDE as UB allows rather than capped at
#      ``TILE_WIDTH_CAP = 1280``.  This was the single largest lever, and it is
#      about op COUNT, not about bandwidth: a ``T.tile.*`` call carries a fixed
#      cost that a 1024-column tile cannot amortise.  Measured on
#      llama-405b-decode: 10.25 us at block_n = 1024 -> 4.50 us at 8192.
#   3. sum(x^2) accumulates elementwise with ``T.tile.mul_add_dst`` and
#      ``T.reduce_sum`` runs ONCE per row group instead of once per n-tile.
#   4. the affine epilogue multiplies each row against the ``(block_n,)``
#      weight vector directly, so the ``(sub_m, block_n)`` broadcast replica of
#      the weight that the two-pass path rebuilds for every n-tile is gone.
#      Measured on llama-405b-prefill/bf16: 262.75 -> 234.50 us.
#
# And the row group stays in UB as fp32 between the two passes, so x is read
# from GM once rather than twice.  That is why the path is only taken when
# ``sub_m * n * 4`` bytes fit: ``_plan_rms_resident`` returns None otherwise and
# the caller falls back to ``_compile_norm``.
#
# Ideas 3 and 4 were arrived at by READING (not copying) MIT-licensed Tile-AI
# code: tilelang-ascend/examples/normalization/rms_norm.py keeps a full-tile
# square accumulator and reduces once, and tilelang-mlir-ascend/examples/norm/
# example_rms_norm.py has a split-K variant with the same shape.  Both are
# re-expressed here against a different buffer plan (padded statistics lanes, a
# UB-resident fp32 row group, per-row vector multiply); no code was pasted, so
# NOTICE is unchanged.  See docs/DECISIONS.md D039 and FAILED_ATTEMPTS F035.

#: Lanes every ``(...,)`` fp32 statistics buffer is padded to.  Same number as
#: ``kernels.common.ROW_EXTENT_GRAIN`` and for the same reason -- an Ascend
#: vector instruction addresses whole 32-byte blocks -- but here it is a buffer
#: WIDTH, not a floor on ``sub_m``.
_STAT_LANES = 8

#: Column tiles per row this planner aims for.  Measured, not guessed: with
#: everything else fixed, 2 beat both 1 and 4 on all 9 RMSNormFwdOp cases
#: (docs/reports/R261-data/05-step4-ntiles.txt).  1 loses because a single tile
#: leaves the loop no second iteration to overlap MTE2 against; 4 loses because
#: the per-``T.tile.*``-call fixed cost comes back.
_TARGET_N_TILES = 2

#: Hard ceiling on column tiles per row for this path.  Beyond it the resident
#: fp32 row has squeezed the tile so narrow that the two-pass kernel is faster,
#: so the planner declines instead (n = 16384: nt = 4 -> 202 us, nt = 8 -> 282 us,
#: two-pass -> 231 us; docs/reports/R261-data/06-step3-widths.txt).
_MAX_N_TILES = 4


def _plan_rms_resident(m: int, n: int, itemsize: int):
    """Geometry for the resident-row path, or ``None`` if it does not apply.

    ``None`` is a normal answer, not a failure: the caller then compiles the
    two-pass kernel, which has no residency requirement and no width rule.
    """
    m = max(1, int(m))
    n = max(1, int(n))
    if n % TILE_GRAIN:
        # The ragged-column path is exactly what the two-pass kernel already
        # handles (its scalar n-tail loop); duplicating it here would double the
        # surface without touching any of the four measured levers.
        return None
    # UB census, read off the allocation list in ``factory`` below.  One wrong
    # entry is a silent UB overflow (R194 2.2), so it is spelled out:
    #   per row of the tile : src[itemsize] + accb[4] + hold[4 * n_tiles]
    #                         ... and hold spans the WHOLE row, hence 4 * n
    #   fixed               : wub[itemsize] + w32[4], both (block_n,)
    #                         + 5 statistics buffers of _STAT_LANES fp32
    # ⚠️ The budget is ``UB_BUDGET_NORM_BYTES`` (147456), the conservative
    # waterline that reserves 32 KB for the reduce scratch that never appears in
    # the Python allocation list.  A one-off probe ran correctly at a DECLARED
    # 164000 bytes (block_n = 8192 at n = 16384: 162 us instead of 202 us on
    # llama-405b-prefill/bf16), and block_n = 16384 there killed the compiling
    # process outright rather than raising -- so the extra width is real but the
    # cliff behind it is fatal, and no case needs it to clear the threshold.
    # docs/reports/R261.md section 6 has the numbers.
    occupancy = max(1, m // (2 * LAUNCH_BLOCK_CAP))
    fixed_rows = 5 * _STAT_LANES * 4
    width = (n // _TARGET_N_TILES) - (n // _TARGET_N_TILES) % TILE_GRAIN
    while width >= TILE_GRAIN:
        n_tiles = n // width
        # nt > _MAX_N_TILES is where residency stops paying: it forces the tile
        # narrow, and a narrow tile is what the whole exercise removed.  Measured
        # at n = 16384: nt = 4 -> 202 us, nt = 8 -> 282 us, while the two-pass
        # kernel does 231 us.  So hand those shapes back rather than regress.
        if n % width == 0 and n_tiles <= _MAX_N_TILES:
            per_row = width * itemsize + width * 4 + n * 4
            fixed = width * itemsize + width * 4 + fixed_rows
            room = (UB_BUDGET_NORM_BYTES - fixed) // per_row
            if room >= 1:
                sub_m = 1
                while sub_m * 2 <= min(room, occupancy):
                    sub_m *= 2
                block_m = sub_m * 2
                m_tiles = (m + block_m - 1) // block_m
                launch_blocks = launch_block_count(min(m_tiles, LAUNCH_BLOCK_CAP))
                return {
                    "sub_m": sub_m,
                    "block_m": block_m,
                    "block_n": width,
                    "m_tiles": m_tiles,
                    "n_tiles": n_tiles,
                    "m_pad": m_tiles * block_m,
                    "launch_blocks": launch_blocks,
                    "grid_repeats": grid_repeat_count(m_tiles, launch_blocks),
                    "ub_bytes": sub_m * per_row + fixed,
                }
        width -= TILE_GRAIN
    return None


@lru_cache(maxsize=128)
def _compile_rms_resident(m: int, n: int, dtype_name: str, eps: float):
    """RMSNorm with the row group resident in UB, or ``None`` if it will not fit."""
    if dtype_name not in {"float16", "bfloat16", "float32"}:
        raise TypeError(f"normalization supports floating dtypes, got {dtype_name}")
    itemsize = 2 if dtype_name in ("float16", "bfloat16") else 4
    plan = _plan_rms_resident(m, n, itemsize)
    if plan is None:
        return None
    sub_m = plan["sub_m"]
    block_m = plan["block_m"]
    block_n = plan["block_n"]
    m_tiles = plan["m_tiles"]
    n_tiles = plan["n_tiles"]
    m_pad = plan["m_pad"]
    launch_blocks = plan["launch_blocks"]
    grid_repeats = plan["grid_repeats"]
    need_cast = dtype_name != "float32"

    @tilelang.jit(
        out_idx=[1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor([m, n], dtype_name),
            B: T.Tensor([m_pad, n], dtype_name),
            W: T.Tensor([n], dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                # The whole row group, in fp32, for the lifetime of both passes.
                hold = T.alloc_ub((n_tiles, sub_m, block_n), "float32")
                src = T.alloc_ub((sub_m, block_n), dtype_name)
                accb = T.alloc_ub((sub_m, block_n), "float32")
                wub = T.alloc_ub((block_n,), dtype_name)
                w32 = T.alloc_ub((block_n,), "float32")
                tot = T.alloc_ub((_STAT_LANES,), "float32")
                den = T.alloc_ub((_STAT_LANES,), "float32")
                inv = T.alloc_ub((_STAT_LANES,), "float32")
                ref = T.alloc_ub((_STAT_LANES,), "float32")
                spare = T.alloc_ub((_STAT_LANES,), "float32")

                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * block_m + vid * sub_m
                            # ⚠️ No guarded per-row copy loop here, unlike
                            # ``_compile_norm``.  A GM->UB ``T.copy`` is clamped
                            # to the real tensor extent at lowering time
                            # (src/op/ascend.cc :: compute_valid_extent), so an
                            # m tail loads fewer rows and leaves the rest of the
                            # UB tile untouched.  That is safe here because
                            # every statistic is per row: garbage in a row this
                            # lane does not own cannot reach a row it does.  The
                            # STORE is in bounds because B is padded to m_pad.
                            T.tile.fill(accb, 0.0)
                            T.tile.fill(tot, 0.0)
                            for nt in T.serial(n_tiles):
                                T.copy(A[row_base, nt * block_n], src)
                                if need_cast:
                                    T.tile.cast(hold[nt, :, :], src, "CAST_NONE",
                                                sub_m * block_n)
                                else:
                                    T.copy(src, hold[nt, :, :])
                                # dst += src0 * src1 in one instruction: this is
                                # lever 3, and it is also what lets the reduction
                                # happen once instead of n_tiles times.
                                T.tile.mul_add_dst(accb, hold[nt, :, :], hold[nt, :, :])
                            T.reduce_sum(accb, tot[0:sub_m], dim=-1)
                            T.tile.mul(den, tot, T.cast(1.0 / n, "float32"))
                            T.tile.add(den, den, T.cast(eps, "float32"))
                            T.tile.rsqrt(inv, den)
                            # Ascend's vector rsqrt is intentionally fast and
                            # approximate.  One Newton step restores fp32
                            # accuracy before the final dtype cast -- same step,
                            # same order as ``_compile_norm``.
                            T.tile.mul(ref, inv, inv)
                            T.tile.mul(ref, ref, den)
                            T.tile.mul(ref, ref, 0.5)
                            T.tile.fill(spare, 1.5)
                            T.tile.sub(spare, spare, ref)
                            T.tile.mul(inv, inv, spare)
                            for nt in T.serial(n_tiles):
                                T.copy(W[nt * block_n], wub)
                                if need_cast:
                                    T.tile.cast(w32, wub, "CAST_NONE", block_n)
                                else:
                                    T.copy(wub, w32)
                                # Lever 4: one (block_n,) vector multiply per
                                # row and one vector-scalar multiply per row,
                                # instead of broadcasting w and inv into two
                                # (sub_m, block_n) replicas first.
                                for r in T.serial(sub_m):
                                    T.tile.mul(hold[nt, r, :], hold[nt, r, :], w32)
                                    T.tile.mul(hold[nt, r, :], hold[nt, r, :], inv[r])
                                if need_cast:
                                    T.tile.cast(src, hold[nt, :, :], "CAST_RINT",
                                                sub_m * block_n)
                                    T.copy(src, B[row_base, nt * block_n])
                                else:
                                    T.copy(hold[nt, :, :], B[row_base, nt * block_n])
        return main

    return factory()


# ---------------------------------------------------------------------------
# R335: the resident-row LayerNorm path
# ---------------------------------------------------------------------------
#
# ``_compile_rms_resident`` above is UNCHANGED and still serves RMSNorm.  This is
# its LayerNorm sibling, and it exists because R274 predicted it and R271/R335
# measured the gap it closes: ``LayerNormFwdOp`` was the only member of this
# family still compiling through ``_compile_norm`` for every shape, and
# ``plan_rowwise_norm`` gives that template a geometry that is catastrophic at
# m = 1 (the decode rows).  For llama-405b-decode / n = 16384 / bf16 it plans
# ``sub_m = 8, block_n = 1024, n_tiles = 16, has_m_tail = True``, i.e.
#
#   * 7 of the 8 rows every ``T.tile.*`` touches DO NOT EXIST (ROW_EXTENT_GRAIN
#     is a hard floor on ``sub_m`` for that template, see kernels/common.py),
#   * the row is walked in 16 narrow tiles rather than 2-4 wide ones, and
#   * ``has_m_tail`` replaces each whole-tile ``T.copy`` with a guarded per-row
#     copy loop that iterates ``sub_m`` times.
#
# R261 fixed exactly these three things for RMSNorm and R274 fixed them for the
# fused residual-add variants; the ``kind="layer"`` body below is the same
# textbook two-pass mean/variance R274 put in
# ``normalization_spatial._compile_fused_add_resident`` (mean over the resident
# row, then sum of squared deviations over it), minus the residual add.
#
# ⚠️ ONE DELIBERATE DIFFERENCE from R274's layer body: the mean pass reduces
# each n-tile straight into a ``(_STAT_LANES,)`` partial and accumulates that,
# instead of summing the tiles into a second ``(sub_m, block_n)`` fp32
# accumulator and reducing once.  That costs ``n_tiles - 1`` extra 8-lane adds
# and saves a WHOLE ``(sub_m, block_n)`` fp32 buffer -- and that buffer is the
# difference between fitting and not fitting at n = 16384: with it,
# ``_plan_layer_resident`` declines every width (room = 0 at n_tiles = 2 and
# 4, and n_tiles = 8 is past _MAX_N_TILES), which is precisely why R274 recorded
# that its layer path "falls back at n = 16384".  llama-405b-decode is the case
# that sets LayerNormFwdOp's ratio_min, so declining it would have missed the
# target.


def _plan_layer_resident(m: int, n: int, itemsize: int):
    """Geometry for the LayerNorm resident-row path, or ``None`` if it declines.

    ``None`` is a normal answer, not a failure: the caller then compiles
    ``_compile_norm``, which has no residency requirement and no width rule.
    """
    m = max(1, int(m))
    n = max(1, int(n))
    if n % TILE_GRAIN:
        # Same reason as ``_plan_rms_resident``: the ragged-column path is what
        # ``_compile_norm`` already handles, and duplicating its scalar n-tail
        # here would double the surface without touching any measured lever.
        return None
    # UB census, read off the allocation list in ``_compile_layer_resident``.
    # One wrong entry is a silent UB overflow (R194 2.2), so it is spelled out:
    #   per row of the tile : src[itemsize] + accb[4] + hold[4 * n_tiles * width]
    #                         ... and hold spans the WHOLE row, hence 4 * n
    #   fixed               : wub[itemsize] + w32[4] + bub[itemsize] + b32[4],
    #                         all (block_n,)
    #                         + 6 statistics buffers of _STAT_LANES fp32
    occupancy = max(1, m // (2 * LAUNCH_BLOCK_CAP))
    fixed_rows = 6 * _STAT_LANES * 4
    width = (n // _TARGET_N_TILES) - (n // _TARGET_N_TILES) % TILE_GRAIN
    while width >= TILE_GRAIN:
        n_tiles = n // width
        if n % width == 0 and n_tiles <= _MAX_N_TILES:
            per_row = width * itemsize + width * 4 + n * 4
            fixed = 2 * width * itemsize + 2 * width * 4 + fixed_rows
            room = (UB_BUDGET_NORM_BYTES - fixed) // per_row
            if room >= 1:
                sub_m = 1
                while sub_m * 2 <= min(room, occupancy):
                    sub_m *= 2
                block_m = sub_m * 2
                m_tiles = (m + block_m - 1) // block_m
                launch_blocks = launch_block_count(min(m_tiles, LAUNCH_BLOCK_CAP))
                return {
                    "sub_m": sub_m,
                    "block_m": block_m,
                    "block_n": width,
                    "m_tiles": m_tiles,
                    "n_tiles": n_tiles,
                    "m_pad": m_tiles * block_m,
                    "launch_blocks": launch_blocks,
                    "grid_repeats": grid_repeat_count(m_tiles, launch_blocks),
                    "ub_bytes": sub_m * per_row + fixed,
                }
        width -= TILE_GRAIN
    return None


@lru_cache(maxsize=128)
def _compile_layer_resident(m: int, n: int, dtype_name: str, eps: float):
    """LayerNorm with the row group resident in UB, or ``None`` if it will not fit."""
    if dtype_name not in {"float16", "bfloat16", "float32"}:
        raise TypeError(f"normalization supports floating dtypes, got {dtype_name}")
    itemsize = 2 if dtype_name in ("float16", "bfloat16") else 4
    plan = _plan_layer_resident(m, n, itemsize)
    if plan is None:
        return None
    sub_m = plan["sub_m"]
    block_m = plan["block_m"]
    block_n = plan["block_n"]
    m_tiles = plan["m_tiles"]
    n_tiles = plan["n_tiles"]
    m_pad = plan["m_pad"]
    launch_blocks = plan["launch_blocks"]
    grid_repeats = plan["grid_repeats"]
    need_cast = dtype_name != "float32"

    @tilelang.jit(
        out_idx=[1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor([m, n], dtype_name),
            B: T.Tensor([m_pad, n], dtype_name),
            W: T.Tensor([n], dtype_name),
            Bias: T.Tensor([n], dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                # The whole row group, in fp32, for the lifetime of all passes.
                hold = T.alloc_ub((n_tiles, sub_m, block_n), "float32")
                src = T.alloc_ub((sub_m, block_n), dtype_name)
                accb = T.alloc_ub((sub_m, block_n), "float32")
                wub = T.alloc_ub((block_n,), dtype_name)
                w32 = T.alloc_ub((block_n,), "float32")
                bub = T.alloc_ub((block_n,), dtype_name)
                b32 = T.alloc_ub((block_n,), "float32")
                tot = T.alloc_ub((_STAT_LANES,), "float32")
                part = T.alloc_ub((_STAT_LANES,), "float32")
                mean = T.alloc_ub((_STAT_LANES,), "float32")
                den = T.alloc_ub((_STAT_LANES,), "float32")
                inv = T.alloc_ub((_STAT_LANES,), "float32")
                ref = T.alloc_ub((_STAT_LANES,), "float32")
                spare = T.alloc_ub((_STAT_LANES,), "float32")

                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * block_m + vid * sub_m
                            # ⚠️ No guarded per-row copy loop here, unlike
                            # ``_compile_norm``.  A GM->UB ``T.copy`` is clamped
                            # to the real tensor extent at lowering time
                            # (src/op/ascend.cc :: compute_valid_extent), so an
                            # m tail loads fewer rows and leaves the rest of the
                            # UB tile untouched.  That is safe here because
                            # every statistic is per row: garbage in a row this
                            # lane does not own cannot reach a row it does.  The
                            # STORE is in bounds because B is padded to m_pad.
                            #
                            # Pass 1: load the row group once, keep it in fp32,
                            # and accumulate the row sums.  ``part`` is zeroed
                            # ONCE: ``T.reduce_sum`` narrows its destination to
                            # ``[0:sub_m]``, so lanes above ``sub_m`` keep the
                            # 0.0 written here for every iteration and can never
                            # feed a NaN into ``tot``.
                            T.tile.fill(tot, 0.0)
                            T.tile.fill(part, 0.0)
                            for nt in T.serial(n_tiles):
                                T.copy(A[row_base, nt * block_n], src)
                                if need_cast:
                                    T.tile.cast(hold[nt, :, :], src, "CAST_NONE",
                                                sub_m * block_n)
                                else:
                                    T.copy(src, hold[nt, :, :])
                                T.reduce_sum(hold[nt, :, :], part[0:sub_m], dim=-1)
                                T.tile.add(tot, tot, part)
                            T.tile.mul(mean, tot, T.cast(1.0 / n, "float32"))
                            # Pass 2: textbook two-pass variance over the
                            # RESIDENT row -- centre in place, then accumulate
                            # the squares with ``mul_add_dst`` so the reduction
                            # happens once instead of once per n-tile.  Centring
                            # in place is what lets the epilogue skip the second
                            # subtraction ``_compile_norm`` pays.
                            T.tile.fill(accb, 0.0)
                            for nt in T.serial(n_tiles):
                                for r in T.serial(sub_m):
                                    T.tile.sub(hold[nt, r, :], hold[nt, r, :], mean[r])
                                T.tile.mul_add_dst(accb, hold[nt, :, :], hold[nt, :, :])
                            T.reduce_sum(accb, tot[0:sub_m], dim=-1)
                            T.tile.mul(den, tot, T.cast(1.0 / n, "float32"))
                            T.tile.add(den, den, T.cast(eps, "float32"))
                            T.tile.rsqrt(inv, den)
                            # Ascend's vector rsqrt is intentionally fast and
                            # approximate.  One Newton step restores fp32
                            # accuracy before the final dtype cast -- same step,
                            # same order as ``_compile_norm``.
                            T.tile.mul(ref, inv, inv)
                            T.tile.mul(ref, ref, den)
                            T.tile.mul(ref, ref, 0.5)
                            T.tile.fill(spare, 1.5)
                            T.tile.sub(spare, spare, ref)
                            T.tile.mul(inv, inv, spare)
                            # Pass 3: affine epilogue, one (block_n,) vector
                            # multiply/add per row and one vector-scalar
                            # multiply per row, instead of broadcasting w, b and
                            # inv into (sub_m, block_n) replicas first.
                            for nt in T.serial(n_tiles):
                                T.copy(W[nt * block_n], wub)
                                T.copy(Bias[nt * block_n], bub)
                                if need_cast:
                                    T.tile.cast(w32, wub, "CAST_NONE", block_n)
                                    T.tile.cast(b32, bub, "CAST_NONE", block_n)
                                else:
                                    T.copy(wub, w32)
                                    T.copy(bub, b32)
                                for r in T.serial(sub_m):
                                    T.tile.mul(hold[nt, r, :], hold[nt, r, :], inv[r])
                                    T.tile.mul(hold[nt, r, :], hold[nt, r, :], w32)
                                    T.tile.add(hold[nt, r, :], hold[nt, r, :], b32)
                                if need_cast:
                                    T.tile.cast(src, hold[nt, :, :], "CAST_RINT",
                                                sub_m * block_n)
                                    T.copy(src, B[row_base, nt * block_n])
                                else:
                                    T.copy(hold[nt, :, :], B[row_base, nt * block_n])
        return main

    return factory()


def _build_row_norm(x, weight, bias, *, eps, kind, op_name, resident=False):
    if x is None or weight is None:
        raise ValueError(f"{op_name} requires tensor inputs")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"{op_name} supports floating dtypes, got {x.dtype}")
    x_shape = tuple(int(v) for v in x.shape)
    n = int(x_shape[-1])
    if tuple(weight.shape) != (n,):
        raise ValueError(f"{op_name} expects weight shape ({n},), got {tuple(weight.shape)}")
    if weight.dtype != x.dtype:
        raise ValueError(f"{op_name} weight must match input dtype")
    if kind == "layer":
        if bias is None or tuple(bias.shape) != (n,) or bias.dtype != x.dtype:
            raise ValueError(f"{op_name} bias must match weight shape and dtype")
    m = math.prod(x_shape) // n
    if resident and kind == "rms":
        kernel = _compile_rms_resident(m, n, _dtype_name(x.dtype), float(eps))
        if kernel is not None:
            def launch_resident(inp, scale=weight):
                out = kernel(inp.contiguous().reshape(m, n), scale.contiguous())
                return out[:m, :n].reshape(x_shape)

            return launch_resident
    # R335: the LayerNorm resident-row path when the shape admits it, the
    # shipped two-pass template otherwise.  ``None`` is a normal answer.
    # ``TILEOPS_LAYER_NORM_RESIDENT=0`` forces the old path; it is the A/B
    # switch this change was attributed with, not a shipping mode.
    if resident and kind == "layer" and os.environ.get(
        "TILEOPS_LAYER_NORM_RESIDENT", "1"
    ) != "0":
        kernel = _compile_layer_resident(m, n, _dtype_name(x.dtype), float(eps))
        if kernel is not None:
            def launch_layer_resident(inp, scale=weight, shift=bias):
                out = kernel(
                    inp.contiguous().reshape(m, n),
                    scale.contiguous(),
                    shift.contiguous(),
                )
                return out[:m, :n].reshape(x_shape)

            return launch_layer_resident
    kernel = _compile_norm(m, n, _dtype_name(x.dtype), kind, float(eps))
    def launch(inp, scale=weight, shift=bias):
        out = kernel(inp.contiguous().reshape(m, n), scale.contiguous(), shift.contiguous() if shift is not None else scale.contiguous())
        return out[:m, :n].reshape(x_shape)

    return launch


@register("RMSNormFwdOp")
def build_rms_norm(x, weight, *, normalized_shape=None, eps=1e-6):
    if normalized_shape is not None and tuple(int(v) for v in normalized_shape) != (int(x.shape[-1]),):
        raise ValueError("RMSNormFwdOp currently supports one trailing normalized axis")
    return _build_row_norm(x, weight, None, eps=eps, kind="rms", op_name="RMSNormFwdOp",
                           resident=True)


@register("LayerNormFwdOp")
def build_layer_norm(x, weight, bias, *, normalized_shape=None, eps=1e-5):
    if normalized_shape is not None and tuple(int(v) for v in normalized_shape) != (int(x.shape[-1]),):
        raise ValueError("LayerNormFwdOp currently supports one trailing normalized axis")
    return _build_row_norm(x, weight, bias, eps=eps, kind="layer", op_name="LayerNormFwdOp",
                           resident=True)


def _unsupported(name):
    @register(name)
    def builder(*args, **kwargs):
        raise NotImplementedError(f"{name} is outside the verified row-wise two-pass template")
    return builder


# BatchNormBwdOp is owned by families.normalization_spatial.py.  Keeping its
# fail-closed stub here would register the same name twice during package import.


@lru_cache(maxsize=128)
def _compile(m: int, n: int, dtype_name: str, kind: str = "softmax"):
    # Small/non-divisible probe shapes get one exact-width tile. Production
    # workloads use the bounded 256-column tile and therefore exercise the
    # two-pass scan over many chunks.
    # Vector instructions require a 32-element aligned extent.  Keep small
    # probe shapes in one tile while padding only the UB extent; tail accesses
    # below are still guarded by the real ``n``.
    itemsize = 2 if dtype_name in ("float16", "bfloat16") else 4
    # --- Geometry (R200) -------------------------------------------------
    # UB census: (sub_m, block_n) -> src[itemsize] + x32[4] + max_2d[4]
    #                                + sum_2d[4];  (sub_m,) -> 6 fp32 scalars.
    # ``grain=32`` keeps the old "small probe shapes get one exact-width tile"
    # behaviour reachable; the vector extent requirement is 32 elements.
    plan = plan_rowwise_norm(
        m, n, itemsize,
        bytes_per_cell=itemsize + 12,
        bytes_per_col=0,
        bytes_per_row=6 * 4,
        grain=32 if n <= 512 else None,
    )
    sub_m = plan["sub_m"]
    block_m = plan["block_m"]
    block_n = plan["block_n"]
    m_tiles = plan["m_tiles"]
    n_tiles = plan["n_tiles"]
    m_pad = plan["m_pad"]
    n_pad = plan["n_pad"]
    has_m_tail = plan["has_m_tail"]
    has_n_tail = plan["has_n_tail"]
    launch_blocks = plan["launch_blocks"]
    grid_repeats = plan["grid_repeats"]
    calc_dtype = "float32"
    need_cast = dtype_name != calc_dtype
    pad_value = {
        "float16": -65504.0,
        "bfloat16": -3.38953139e38,
        "float32": -3.402823466e38,
    }[dtype_name]

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
            A: T.Tensor([m, n], dtype_name),
            B: T.Tensor([m_pad, n_pad], dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((sub_m, block_n), dtype_name)
                x32 = T.alloc_ub((sub_m, block_n), calc_dtype)
                tile_max = T.alloc_ub((sub_m,), calc_dtype)
                prev_max = T.alloc_ub((sub_m,), calc_dtype)
                tile_sum = T.alloc_ub((sub_m,), calc_dtype)
                prev_sum = T.alloc_ub((sub_m,), calc_dtype)
                tmp = T.alloc_ub((sub_m,), calc_dtype)
                max_2d = T.alloc_ub((sub_m, block_n), calc_dtype)
                sum_2d = T.alloc_ub((sub_m, block_n), calc_dtype)
                log_sum = T.alloc_ub((sub_m,), calc_dtype)

                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * block_m + vid * sub_m
                            T.tile.fill(prev_max, -T.infinity(calc_dtype))
                            T.tile.fill(prev_sum, 0.0)

                            # Pass 1: online max and exp-sum. Padded columns are -inf,
                            # so they contribute zero without a host-side pad/copy.
                            for nt in T.serial(n_tiles):
                                if has_n_tail:
                                    T.tile.fill(src, pad_value)
                                if not has_n_tail:
                                    # R200: the m-tail predicate used to be
                                    # missing here, so a full sub_m-row vector
                                    # copy read up to sub_m-1 rows PAST the end
                                    # of A whenever n divided block_n but m did
                                    # not divide block_m (e.g. SoftmaxFwdOp
                                    # lm-head-logits, m=4 sub_m=64: 60 rows x
                                    # 102400 cols of out-of-bounds GM).
                                    if not has_m_tail:
                                        T.copy(A[row_base, nt * block_n], src)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(A[row_base + r, nt * block_n], src[r, :])
                                elif nt < n_tiles - 1:
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
                                T.reduce_max(x32, tile_max, dim=-1)
                                T.tile.max(tile_max, prev_max, tile_max)
                                T.tile.sub(tmp, prev_max, tile_max)
                                T.tile.exp(tmp, tmp)
                                T.tile.mul(tmp, prev_sum, tmp)
                                T.tile.broadcast(max_2d, tile_max)
                                T.tile.sub(x32, x32, max_2d)
                                T.tile.exp(x32, x32)
                                T.reduce_sum(x32, tile_sum, dim=-1)
                                T.tile.add(prev_sum, tile_sum, tmp)
                                T.copy(tile_max, prev_max)

                            # Pass 2: normalize each input tile using the final row
                            # statistics. All output writes are vector T.copy operations.
                            T.tile.broadcast(max_2d, prev_max)
                            T.tile.broadcast(sum_2d, prev_sum)
                            for nt in T.serial(n_tiles):
                                if has_n_tail:
                                    T.tile.fill(src, pad_value)
                                if not has_n_tail:
                                    # R200: the m-tail predicate used to be
                                    # missing here, so a full sub_m-row vector
                                    # copy read up to sub_m-1 rows PAST the end
                                    # of A whenever n divided block_n but m did
                                    # not divide block_m (e.g. SoftmaxFwdOp
                                    # lm-head-logits, m=4 sub_m=64: 60 rows x
                                    # 102400 cols of out-of-bounds GM).
                                    if not has_m_tail:
                                        T.copy(A[row_base, nt * block_n], src)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(A[row_base + r, nt * block_n], src[r, :])
                                elif nt < n_tiles - 1:
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
                                if kind == "log_softmax":
                                    T.tile.sub(x32, x32, max_2d)
                                    T.tile.ln(log_sum, prev_sum)
                                    T.tile.broadcast(sum_2d, log_sum)
                                    T.tile.sub(x32, x32, sum_2d)
                                else:
                                    T.tile.sub(x32, x32, max_2d)
                                    T.tile.exp(x32, x32)
                                    T.tile.div(x32, x32, sum_2d)
                                if need_cast:
                                    T.tile.cast(src, x32, "CAST_RINT", sub_m * block_n)
                                if need_cast:
                                    T.copy(src, B[row_base, nt * block_n])
                                else:
                                    T.copy(x32, B[row_base, nt * block_n])

        return main

    return factory()


def build_softmax(x, *, dim=None):
    """Build a callable preserving the public tensor shape and dtype."""
    if x is None:
        raise ValueError("SoftmaxFwdOp requires an input tensor")
    shape = tuple(int(v) for v in x.shape)
    ndim = len(shape)
    axis = _implicit_dim(ndim) if dim is None else int(dim)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"softmax dim {dim!r} out of range for rank {ndim}")
    order = tuple(i for i in range(ndim) if i != axis) + (axis,)
    inverse = tuple(order.index(i) for i in range(ndim))
    m = math.prod(shape[i] for i in range(ndim) if i != axis)
    n = shape[axis]
    kernel = _compile(m, n, _dtype_name(x.dtype), "softmax")

    def launch(inp):
        view = inp if order == tuple(range(ndim)) else inp.permute(order)
        result = kernel(view.reshape(m, n).contiguous())[:m, :n]
        result = result.reshape(tuple(shape[i] for i in order))
        return result if inverse == tuple(range(ndim)) else result.permute(inverse)

    return launch


def build_log_softmax(x, *, dim=None):
    """Build log-softmax with the same online-max two-pass traversal."""
    if x is None:
        raise ValueError("LogSoftmaxFwdOp requires an input tensor")
    shape = tuple(int(v) for v in x.shape)
    ndim = len(shape)
    axis = _implicit_dim(ndim) if dim is None else int(dim)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"log_softmax dim {dim!r} out of range for rank {ndim}")
    order = tuple(i for i in range(ndim) if i != axis) + (axis,)
    inverse = tuple(order.index(i) for i in range(ndim))
    m = math.prod(shape[i] for i in range(ndim) if i != axis)
    n = shape[axis]
    kernel = _compile(m, n, _dtype_name(x.dtype), "log_softmax")

    def launch(inp):
        view = inp if order == tuple(range(ndim)) else inp.permute(order)
        result = kernel(view.reshape(m, n).contiguous())[:m, :n]
        result = result.reshape(tuple(shape[i] for i in order))
        return result if inverse == tuple(range(ndim)) else result.permute(inverse)

    return launch


@register("SoftmaxFwdOp")
def build_softmax_registered(x, *, dim=None):
    return build_softmax(x, dim=dim)


@register("LogSoftmaxFwdOp")
def build_log_softmax_registered(x, *, dim=None):
    return build_log_softmax(x, dim=dim)


# ======================================================================================
# R355 / T355 #59 -- FP8QuantFwdOp.  Per-(b,s,g)-row absmax -> fp32 scale + e4m3fn bytes.
#
# WHY HERE.  ``kernels/families/__init__.py`` is PM-owned and frozen (R069), so a new
# builder has to live in a module that file already imports.  Of those, this one is the
# semantic fit the dispatch note asked for: FP8 dynamic quantization IS a two-pass
# row-wise reduction -- pass 1 reduces |x| to a row absmax, pass 2 rescales and writes
# the row -- which is the same archetype the softmax above implements.  Nothing here
# touches the softmax path: the two share the module, not a line of code.
#
# WHY THE ENCODE LOOKS LIKE THIS.  dav-2201 has no FP8 unit, so the e4m3fn byte is
# built arithmetically.  The algorithm is PyTorch's own
# ``c10::detail::fp8e4m3fn_from_fp32_value`` (c10/util/Float8_e4m3fn.h) re-expressed in
# the intrinsics that R355-data/p2-intrinsics.json proved BOTH lower AND are right on
# this device.  Four of them are not available and the shape of this code is what is
# left after removing them:
#
#   * ``T.tile.bitwise_and`` / ``_or`` on an int32 buffer process only HALF the tile
#     (measured: 512 int32 in, elements 0..255 right, 256..511 wrong -- the intrinsic
#     counts 16-bit elements).  So there is no AND anywhere below: ``|v|`` comes from
#     ``T.tile.abs`` on the float, and ``(bits >> 20) & 1`` is written as
#     ``(bits >> 20) - ((bits >> 21) << 1)``.
#   * ``T.tile.bitwise_and`` has no scalar form at all (T351 recorded the same thing in
#     families/gemm.py:288).
#   * ``T.tile.select`` has no int32 lowering (bisheng compile error), so every branch
#     merge happens on float32 -- where the same probe measures it correct.
#   * ``cast int32 -> float16`` silently yields zeros, so the byte is produced as
#     int32 -> float32 -> float16 -> uint8.  ``cast float16 -> uint8 CAST_RINT`` is the
#     one narrowing R346 measured as lowering in all four rounding modes.
#   * ``T.tile.bitwise_rshift`` IS arithmetic (measured: -1 >> 31 == -1), which is what
#     lets ``abs(bits >> 31) * 128`` be the sign byte and keeps -0.0 at 0x80.
#
# The encode is verified by enumeration, not by sampling: every one of the 65536 fp16
# values, every e4m3 grid point and midpoint and their +-1/+-2 ulp neighbours, 2.3M
# random fp32 across the saturating and subnormal ranges -- 2426890 values, 0 wrong,
# against CPU ATen's own conversion.  R355-data/probe/p3-encode.json.
# ======================================================================================

#: The manifest's declared input dtypes.  Spelled out rather than reusing the softmax
#: helper above, so a wrong dtype names THIS op in the error.
_FP8Q_DTYPES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}

#: e4m3fn's largest finite magnitude.  The manifest's scale is ``absmax / 448``.
_E4M3_MAX = 448.0

#: ``f_bits >= (1087 << 20)`` is PyTorch's "no longer representable" test; 1087 << 20
#: is exactly 480.0f, so the same test is a float compare here -- and the float compare
#: is what handles NaN, because ``NaN < 480.0`` is false on this device (measured).
_E4M3_NAN_EDGE = 480.0

#: ``121 << 23`` is 2^-6, e4m3fn's smallest NORMAL magnitude.  Below it PyTorch takes
#: the denormal path.
_E4M3_MIN_NORMAL = 0.015625

#: ``141 << 23`` = 2^14.  Adding it to |v| parks |v| in a binade whose fp32 ulp is
#: exactly 2^-9 -- e4m3fn's subnormal step -- so the mantissa field of the sum, read as
#: an integer, IS the subnormal code, already rounded to nearest-even by the fp32 add.
_DENORM_MAGIC = 16384.0
_DENORM_MAGIC_BITS = 0x46800000

#: ``(7 - 127) << 23`` rebiases fp32's exponent to e4m3's, and ``0x7FFFF`` is the
#: half-ulp-minus-one rounding bias for the 20 bits that get dropped.  Folded into one
#: scalar add because ``T.tile.add`` takes a scalar but every extra pass costs a tile.
_NORMAL_REBIAS = -0x3C000000 + 0x7FFFF

#: Bytes per element of the UB census below.  Spelled out because a wrong entry here is
#: a silent UB overflow (aicore 507015), not a compile error:
#:   src  in_dtype | x32 a g  fp32 | ab s1 s2 tb cn cs int32
#:   | cnf csf sgn code c127 inv2d fp32 | c16 fp16 | ob uint8 | 2 packed masks at 1/8 byte
_FP8Q_FIXED_BYTES = 3 * 4 + 6 * 4 + 6 * 4 + 2 + 1
_FP8Q_MASK_EIGHTHS = 2


#: Dekker's splitter for binary32 (2^12 + 1).  ``x * _SPLIT`` must not overflow, which
#: is what ``_AMAX_LO`` / ``_AMAX_HI`` below guarantee.
_SPLIT = 4097.0

#: 🚨 DEFINITIONAL, and the CPU reference implements the identical clamp.
#: The row absmax is clamped to [2^-100, 2^100] before either division.  Two reasons,
#: both measured:
#:   1. an all-zero row makes ``448 / amax`` infinite and ``0 * inf`` a NaN, so the whole
#:      row would quantize to the e4m3 NaN code 0x7F;
#:   2. the Dekker two-product below multiplies by 4097, which overflows fp32 above
#:      ~8.3e34 -- outside the clamp the exact residual would be +-inf and the refined
#:      quotient NaN, i.e. a SILENT wrong answer rather than a loud one.
#: Inside the clamp nothing is changed: every manifest workload has amax of order 1.
#: R355.md section 5.3 has the enumeration of what the clamp touches.
#:
#: 🚨 THE BOUNDS ARE 1e-30 / 1e30 AND NOT 2**-100 / 2**100 FOR A MEASURED REASON.
#: The Ascend codegen emits a float scalar with SEVEN significant digits.  With
#: ``2.0 ** 100`` the generated source carried
#:     AscendC::Mins(amax[0], amax[0], 1.267651e+30f, 37);
#: (R355-data/logs/p15-scalar-literal.txt) -- 1.267651e+30 is 0x71800003, three ulp
#: away from 2^100's 0x71800000, so the device clamped to a different number than the
#: reference did and every derived value inherited the gap.  A bound whose decimal
#: form is exact in seven digits round-trips, and 1e30 / 1e-30 are still well inside
#: the overflow limit the Dekker split needs (|x| * 4097 must stay finite, i.e.
#: |x| < 8.3e34, and 448 / 1e-30 = 4.5e32 also splits safely).
_AMAX_LO = 1e-30
_AMAX_HI = 1e30


@T.macro
def _refined_div(q, a, b, t1, t2, hq, lq, hb, lb, pp, ee):
    """``q = correctly-rounded fl32(a / b)``, for a, b inside the amax clamp.

    ``T.tile.div`` alone is NOT correctly rounded on dav-2201: measured 1 ulp off CPU
    on 234/4096 random pairs and on 464/4096 of a dense mantissa sweep
    (R355-data/probe/p5-markstein.json).  That 1 ulp is not cosmetic here -- p4 traced
    38/524288 wrong e4m3 bytes to it, every one a value sitting exactly on an e4m3
    rounding midpoint, which fp16/bf16 inputs hit often because ``x / amax`` is a ratio
    of two low-mantissa numbers.

    This is Markstein's refinement without an FMA: the hardware quotient q0, the EXACT
    residual ``a - q0*b`` from a Dekker two-product, and one more divide for the
    correction.  Measured bit-identical to CPU on all 16384 divisors of p5, including
    the dense mantissa sweep that is worst for a divider.

    MUST be a ``T.macro``: a plain Python helper is evaluated by the TVMScript parser
    but its statements are never emitted -- T351 measured that exact failure
    (families/gemm.py:288) and it is silent.
    """
    T.tile.div(q, a, b)
    T.tile.mul(t1, q, _SPLIT)          # split(q)
    T.tile.sub(t2, t1, q)
    T.tile.sub(hq, t1, t2)
    T.tile.sub(lq, q, hq)
    T.tile.mul(t1, b, _SPLIT)          # split(b)
    T.tile.sub(t2, t1, b)
    T.tile.sub(hb, t1, t2)
    T.tile.sub(lb, b, hb)
    T.tile.mul(pp, q, b)               # pp = fl(q*b)
    T.tile.mul(ee, hq, hb)             # ee = q*b - pp, exactly
    T.tile.sub(ee, ee, pp)
    T.tile.mul(t1, hq, lb)
    T.tile.add(ee, ee, t1)
    T.tile.mul(t1, lq, hb)
    T.tile.add(ee, ee, t1)
    T.tile.mul(t1, lq, lb)
    T.tile.add(ee, ee, t1)
    T.tile.sub(t1, a, pp)              # a - pp is exact (Sterbenz: pp is within 2x of a)
    T.tile.sub(t1, t1, ee)
    T.tile.div(t2, t1, b)
    T.tile.add(q, q, t2)


@lru_cache(maxsize=64)
def _compile_fp8_quant(rows: int, cols: int, in_dtype: str):
    """Compile the [rows, cols] -> (scale[rows], byte[rows, cols]) quantizer.

    GEOMETRY, and why it is not free-form.  ``sub_m`` is a POWER OF TWO and at least 8,
    and both outputs are declared padded to ``m_tiles * block_m`` rows.  Both come from
    a measured failure, not from taste (R355-data/probe/p6-debug.json):

      * with ``sub_m = 45`` the scale row vector is 180 bytes, so consecutive vector
        contexts wrote GM at offsets that are not a multiple of 32.  The tail path then
        fell back to ``SCALE[row_base + r] = scl[r]`` -- a 4-byte SCALAR GM store,
        issued from both AIV contexts of six blocks at once.  Rows 32..44 of the first
        tile came back holding the previous kernel's bytes: the same three float values
        appeared, bit for bit, in two runs with completely different inputs.
      * padding the outputs removes the tail predicate from the STORE side entirely, so
        every write is one whole-tile ``T.copy`` -- a vector DMA, which is coherent.
        The LOAD side keeps its predicate (reading past the real input is the one thing
        padding cannot fix) and zero-fills first.

    With ``sub_m`` a power of two, both manifest row counts (8192 and 4096) divide
    ``block_m`` exactly, so the manifest workloads take the untailed path on both sides.
    """
    if in_dtype not in {"float16", "bfloat16", "float32"}:
        raise TypeError(f"FP8QuantFwdOp supports floating dtypes, got {in_dtype}")
    in_bytes = 2 if in_dtype in {"float16", "bfloat16"} else 4
    live_eighths = 8 * (in_bytes + _FP8Q_FIXED_BYTES) + _FP8Q_MASK_EIGHTHS
    cells = (UB_BUDGET_NORM_BYTES * 8) // live_eighths
    cap = min(rows_pow2 := 1 << max(0, (cells // max(cols, 1)).bit_length() - 1), cells // max(cols, 1))
    if cap < 8:
        raise NotImplementedError(
            f"FP8QuantFwdOp needs at least 8 rows of {cols} columns in UB "
            f"({UB_BUDGET_NORM_BYTES} bytes, {live_eighths / 8:.2f} bytes per element); "
            f"index_dim {cols} leaves room for {cap}.  A column-tiled variant would be "
            f"needed, and no manifest workload asks for one."
        )
    sub_m = cap
    block_m = sub_m * _VEC
    m_tiles = max(1, math.ceil(rows / block_m))
    rows_pad = m_tiles * block_m
    launch_blocks = launch_block_count(min(m_tiles, LAUNCH_BLOCK_CAP))
    grid_repeats = grid_repeat_count(m_tiles, launch_blocks)
    has_m_tail = rows % block_m != 0
    need_cast = in_dtype != "float32"
    tile = sub_m * cols

    @tilelang.jit(
        out_idx=[1, 2],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            X: T.Tensor([rows, cols], in_dtype),
            SCALE: T.Tensor([rows_pad], "float32"),
            OUTB: T.Tensor([rows_pad, cols], "uint8"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((sub_m, cols), in_dtype)
                x32 = T.alloc_ub((sub_m, cols), "float32")
                a = T.alloc_ub((sub_m, cols), "float32")
                g = T.alloc_ub((sub_m, cols), "float32")
                ab = T.alloc_ub((sub_m, cols), "int32")
                s1 = T.alloc_ub((sub_m, cols), "int32")
                s2 = T.alloc_ub((sub_m, cols), "int32")
                tb = T.alloc_ub((sub_m, cols), "int32")
                cn = T.alloc_ub((sub_m, cols), "int32")
                cs = T.alloc_ub((sub_m, cols), "int32")
                cnf = T.alloc_ub((sub_m, cols), "float32")
                csf = T.alloc_ub((sub_m, cols), "float32")
                sgn = T.alloc_ub((sub_m, cols), "float32")
                code = T.alloc_ub((sub_m, cols), "float32")
                c127 = T.alloc_ub((sub_m, cols), "float32")
                inv2d = T.alloc_ub((sub_m, cols), "float32")
                m_sub = T.alloc_ub((sub_m * cols // 8,), "uint8")
                m_small = T.alloc_ub((sub_m * cols // 8,), "uint8")
                c16 = T.alloc_ub((sub_m, cols), "float16")
                ob = T.alloc_ub((sub_m, cols), "uint8")
                amax = T.alloc_ub((sub_m,), "float32")
                inv = T.alloc_ub((sub_m,), "float32")
                scl = T.alloc_ub((sub_m,), "float32")
                cmax = T.alloc_ub((sub_m,), "float32")
                # ``_refined_div`` scratch.  One value per row, so the whole block costs
                # 8 * sub_m * 4 bytes -- not a per-element cost, and it does not enter
                # the tile census above.
                rt1 = T.alloc_ub((sub_m,), "float32")
                rt2 = T.alloc_ub((sub_m,), "float32")
                rhq = T.alloc_ub((sub_m,), "float32")
                rlq = T.alloc_ub((sub_m,), "float32")
                rhb = T.alloc_ub((sub_m,), "float32")
                rlb = T.alloc_ub((sub_m,), "float32")
                rpp = T.alloc_ub((sub_m,), "float32")
                ree = T.alloc_ub((sub_m,), "float32")

                with T.Scope("V"):
                    T.tile.fill(c127, 127.0)
                    T.tile.fill(cmax, _E4M3_MAX)
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * block_m + vid * sub_m
                            # --- load (the only side that still needs a predicate) --
                            if not has_m_tail:
                                T.copy(X[row_base, 0], src)
                            else:
                                T.tile.fill(src, 0.0)
                                for r in T.serial(sub_m):
                                    if row_base + r < rows:
                                        T.copy(X[row_base + r, 0], src[r, :])
                            T.barrier_all()
                            if need_cast:
                                T.tile.cast(x32, src, "CAST_NONE", tile)
                            else:
                                T.copy(src, x32)
                            # --- pass 1: row absmax, clamp, and the two divisions ---
                            T.tile.abs(a, x32)
                            T.reduce_max(a, amax, dim=-1)
                            # The clamp is part of the op's declared arithmetic; the
                            # reference applies the identical one.
                            T.tile.max(amax, amax, _AMAX_LO)
                            T.tile.min(amax, amax, _AMAX_HI)
                            # The ONLY divisions in this kernel, and they run on one
                            # value per ROW.  Every per-element operation is a multiply,
                            # which IS IEEE-exact; both quotients are refined to
                            # correctly rounded because a 1-ulp quotient moves values
                            # that sit exactly on an e4m3 midpoint to the wrong side.
                            _refined_div(scl, amax, cmax, rt1, rt2, rhq, rlq, rhb, rlb, rpp, ree)
                            _refined_div(inv, cmax, amax, rt1, rt2, rhq, rlq, rhb, rlb, rpp, ree)
                            # 🚨 EXPLICIT, not redundant.  ``TL_ASCEND_AUTO_SYNC`` did not
                            # insert a barrier between the refinement's last write to
                            # ``inv`` and the broadcast that reads it, and the broadcast
                            # then picked up the UNREFINED quotient on a few rows per
                            # tensor -- 1 ulp, which only shows up on values sitting
                            # exactly on an e4m3 midpoint, and which varied from process
                            # to process (p6 saw 1 and 19 wrong bytes where p9, same code
                            # and same input, saw 0).  R355.md section 5.4.
                            T.barrier_all()
                            # --- pass 2: rescale and encode ------------------------
                            T.tile.broadcast(inv2d, inv)
                            T.barrier_all()
                            T.tile.mul(x32, x32, inv2d)
                            # sign byte, straight off the bit pattern so -0.0 -> 0x80
                            T.reinterpretcast(s1, x32, "int32_t")
                            T.tile.bitwise_rshift(s2, s1, 31)
                            T.tile.cast(sgn, s2, "CAST_RINT", tile)
                            T.tile.abs(sgn, sgn)
                            T.tile.mul(sgn, sgn, 128.0)
                            # |v| and its bits
                            T.tile.abs(a, x32)
                            T.reinterpretcast(ab, a, "int32_t")
                            # normal branch: round-to-nearest-EVEN via an integer bias
                            T.tile.bitwise_rshift(s1, ab, 20)
                            T.tile.bitwise_rshift(s2, ab, 21)
                            T.tile.bitwise_lshift(s2, s2, 1)
                            T.tile.sub(s1, s1, s2)
                            T.tile.add(tb, ab, _NORMAL_REBIAS)
                            T.tile.add(tb, tb, s1)
                            T.tile.bitwise_rshift(cn, tb, 20)
                            # subnormal branch: the 2^14 magic add
                            T.tile.add(g, a, _DENORM_MAGIC)
                            T.reinterpretcast(cs, g, "int32_t")
                            T.tile.add(cs, cs, -_DENORM_MAGIC_BITS)
                            # merge on float32 (int32 select does not lower)
                            T.tile.cast(cnf, cn, "CAST_RINT", tile)
                            T.tile.cast(csf, cs, "CAST_RINT", tile)
                            T.tile.compare(m_sub, a, _E4M3_MIN_NORMAL, "LT")
                            T.tile.compare(m_small, a, _E4M3_NAN_EDGE, "LT")
                            T.tile.select(code, m_sub, csf, cnf, "VSEL_TENSOR_TENSOR_MODE")
                            T.tile.select(code, m_small, code, c127, "VSEL_TENSOR_TENSOR_MODE")
                            T.tile.add(code, code, sgn)
                            T.tile.cast(c16, code, "CAST_NONE", tile)
                            T.tile.cast(ob, c16, "CAST_RINT", tile)
                            T.barrier_all()
                            # --- store: always a whole-tile vector copy -------------
                            T.copy(ob, OUTB[row_base, 0])
                            T.copy(scl, SCALE[row_base])

        return main

    return factory()


@register("FP8QuantFwdOp")
def build_fp8_quant(input_tensor):
    """Manifest signature: ``input_tensor[b, s, g, d]`` -> ``scale[b,s,g]``, ``out[...]``.

    The op layer hands a ``TensorSpec`` (dtype + shape), so the shape is committed at
    build time exactly as every other builder in this package commits it.
    """
    if input_tensor is None:
        raise ValueError("FP8QuantFwdOp requires an input tensor")
    shape = tuple(int(d) for d in input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"FP8QuantFwdOp expects input_tensor shape [B, S, G, D], got {shape}")
    rows = shape[0] * shape[1] * shape[2]
    cols = shape[3]
    if min(shape) <= 0:
        raise ValueError("FP8QuantFwdOp input dimensions must be positive")
    try:
        name = _FP8Q_DTYPES[input_tensor.dtype]
    except KeyError:
        raise TypeError(
            f"FP8QuantFwdOp does not support dtype {input_tensor.dtype}; supported "
            f"dtypes are [torch.float16, torch.bfloat16, torch.float32]"
        ) from None
    kernel = _compile_fp8_quant(rows, cols, name)

    def invoke(x):
        if tuple(x.shape) != shape:
            raise ValueError(
                f"FP8QuantFwdOp kernel shape mismatch: expected {shape}, received {tuple(x.shape)}"
            )
        flat = x.reshape(rows, cols)
        scale, byte = kernel(flat if flat.is_contiguous() else flat.contiguous())
        # Both outputs are declared padded to whole blocks (see _compile_fp8_quant);
        # the padding rows hold whatever the last partial tile computed and are dropped
        # here, exactly as build_softmax slices [:m, :n].
        scale, byte = scale[:rows], byte[:rows]
        # ``.view`` only relabels the bytes.  Nothing on the device converts to fp8:
        # ``Tensor.to(torch.float8_e4m3fn)`` on an NPU tensor goes through
        # aclnnInplaceCopy, which has no e4m3 path and fails with 561103 (T355 section 1.2).
        return scale.reshape(shape[:3]), byte.view(torch.float8_e4m3fn).reshape(shape)

    return invoke


__all__ = ["build_fp8_quant", "build_log_softmax", "build_softmax", "build_softmax_registered"]
