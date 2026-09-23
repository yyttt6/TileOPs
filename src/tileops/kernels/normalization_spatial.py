"""NCHW spatial normalization kernels.

The reduction is deliberately expressed over the physical channel/spatial
coordinates.  Flattening an NCHW tensor to rows would change the reduction
domain for GroupNorm and InstanceNorm, so each logical group/channel owns one
tile task and walks its strided channel slices.  Mean and M2 use fp32 tile
Welford state throughout; no second-moment subtraction is used.
"""

from functools import lru_cache
import math
import os

import tilelang
import tilelang.language as T
import torch

from .common import (
    LAUNCH_BLOCK_CAP,
    SPATIAL_ELEM_GRAIN,
    TILE_GRAIN,
    TILE_WIDTH_CAP,
    UB_BUDGET_BYTES,
    UB_BUDGET_NORM_BYTES,
    grid_repeat_count,
    launch_block_count,
    plan_rowwise_norm,
    plan_spatial_tile,
)


_TILE = 128
_VEC = 2

#: Event ids reserved for the manual write-after-read drains in
#: ``_compile_fused_add_norm``.  ``TL_ASCEND_AUTO_SYNC`` allocates its own ids
#: from the low end, so these sit above the range moe.py already uses (2, 3).
_WAR_EVENT_V = 4
_WAR_EVENT_MTE2 = 5
_WAR_EVENT_S = 6


def _drain_out_dma(has_tail: bool):
    """Block every pipe that may rewrite the just-copied-out UB buffer.

    ``T.copy(ub, gm)`` issues an asynchronous MTE3 transfer that reads ``ub``.
    Reusing ``ub`` before that transfer drains is a write-after-read hazard, and
    ``TL_ASCEND_AUTO_SYNC`` does not cover it for a copy in the middle of a loop
    body: the next iteration's reload wins the race and the tile written to GM
    holds the *following* tile's bytes.  (AUTO_SYNC does emit exactly this drain
    for the second loop's ``Y`` writeback -- ``HardEvent::MTE3_MTE2`` -- which is
    why only ``residual_out`` was ever wrong.)  The set/wait pair per consuming
    pipe is the idiom kernels/moe.py:116-117 already uses for MTE3 -> MTE2.

    Only MTE2 rewrites the buffer on the divisible path, so the V and S drains
    are emitted solely for ``has_tail``, where the next iteration opens with a
    ``T.tile.fill`` (V) and the last column tile reloads with a scalar loop (S).
    Emitting all three unconditionally costs ~21% on 2048x16384; see R186.
    """
    # T.evaluate is required: these intrinsics are built here rather than in the
    # parsed prim_func body, so nothing would emit them into the IR otherwise.
    T.evaluate(T.set_flag("mte3", "mte2", _WAR_EVENT_MTE2))
    T.evaluate(T.wait_flag("mte3", "mte2", _WAR_EVENT_MTE2))
    if has_tail:
        T.evaluate(T.set_flag("mte3", "v", _WAR_EVENT_V))
        T.evaluate(T.wait_flag("mte3", "v", _WAR_EVENT_V))
        T.evaluate(T.set_flag("mte3", "s", _WAR_EVENT_S))
        T.evaluate(T.wait_flag("mte3", "s", _WAR_EVENT_S))


def _dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise TypeError(f"spatial normalization supports floating dtypes, got {dtype}") from exc


def _launch_tasks(tasks: int) -> tuple[int, int]:
    logical_blocks = max(1, math.ceil(tasks / _VEC))
    launch_blocks = launch_block_count(logical_blocks)
    return launch_blocks, grid_repeat_count(logical_blocks, launch_blocks)


def _bn_fwd_arm() -> str:
    """R362's control-arm switch.  "base" = the tree as T362 found it."""
    return os.environ.get("TILEOPS_T362_ARM", "r362")


#: R362.  Elements in one ``(rows, c)`` tile of the flat rank-2 BatchNorm path.
#: Deliberately the same 8192 the other templates operate at.
_FLAT_TILE_CELLS = 8192
#: ... and how many tasks to aim for: 25 AI cores x ``_VEC`` vector units.
_FLAT_TASK_TARGET = 50

#: R362.  ``True`` applies the four affine steps with the per-channel scalar as a
#: VECTOR-SCALAR operand (``T.tile.sub(x32, x32, mean[0])``) instead of
#: ``broadcast`` into a full tile followed by a vector-vector op.  The precedent
#: is the resident norm's affine epilogue, which already multiplies a tile row by
#: ``inv[r]`` this way.  Flip to ``False`` to fall back to the broadcast form,
#: which is the instruction-for-instruction twin of ``_compile_batch``.
#: ⚠️ Whichever is used, the two arms' md5s must match -- see R362 section 5.
_PLANE_SCALAR_OPERAND = True


def _batch_flat_rows(n: int, c: int) -> int:
    """Rows per task for the flat ``[N, C]`` path.

    A DIVISOR of ``n`` (so no task is ragged -- the ragged branch is the whole
    disease being cured here) that fits ``_FLAT_TILE_CELLS`` elements and lands
    near ``_FLAT_TASK_TARGET`` tasks.  ``n = 32`` returns 1, i.e. one row per
    vector unit and 16 launch blocks.
    """
    rows = max(1, n // _FLAT_TASK_TARGET)
    while rows > 1 and (n % rows or rows * c > _FLAT_TILE_CELLS):
        rows -= 1
    return rows



#: ⚠️ ``_TILE`` was the chunk width for every ``(1, tile)`` spatial kernel below,
#: for every shape.  R200 replaced it with ``_spatial_tile``; it survives only as
#: the granularity handed to the planner.  A channel with ``spatial = 1024*1024``
#: used to walk 8192 chunks per pass, each paying the whole ~20-op vector chain
#: on 128 elements (256 bytes for fp16).
def _spatial_tile(spatial: int, dtype_name: str, cell_extra: int = 8) -> int:
    """Chunk width for the ``(1, tile)`` spatial kernels.

    ``cell_extra`` is the per-element UB cost of the fp32 ``(1, tile)`` buffers
    (the two dtype-width buffers are added here).  It MUST match the allocation
    list of the caller -- see the census comment in each ``_compile_*``.

    ⚠️ Both the ``_compile_*`` function and its ``launch_*`` wrapper call this,
    because the wrapper sizes the padded output buffer from it.  They must never
    disagree or the kernel writes outside the tensor.
    """
    itemsize = 2 if dtype_name in ("float16", "bfloat16") else 4
    return plan_spatial_tile(spatial, 2 * itemsize + cell_extra, bytes_fixed=1024)


#: ---------------------------------------------------------------------------
#: RAGGED-CHUNK LOADS (R200 round 2) -- read this before touching them.
#:
#: Every ``(1, tile)`` kernel below used, for its ragged chunk::
#:
#:     for j in T.serial(tile):
#:         if j < tail:
#:             src[0, j] = A[base + j]
#:
#: which iterates ``tile`` times no matter how short the tail is.  That is what
#: made R200 round 1's wider tiles a 4x REGRESSION on ``spatial = 900``
#: (GroupNorm 0.113 -> 0.038); see ``plan_spatial_tile`` for the full history.
#:
#: The replacement, written inline at every site, is::
#:
#:     if tail_vec:
#:         T.copy(A[base:base + tail_vec], src[0, 0:tail_vec])
#:     for j in T.serial(blk):
#:         if tail_vec + j < tail:
#:             src[0, tail_vec + j] = A[base + tail_vec + j]
#:
#: with ``blk = 32 // itemsize`` and ``tail_vec = tail - tail % blk``.  fp16,
#: tile=1024, tail=900: 4 scalar iterations instead of 1024.
#:
#: TRAP 1 -- the obvious one-liner is WRONG.  ``T.copy(A[base:base + tail],
#: src[0, 0:tail])`` is a partial-extent load, and a partial-extent GM->UB copy
#: moves WHOLE 32-BYTE BLOCKS: when ``tail * itemsize`` is not a multiple of 32
#: it reads past ``base + tail`` and drops whatever lives there into the padding
#: lanes, which then feed ``reduce_sum``.  Measured against a deliberately
#: poisoned slab behind the tensor (R200-data/r2/probe_ragged_load.log):
#:
#:     tail*itemsize % 32 == 0  ->  matches the scalar loop exactly
#:     tail=1   (2 bytes)       ->  rel_err 2.9e-02  (scalar loop: 0.0)
#:     tail=63  (252 bytes)     ->  rel_err 5.0e-04  (scalar loop: 0.0)
#:     tail=895 (1790 bytes)    ->  rel_err 1.1e-03  (scalar loop: 3.6e-04)
#:
#: Hence ``tail_vec``, never ``tail``.
#:
#: TRAP 2 -- ``run_to_run_identical`` was TRUE for every one of those wrong
#: cases, because the over-read lands in deterministic neighbouring memory.
#: PROJECT_STATE 13.54's run-to-run judge does NOT catch this class.  What
#: caught it was poisoning the memory behind the tensor.
#:
#: TRAP 3 -- IT CANNOT BE FACTORED INTO A HELPER, which is why the same handful
#: of lines is repeated at every site instead of being called.  A module-level
#: helper invoked from inside a ``@T.prim_func`` may not contain a ``T.serial``
#: loop ("'ForFrame' object is not iterable"), may not assign a buffer element
#: ("'Buffer' object does not support item assignment"), and -- the nasty one --
#: a sliced ``T.copy`` emitted from a helper COMPILES AND SILENTLY DOES NOTHING.
#: Measured side by side, identical code inline vs in a helper
#: (R200-data/r2/probe_spatial_tile.py, modes ``hybrid`` vs ``helper``):
#:
#:     spatial=900  fp16   inline rel_err 2.8e-04   helper rel_err 9.96e-01
#:     spatial=63   fp32   inline rel_err 0.0       helper rel_err 8.89e-01
#:
#: ``_drain_out_dma`` above is fine because it emits flag intrinsics only.
#: ---------------------------------------------------------------------------


def _blk(dtype_name: str) -> int:
    """Elements per 32-byte vector block for ``dtype_name``."""
    return 32 // (2 if dtype_name in ("float16", "bfloat16") else 4)


@lru_cache(maxsize=128)
def _compile_group(
    n: int,
    c: int,
    spatial: int,
    groups: int,
    channels_per_group: int,
    dtype_name: str,
    affine: bool,
    eps: float,
):
    launch_blocks, grid_repeats = _launch_tasks(n * groups)
    # --- Geometry (R200) ---------------------------------------------
    # (1, tile) UB census: src[itemsize] out[itemsize] x32[4] work[4]
    tile = _spatial_tile(spatial, dtype_name, cell_extra=8)
    blk = _blk(dtype_name)
    chunks = math.ceil(spatial / tile)
    tail = spatial % tile
    # 32-byte-aligned part of the ragged chunk, and where the fp32 padding
    # cleanup may start.  Both compile-time constants; see the RAGGED-CHUNK
    # LOADS block above -- in particular why it is inlined, not factored out.
    tail_vec = tail - (tail % blk)
    zpad_start = min(tile, ((tail + 7) // 8) * 8)

    need_cast = dtype_name != "float32"

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((n * c * spatial,), dtype_name),
            B: T.Tensor((n * c * chunks * tile,), dtype_name),
            W: T.Tensor((c,), dtype_name),
            Bias: T.Tensor((c,), dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((1, tile), dtype_name)
                out = T.alloc_ub((1, tile), dtype_name)
                x32 = T.alloc_ub((1, tile), "float32")
                work = T.alloc_ub((1, tile), "float32")
                count = T.alloc_ub((1,), "float32")
                mean = T.alloc_ub((1,), "float32")
                m2 = T.alloc_ub((1,), "float32")
                row_sum = T.alloc_ub((1,), "float32")
                row_m2 = T.alloc_ub((1,), "float32")
                tile_mean = T.alloc_ub((1,), "float32")
                tile_count = T.alloc_ub((1,), "float32")
                delta = T.alloc_ub((1,), "float32")
                ratio = T.alloc_ub((1,), "float32")
                new_count = T.alloc_ub((1,), "float32")
                correction = T.alloc_ub((1,), "float32")
                denom = T.alloc_ub((1,), "float32")
                inv = T.alloc_ub((1,), "float32")
                refine = T.alloc_ub((1,), "float32")
                affine_raw = T.alloc_ub((1,), dtype_name)
                affine32 = T.alloc_ub((1,), "float32")
                w32v = T.alloc_ub((1,), "float32")
                b32v = T.alloc_ub((1,), "float32")
                with T.Scope("V"):
                    for repeat in T.serial(grid_repeats):
                        block = cid + repeat * launch_blocks
                        task = block * _VEC + vid
                        if task < n * groups:
                            batch = task // groups
                            group = task % groups
                            cbase = group * channels_per_group

                            T.tile.fill(count, 0.0)
                            T.tile.fill(mean, 0.0)
                            T.tile.fill(m2, 0.0)
                            for ci in T.serial(channels_per_group):
                                channel = cbase + ci
                                for chunk in T.serial(chunks):
                                    start = chunk * tile
                                    T.tile.fill(src, 0.0)
                                    if tail == 0 or chunk < chunks - 1:
                                        T.copy(A[(batch * c + channel) * spatial + start], src[0, :])
                                    else:
                                        if tail_vec:
                                            T.copy(A[(batch * c + channel) * spatial + start:
                                                     (batch * c + channel) * spatial + start + tail_vec],
                                                   src[0, 0:tail_vec])
                                        for j in T.serial(blk):
                                            if tail_vec + j < tail:
                                                src[0, tail_vec + j] = A[
                                                    (batch * c + channel) * spatial + start + tail_vec + j]
                                    if need_cast:
                                        T.tile.cast(x32, src, "CAST_NONE", tile)
                                    else:
                                        T.copy(src, x32)
                                    T.reduce_sum(x32, row_sum, dim=-1)
                                    if tail == 0 or chunk < chunks - 1:
                                        T.tile.fill(tile_count, T.cast(float(tile), "float32"))
                                    else:
                                        T.tile.fill(tile_count, T.cast(float(tail), "float32"))
                                    T.tile.div(tile_mean, row_sum, tile_count)
                                    T.tile.broadcast(work, tile_mean)
                                    T.tile.sub(x32, x32, work)
                                    if tail != 0 and chunk == chunks - 1:
                                        if zpad_start < tile:
                                            T.tile.fill(x32[0, zpad_start:tile], 0.0)
                                        for j in T.serial(8):
                                            if tail + j < zpad_start:
                                                x32[0, tail + j] = 0.0
                                    T.tile.mul(work, x32, x32)
                                    T.reduce_sum(work, row_m2, dim=-1)

                                    T.tile.sub(delta, tile_mean, mean)
                                    T.tile.add(new_count, count, tile_count)
                                    T.tile.div(ratio, tile_count, new_count)
                                    T.tile.mul(correction, delta, ratio)
                                    T.tile.add(mean, mean, correction)
                                    T.tile.mul(correction, delta, delta)
                                    T.tile.mul(ratio, count, tile_count)
                                    T.tile.div(ratio, ratio, new_count)
                                    T.tile.mul(correction, correction, ratio)
                                    T.tile.add(row_m2, row_m2, correction)
                                    T.tile.add(m2, m2, row_m2)
                                    T.tile.add(count, count, tile_count)

                            T.tile.div(denom, m2, count)
                            T.tile.add(denom, denom, T.cast(eps, "float32"))
                            T.tile.rsqrt(inv, denom)
                            T.tile.mul(refine, inv, inv)
                            T.tile.mul(refine, refine, denom)
                            T.tile.mul(refine, refine, 0.5)
                            T.tile.fill(denom, 1.5)
                            T.tile.sub(denom, denom, refine)
                            T.tile.mul(inv, inv, denom)

                            # Second pass writes fixed-width vector tiles.  The
                            # final partial tile is copied only over its valid
                            # prefix, so no scalar GM store is emitted.
                            for ci in T.serial(channels_per_group):
                                channel = cbase + ci
                                # --- Hoisted affine load (R200 r2) -----------------
                                # W[channel] / Bias[channel] are loop-invariant in
                                # `chunk`, but the load sat INSIDE the chunk loop, so
                                # every chunk paid two 1-element GM reads plus their
                                # casts.  Those are latency, not bandwidth, and they
                                # serialise.  Measured cost, same shape, affine vs
                                # non-affine (R200 r2 per-case table):
                                #     GroupNorm  image-g32        0.841 -> 0.653
                                #     InstanceNorm wider-channel  1.078 -> 0.642
                                # i.e. the affine path was the whole remaining gap.
                                # Hoisting a loop-invariant load is bit-exact.
                                if affine:
                                    T.copy(W[channel], affine_raw)
                                    if need_cast:
                                        T.tile.cast(w32v, affine_raw, "CAST_NONE", 1)
                                    else:
                                        T.copy(affine_raw, w32v)
                                    T.copy(Bias[channel], affine_raw)
                                    if need_cast:
                                        T.tile.cast(b32v, affine_raw, "CAST_NONE", 1)
                                    else:
                                        T.copy(affine_raw, b32v)
                                for chunk in T.serial(chunks):
                                    start = chunk * tile
                                    T.tile.fill(src, 0.0)
                                    if tail == 0 or chunk < chunks - 1:
                                        T.copy(A[(batch * c + channel) * spatial + start], src[0, :])
                                    else:
                                        if tail_vec:
                                            T.copy(A[(batch * c + channel) * spatial + start:
                                                     (batch * c + channel) * spatial + start + tail_vec],
                                                   src[0, 0:tail_vec])
                                        for j in T.serial(blk):
                                            if tail_vec + j < tail:
                                                src[0, tail_vec + j] = A[
                                                    (batch * c + channel) * spatial + start + tail_vec + j]
                                    if need_cast:
                                        T.tile.cast(x32, src, "CAST_NONE", tile)
                                    else:
                                        T.copy(src, x32)
                                    T.tile.broadcast(work, mean)
                                    T.tile.sub(x32, x32, work)
                                    T.tile.broadcast(work, inv)
                                    T.tile.mul(x32, x32, work)
                                    if affine:
                                        T.tile.broadcast(work, w32v)
                                        T.tile.mul(x32, x32, work)
                                        T.tile.broadcast(work, b32v)
                                        T.tile.add(x32, x32, work)
                                    if need_cast:
                                        T.tile.cast(out, x32, "CAST_RINT", tile)
                                    else:
                                        T.copy(x32, out)
                                    if tail == 0 or chunk < chunks - 1:
                                        T.copy(out[0, :], B[(batch * c + channel) * chunks * tile + start])
                                    else:
                                        T.copy(out[0, 0:tail], B[(batch * c + channel) * chunks * tile + start : (batch * c + channel) * chunks * tile + start + tail])

        return main

    return factory()


# ---------------------------------------------------------------------------
# R335: the resident row-group path for GroupNorm / InstanceNorm(affine)
# ---------------------------------------------------------------------------
#
# ``_compile_group`` above is UNCHANGED and still serves every shape this path
# declines.  What follows replaces it when the shape allows, and it exists
# because R335 measured where ``_compile_group``'s time actually goes.
#
# Double-sided PipeUtilization, GroupNormFwdOp / image-g32-affine / fp16, ABBA
# 30+30, counter perturbation -0.02% / +0.05%
# (docs/reports/R335-data/pipe/gn-image-g32-affine-fp16.json):
#
#              kernel interval   aiv_time   VEC      MTE2     MTE3    Scalar
#   ours        51.571 us        52.33      19.134   20.105   3.471   7.081
#   aclnnGroupNorm 21.661 us     18.07       3.444    3.979   1.450   4.626
#
# i.e. the AIV is busy for the whole kernel (no launch-overhead story), and we
# are 5.6x the opponent on VEC **and** 5.1x on MTE2 at the same time.  The three
# things ``_compile_group`` does that this path does not:
#
#   1. it walks ONE ``(1, tile)`` row at a time, so every ``T.tile.*`` carries
#      its fixed cost for a single channel slice and every GM->UB transfer is
#      one ``spatial``-sized read (MTE2);
#   2. it reads ``A`` from GM a SECOND time in pass 2 and re-casts it (MTE2 and
#      VEC again), because nothing is kept between the passes;
#   3. it runs the full ~13-op Welford combine per (channel, chunk) even though
#      every production shape here has ``chunks == 1``, where the combine is the
#      identity.
#
# This path keeps the whole task resident in fp32 (one GM read), makes the UB
# tile 2-D so one ``T.tile.*`` covers every row of the task, and replaces
# Welford with the textbook two-pass mean/variance over the resident rows --
# the same recipe R261 used for RMSNorm and R274 for the fused residual-add
# norms, which is where the other members of this family got their speed.
#
# ROW ORDER.  The kernel views the input as ``(n * c, spatial)``: row
# ``b * c + ch``.  A task owns ``rows`` CONSECUTIVE rows of that matrix, so a
# single 2-D ``T.copy`` moves the whole task when ``tile == spatial``.
#   * ``cpg > 1`` (GroupNorm): ``rows = cpg``, one group per task.  The group
#     statistic is the sum over all ``cpg`` rows, so it is accumulated into a
#     ``(1, tile)`` scratch row and reduced once.
#   * ``cpg == 1`` (InstanceNorm with affine, and GroupNorm with C == groups):
#     each row IS a group, so ``T.reduce_sum`` already produces one statistic
#     per row and a task can own several rows.  ``rows`` is raised only while at
#     least ``LAUNCH_BLOCK_CAP * _VEC`` tasks survive, so batching never buys
#     instruction sharing by leaving vector cores idle.


#: Statistics-buffer width, and therefore the cap on ``rows`` for the
#: ``cpg == 1`` branch: an Ascend vector instruction addresses whole 32-byte
#: blocks, so a ``(rows,)`` fp32 buffer is padded to this and only the
#: ``T.reduce_sum`` destination is narrowed to ``[0:rows]``.  Same number and
#: same reason as ``kernels.common.ROW_EXTENT_GRAIN`` / ``_RES_STAT_LANES``.
_GRP_STAT_LANES = 8

#: Hard cap on ``rows`` for the ``cpg > 1`` branch.  ``rows = cpg`` there and the
#: cross-channel accumulation is an unrolled add chain of that length, so a
#: pathological ``channels_per_group`` would trade one fixed cost for another.
#: Above it the planner declines and ``_compile_group`` runs unchanged.
_GRP_MAX_ROWS = 16


def _plan_group_resident(n: int, c: int, spatial: int, groups: int,
                         channels_per_group: int, dtype_name: str):
    """Geometry for the resident spatial-norm path, or ``None`` if it declines.

    ``None`` is a normal answer, not a failure: the caller then compiles
    ``_compile_group``, which has no residency requirement.
    """
    itemsize = 2 if dtype_name in ("float16", "bfloat16") else 4
    m = int(n) * int(c)
    spatial = int(spatial)
    cpg = int(channels_per_group)
    if m <= 0 or spatial <= 0 or cpg <= 0:
        return None
    grain = SPATIAL_ELEM_GRAIN
    tile = spatial if spatial % grain == 0 else ((spatial + grain - 1) // grain) * grain
    exact = tile == spatial
    if cpg > 1:
        if cpg > _GRP_MAX_ROWS:
            return None
        rows = cpg
    else:
        rows = 1
        while (rows * 2 <= _GRP_STAT_LANES and m % (rows * 2) == 0
               and m // (rows * 2) >= LAUNCH_BLOCK_CAP * _VEC):
            rows *= 2
    if m % rows:
        return None
    # UB census, read off the allocation list in ``_compile_group_resident``.
    # One wrong entry is a silent UB overflow (R194 2.2), so it is spelled out:
    #   (rows, tile) : src[itemsize] + hold[4] + accb[4]
    #   (1, tile)    : acc[4], allocated for both branches (TVMScript parses the
    #                  whole body, so the buffer cannot appear conditionally) but
    #                  only USED when cpg > 1; it shrinks to one grain otherwise.
    #   fixed        : 8 statistics buffers of _GRP_STAT_LANES fp32
    #                  + 4 one-element affine buffers
    per_cell = itemsize + 8
    acc_cells = tile if cpg > 1 else grain
    ub = (rows * tile * per_cell + acc_cells * 4 + 8 * _GRP_STAT_LANES * 4
          + 2 * 32 * itemsize + 2 * 32 * 4)
    if ub > UB_BUDGET_NORM_BYTES:
        return None
    tasks = m // rows
    launch_blocks, grid_repeats = _launch_tasks(tasks)
    return {
        "rows": rows,
        "tile": tile,
        "exact": exact,
        # A task owns rows ``row0 .. row0+rows-1`` of the (n*c, spatial) view and
        # channel ``row % c``.  When ``c % rows == 0`` those are ``rows``
        # CONSECUTIVE channels with no wrap, so the whole task's affine pair can
        # be fetched with two vector loads instead of ``2 * rows`` one-element
        # GM reads sitting on the dependency chain.
        "vector_affine": (int(c) % rows) == 0,
        "tasks": tasks,
        "launch_blocks": launch_blocks,
        "grid_repeats": grid_repeats,
        "ub_bytes": ub,
    }


@lru_cache(maxsize=128)
def _compile_group_resident(
    n: int,
    c: int,
    spatial: int,
    groups: int,
    channels_per_group: int,
    dtype_name: str,
    affine: bool,
    eps: float,
):
    """GroupNorm / InstanceNorm with the task's rows resident in UB, or ``None``."""
    plan = _plan_group_resident(n, c, spatial, groups, channels_per_group, dtype_name)
    if plan is None:
        return None
    rows = plan["rows"]
    tile = plan["tile"]
    exact = plan["exact"]
    tasks = plan["tasks"]
    vector_affine = plan["vector_affine"]
    launch_blocks = plan["launch_blocks"]
    grid_repeats = plan["grid_repeats"]
    cpg = int(channels_per_group)
    m = int(n) * int(c)
    grain = SPATIAL_ELEM_GRAIN
    need_cast = dtype_name != "float32"
    inv_count = 1.0 / float(cpg * spatial)
    # Where the fp32 padding cleanup may start when ``tile > spatial``: an
    # 8-aligned span that ``T.tile.fill`` can cover, plus at most 8 scalar lanes
    # below it.  Same constant, same reason as ``_compile_group``.
    zpad_start = min(tile, ((spatial + 7) // 8) * 8)
    acc_cells = tile if cpg > 1 else grain
    # R341 (G2): how many UB cells of the statistic's domain are PAD rather than data.
    # A task's variance reduction runs over ``rows * tile`` cells when cpg > 1 (one group
    # spans every row) and over ``tile`` cells per row when cpg == 1 (each row is its own
    # group), so the pad count per reduction differs by that factor.  Zero when the tile
    # is exact, which is what makes G2 a compile-time no-op on those shapes.
    pad_cells = (tile - spatial) * (rows if cpg > 1 else 1)

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((m, spatial), dtype_name),
            B: T.Tensor((m, tile), dtype_name),
            W: T.Tensor((c,), dtype_name),
            Bias: T.Tensor((c,), dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((rows, tile), dtype_name)
                hold = T.alloc_ub((rows, tile), "float32")
                accb = T.alloc_ub((rows, tile), "float32")
                acc = T.alloc_ub((1, acc_cells), "float32")
                tot = T.alloc_ub((_GRP_STAT_LANES,), "float32")
                mean = T.alloc_ub((_GRP_STAT_LANES,), "float32")
                den = T.alloc_ub((_GRP_STAT_LANES,), "float32")
                inv = T.alloc_ub((_GRP_STAT_LANES,), "float32")
                ref = T.alloc_ub((_GRP_STAT_LANES,), "float32")
                spare = T.alloc_ub((_GRP_STAT_LANES,), "float32")
                # ⚠️ 32 LANES, not ``rows``.  A GM->UB ``T.copy`` moves whole
                # 32-byte blocks (RAGGED-CHUNK LOADS, TRAP 1 above), so a
                # ``rows``-element load into a ``rows``-element buffer would
                # write past it -- into whatever UB buffer follows.  32 lanes is
                # 64 bytes at fp16 and 128 at fp32, so the rounded-up write
                # always lands inside.  The matching GM over-READ is bounded by
                # 32 bytes past ``W[chan_base]`` and is strictly smaller than the
                # one ``_compile_group`` already performs today with
                # ``T.copy(W[channel], affine_raw)`` on a (1,) destination.
                affine_raw = T.alloc_ub((32,), dtype_name)
                bias_raw = T.alloc_ub((32,), dtype_name)
                w32v = T.alloc_ub((32,), "float32")
                b32v = T.alloc_ub((32,), "float32")

                with T.Scope("V"):
                    for repeat in T.serial(grid_repeats):
                        block = cid + repeat * launch_blocks
                        task = block * _VEC + vid
                        if task < tasks:
                            row0 = task * rows
                            # --- pass 1: one GM read, kept in fp32 ------------
                            T.tile.fill(tot, 0.0)
                            if affine and vector_affine:
                                # Issued BEFORE pass 1 so MTE2 has the whole
                                # reduction to hide the GM latency behind.  In
                                # the first R335 kernel these were 2*rows
                                # one-element loads sitting between two
                                # dependent vector ops in the epilogue, and they
                                # cost +24.75 us on GroupNorm/image-g32/fp16
                                # (19.750 us non-affine vs 44.500 us affine,
                                # docs/reports/R335-data/15-groupnorm-ab.txt).
                                T.copy(W[row0 % c], affine_raw[0:rows])
                                T.copy(Bias[row0 % c], bias_raw[0:rows])
                                if need_cast:
                                    T.tile.cast(w32v, affine_raw, "CAST_NONE", 32)
                                    T.tile.cast(b32v, bias_raw, "CAST_NONE", 32)
                                else:
                                    T.copy(affine_raw, w32v)
                                    T.copy(bias_raw, b32v)
                            # ONE 2-D GM->UB transfer for the whole task, ragged
                            # ``spatial`` included.  The shipped ``_compile_group``
                            # cannot do this because its tile is one row wide; the
                            # RAGGED-CHUNK LOADS block above therefore has it walk
                            # a vector prefix plus a guarded scalar loop PER ROW,
                            # and R335 measured what that costs: tail-spatial-g16
                            # (spatial = 900) 27.375 us against image-g32's
                            # 19.750 us on 2.3x MORE data.
                            #
                            # When ``tile > spatial`` the columns [spatial, tile)
                            # of each UB row are NOT this row's data -- they are
                            # either the next row's head (if the lowering does not
                            # clamp the column extent) or stale UB.  Either way
                            # they are zeroed here, BEFORE the sum, and again after
                            # centring turns the zeros into -mean.  Zeroing is over
                            # an 8-aligned span plus at most 8 scalar lanes, the
                            # same shape ``_compile_group`` uses.
                            T.copy(A[row0, 0], src)
                            if need_cast:
                                T.tile.cast(hold, src, "CAST_NONE", rows * tile)
                            else:
                                T.copy(src, hold)
                            if not exact:
                                for r in T.serial(rows):
                                    if zpad_start < tile:
                                        T.tile.fill(hold[r, zpad_start:tile], 0.0)
                                    for j in T.serial(8):
                                        if spatial + j < zpad_start:
                                            hold[r, spatial + j] = 0.0
                            if cpg > 1:
                                # One group per task: the statistic spans every
                                # row, so fold the rows together first and reduce
                                # once.  ``acc`` is (1, tile) here.
                                T.tile.fill(acc, 0.0)
                                for r in T.serial(rows):
                                    T.tile.add(acc[0, :], acc[0, :], hold[r, :])
                                T.reduce_sum(acc, tot[0:1], dim=-1)
                            else:
                                # Each row is its own group: ``T.reduce_sum``
                                # already produces one statistic per row.
                                T.reduce_sum(hold, tot[0:rows], dim=-1)
                            T.tile.mul(mean, tot, T.cast(inv_count, "float32"))
                            # --- centre in place, then sum the squares --------
                            for r in T.serial(rows):
                                T.tile.sub(hold[r, :], hold[r, :],
                                           mean[0] if cpg > 1 else mean[r])
                            # R341 (G2): the second cleanup used to live in the loop above
                            # and re-zeroed the pad because centring had just turned this
                            # task's zeros into ``-mean``.  It is gone.  After the FIRST
                            # cleanup the pad is exactly 0.0, so after centring it is
                            # exactly ``-mean``, so its contribution to the sum of squared
                            # deviations is exactly ``pad_cells * mean**2`` -- an algebraic
                            # identity, not an approximation, and it is subtracted below
                            # with three 8-lane ops instead of ``rows`` fills plus
                            # ``8 * rows`` predicated scalar stores.
                            #
                            # Measured (docs/reports/R341-data/14-copyfloor-decomposition.txt,
                            # stages s4_folded vs s4_folded_padfix): tail-spatial-g16-affine
                            # 13.500 -> 11.125 us; the three exact-tile cases move by
                            # EXACTLY 0.000 us, as a pad-only change must.
                            T.tile.mul(accb, hold, hold)
                            if cpg > 1:
                                T.tile.fill(acc, 0.0)
                                for r in T.serial(rows):
                                    T.tile.add(acc[0, :], acc[0, :], accb[r, :])
                                T.reduce_sum(acc, tot[0:1], dim=-1)
                            else:
                                T.reduce_sum(accb, tot[0:rows], dim=-1)
                            if pad_cells:
                                # R341 (G2): remove the pad's exact contribution,
                                # ``pad_cells`` copies of ``(-mean)**2``.  ``spare`` is
                                # free here -- the Newton step below fills it before use.
                                T.tile.mul(spare, mean, mean)
                                T.tile.mul(spare, spare,
                                           T.cast(float(pad_cells), "float32"))
                                T.tile.sub(tot, tot, spare)
                            T.tile.mul(den, tot, T.cast(inv_count, "float32"))
                            T.tile.add(den, den, T.cast(eps, "float32"))
                            T.tile.rsqrt(inv, den)
                            # Ascend's vector rsqrt is intentionally fast and
                            # approximate.  One Newton step restores fp32
                            # accuracy before the final dtype cast -- same step,
                            # same order as ``_compile_group``.
                            T.tile.mul(ref, inv, inv)
                            T.tile.mul(ref, ref, den)
                            T.tile.mul(ref, ref, 0.5)
                            T.tile.fill(spare, 1.5)
                            T.tile.sub(spare, spare, ref)
                            T.tile.mul(inv, inv, spare)
                            # R341 (G1): R335 section 8.1's named next step ("replace the
                            # per-row scalar multiplies with fewer, bigger instructions;
                            # NOT DONE, no predicted number given").  ``inv`` is one value
                            # per GROUP (index 0 when cpg > 1) and the weight is one value
                            # per ROW, so ``inv * w`` is a SINGLE 32-lane multiply done
                            # once per task.  The epilogue then does two whole-row ops per
                            # row instead of three: ``rows`` fewer ``T.tile.*`` calls and
                            # ``rows * tile`` fewer element-ops.  R261's lever 2 recorded
                            # that these templates pay for ``T.tile.*`` CALL COUNT.
                            #
                            # ⚠️ Only the ``cpg > 1`` branch (GroupNorm).  When cpg == 1
                            # (InstanceNorm's affine cases) ``inv`` is per row in an
                            # 8-lane buffer while the weight is in a 32-lane one, so the
                            # fold is not a single op there; that branch ships unchanged.
                            #
                            # ⚠️ Numerics change, and it is not a relaxation:
                            # ``(x-mean) * inv * w`` becomes ``(x-mean) * (inv*w)``.  Both
                            # fp32, different last-ulp rounding.  Gated by
                            # docs/reports/R341-data/p_correctness.py over all 8 operators
                            # this builder serves, at the unchanged tolerance.
                            #
                            # Measured: 14-copyfloor-decomposition.txt s4_full ->
                            # s4_folded, -0.750 / -0.625 / -0.750 / -0.500 us on the four
                            # affine cases.
                            if affine and vector_affine and cpg > 1:
                                T.tile.mul(w32v, w32v, inv[0])
                            # --- pass 2 over the RESIDENT rows ----------------
                            for r in T.serial(rows):
                                if not (affine and vector_affine and cpg > 1):
                                    T.tile.mul(hold[r, :], hold[r, :],
                                               inv[0] if cpg > 1 else inv[r])
                                if affine and vector_affine:
                                    T.tile.mul(hold[r, :], hold[r, :], w32v[r])
                                    T.tile.add(hold[r, :], hold[r, :], b32v[r])
                                elif affine:
                                    # ``c % rows != 0``: the task's rows wrap past
                                    # a batch boundary, so the channels are not
                                    # consecutive and the vector load above does
                                    # not apply.  Fall back to the shipped
                                    # kernel's one-element loads.
                                    T.copy(W[(row0 + r) % c], affine_raw[0:1])
                                    if need_cast:
                                        T.tile.cast(w32v, affine_raw, "CAST_NONE", 32)
                                    else:
                                        T.copy(affine_raw, w32v)
                                    T.tile.mul(hold[r, :], hold[r, :], w32v[0])
                                    T.copy(Bias[(row0 + r) % c], bias_raw[0:1])
                                    if need_cast:
                                        T.tile.cast(b32v, bias_raw, "CAST_NONE", 32)
                                    else:
                                        T.copy(bias_raw, b32v)
                                    T.tile.add(hold[r, :], hold[r, :], b32v[0])
                            if need_cast:
                                T.tile.cast(src, hold, "CAST_RINT", rows * tile)
                            else:
                                T.copy(hold, src)
                            T.copy(src, B[row0, 0])
        return main

    return factory()


@lru_cache(maxsize=128)
def _compile_instance_infer(
    n: int,
    c: int,
    spatial: int,
    dtype_name: str,
    affine: bool,
    eps: float,
):
    launch_blocks, grid_repeats = _launch_tasks(n * c)
    # --- Geometry (R200) ---------------------------------------------
    # (1, tile) UB census: src[itemsize] out[itemsize] x32[4] work[4]
    tile = _spatial_tile(spatial, dtype_name, cell_extra=8)
    blk = _blk(dtype_name)
    chunks = math.ceil(spatial / tile)
    tail = spatial % tile
    # 32-byte-aligned part of the ragged chunk, and where the fp32 padding
    # cleanup may start.  Both compile-time constants; see the RAGGED-CHUNK
    # LOADS block above -- in particular why it is inlined, not factored out.
    tail_vec = tail - (tail % blk)
    zpad_start = min(tile, ((tail + 7) // 8) * 8)

    need_cast = dtype_name != "float32"

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((n * c * spatial,), dtype_name),
            B: T.Tensor((n * c * chunks * tile,), dtype_name),
            RunningMean: T.Tensor((c,), "float32"),
            RunningVar: T.Tensor((c,), "float32"),
            W: T.Tensor((c,), dtype_name),
            Bias: T.Tensor((c,), dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((1, tile), dtype_name)
                out = T.alloc_ub((1, tile), dtype_name)
                x32 = T.alloc_ub((1, tile), "float32")
                work = T.alloc_ub((1, tile), "float32")
                mean = T.alloc_ub((1,), "float32")
                denom = T.alloc_ub((1,), "float32")
                inv = T.alloc_ub((1,), "float32")
                refine = T.alloc_ub((1,), "float32")
                affine_raw = T.alloc_ub((1,), dtype_name)
                affine32 = T.alloc_ub((1,), "float32")
                w32v = T.alloc_ub((1,), "float32")
                b32v = T.alloc_ub((1,), "float32")
                with T.Scope("V"):
                    for repeat in T.serial(grid_repeats):
                        block = cid + repeat * launch_blocks
                        task = block * _VEC + vid
                        if task < n * c:
                            batch = task // c
                            channel = task % c
                            T.copy(RunningMean[channel], mean)
                            T.copy(RunningVar[channel], denom)
                            T.tile.add(denom, denom, T.cast(eps, "float32"))
                            T.tile.rsqrt(inv, denom)
                            T.tile.mul(refine, inv, inv)
                            T.tile.mul(refine, refine, denom)
                            T.tile.mul(refine, refine, 0.5)
                            T.tile.fill(denom, 1.5)
                            T.tile.sub(denom, denom, refine)
                            T.tile.mul(inv, inv, denom)
                            # --- Hoisted affine load (R200 r2) -----------------
                            # W[channel] / Bias[channel] are loop-invariant in
                            # `chunk`, but the load sat INSIDE the chunk loop, so
                            # every chunk paid two 1-element GM reads plus their
                            # casts.  Those are latency, not bandwidth, and they
                            # serialise.  Measured cost, same shape, affine vs
                            # non-affine (R200 r2 per-case table):
                            #     GroupNorm  image-g32        0.841 -> 0.653
                            #     InstanceNorm wider-channel  1.078 -> 0.642
                            # i.e. the affine path was the whole remaining gap.
                            # Hoisting a loop-invariant load is bit-exact.
                            if affine:
                                T.copy(W[channel], affine_raw)
                                if need_cast:
                                    T.tile.cast(w32v, affine_raw, "CAST_NONE", 1)
                                else:
                                    T.copy(affine_raw, w32v)
                                T.copy(Bias[channel], affine_raw)
                                if need_cast:
                                    T.tile.cast(b32v, affine_raw, "CAST_NONE", 1)
                                else:
                                    T.copy(affine_raw, b32v)
                            for chunk in T.serial(chunks):
                                start = chunk * tile
                                T.tile.fill(src, 0.0)
                                if tail == 0 or chunk < chunks - 1:
                                    T.copy(A[(batch * c + channel) * spatial + start], src[0, :])
                                else:
                                    if tail_vec:
                                        T.copy(A[(batch * c + channel) * spatial + start:
                                                 (batch * c + channel) * spatial + start + tail_vec],
                                               src[0, 0:tail_vec])
                                    for j in T.serial(blk):
                                        if tail_vec + j < tail:
                                            src[0, tail_vec + j] = A[
                                                (batch * c + channel) * spatial + start + tail_vec + j]
                                if need_cast:
                                    T.tile.cast(x32, src, "CAST_NONE", tile)
                                else:
                                    T.copy(src, x32)
                                T.tile.broadcast(work, mean)
                                T.tile.sub(x32, x32, work)
                                T.tile.broadcast(work, inv)
                                T.tile.mul(x32, x32, work)
                                if affine:
                                    T.tile.broadcast(work, w32v)
                                    T.tile.mul(x32, x32, work)
                                    T.tile.broadcast(work, b32v)
                                    T.tile.add(x32, x32, work)
                                if need_cast:
                                    T.tile.cast(out, x32, "CAST_RINT", tile)
                                else:
                                    T.copy(x32, out)
                                if tail == 0 or chunk < chunks - 1:
                                    T.copy(out[0, :], B[(batch * c + channel) * chunks * tile + start])
                                else:
                                    T.copy(out[0, 0:tail], B[(batch * c + channel) * chunks * tile + start : (batch * c + channel) * chunks * tile + start + tail])

        return main

    return factory()


@lru_cache(maxsize=128)
def _compile_batch(
    n: int,
    c: int,
    spatial: int,
    dtype_name: str,
    training: bool,
    eps: float,
    momentum: float,
):
    launch_blocks, grid_repeats = _launch_tasks(c)
    # --- Geometry (R200) ---------------------------------------------
    # (1, tile) UB census: src[itemsize] out[itemsize] x32[4] work[4]
    tile = _spatial_tile(spatial, dtype_name, cell_extra=8)
    blk = _blk(dtype_name)
    chunks = math.ceil(spatial / tile)
    tail = spatial % tile
    # 32-byte-aligned part of the ragged chunk, and where the fp32 padding
    # cleanup may start.  Both compile-time constants; see the RAGGED-CHUNK
    # LOADS block above -- in particular why it is inlined, not factored out.
    tail_vec = tail - (tail % blk)
    zpad_start = min(tile, ((tail + 7) // 8) * 8)

    need_cast = dtype_name != "float32"

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((n * c * spatial,), dtype_name),
            B: T.Tensor((n * c * chunks * tile,), dtype_name),
            RunningMean: T.Tensor((c,), "float32"),
            RunningVar: T.Tensor((c,), "float32"),
            W: T.Tensor((c,), "float32"),
            Bias: T.Tensor((c,), "float32"),
            BatchMean: T.Tensor((c,), "float32"),
            BatchRstd: T.Tensor((c,), "float32"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((1, tile), dtype_name)
                out = T.alloc_ub((1, tile), dtype_name)
                x32 = T.alloc_ub((1, tile), "float32")
                work = T.alloc_ub((1, tile), "float32")
                count = T.alloc_ub((1,), "float32")
                mean = T.alloc_ub((1,), "float32")
                m2 = T.alloc_ub((1,), "float32")
                row_sum = T.alloc_ub((1,), "float32")
                row_m2 = T.alloc_ub((1,), "float32")
                tile_mean = T.alloc_ub((1,), "float32")
                tile_count = T.alloc_ub((1,), "float32")
                delta = T.alloc_ub((1,), "float32")
                ratio = T.alloc_ub((1,), "float32")
                new_count = T.alloc_ub((1,), "float32")
                correction = T.alloc_ub((1,), "float32")
                denom = T.alloc_ub((1,), "float32")
                inv = T.alloc_ub((1,), "float32")
                refine = T.alloc_ub((1,), "float32")
                old_stat = T.alloc_ub((1,), "float32")
                update = T.alloc_ub((1,), "float32")
                affine = T.alloc_ub((1,), "float32")
                w32v = T.alloc_ub((1,), "float32")
                b32v = T.alloc_ub((1,), "float32")
                with T.Scope("V"):
                    for repeat in T.serial(grid_repeats):
                        block = cid + repeat * launch_blocks
                        channel = block * _VEC + vid
                        if channel < c:
                            if training:
                                T.tile.fill(count, 0.0)
                                T.tile.fill(mean, 0.0)
                                T.tile.fill(m2, 0.0)
                                for batch in T.serial(n):
                                    for chunk in T.serial(chunks):
                                        start = chunk * tile
                                        T.tile.fill(src, 0.0)
                                        if tail == 0 or chunk < chunks - 1:
                                            T.copy(A[(batch * c + channel) * spatial + start], src[0, :])
                                        else:
                                            if tail_vec:
                                                T.copy(A[(batch * c + channel) * spatial + start:
                                                         (batch * c + channel) * spatial + start + tail_vec],
                                                       src[0, 0:tail_vec])
                                            for j in T.serial(blk):
                                                if tail_vec + j < tail:
                                                    src[0, tail_vec + j] = A[
                                                        (batch * c + channel) * spatial + start + tail_vec + j]
                                        if need_cast:
                                            T.tile.cast(x32, src, "CAST_NONE", tile)
                                        else:
                                            T.copy(src, x32)
                                        T.reduce_sum(x32, row_sum, dim=-1)
                                        if tail == 0 or chunk < chunks - 1:
                                            T.tile.fill(tile_count, T.cast(float(tile), "float32"))
                                        else:
                                            T.tile.fill(tile_count, T.cast(float(tail), "float32"))
                                        T.tile.div(tile_mean, row_sum, tile_count)
                                        T.tile.broadcast(work, tile_mean)
                                        T.tile.sub(x32, x32, work)
                                        if tail != 0 and chunk == chunks - 1:
                                            if zpad_start < tile:
                                                T.tile.fill(x32[0, zpad_start:tile], 0.0)
                                            for j in T.serial(8):
                                                if tail + j < zpad_start:
                                                    x32[0, tail + j] = 0.0
                                        T.tile.mul(work, x32, x32)
                                        T.reduce_sum(work, row_m2, dim=-1)

                                        T.tile.sub(delta, tile_mean, mean)
                                        T.tile.add(new_count, count, tile_count)
                                        T.tile.div(ratio, tile_count, new_count)
                                        T.tile.mul(correction, delta, ratio)
                                        T.tile.add(mean, mean, correction)
                                        T.tile.mul(correction, delta, delta)
                                        T.tile.mul(ratio, count, tile_count)
                                        T.tile.div(ratio, ratio, new_count)
                                        T.tile.mul(correction, correction, ratio)
                                        T.tile.add(row_m2, row_m2, correction)
                                        T.tile.add(m2, m2, row_m2)
                                        T.tile.add(count, count, tile_count)

                                # BatchNorm normalizes with biased batch variance;
                                # running_var tracks the unbiased estimate.
                                T.copy(RunningMean[channel], old_stat)
                                T.tile.mul(old_stat, old_stat, 1.0 - momentum)
                                T.tile.mul(update, mean, momentum)
                                T.tile.add(update, old_stat, update)
                                T.copy(update, RunningMean[channel])
                                T.tile.add(denom, count, -1.0)
                                T.tile.div(update, m2, denom)
                                T.copy(RunningVar[channel], old_stat)
                                T.tile.mul(old_stat, old_stat, 1.0 - momentum)
                                T.tile.mul(update, update, momentum)
                                T.tile.add(update, old_stat, update)
                                T.copy(update, RunningVar[channel])
                                T.tile.div(denom, m2, count)
                            else:
                                T.copy(RunningMean[channel], mean)
                                T.copy(RunningVar[channel], denom)
                            T.tile.add(denom, denom, T.cast(eps, "float32"))
                            T.tile.rsqrt(inv, denom)
                            T.tile.mul(refine, inv, inv)
                            T.tile.mul(refine, refine, denom)
                            T.tile.mul(refine, refine, 0.5)
                            T.tile.fill(denom, 1.5)
                            T.tile.sub(denom, denom, refine)
                            T.tile.mul(inv, inv, denom)
                            if training:
                                T.copy(mean, BatchMean[channel])
                                T.copy(inv, BatchRstd[channel])
                            # --- Hoisted affine load (R200 r2) -----------------
                            # W[channel] / Bias[channel] are loop-invariant in
                            # `chunk`, but the load sat INSIDE the chunk loop, so
                            # every chunk paid two 1-element GM reads plus their
                            # casts.  Those are latency, not bandwidth, and they
                            # serialise.  Measured cost, same shape, affine vs
                            # non-affine (R200 r2 per-case table):
                            #     GroupNorm  image-g32        0.841 -> 0.653
                            #     InstanceNorm wider-channel  1.078 -> 0.642
                            # i.e. the affine path was the whole remaining gap.
                            # Hoisting a loop-invariant load is bit-exact.
                            T.copy(W[channel], w32v)
                            T.copy(Bias[channel], b32v)
                            for batch in T.serial(n):
                                for chunk in T.serial(chunks):
                                    start = chunk * tile
                                    T.tile.fill(src, 0.0)
                                    if tail == 0 or chunk < chunks - 1:
                                        T.copy(A[(batch * c + channel) * spatial + start], src[0, :])
                                    else:
                                        if tail_vec:
                                            T.copy(A[(batch * c + channel) * spatial + start:
                                                     (batch * c + channel) * spatial + start + tail_vec],
                                                   src[0, 0:tail_vec])
                                        for j in T.serial(blk):
                                            if tail_vec + j < tail:
                                                src[0, tail_vec + j] = A[
                                                    (batch * c + channel) * spatial + start + tail_vec + j]
                                    if need_cast:
                                        T.tile.cast(x32, src, "CAST_NONE", tile)
                                    else:
                                        T.copy(src, x32)
                                    T.tile.broadcast(work, mean)
                                    T.tile.sub(x32, x32, work)
                                    T.tile.broadcast(work, inv)
                                    T.tile.mul(x32, x32, work)
                                    T.tile.broadcast(work, w32v)
                                    T.tile.mul(x32, x32, work)
                                    T.tile.broadcast(work, b32v)
                                    T.tile.add(x32, x32, work)
                                    if need_cast:
                                        T.tile.cast(out, x32, "CAST_RINT", tile)
                                    else:
                                        T.copy(x32, out)
                                    if tail == 0 or chunk < chunks - 1:
                                        T.copy(out[0, :], B[(batch * c + channel) * chunks * tile + start])
                                    else:
                                        T.copy(out[0, 0:tail], B[(batch * c + channel) * chunks * tile + start : (batch * c + channel) * chunks * tile + start + tail])

        return main

    return factory()


# ---------------------------------------------------------------------------
# R274: the resident-row fused residual-add + normalization path
# ---------------------------------------------------------------------------
#
# ``_compile_fused_add_norm`` below is unchanged and still serves every shape
# this path declines.  What follows is the R261 recipe
# (``docs/recipes/R261-norm.md``, ``families/two_pass.py::_compile_rms_resident``)
# applied to the FUSED op, which is the same row-wise two-pass shape with a
# residual add welded onto the front:
#
#   1. the ``(sub_m,)`` statistics buffers are padded to ``_RES_STAT_LANES``
#      independently of how many rows a lane owns, so ``sub_m`` may be 1.
#      ``kernels.common.ROW_EXTENT_GRAIN`` documents 8 as a HARD FLOOR and
#      ``plan_rowwise_norm`` obeys it, which is why an m = 1 decode row is
#      processed as an 8-row tile with 7 dead rows -- 8x the vector work.
#   2. the column tile is as WIDE as UB allows rather than capped at
#      ``TILE_WIDTH_CAP = 1280``.
#   3. sum(x^2) accumulates elementwise with ``T.tile.mul_add_dst`` and
#      ``T.reduce_sum`` runs ONCE per row group instead of once per n-tile.
#   4. the affine epilogue multiplies each row against the ``(block_n,)``
#      weight vector directly instead of broadcasting it into a
#      ``(sub_m, block_n)`` replica per n-tile.
#
# And -- the part that matters most for THIS op -- the added row group stays in
# UB as fp32 between the two passes, so ``x`` and ``residual`` are read from GM
# ONCE rather than twice and the add is performed once rather than twice.  The
# shipped template moves ``6 * M * N`` bytes where the manifest roofline says
# ``4 * M * N``; this path moves ``4 * M * N``.
#
# ⚠️ Semantics are preserved exactly, including the bf16 double rounding: the
# resident row holds ``fp32(dtype(x + residual))``, i.e. the value that was
# already written to ``residual_out``, not the unrounded fp32 sum.

#: Lanes every ``(...,)`` fp32 statistics buffer is padded to.  Same number and
#: same reason as ``kernels.common.ROW_EXTENT_GRAIN`` (an Ascend vector
#: instruction addresses whole 32-byte blocks), but here it is a buffer WIDTH,
#: not a floor on ``sub_m``.
_RES_STAT_LANES = 8

#: Column tiles per row this planner aims for, and the hard ceiling on it.
#: Both carried over from R261's measured sweep (R261-data/05-step4-ntiles.txt,
#: 06-step3-widths.txt): 2 beat 1 and 4 on all 9 RMSNormFwdOp cases, and beyond
#: 4 the resident fp32 row has squeezed the tile so narrow that the two-pass
#: kernel wins.  ⚠️ NOT re-derived for the fused op -- the fused row group costs
#: two more dtype-width tiles per row, so the width that fits is smaller at the
#: same n.  See R274 section "measured but not adopted" for the sweep that
#: checked this.
_RES_TARGET_N_TILES = 2
_RES_MAX_N_TILES = 4


def _plan_fused_add_resident(m: int, n: int, itemsize: int, kind: str):
    """Geometry for the resident-row fused-add path, or ``None`` if it declines.

    ``None`` is a normal answer, not a failure: the caller then compiles
    ``_compile_fused_add_norm``, which has no residency requirement.
    """
    m = max(1, int(m))
    n = max(1, int(n))
    if n % TILE_GRAIN:
        # The ragged-column path is exactly what the shipped kernel already
        # handles (its scalar n-tail double loop); duplicating it here would
        # double the surface without touching the lever.
        return None
    # UB census, read off the allocation list in ``factory`` below.  One wrong
    # entry is a silent UB overflow (R194 2.2), so it is spelled out:
    #   per row of the tile : src[itemsize] + rhs[itemsize] + accb[4]
    #                         + hold[4 * n]      (hold spans the WHOLE row)
    #                         + (acc_sum[4] for kind == "layer")
    #   fixed               : wub[itemsize] + w32[4]           (block_n,)
    #                         + bub[itemsize] + b32[4]         (layer only)
    #                         + _RES_STATS * _RES_STAT_LANES fp32
    #   per row of the tile : src[itemsize] + rhs[itemsize] + accb[4] + accs[4]
    #                         + hold[4 * n]      (hold spans the WHOLE row)
    #   fixed               : wub[itemsize] + w32[4]           (block_n,)
    #                         + bub[itemsize] + b32[4]         (layer only)
    #                         + 6 * _RES_STAT_LANES fp32
    # ⚠️ ``accs`` is allocated for BOTH kinds.  It is the layer sum accumulator,
    # but it is also the fp32 landing buffer for the residual under bf16, which
    # cannot alias ``accb``: ``accb`` is the sum-of-squares accumulator and a
    # cast into it silently seeds the reduction with one tile of residual.
    stats = 6
    fixed_rows = stats * _RES_STAT_LANES * 4
    per_col_fixed = (2 * itemsize + 8) if kind == "layer" else (itemsize + 4)
    per_col_row = 2 * itemsize + 8
    occupancy = max(1, m // (2 * LAUNCH_BLOCK_CAP))
    width = (n // _RES_TARGET_N_TILES) - (n // _RES_TARGET_N_TILES) % TILE_GRAIN
    while width >= TILE_GRAIN:
        n_tiles = n // width
        if n % width == 0 and n_tiles <= _RES_MAX_N_TILES:
            per_row = width * per_col_row + n * 4
            fixed = width * per_col_fixed + fixed_rows
            room = (UB_BUDGET_NORM_BYTES - fixed) // per_row if per_row > 0 else 0
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
def _compile_fused_add_resident(
    m: int,
    n: int,
    dtype_name: str,
    kind: str,
    eps: float,
    reread: bool = False,
):
    """Fused residual-add + row norm with the row group resident in UB.

    Returns ``None`` when the shape does not admit the residency requirement;
    the caller then falls back to ``_compile_fused_add_norm``.

    ``reread=True`` is the ABLATION, not a shipping path: identical geometry,
    identical statistics, but the second pass reloads ``A`` and ``Residual``
    from GM and redoes the add instead of reading the resident row.  It exists
    so that R274 can attribute the delta between "new geometry" and "one GM
    read instead of two" separately (T273 methodology requirement: every change
    measured on its own).  Selected by ``TILEOPS_FUSED_ADD_RESIDENT=reread``.
    """
    if kind not in {"rms", "layer"}:
        raise ValueError(f"unsupported fused normalization kind: {kind}")
    if reread and kind != "rms":
        # The layer body needs the resident row twice more (mean, then
        # variance); there is nothing to ablate against.
        return None
    itemsize = 2 if dtype_name in ("float16", "bfloat16") else 4
    plan = _plan_fused_add_resident(m, n, itemsize, kind)
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
    # R263 hardware fact 1: ``T.tile.add`` has no bf16 UB instantiation on
    # dav-2201, so bf16 must round-trip through fp32.  fp16 adds in place, which
    # is also what the shipped kernel does -- keeping both paths identical to it
    # keeps the numerics identical to it.
    add_via_fp32 = dtype_name == "bfloat16"

    @tilelang.jit(
        out_idx=[4, 5],
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
            Residual: T.Tensor([m, n], dtype_name),
            W: T.Tensor([n], dtype_name),
            Bias: T.Tensor([n], dtype_name),
            Y: T.Tensor([m_pad, n], dtype_name),
            Add: T.Tensor([m_pad, n], dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                # The whole added row group, in fp32, for the lifetime of both
                # passes.  This is the read that the shipped kernel repeats.
                hold = T.alloc_ub(
                    (1, 1, TILE_GRAIN) if reread else (n_tiles, sub_m, block_n),
                    "float32",
                )
                # Under the ablation the fp32 working tile is a single n-tile
                # rather than the whole row; ``keep`` stands in for
                # ``hold[nt]`` so the two bodies differ in exactly one thing.
                keep = T.alloc_ub(
                    (sub_m, block_n) if reread else (1, TILE_GRAIN), "float32"
                )
                src = T.alloc_ub((sub_m, block_n), dtype_name)
                rhs = T.alloc_ub((sub_m, block_n), dtype_name)
                accb = T.alloc_ub((sub_m, block_n), "float32")
                # ⚠️ Every ``T.alloc_ub`` must be an UNCONDITIONAL statement:
                # TVMScript parses the whole body, so a name bound inside a
                # Python ``if`` is undefined at the use site ("Undefined
                # variable: accs").  The kind-specific buffers therefore shrink
                # to a one-block dummy instead of disappearing -- which is also
                # what ``_plan_fused_add_resident``'s census assumes.
                accs = T.alloc_ub((sub_m, block_n), "float32")
                wub = T.alloc_ub((block_n,), dtype_name)
                w32 = T.alloc_ub((block_n,), "float32")
                bub = T.alloc_ub(
                    (block_n,) if kind == "layer" else (TILE_GRAIN,), dtype_name
                )
                b32 = T.alloc_ub(
                    (block_n,) if kind == "layer" else (TILE_GRAIN,), "float32"
                )
                tot = T.alloc_ub((_RES_STAT_LANES,), "float32")
                den = T.alloc_ub((_RES_STAT_LANES,), "float32")
                inv = T.alloc_ub((_RES_STAT_LANES,), "float32")
                ref = T.alloc_ub((_RES_STAT_LANES,), "float32")
                spare = T.alloc_ub((_RES_STAT_LANES,), "float32")
                mean = T.alloc_ub((_RES_STAT_LANES,), "float32")

                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * block_m + vid * sub_m
                            # ⚠️ No guarded per-row copy loop, unlike
                            # ``_compile_fused_add_norm``.  A GM->UB ``T.copy``
                            # is clamped to the real tensor extent at lowering
                            # time (tilelang-ascend src/op/ascend.cc ::
                            # compute_valid_extent), so an m tail loads fewer
                            # rows and leaves the rest of the UB tile untouched.
                            # Safe because every statistic is per row: garbage
                            # in a row this lane does not own cannot reach a row
                            # it does.  The STORES are in bounds because Y and
                            # Add are padded to m_pad rows.
                            T.tile.fill(accb, 0.0)
                            for nt in T.serial(n_tiles):
                                T.copy(A[row_base, nt * block_n], src)
                                T.copy(Residual[row_base, nt * block_n], rhs)
                                if add_via_fp32:
                                    if reread:
                                        T.tile.cast(keep, src, "CAST_NONE",
                                                    sub_m * block_n)
                                        T.tile.cast(accs, rhs, "CAST_NONE",
                                                    sub_m * block_n)
                                        T.tile.add(keep, keep, accs)
                                        T.tile.cast(src, keep, "CAST_RINT",
                                                    sub_m * block_n)
                                    else:
                                        T.tile.cast(hold[nt, :, :], src, "CAST_NONE",
                                                    sub_m * block_n)
                                        T.tile.cast(accs, rhs, "CAST_NONE",
                                                    sub_m * block_n)
                                        T.tile.add(hold[nt, :, :], hold[nt, :, :], accs)
                                        T.tile.cast(src, hold[nt, :, :], "CAST_RINT",
                                                    sub_m * block_n)
                                else:
                                    T.tile.add(src, src, rhs)
                                T.copy(src, Add[row_base, nt * block_n])
                                # WAR: the store above is an asynchronous MTE3
                                # read of ``src``; the next iteration's MTE2
                                # reload and this iteration's V cast both
                                # rewrite it.  AUTO_SYNC does not cover a copy
                                # in the middle of a loop body (see
                                # ``_drain_out_dma``).
                                T.evaluate(T.set_flag("mte3", "mte2", _WAR_EVENT_MTE2))
                                T.evaluate(T.wait_flag("mte3", "mte2", _WAR_EVENT_MTE2))
                                T.evaluate(T.set_flag("mte3", "v", _WAR_EVENT_V))
                                T.evaluate(T.wait_flag("mte3", "v", _WAR_EVENT_V))
                                if reread:
                                    if need_cast:
                                        T.tile.cast(keep, src, "CAST_NONE",
                                                    sub_m * block_n)
                                    else:
                                        T.copy(src, keep)
                                    T.tile.mul_add_dst(accb, keep, keep)
                                else:
                                    if need_cast:
                                        T.tile.cast(hold[nt, :, :], src, "CAST_NONE",
                                                    sub_m * block_n)
                                    else:
                                        T.copy(src, hold[nt, :, :])
                            if reread:
                                # ``accb`` already carries sum(x^2): the
                                # ablation accumulates it inside pass 1 because
                                # it has no resident row to walk afterwards.
                                pass
                            elif kind == "layer":
                                # Textbook two-pass mean/variance, both passes
                                # over the RESIDENT row -- no GM re-read and no
                                # second-moment subtraction.  ``accs`` is zeroed
                                # HERE, not before pass 1: pass 1 used it as the
                                # fp32 residual landing buffer.
                                T.tile.fill(accs, 0.0)
                                for nt in T.serial(n_tiles):
                                    T.tile.add(accs, accs, hold[nt, :, :])
                                T.reduce_sum(accs, tot[0:sub_m], dim=-1)
                                T.tile.mul(mean, tot, T.cast(1.0 / n, "float32"))
                                T.tile.fill(accb, 0.0)
                                for nt in T.serial(n_tiles):
                                    for r in T.serial(sub_m):
                                        T.tile.sub(hold[nt, r, :], hold[nt, r, :], mean[r])
                                    T.tile.mul_add_dst(accb, hold[nt, :, :],
                                                       hold[nt, :, :])
                            else:
                                for nt in T.serial(n_tiles):
                                    # dst += src0 * src1 in one instruction, and
                                    # it is also what lets the reduction happen
                                    # once instead of n_tiles times.
                                    T.tile.mul_add_dst(accb, hold[nt, :, :],
                                                       hold[nt, :, :])
                            T.reduce_sum(accb, tot[0:sub_m], dim=-1)
                            T.tile.mul(den, tot, T.cast(1.0 / n, "float32"))
                            T.tile.add(den, den, T.cast(eps, "float32"))
                            T.tile.rsqrt(inv, den)
                            # Ascend's vector rsqrt is intentionally fast and
                            # approximate.  One Newton step restores fp32
                            # accuracy before the final dtype cast -- same step,
                            # same order as the shipped kernel.
                            T.tile.mul(ref, inv, inv)
                            T.tile.mul(ref, ref, den)
                            T.tile.mul(ref, ref, 0.5)
                            T.tile.fill(spare, 1.5)
                            T.tile.sub(spare, spare, ref)
                            T.tile.mul(inv, inv, spare)
                            for nt in T.serial(n_tiles):
                                if reread:
                                    # THE ABLATED LINE: read x and residual from
                                    # GM a second time and redo the add, instead
                                    # of reading the row that is already in UB.
                                    T.copy(A[row_base, nt * block_n], src)
                                    T.copy(Residual[row_base, nt * block_n], rhs)
                                    if add_via_fp32:
                                        T.tile.cast(keep, src, "CAST_NONE",
                                                    sub_m * block_n)
                                        T.tile.cast(accs, rhs, "CAST_NONE",
                                                    sub_m * block_n)
                                        T.tile.add(keep, keep, accs)
                                        T.tile.cast(src, keep, "CAST_RINT",
                                                    sub_m * block_n)
                                        T.tile.cast(keep, src, "CAST_NONE",
                                                    sub_m * block_n)
                                    else:
                                        T.tile.add(src, src, rhs)
                                        if need_cast:
                                            T.tile.cast(keep, src, "CAST_NONE",
                                                        sub_m * block_n)
                                        else:
                                            T.copy(src, keep)
                                T.copy(W[nt * block_n], wub)
                                if need_cast:
                                    T.tile.cast(w32, wub, "CAST_NONE", block_n)
                                else:
                                    T.copy(wub, w32)
                                if kind == "layer":
                                    T.copy(Bias[nt * block_n], bub)
                                    if need_cast:
                                        T.tile.cast(b32, bub, "CAST_NONE", block_n)
                                    else:
                                        T.copy(bub, b32)
                                # One (block_n,) vector multiply per row and one
                                # vector-scalar multiply per row, instead of
                                # broadcasting w and inv into two
                                # (sub_m, block_n) replicas first.
                                for r in T.serial(sub_m):
                                    if reread:
                                        T.tile.mul(keep[r, :], keep[r, :], inv[r])
                                        T.tile.mul(keep[r, :], keep[r, :], w32)
                                    else:
                                        T.tile.mul(hold[nt, r, :], hold[nt, r, :], inv[r])
                                        T.tile.mul(hold[nt, r, :], hold[nt, r, :], w32)
                                        if kind == "layer":
                                            T.tile.add(hold[nt, r, :], hold[nt, r, :], b32)
                                if reread:
                                    if need_cast:
                                        T.tile.cast(src, keep, "CAST_RINT",
                                                    sub_m * block_n)
                                        T.copy(src, Y[row_base, nt * block_n])
                                    else:
                                        T.copy(keep, Y[row_base, nt * block_n])
                                else:
                                    if need_cast:
                                        T.tile.cast(src, hold[nt, :, :], "CAST_RINT",
                                                    sub_m * block_n)
                                        T.copy(src, Y[row_base, nt * block_n])
                                    else:
                                        T.copy(hold[nt, :, :], Y[row_base, nt * block_n])
        return main

    return factory()


@lru_cache(maxsize=128)
def _compile_fused_add_norm(
    m: int,
    n: int,
    dtype_name: str,
    kind: str,
    eps: float,
):
    """Compile residual-add + row normalization as one AIV launch.

    Shape and tail facts stay as Python compile-time constants.  Only the row
    and column predicates remain in the generated program; this mirrors the
    proven non-divisible path in ``families.two_pass``.
    """
    if kind not in {"rms", "layer"}:
        raise ValueError(f"unsupported fused normalization kind: {kind}")
    itemsize = 2 if dtype_name in ("float16", "bfloat16") else 4
    # --- Geometry (R200) -------------------------------------------------
    # UB census read off the allocation list in ``factory`` below.  One wrong
    # entry here is a silent UB overflow, so it is spelled out:
    #   (sub_m, block_n) : src[itemsize] rhs[itemsize] x32[4] work[4] bc[4]
    #   (block_n,)       : wub[itemsize] bub[itemsize] w32[4] b32[4]
    #   (sub_m,)         : 18 fp32 scalars
    plan = plan_rowwise_norm(
        m, n, itemsize,
        bytes_per_cell=2 * itemsize + 12,
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
        out_idx=[4, 5],
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
            Residual: T.Tensor([m, n], dtype_name),
            W: T.Tensor([n], dtype_name),
            Bias: T.Tensor([n], dtype_name),
            Y: T.Tensor([m_pad, n_pad], dtype_name),
            Add: T.Tensor([m_pad, n_pad], dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((sub_m, block_n), dtype_name)
                rhs = T.alloc_ub((sub_m, block_n), dtype_name)
                x32 = T.alloc_ub((sub_m, block_n), "float32")
                work = T.alloc_ub((sub_m, block_n), "float32")
                stat = T.alloc_ub((sub_m,), "float32")
                stat2 = T.alloc_ub((sub_m,), "float32")
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
                                    T.tile.fill(rhs, 0.0)
                                if not has_n_tail:
                                    if not has_m_tail:
                                        T.copy(A[row_base, nt * block_n], src)
                                        T.copy(Residual[row_base, nt * block_n], rhs)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(A[row_base + r, nt * block_n], src[r, :])
                                                T.copy(Residual[row_base + r, nt * block_n], rhs[r, :])
                                elif nt < n_tiles - 1:
                                    if not has_m_tail:
                                        T.copy(A[row_base, nt * block_n], src)
                                        T.copy(Residual[row_base, nt * block_n], rhs)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(A[row_base + r, nt * block_n], src[r, :])
                                                T.copy(Residual[row_base + r, nt * block_n], rhs[r, :])
                                else:
                                    for r in T.serial(sub_m):
                                        if row_base + r < m:
                                            for c in T.serial(block_n):
                                                if c < n - nt * block_n:
                                                    src[r, c] = A[row_base + r, nt * block_n + c]
                                                    rhs[r, c] = Residual[row_base + r, nt * block_n + c]
                                if dtype_name == "bfloat16":
                                    T.tile.cast(x32, src, "CAST_NONE", sub_m * block_n)
                                    T.tile.cast(work, rhs, "CAST_NONE", sub_m * block_n)
                                    T.tile.add(x32, x32, work)
                                    T.tile.cast(src, x32, "CAST_RINT", sub_m * block_n)
                                    T.copy(src, Add[row_base, nt * block_n])
                                    _drain_out_dma(has_n_tail)
                                    T.tile.cast(x32, src, "CAST_NONE", sub_m * block_n)
                                else:
                                    T.tile.add(src, src, rhs)
                                    T.copy(src, Add[row_base, nt * block_n])
                                    _drain_out_dma(has_n_tail)
                                if need_cast and dtype_name != "bfloat16":
                                    T.tile.cast(x32, src, "CAST_NONE", sub_m * block_n)
                                elif not need_cast:
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

                            if kind != "rms":
                                T.copy(stat, mean)
                            if kind == "rms":
                                T.tile.mul(denom, stat2, T.cast(1.0 / n, "float32"))
                                T.tile.add(denom, denom, T.cast(eps, "float32"))
                            else:
                                T.tile.div(denom, m2, count)
                                T.tile.add(denom, denom, T.cast(eps, "float32"))
                            T.tile.rsqrt(inv, denom)
                            T.tile.mul(refine, inv, inv)
                            T.tile.mul(refine, refine, denom)
                            T.tile.mul(refine, refine, 0.5)
                            T.tile.fill(denom, 1.5)
                            T.tile.sub(denom, denom, refine)
                            T.tile.mul(inv, inv, denom)
                            T.tile.broadcast(bc, inv)

                            for nt in T.serial(n_tiles):
                                if has_n_tail:
                                    T.tile.fill(src, 0.0)
                                    T.tile.fill(rhs, 0.0)
                                if not has_n_tail or nt < n_tiles - 1:
                                    if not has_m_tail:
                                        T.copy(A[row_base, nt * block_n], src)
                                        T.copy(Residual[row_base, nt * block_n], rhs)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(A[row_base + r, nt * block_n], src[r, :])
                                                T.copy(Residual[row_base + r, nt * block_n], rhs[r, :])
                                else:
                                    for r in T.serial(sub_m):
                                        if row_base + r < m:
                                            for c in T.serial(block_n):
                                                if c < n - nt * block_n:
                                                    src[r, c] = A[row_base + r, nt * block_n + c]
                                                    rhs[r, c] = Residual[row_base + r, nt * block_n + c]
                                if dtype_name == "bfloat16":
                                    T.tile.cast(x32, src, "CAST_NONE", sub_m * block_n)
                                    T.tile.cast(work, rhs, "CAST_NONE", sub_m * block_n)
                                    T.tile.add(x32, x32, work)
                                    T.tile.cast(src, x32, "CAST_RINT", sub_m * block_n)
                                    T.tile.cast(x32, src, "CAST_NONE", sub_m * block_n)
                                else:
                                    T.tile.add(src, src, rhs)
                                if need_cast and dtype_name != "bfloat16":
                                    T.tile.cast(x32, src, "CAST_NONE", sub_m * block_n)
                                elif not need_cast:
                                    T.copy(src, x32)
                                if kind == "layer":
                                    T.tile.broadcast(work, mean)
                                    T.tile.sub(x32, x32, work)
                                T.tile.mul(x32, x32, bc)
                                # R200: was a per-ELEMENT scalar GM read of the
                                # whole weight (and bias) vector, re-run by every
                                # task -- n * n_tasks scalar iterations, and the
                                # dominant cost of this template.  Differential
                                # in R200-data/probe_vecweight_decode.json:
                                # 2.16x (rms) / 2.81x (layer) at unchanged
                                # geometry, 8.5x-12.0x combined with geometry.
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
                                    T.copy(src, Y[row_base, nt * block_n])
                                else:
                                    T.tile.broadcast(work, wub)
                                    T.tile.mul(x32, x32, work)
                                    if kind == "layer":
                                        T.tile.broadcast(work, bub)
                                        T.tile.add(x32, x32, work)
                                    T.copy(x32, Y[row_base, nt * block_n])
        return main

    return factory()


def launch_fused_add_norm(x, residual, weight, bias, *, kind: str, eps: float):
    shape = tuple(int(v) for v in x.shape)
    n = shape[-1]
    m = math.prod(shape) // n
    # R274: the resident-row path when the shape admits it, the shipped
    # two-GM-read template otherwise.  ``None`` is a normal answer.
    kernel = None
    mode = os.environ.get("TILEOPS_FUSED_ADD_RESIDENT", "1")
    if mode != "0":
        kernel = _compile_fused_add_resident(
            m, n, _dtype_name(x.dtype), kind, float(eps), mode == "reread"
        )
    if kernel is None:
        kernel = _compile_fused_add_norm(m, n, _dtype_name(x.dtype), kind, float(eps))
    bias_arg = weight if bias is None else bias
    y, add = kernel(
        x.contiguous().reshape(m, n),
        residual.contiguous().reshape(m, n),
        weight.contiguous(),
        bias_arg.contiguous(),
    )
    return y[:m, :n].reshape(shape), add[:m, :n].reshape(shape)


@lru_cache(maxsize=128)
def _compile_ada_norm(m: int, n: int, dtype_name: str, has_gate: bool, eps: float):
    itemsize = 2 if dtype_name in ("float16", "bfloat16") else 4
    # --- Geometry (R200) -------------------------------------------------
    # UB census read off the allocation list in ``factory`` below:
    #   (sub_m, block_n) : src[itemsize] x32[4] work[4] bc[4]
    #   (sub_m,)         : 14 fp32 scalars
    # NOTE: Scale/Shift/Gate are per-row (m, n) tensors already loaded with
    # vector ``T.copy``, which is why AdaLayerNormFwdOp is the only op in this
    # family that already beat the vendor baseline (graph ratio 1.04-1.32) --
    # it never had the scalar parameter-load defect.  It only needs geometry.
    plan = plan_rowwise_norm(
        m, n, itemsize,
        bytes_per_cell=itemsize + 12,
        bytes_per_col=0,
        bytes_per_row=14 * 4,
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
        out_idx=[4],
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
            Scale: T.Tensor([m, n], dtype_name),
            Shift: T.Tensor([m, n], dtype_name),
            Gate: T.Tensor([m, n], dtype_name),
            Y: T.Tensor([m_pad, n_pad], dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((sub_m, block_n), dtype_name)
                x32 = T.alloc_ub((sub_m, block_n), "float32")
                work = T.alloc_ub((sub_m, block_n), "float32")
                count = T.alloc_ub((sub_m,), "float32")
                mean = T.alloc_ub((sub_m,), "float32")
                m2 = T.alloc_ub((sub_m,), "float32")
                row_sum = T.alloc_ub((sub_m,), "float32")
                row_m2 = T.alloc_ub((sub_m,), "float32")
                tile_mean = T.alloc_ub((sub_m,), "float32")
                tile_count = T.alloc_ub((sub_m,), "float32")
                delta = T.alloc_ub((sub_m,), "float32")
                ratio = T.alloc_ub((sub_m,), "float32")
                new_count = T.alloc_ub((sub_m,), "float32")
                correction = T.alloc_ub((sub_m,), "float32")
                denom = T.alloc_ub((sub_m,), "float32")
                refine = T.alloc_ub((sub_m,), "float32")
                inv = T.alloc_ub((sub_m,), "float32")
                bc = T.alloc_ub((sub_m, block_n), "float32")

                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * block_m + vid * sub_m
                            T.tile.fill(count, 0.0)
                            T.tile.fill(mean, 0.0)
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
                                if has_n_tail and nt == n_tiles - 1:
                                    for r in T.serial(sub_m):
                                        for c in T.serial(block_n):
                                            if c >= n - (n_tiles - 1) * block_n:
                                                x32[r, c] = 0.0
                                T.tile.mul(work, x32, x32)
                                T.reduce_sum(work, row_m2, dim=-1)
                                if single_n_tile:
                                    # Bit-exact shortcut, see ``single_n_tile``.
                                    T.copy(tile_mean, mean)
                                    T.copy(row_m2, m2)
                                    T.copy(tile_count, count)
                                else:
                                    T.tile.sub(delta, tile_mean, mean)
                                    T.tile.add(new_count, count, tile_count)
                                    T.tile.div(ratio, tile_count, new_count)
                                    T.tile.mul(correction, delta, ratio)
                                    T.tile.add(mean, mean, correction)
                                    T.tile.mul(correction, delta, delta)
                                    T.tile.mul(ratio, count, tile_count)
                                    T.tile.div(ratio, ratio, new_count)
                                    T.tile.mul(correction, correction, ratio)
                                    T.tile.add(row_m2, row_m2, correction)
                                    T.tile.add(m2, m2, row_m2)
                                    T.tile.add(count, count, tile_count)

                            T.tile.div(denom, m2, count)
                            T.tile.add(denom, denom, T.cast(eps, "float32"))
                            T.tile.rsqrt(inv, denom)
                            T.tile.mul(refine, inv, inv)
                            T.tile.mul(refine, refine, denom)
                            T.tile.mul(refine, refine, 0.5)
                            T.tile.fill(denom, 1.5)
                            T.tile.sub(denom, denom, refine)
                            T.tile.mul(inv, inv, denom)
                            T.tile.broadcast(bc, inv)

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
                                T.tile.broadcast(work, mean)
                                T.tile.sub(x32, x32, work)
                                T.tile.mul(x32, x32, bc)

                                if has_n_tail:
                                    T.tile.fill(src, 0.0)
                                if not has_n_tail or nt < n_tiles - 1:
                                    if not has_m_tail:
                                        T.copy(Scale[row_base, nt * block_n], src)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(Scale[row_base + r, nt * block_n], src[r, :])
                                else:
                                    for r in T.serial(sub_m):
                                        if row_base + r < m:
                                            for c in T.serial(block_n):
                                                if c < n - nt * block_n:
                                                    src[r, c] = Scale[row_base + r, nt * block_n + c]
                                if need_cast:
                                    T.tile.cast(work, src, "CAST_NONE", sub_m * block_n)
                                else:
                                    T.copy(src, work)
                                T.tile.mul(x32, x32, work)

                                if has_n_tail:
                                    T.tile.fill(src, 0.0)
                                if not has_n_tail or nt < n_tiles - 1:
                                    if not has_m_tail:
                                        T.copy(Shift[row_base, nt * block_n], src)
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                T.copy(Shift[row_base + r, nt * block_n], src[r, :])
                                else:
                                    for r in T.serial(sub_m):
                                        if row_base + r < m:
                                            for c in T.serial(block_n):
                                                if c < n - nt * block_n:
                                                    src[r, c] = Shift[row_base + r, nt * block_n + c]
                                if need_cast:
                                    T.tile.cast(work, src, "CAST_NONE", sub_m * block_n)
                                else:
                                    T.copy(src, work)
                                T.tile.add(x32, x32, work)

                                if has_gate:
                                    if has_n_tail:
                                        T.tile.fill(src, 0.0)
                                    if not has_n_tail or nt < n_tiles - 1:
                                        if not has_m_tail:
                                            T.copy(Gate[row_base, nt * block_n], src)
                                        else:
                                            for r in T.serial(sub_m):
                                                if row_base + r < m:
                                                    T.copy(Gate[row_base + r, nt * block_n], src[r, :])
                                    else:
                                        for r in T.serial(sub_m):
                                            if row_base + r < m:
                                                for c in T.serial(block_n):
                                                    if c < n - nt * block_n:
                                                        src[r, c] = Gate[row_base + r, nt * block_n + c]
                                    if need_cast:
                                        T.tile.cast(work, src, "CAST_NONE", sub_m * block_n)
                                    else:
                                        T.copy(src, work)
                                    T.tile.mul(x32, x32, work)
                                if need_cast:
                                    T.tile.cast(src, x32, "CAST_RINT", sub_m * block_n)
                                    T.copy(src, Y[row_base, nt * block_n])
                                else:
                                    T.copy(x32, Y[row_base, nt * block_n])
        return main

    return factory()


# ---------------------------------------------------------------------------
# R341: the resident-row AdaLayerNorm path
# ---------------------------------------------------------------------------
#
# ``_compile_ada_norm`` above is unchanged and still serves every shape this
# path declines.  What follows replaces it when the shape allows, and it exists
# because ``_compile_ada_norm`` was the LAST member of the row-wise
# normalization family still planning through ``kernels.common.
# plan_rowwise_norm`` for every shape.  R261 migrated RMSNorm, R274 the
# fused residual-add pair, R335 LayerNorm; these two were never migrated, and
# ``families/two_pass.py``'s own comment (the ``_plan_layer_resident`` preamble)
# already names the three symptoms.  All three hold here, computed for
# ``AdaLayerNorm{,Zero}FwdOp / llama-8b-decode / bf16`` (m = 1, n = 4096), the
# case that sets BOTH operators' ``ratio_min`` in 5 of 5 independent passes
# (docs/reports/R341-data/05-forensic-ada-cases.txt):
#
#   * ``sub_m = 8`` -- ``ROW_EXTENT_GRAIN`` is a hard floor for that template,
#     so every ``T.tile.*`` runs on 8 rows of which 7 DO NOT EXIST.
#   * ``block_n = 1024`` -> ``n_tiles = 4`` narrow tiles rather than 1-2 wide
#     ones.  R261 measured this as its single largest lever and recorded that it
#     is about ``T.tile.*`` CALL COUNT, not bandwidth.
#   * ``has_m_tail = True`` -> every whole-tile ``T.copy`` becomes a guarded
#     per-row copy loop that iterates ``sub_m`` = 8 times.
#
# and one more that is specific to this template rather than inherited:
#
#   * ``_compile_ada_norm`` reads A from GM TWICE (once per pass).  The
#     manifest's own roofline for these operators says ``4 * M * N`` /
#     ``5 * M * N`` bytes, i.e. x read ONCE; the kernel moves ``5 * M * N`` /
#     ``6 * M * N``.  Keeping the row group resident in UB as fp32 between the
#     passes removes that whole extra GM pass.
#
# Evidence that this is schedule and not a floor: on the SAME tree, the SAME
# machine and the SAME geometry (m = 1, n = 4096, bf16), the already-migrated
# siblings run at RMSNorm 2.75 us / LayerNorm 3.50 us / FusedAddLayerNorm 4.00
# us while ``AdaLayerNormFwdOp`` takes 16.00 us and ``AdaLayerNormZeroFwdOp``
# 18.50 us (docs/reports/R341-data/06-layernorm-decode-reference.txt).
#
# The body is ``families.two_pass._compile_layer_resident`` re-expressed for
# this operator's inputs.  ONE structural difference, and it is forced by the
# signature: LayerNorm's affine parameters are ``(n,)`` vectors, so R261's
# fourth lever (multiply each row against a ``(block_n,)`` weight rather than
# broadcasting it into a ``(sub_m, block_n)`` replica) applies there.  Here
# Scale/Shift/Gate are full ``(m, n)`` tensors -- there is no broadcast to
# remove, the epilogue simply streams each tile in.  Their GM reads are
# irreducible and are NOT counted as a lever.
#
# ⚠️ Numerics change on purpose, and it is not a relaxation: ``_compile_ada_norm``
# runs fp32 Welford ACROSS n-tiles, this runs the textbook two-pass mean /
# sum-of-squared-deviations over the resident row (mean = sum/n, then centre in
# place).  That is the same estimator ``_compile_layer_resident`` and
# ``_compile_fused_add_resident`` already ship.  Both are fp32; neither is a
# wider tolerance.  The paired correctness gate for this round
# (docs/reports/R341-data/p_correctness.py) reports max_abs_err / max_rel_err
# against the unchanged reference and the unchanged tolerance for all 8
# operators this builder serves, before and after.

#: Lanes every ``(...,)`` fp32 statistics buffer is padded to, so ``sub_m`` is
#: free of ``kernels.common.ROW_EXTENT_GRAIN``'s hard floor of 8.  Same number
#: and same 32-byte-block reason as ``two_pass._STAT_LANES``, but here it is a
#: buffer WIDTH, not a floor on ``sub_m``.
_ADA_STAT_LANES = 8

#: Column tiles per row this planner aims for.  ⚠️ NOT inherited from
#: ``two_pass._TARGET_N_TILES = 2``: that 2 was measured on RMSNorm's buffer
#: plan (R261-data/05-step4-ntiles.txt), and this template streams two or three
#: extra ``(m, n)`` operands through the epilogue, which changes the MTE2/VEC
#: balance the choice trades off.  Swept on all 10 real manifest cases in
#: docs/reports/R341-data/13-ada-ntiles-ablation.txt (raw: probe/abl1.log,
#: probe/abl1/abl-abl1.json) and set to the winner.  1 won EVERY case, and by a
#: lot where it matters: llama-8b-decode/bf16 16.250 -> 3.500 us at nt=1, against
#: 4.500 at nt=2 and 6.750 at nt=4; llama-8b-prefill 111.9 -> 91.3 at nt=1 against
#: 96.0 at nt=2.  For n = 1152 (dit-xl-2) nt=2 is actively BAD (17.0 -> 26.0)
#: because the width search then lands on 384 x 3 rather than one 1152 tile.
#: ``TILEOPS_ADA_TARGET_N_TILES`` overrides it, for that sweep only.
_ADA_TARGET_N_TILES = 1

#: Hard ceiling on column tiles per row; past it the resident fp32 row has
#: squeezed the tile so narrow that the two-pass kernel wins, so the planner
#: declines and the caller falls back.  Same ceiling as ``two_pass``.
_ADA_MAX_N_TILES = 4


def _ada_target_n_tiles() -> int:
    raw = os.environ.get("TILEOPS_ADA_TARGET_N_TILES")
    if not raw:
        return _ADA_TARGET_N_TILES
    try:
        value = int(raw)
    except ValueError:
        return _ADA_TARGET_N_TILES
    return value if value >= 1 else _ADA_TARGET_N_TILES


def _plan_ada_resident(m: int, n: int, itemsize: int):
    """Geometry for the resident-row AdaLayerNorm path, or ``None`` if it declines.

    ``None`` is a normal answer, not a failure: the caller then compiles
    ``_compile_ada_norm``, which has no residency requirement and no width rule.
    """
    m = max(1, int(m))
    n = max(1, int(n))
    if n % TILE_GRAIN:
        # The ragged-column path is exactly what ``_compile_ada_norm`` already
        # handles (its scalar n-tail double loop); duplicating it here would
        # double the surface without touching any measured lever.
        return None
    # UB census, read off the allocation list in ``_compile_ada_resident``.
    # One wrong entry is a silent UB overflow (R194 2.2), so it is spelled out:
    #   per row of the tile : src[itemsize] + accb[4] + hold[4 * n_tiles * width]
    #                         ... and hold spans the WHOLE row, hence 4 * n
    #   fixed               : 7 statistics buffers of _ADA_STAT_LANES fp32
    #                         (tot, part, mean, den, inv, ref, spare)
    occupancy = max(1, m // (2 * LAUNCH_BLOCK_CAP))
    fixed = 7 * _ADA_STAT_LANES * 4
    target = _ada_target_n_tiles()
    width = (n // target) - (n // target) % TILE_GRAIN
    while width >= TILE_GRAIN:
        n_tiles = n // width
        if n % width == 0 and n_tiles <= _ADA_MAX_N_TILES:
            per_row = width * itemsize + width * 4 + n * 4
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
def _compile_ada_resident(m: int, n: int, dtype_name: str, has_gate: bool, eps: float):
    """AdaLayerNorm with the row group resident in UB, or ``None`` if it will not fit."""
    if dtype_name not in {"float16", "bfloat16", "float32"}:
        raise TypeError(f"normalization supports floating dtypes, got {dtype_name}")
    itemsize = 2 if dtype_name in ("float16", "bfloat16") else 4
    plan = _plan_ada_resident(m, n, itemsize)
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
        out_idx=[4],
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
            Scale: T.Tensor([m, n], dtype_name),
            Shift: T.Tensor([m, n], dtype_name),
            Gate: T.Tensor([m, n], dtype_name),
            Y: T.Tensor([m_pad, n], dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                # The whole row group, in fp32, for the lifetime of all passes.
                hold = T.alloc_ub((n_tiles, sub_m, block_n), "float32")
                src = T.alloc_ub((sub_m, block_n), dtype_name)
                accb = T.alloc_ub((sub_m, block_n), "float32")
                tot = T.alloc_ub((_ADA_STAT_LANES,), "float32")
                part = T.alloc_ub((_ADA_STAT_LANES,), "float32")
                mean = T.alloc_ub((_ADA_STAT_LANES,), "float32")
                den = T.alloc_ub((_ADA_STAT_LANES,), "float32")
                inv = T.alloc_ub((_ADA_STAT_LANES,), "float32")
                ref = T.alloc_ub((_ADA_STAT_LANES,), "float32")
                spare = T.alloc_ub((_ADA_STAT_LANES,), "float32")

                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < m_tiles:
                            row_base = logical_cid * block_m + vid * sub_m
                            # ⚠️ No guarded per-row copy loop, unlike
                            # ``_compile_ada_norm``.  A GM->UB ``T.copy`` is
                            # clamped to the real tensor extent at lowering time
                            # (tilelang-ascend src/op/ascend.cc ::
                            # compute_valid_extent), so an m tail loads fewer
                            # rows and leaves the rest of the UB tile untouched.
                            # Safe because every statistic is per row: garbage in
                            # a row this lane does not own cannot reach a row it
                            # does.  The STORE is in bounds because Y is padded
                            # to m_pad and ``launch_ada_norm`` slices [:m].
                            #
                            # Pass 1: load the row group ONCE, keep it in fp32,
                            # accumulate the row sums.  ``part`` is zeroed once:
                            # ``T.reduce_sum`` narrows its destination to
                            # ``[0:sub_m]``, so lanes above ``sub_m`` keep the
                            # 0.0 written here and can never feed a NaN to
                            # ``tot``.
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
                            # Pass 2: textbook two-pass variance over the RESIDENT
                            # row -- centre in place, then accumulate the squares
                            # with ``mul_add_dst`` so the reduction happens once
                            # instead of once per n-tile.  Centring in place is
                            # what lets the epilogue skip the second subtraction
                            # ``_compile_ada_norm`` pays, AND it is why A is never
                            # re-read.
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
                            # approximate.  One Newton step restores fp32 accuracy
                            # before the final dtype cast -- same step, same order
                            # as ``_compile_ada_norm``.
                            T.tile.mul(ref, inv, inv)
                            T.tile.mul(ref, ref, den)
                            T.tile.mul(ref, ref, 0.5)
                            T.tile.fill(spare, 1.5)
                            T.tile.sub(spare, spare, ref)
                            T.tile.mul(inv, inv, spare)
                            # Pass 3: epilogue.  ``hold`` already holds x - mean,
                            # so this is one vector-scalar multiply per row plus
                            # one streamed ``(m, n)`` operand per factor.  ``accb``
                            # is free again here -- the variance reduction above
                            # consumed it.
                            for nt in T.serial(n_tiles):
                                for r in T.serial(sub_m):
                                    T.tile.mul(hold[nt, r, :], hold[nt, r, :], inv[r])
                                T.copy(Scale[row_base, nt * block_n], src)
                                if need_cast:
                                    T.tile.cast(accb, src, "CAST_NONE", sub_m * block_n)
                                else:
                                    T.copy(src, accb)
                                T.tile.mul(hold[nt, :, :], hold[nt, :, :], accb)
                                T.copy(Shift[row_base, nt * block_n], src)
                                if need_cast:
                                    T.tile.cast(accb, src, "CAST_NONE", sub_m * block_n)
                                else:
                                    T.copy(src, accb)
                                T.tile.add(hold[nt, :, :], hold[nt, :, :], accb)
                                if has_gate:
                                    T.copy(Gate[row_base, nt * block_n], src)
                                    if need_cast:
                                        T.tile.cast(accb, src, "CAST_NONE",
                                                    sub_m * block_n)
                                    else:
                                        T.copy(src, accb)
                                    T.tile.mul(hold[nt, :, :], hold[nt, :, :], accb)
                                if need_cast:
                                    T.tile.cast(src, hold[nt, :, :], "CAST_RINT",
                                                sub_m * block_n)
                                    T.copy(src, Y[row_base, nt * block_n])
                                else:
                                    T.copy(hold[nt, :, :], Y[row_base, nt * block_n])
        return main

    return factory()


def launch_ada_norm(x, scale, shift, gate, *, eps: float):
    shape = tuple(int(v) for v in x.shape)
    n = shape[-1]
    m = math.prod(shape) // n
    dtype_name = _dtype_name(x.dtype)
    gate_arg = shift if gate is None else gate
    args = (
        x.contiguous().reshape(m, n),
        scale.contiguous().reshape(m, n),
        shift.contiguous().reshape(m, n),
        gate_arg.contiguous().reshape(m, n),
    )
    # R341: the resident row-group path when the shape admits it, the shipped
    # ``plan_rowwise_norm`` template otherwise.  ``None`` is a normal answer.
    # ``TILEOPS_ADA_RESIDENT=0`` forces the old path; it is the A/B switch this
    # change was attributed with, not a shipping mode.
    if os.environ.get("TILEOPS_ADA_RESIDENT", "1") != "0":
        kernel = _compile_ada_resident(m, n, dtype_name, gate is not None, float(eps))
        if kernel is not None:
            y = kernel(*args)
            return y[:m, :n].reshape(shape)
    kernel = _compile_ada_norm(m, n, dtype_name, gate is not None, float(eps))
    y = kernel(*args)
    return y[:m, :n].reshape(shape)


def launch_group(x, weight, bias, *, groups: int, eps: float):
    shape = tuple(int(v) for v in x.shape)
    n, c, spatial_shape = shape[0], shape[1], shape[2:]
    spatial = math.prod(spatial_shape)
    cpg = c // groups
    dtype_name = _dtype_name(x.dtype)
    affine = weight is not None
    if not affine:
        # R335: this used to be ``torch.ones``/``torch.zeros``, and each of those
        # launches a DEVICE kernel inside the timed region -- measured as
        # ``aclnnInplaceOne_OnesLike`` + ``aclnnInplaceZero_ZerosLike``, 2 of the 3
        # kernels per call on every non-affine case, 2.06 us of GroupNorm /
        # image-g32 / fp16's 39.875 us (docs/reports/R335-data/04-groupnorm-per-kernel.txt).
        # Both compiled kernels put the affine loads behind a COMPILE-TIME
        # ``if affine``, so with no affine pair they never read W or Bias and an
        # uninitialised buffer of the right shape and dtype is enough.  ``affine``
        # is now passed to the compiler explicitly rather than re-derived from
        # ``weight is not None``, which these placeholders would have flipped.
        weight = torch.empty((c,), device=x.device, dtype=x.dtype)
        bias = torch.empty((c,), device=x.device, dtype=x.dtype)
    # R335: the resident row-group path when the shape admits it, the shipped
    # (1, tile) template otherwise.  ``None`` is a normal answer.
    # ``TILEOPS_GROUP_NORM_RESIDENT=0`` forces the old path; it is the A/B switch
    # this change was attributed with, not a shipping mode.
    if os.environ.get("TILEOPS_GROUP_NORM_RESIDENT", "1") != "0":
        plan = _plan_group_resident(n, c, spatial, groups, cpg, dtype_name)
        kernel = _compile_group_resident(n, c, spatial, groups, cpg, dtype_name,
                                         affine, float(eps))
        if kernel is not None:
            out = torch.empty((n * c, plan["tile"]), device=x.device, dtype=x.dtype)
            kernel(x.contiguous().reshape(n * c, spatial), out,
                   weight.contiguous(), bias.contiguous())
            return out[:, :spatial].reshape(shape)
    tile = _spatial_tile(spatial, dtype_name, cell_extra=8)
    padded = math.ceil(spatial / tile) * tile
    kernel = _compile_group(n, c, spatial, groups, cpg, dtype_name, affine, float(eps))
    out = torch.empty((n * c * padded,), device=x.device, dtype=x.dtype)
    kernel(x.contiguous().reshape(-1), out, weight.contiguous(), bias.contiguous())
    return out.reshape(n, c, padded)[:, :, :spatial].reshape(shape)


def launch_instance_infer(x, running_mean, running_var, weight, bias, *, eps: float):
    shape = tuple(int(v) for v in x.shape)
    n, c, spatial_shape = shape[0], shape[1], shape[2:]
    spatial = math.prod(spatial_shape)
    tile = _spatial_tile(spatial, _dtype_name(x.dtype), cell_extra=8)
    padded = math.ceil(spatial / tile) * tile
    kernel = _compile_instance_infer(n, c, spatial, _dtype_name(x.dtype), weight is not None, float(eps))
    out = torch.empty((n * c * padded,), device=x.device, dtype=x.dtype)
    if weight is None:
        # R335, same reason as ``launch_group``: ``torch.ones``/``torch.zeros``
        # each launch a device kernel inside the timed region, and
        # ``_compile_instance_infer`` puts its affine loads behind a
        # compile-time ``if affine``, so it never reads these.
        weight = torch.empty((c,), device=x.device, dtype=x.dtype)
        bias = torch.empty((c,), device=x.device, dtype=x.dtype)
    kernel(
        x.contiguous().reshape(-1),
        out,
        running_mean.contiguous(),
        running_var.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
    )
    return out.reshape(n, c, padded)[:, :, :spatial].reshape(shape)


@lru_cache(maxsize=128)
def _compile_batch_flat(n: int, c: int, dtype_name: str, eps: float):
    """R362: BatchNorm INFERENCE on a rank-2 ``[N, C]`` input (``spatial == 1``).

    The shipped ``_compile_batch`` is channel-parallel with ``for batch in
    T.serial(n)`` inside, and one channel's ``n`` values are strided by ``c`` in
    GM.  At ``spatial == 1`` that makes every one of those ``n`` iterations a
    ragged chunk of ONE element: ``tile = 16``, ``tail = 1``, ``tail_vec = 0``,
    so the load is 16 guarded scalar GM reads and the store is a 1-element MTE3
    transfer.  Measured on ``resnet50-fc`` [32, 64] fp16 (R362
    probe/base/p1_BatchNormFwdOp.stdout, Level1, reps=20):

        duration   35.706 us
        aiv_scalar 24.229 us  (56.2%)   <- 32 batches x 16 guarded reads x 64 ch
        aiv_mte3    8.786 us  (20.3%)   <- 32 one-element stores per channel
        aiv_vec     6.703 us  (15.6%)   <- 9 vector ops per (1, 16) tile, 1/16
                                           of whose lanes are real
        aiv_mte2    0.484 us  ( 1.1%)   <- the whole tensor is 4 KiB

    i.e. the case is SCALAR bound, not bandwidth bound and not launch bound:
    ``resnet50-stage2`` is a 256x larger tensor and takes 9.838 us.

    This kernel runs ROW-parallel instead.  A row is ``[c]``, which is
    contiguous, so the load is one aligned vector transfer; the per-channel
    affine coefficients are ``(c,)`` UB vectors, so the four arithmetic steps
    are plain ``(c,)`` vector ops with no broadcast and no scalar loop.

    ⚠️ BIT-EXACT against ``_compile_batch``'s inference path: the same four
    operations are applied in the same order to the same fp32 values --
    ``(cast(x) - mean) * inv * w + bias`` with the SAME Newton-refined
    ``rsqrt``.  Only the lane the value sits in changes.  ``eps`` is added to
    the variance before the rsqrt exactly as before.
    """
    rows = _batch_flat_rows(n, c)
    tasks = n // rows
    launch_blocks, grid_repeats = _launch_tasks(tasks)
    need_cast = dtype_name != "float32"

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((n, c), dtype_name),
            B: T.Tensor((n, c), dtype_name),
            RunningMean: T.Tensor((c,), "float32"),
            RunningVar: T.Tensor((c,), "float32"),
            W: T.Tensor((c,), "float32"),
            Bias: T.Tensor((c,), "float32"),
            BatchMean: T.Tensor((c,), "float32"),
            BatchRstd: T.Tensor((c,), "float32"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((rows, c), dtype_name)
                out = T.alloc_ub((rows, c), dtype_name)
                x32 = T.alloc_ub((rows, c), "float32")
                mean_c = T.alloc_ub((c,), "float32")
                inv_c = T.alloc_ub((c,), "float32")
                denom_c = T.alloc_ub((c,), "float32")
                refine_c = T.alloc_ub((c,), "float32")
                w_c = T.alloc_ub((c,), "float32")
                b_c = T.alloc_ub((c,), "float32")
                with T.Scope("V"):
                    # Same sequence as _compile_batch's inference prologue, on
                    # (c,) lanes instead of one lane at a time.
                    T.copy(RunningMean[0], mean_c)
                    T.copy(RunningVar[0], denom_c)
                    T.tile.add(denom_c, denom_c, T.cast(eps, "float32"))
                    T.tile.rsqrt(inv_c, denom_c)
                    T.tile.mul(refine_c, inv_c, inv_c)
                    T.tile.mul(refine_c, refine_c, denom_c)
                    T.tile.mul(refine_c, refine_c, 0.5)
                    T.tile.fill(denom_c, 1.5)
                    T.tile.sub(denom_c, denom_c, refine_c)
                    T.tile.mul(inv_c, inv_c, denom_c)
                    T.copy(W[0], w_c)
                    T.copy(Bias[0], b_c)
                    for repeat in T.serial(grid_repeats):
                        task = (cid + repeat * launch_blocks) * _VEC + vid
                        if task < tasks:
                            T.copy(A[task * rows, 0], src)
                            if need_cast:
                                T.tile.cast(x32, src, "CAST_NONE", rows * c)
                            else:
                                T.copy(src, x32)
                            # One (c,) vector op per row -- the same pattern the
                            # resident norm's affine epilogue uses (see the
                            # `T.tile.mul(hold[nt, r, :], ..., w32)` block).
                            for r in T.serial(rows):
                                T.tile.sub(x32[r, :], x32[r, :], mean_c)
                                T.tile.mul(x32[r, :], x32[r, :], inv_c)
                                T.tile.mul(x32[r, :], x32[r, :], w_c)
                                T.tile.add(x32[r, :], x32[r, :], b_c)
                            if need_cast:
                                T.tile.cast(out, x32, "CAST_RINT", rows * c)
                            else:
                                T.copy(x32, out)
                            T.copy(out, B[task * rows, 0])

        return main

    return factory()


@lru_cache(maxsize=128)
def _compile_batch_plane(n: int, c: int, spatial: int, dtype_name: str, eps: float):
    """R362: BatchNorm INFERENCE with one channel's WHOLE ``(n, spatial)`` plane
    resident in UB.

    ``_compile_batch`` walks ``for batch in T.serial(n)`` and reloads a
    ``(1, tile)`` chunk each time, so a channel of ``resnet50-stage3``
    (n = 4, spatial = 784) costs 4 loads, 4 stores and 40 vector ops on
    784-element tiles.  Viewed as ``A[(n, c * spatial)]`` the same plane is the
    2-D sub-block ``A[0:n, channel*spatial : (channel+1)*spatial]`` -- one
    strided MTE2 transfer -- and the four affine steps then run ONCE on
    ``n * spatial`` lanes.

    ⚠️ BIT-EXACT: identical arithmetic, identical order, identical fp32 values;
    only the lane a value occupies changes.  It is elementwise, so no reduction
    tree is reshaped (contrast T357 section 5.3, where a tiling change did move
    bits because ``T.reduce_sum`` re-associated).

    ⚠️ Entry is gated on ``spatial * itemsize % 32 == 0`` so every row of that
    2-D transfer is a whole number of 32-byte blocks -- the RAGGED-CHUNK TRAP 1
    partial-extent over-read cannot be reached from here.
    """
    launch_blocks, grid_repeats = _launch_tasks(c)
    need_cast = dtype_name != "float32"
    cells = n * spatial

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((n, c * spatial), dtype_name),
            B: T.Tensor((n, c * spatial), dtype_name),
            RunningMean: T.Tensor((c,), "float32"),
            RunningVar: T.Tensor((c,), "float32"),
            W: T.Tensor((c,), "float32"),
            Bias: T.Tensor((c,), "float32"),
            BatchMean: T.Tensor((c,), "float32"),
            BatchRstd: T.Tensor((c,), "float32"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                src = T.alloc_ub((n, spatial), dtype_name)
                out = T.alloc_ub((n, spatial), dtype_name)
                x32 = T.alloc_ub((n, spatial), "float32")
                work = T.alloc_ub((n, spatial), "float32")
                mean = T.alloc_ub((1,), "float32")
                denom = T.alloc_ub((1,), "float32")
                inv = T.alloc_ub((1,), "float32")
                refine = T.alloc_ub((1,), "float32")
                w32v = T.alloc_ub((1,), "float32")
                b32v = T.alloc_ub((1,), "float32")
                with T.Scope("V"):
                    for repeat in T.serial(grid_repeats):
                        block = cid + repeat * launch_blocks
                        channel = block * _VEC + vid
                        if channel < c:
                            # Identical to _compile_batch's inference prologue.
                            T.copy(RunningMean[channel], mean)
                            T.copy(RunningVar[channel], denom)
                            T.tile.add(denom, denom, T.cast(eps, "float32"))
                            T.tile.rsqrt(inv, denom)
                            T.tile.mul(refine, inv, inv)
                            T.tile.mul(refine, refine, denom)
                            T.tile.mul(refine, refine, 0.5)
                            T.tile.fill(denom, 1.5)
                            T.tile.sub(denom, denom, refine)
                            T.tile.mul(inv, inv, denom)
                            T.copy(W[channel], w32v)
                            T.copy(Bias[channel], b32v)
                            T.copy(A[0, channel * spatial], src)
                            if need_cast:
                                T.tile.cast(x32, src, "CAST_NONE", cells)
                            else:
                                T.copy(src, x32)
                            if _PLANE_SCALAR_OPERAND:
                                T.tile.sub(x32, x32, mean[0])
                                T.tile.mul(x32, x32, inv[0])
                                T.tile.mul(x32, x32, w32v[0])
                                T.tile.add(x32, x32, b32v[0])
                            else:
                                T.tile.broadcast(work, mean)
                                T.tile.sub(x32, x32, work)
                                T.tile.broadcast(work, inv)
                                T.tile.mul(x32, x32, work)
                                T.tile.broadcast(work, w32v)
                                T.tile.mul(x32, x32, work)
                                T.tile.broadcast(work, b32v)
                                T.tile.add(x32, x32, work)
                            if need_cast:
                                T.tile.cast(out, x32, "CAST_RINT", cells)
                            else:
                                T.copy(x32, out)
                            T.copy(out, B[0, channel * spatial])
                            # `out` is rewritten by a V op on the next grid
                            # repeat while MTE3 may still be reading it.
                            _drain_out_dma(True)

        return main

    return factory()


def launch_batch(
    x,
    running_mean,
    running_var,
    weight,
    bias,
    *,
    training: bool,
    eps: float,
    momentum: float,
    return_stats: bool = False,
):
    shape = tuple(int(v) for v in x.shape)
    n, c, spatial_shape = shape[0], shape[1], shape[2:]
    spatial = math.prod(spatial_shape)
    # --- R362: the rank-2 inference fast path -----------------------------
    # Entry conditions are about SAFETY, not about performance:
    #   * `spatial == 1` is the only geometry where the shipped kernel's ragged
    #     branch fires on every iteration (every other manifest case has
    #     `tail == 0`);
    #   * `c % 16 == 0` makes one row 32-byte aligned for BOTH dtype widths, so
    #     no partial-extent GM<->UB transfer is emitted (RAGGED-CHUNK TRAP 1);
    #   * inference only, because the training branch also needs the Welford
    #     pass and the running-stat update, which are not reshaped here;
    #   * `c <= _FLAT_TILE_CELLS` keeps one row inside the UB tile budget.
    if (
        _bn_fwd_arm() != "base"
        and spatial == 1
        and not training
        and c % 16 == 0
        and c <= _FLAT_TILE_CELLS
        and n >= 1
    ):
        kernel = _compile_batch_flat(n, c, _dtype_name(x.dtype), float(eps))
        flat_out = torch.empty((n, c), device=x.device, dtype=x.dtype)
        flat_mean = torch.empty((c,), device=x.device, dtype=torch.float32)
        flat_rstd = torch.empty((c,), device=x.device, dtype=torch.float32)
        kernel(
            x.contiguous().reshape(n, c),
            flat_out,
            running_mean,
            running_var,
            weight,
            bias,
            flat_mean,
            flat_rstd,
        )
        flat_output = flat_out.reshape(shape)
        return (flat_output, flat_mean, flat_rstd) if return_stats else flat_output
    # --- R362: the whole-plane inference path -----------------------------
    # Same three safety conditions as above plus a UB one.  `2 * itemsize + 8`
    # is the census of src + out + x32 + work; `work` is only allocated on the
    # broadcast variant but is budgeted for unconditionally so that flipping
    # `_PLANE_SCALAR_OPERAND` can never overflow UB.
    _itemsize = 2 if _dtype_name(x.dtype) in ("float16", "bfloat16") else 4
    if (
        _bn_fwd_arm() != "base"
        and spatial > 1
        and not training
        and (spatial * _itemsize) % 32 == 0
        and n * spatial * (2 * _itemsize + 8) + 1024 <= UB_BUDGET_NORM_BYTES
    ):
        kernel = _compile_batch_plane(n, c, spatial, _dtype_name(x.dtype), float(eps))
        plane_out = torch.empty((n, c * spatial), device=x.device, dtype=x.dtype)
        plane_mean = torch.empty((c,), device=x.device, dtype=torch.float32)
        plane_rstd = torch.empty((c,), device=x.device, dtype=torch.float32)
        kernel(
            x.contiguous().reshape(n, c * spatial),
            plane_out,
            running_mean,
            running_var,
            weight,
            bias,
            plane_mean,
            plane_rstd,
        )
        plane_output = plane_out.reshape(shape)
        return (plane_output, plane_mean, plane_rstd) if return_stats else plane_output
    tile = _spatial_tile(spatial, _dtype_name(x.dtype), cell_extra=8)
    padded = math.ceil(spatial / tile) * tile
    kernel = _compile_batch(n, c, spatial, _dtype_name(x.dtype), training, float(eps), float(momentum))
    out = torch.empty((n * c * padded,), device=x.device, dtype=x.dtype)
    batch_mean = torch.empty((c,), device=x.device, dtype=torch.float32)
    batch_rstd = torch.empty((c,), device=x.device, dtype=torch.float32)
    kernel(
        x.contiguous().reshape(-1),
        out,
        running_mean,
        running_var,
        weight,
        bias,
        batch_mean,
        batch_rstd,
    )
    output = out.reshape(n, c, padded)[:, :, :spatial].reshape(shape)
    return (output, batch_mean, batch_rstd) if return_stats else output


@lru_cache(maxsize=128)
def _compile_batch_bwd(c: int, spatial: int, dtype_name: str, reduction_spatial: int = None,
                       tile: int = None):
    """Compile the spatial BatchNorm backward two-pass AIV kernel.

    Inputs are channel-major ``[C, N*spatial]`` views.  Reduction state and
    affine gradients stay fp32; only ``grad_x`` is cast back to the input dtype.
    """
    if reduction_spatial is None:
        reduction_spatial = spatial
    logical_blocks = max(1, c)
    launch_blocks = launch_block_count(logical_blocks)
    grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)
    # --- Geometry (R200) ---------------------------------------------
    # (1, tile) UB census: go[itemsize] xv[itemsize] + 9 fp32 buffers
    # (go32 x32 work xhat prod sum_do_tile sum_dw_tile scale_tile
    #  length_tile) = 2*itemsize + 36 bytes per element.
    # ⚠️ ``spatial`` here is ALREADY padded by launch_batch_backward, so the
    # tile must come from the caller rather than be recomputed: a multiple
    # of 896 can also be a multiple of 1024, and the two would disagree.
    if tile is None:
        tile = _spatial_tile(spatial, dtype_name, cell_extra=36)
    blk = _blk(dtype_name)
    chunks = math.ceil(spatial / tile)
    tail = spatial % tile
    # 32-byte-aligned part of the ragged chunk, and where the fp32 padding
    # cleanup may start.  Both compile-time constants; see the RAGGED-CHUNK
    # LOADS block above -- in particular why it is inlined, not factored out.
    tail_vec = tail - (tail % blk)
    zpad_start = min(tile, ((tail + 7) // 8) * 8)
    need_cast = dtype_name != "float32"
    # R347: one chunk covers the whole (already padded) channel, so pass 2 can
    # read pass 1's leftovers out of UB instead of re-reading GM.  Compile-time.
    single_chunk = chunks == 1 and tail == 0

    @tilelang.jit(
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            GradOut: T.Tensor((c * spatial,), dtype_name),
            X: T.Tensor((c * spatial,), dtype_name),
            W: T.Tensor((c,), "float32"),
            Mean: T.Tensor((c,), "float32"),
            Rstd: T.Tensor((c,), "float32"),
            GradX: T.Tensor((c * spatial,), dtype_name),
            GradW: T.Tensor((c,), "float32"),
            GradB: T.Tensor((c,), "float32"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                go = T.alloc_ub((1, tile), dtype_name)
                xv = T.alloc_ub((1, tile), dtype_name)
                go32 = T.alloc_ub((1, tile), "float32")
                x32 = T.alloc_ub((1, tile), "float32")
                work = T.alloc_ub((1, tile), "float32")
                xhat = T.alloc_ub((1, tile), "float32")
                prod = T.alloc_ub((1, tile), "float32")
                sum_do_tile = T.alloc_ub((1, tile), "float32")
                sum_dw_tile = T.alloc_ub((1, tile), "float32")
                scale_tile = T.alloc_ub((1, tile), "float32")
                length_tile = T.alloc_ub((1, tile), "float32")
                sum_do = T.alloc_ub((1,), "float32")
                sum_dw = T.alloc_ub((1,), "float32")
                part = T.alloc_ub((1,), "float32")
                part2 = T.alloc_ub((1,), "float32")
                with T.Scope("V"):
                    for repeat in T.serial(grid_repeats):
                        channel = cid + repeat * launch_blocks
                        if channel < c:
                            T.tile.fill(sum_do, 0.0)
                            T.tile.fill(sum_dw, 0.0)
                            mean_v = T.alloc_ub((1,), "float32")
                            rstd_v = T.alloc_ub((1,), "float32")
                            w_v = T.alloc_ub((1,), "float32")
                            T.copy(Mean[channel], mean_v)
                            T.copy(Rstd[channel], rstd_v)
                            T.copy(W[channel], w_v)
                            T.tile.fill(length_tile, float(reduction_spatial))
                            for chunk in T.serial(chunks):
                                start = chunk * tile
                                T.tile.fill(go, 0.0)
                                T.tile.fill(xv, 0.0)
                                if tail == 0 or chunk < chunks - 1:
                                    T.copy(GradOut[channel * spatial + start], go)
                                    T.copy(X[channel * spatial + start], xv)
                                else:
                                    if tail_vec:
                                        T.copy(GradOut[channel * spatial + start:
                                                       channel * spatial + start + tail_vec],
                                               go[0, 0:tail_vec])
                                        T.copy(X[channel * spatial + start:
                                                 channel * spatial + start + tail_vec],
                                               xv[0, 0:tail_vec])
                                    for j in T.serial(blk):
                                        if tail_vec + j < tail:
                                            go[0, tail_vec + j] = GradOut[
                                                channel * spatial + start + tail_vec + j]
                                            xv[0, tail_vec + j] = X[channel * spatial + start + tail_vec + j]
                                if need_cast:
                                    T.tile.cast(go32, go, "CAST_NONE", tile)
                                    T.tile.cast(x32, xv, "CAST_NONE", tile)
                                else:
                                    T.copy(go, go32)
                                    T.copy(xv, x32)
                                T.tile.broadcast(work, mean_v)
                                T.tile.sub(xhat, x32, work)
                                T.tile.broadcast(work, rstd_v)
                                T.tile.mul(xhat, xhat, work)
                                T.tile.mul(prod, go32, xhat)
                                T.reduce_sum(go32, part, dim=-1)
                                T.reduce_sum(prod, part2, dim=-1)
                                T.tile.add(sum_do, sum_do, part)
                                T.tile.add(sum_dw, sum_dw, part2)
                            T.copy(sum_do, GradB[channel])
                            T.copy(sum_dw, GradW[channel])
                            # --- pass 2: grad_x ------------------------------
                            # R347: this used to be a scalar loop over every
                            # element (two scalar GM loads + one scalar GM store
                            # each), which cost 1.6 ms on resnet50-stage1 while
                            # pass 1 above -- the same traffic, vectorised --
                            # costs microseconds.  Same arithmetic, same order,
                            # now one (1, tile) vector chain per chunk.
                            #
                            # UB census is unchanged: pass 2 reuses `xhat` as
                            # the broadcast mean and `prod` as the broadcast
                            # rstd (neither is live here), so the nine fp32
                            # (1, tile) buffers `_spatial_tile(cell_extra=36)`
                            # was sized for are exactly the nine used.
                            T.tile.broadcast(sum_do_tile, sum_do)
                            T.tile.broadcast(sum_dw_tile, sum_dw)
                            part[0] = w_v[0] * rstd_v[0] / float(reduction_spatial)
                            # Scalar write -> vector read of the same UB slot.  The
                            # in-tree precedent for this pair (kernels/scan.py's
                            # `carry_ub[0] = ...` then `T.tile.broadcast(carry_vec,
                            # carry_ub)`) puts an explicit barrier between them.
                            T.barrier_all()
                            T.tile.broadcast(scale_tile, part)
                            if single_chunk:
                                # The whole channel fitted in one tile, so pass 1 left
                                # cast(grad_out) in `go32` and the normalised
                                # (x - mean) * rstd in `xhat` -- bit-for-bit what the
                                # general path below recomputes.  Skipping the reload
                                # removes the entire second read of grad_out and x
                                # (resnet50-fc and resnet50-stage3 both have chunks == 1).
                                T.tile.mul(xhat, xhat, sum_dw_tile)
                                T.tile.mul(work, go32, length_tile)
                                T.tile.sub(work, work, sum_do_tile)
                                T.tile.sub(work, work, xhat)
                                T.tile.mul(work, work, scale_tile)
                                if need_cast:
                                    T.tile.cast(go, work, "CAST_RINT", tile)
                                else:
                                    T.copy(work, go)
                                T.copy(go[0, :], GradX[channel * spatial])
                                _drain_out_dma(False)
                            else:
                                T.tile.broadcast(xhat, mean_v)
                                T.tile.broadcast(prod, rstd_v)
                                full_chunks = chunks if tail == 0 else chunks - 1
                                for chunk in T.serial(full_chunks):
                                    start = chunk * tile
                                    T.copy(GradOut[channel * spatial + start], go)
                                    T.copy(X[channel * spatial + start], xv)
                                    if need_cast:
                                        T.tile.cast(go32, go, "CAST_NONE", tile)
                                        T.tile.cast(x32, xv, "CAST_NONE", tile)
                                    else:
                                        T.copy(go, go32)
                                        T.copy(xv, x32)
                                    # x32 <- xhat * sum_dw
                                    T.tile.sub(x32, x32, xhat)
                                    T.tile.mul(x32, x32, prod)
                                    T.tile.mul(x32, x32, sum_dw_tile)
                                    # work <- scale * (RS*go - sum_do - xhat*sum_dw)
                                    T.tile.mul(work, go32, length_tile)
                                    T.tile.sub(work, work, sum_do_tile)
                                    T.tile.sub(work, work, x32)
                                    T.tile.mul(work, work, scale_tile)
                                    if need_cast:
                                        T.tile.cast(go, work, "CAST_RINT", tile)
                                    else:
                                        T.copy(work, go)
                                    T.copy(go[0, :], GradX[channel * spatial + start])
                                    # `go` is both the MTE2 landing buffer and the MTE3
                                    # source here, so the next iteration's reload is a
                                    # write-after-read against a store that may still be
                                    # in flight.  AUTO_SYNC is documented to emit this
                                    # MTE3->MTE2 drain for a writeback at the end of a
                                    # loop body (see _drain_out_dma); it is emitted
                                    # explicitly anyway because the failure mode is
                                    # silent wrong data, not a crash.
                                    _drain_out_dma(False)
                                if tail != 0:
                                    start = (chunks - 1) * tile
                                    for j in T.serial(tile):
                                        if j < tail:
                                            idx = channel * spatial + start + j
                                            go_val = T.cast(GradOut[idx], "float32")
                                            x_val = T.cast(X[idx], "float32")
                                            xh_val = (x_val - mean_v[0]) * rstd_v[0]
                                            gx_val = (w_v[0] * rstd_v[0] / float(reduction_spatial)) * (
                                                float(reduction_spatial) * go_val - sum_do[0] - xh_val * sum_dw[0]
                                            )
                                            GradX[idx] = T.cast(gx_val, dtype_name)

        return main

    return factory()



def _bn_bwd_arm() -> str:
    """T357's control-arm switch.  ``base`` reproduces the pre-T357 tree exactly."""
    return os.environ.get("TILEOPS_T357_ARM", "r357")


@lru_cache(maxsize=128)
def _compile_batch_bwd_nchw(n_batch: int, c: int, s: int, dtype_name: str):
    """R357.  The spatial BatchNorm backward, reading ``[N, C, S]`` WHERE IT LIES.

    ``_compile_batch_bwd`` above consumes a channel-major ``[C, N*S]`` view, so
    ``launch_batch_backward`` had to build one: ``grad_out.permute(1,0,2,3)
    .contiguous()``, the same for ``x``, and then ``grad_x.permute(1,0,2,3)
    .contiguous()`` on the way out.  Those are THREE extra device kernels and
    ``2 x nbytes`` of HBM traffic each.  Measured share on the manifest's
    ``large-spatial`` workload -- see R357 section 4.

    The FORWARD twin ``_compile_batch`` never paid it: it indexes
    ``A[(batch * c + channel) * spatial + start]`` and walks ``for batch in
    T.serial(n)``.  This function is that indexing applied to the backward's two
    passes; the arithmetic, the fp32 reduction state and the order of the vector
    chain are unchanged from ``_compile_batch_bwd``.

    ``vid``: ``_compile_batch_bwd`` computes ``channel = cid + repeat *
    launch_blocks``, i.e. it never reads the second element of ``T.Kernel(...) as
    (cid, vid)``, so both vector units of an AI core run the SAME channel and
    issue the same loads and the same stores.  ``channel = block * _VEC + vid`` is
    what every other kernel in this file does (``_compile_group`` :241,
    ``_compile_group_resident`` :582, ``_compile_instance_infer`` :812,
    ``_compile_batch`` :960) and what R347 fixed in ``kernels/scan.py``.
    """
    launch_blocks, grid_repeats = _launch_tasks(c)
    # --- Geometry ------------------------------------------------------
    # (1, tile) UB census, identical to _compile_batch_bwd: go[itemsize]
    # xv[itemsize] + 9 fp32 buffers (go32 x32 work xhat prod sum_do_tile
    # sum_dw_tile scale_tile length_tile) = 2*itemsize + 36 bytes per element.
    # ``s`` -- ONE (n, c) plane -- is what the tile must cover here, not the whole
    # channel: a chunk may not straddle a plane boundary, because two planes of the
    # same channel are ``c * s`` elements apart in GM.
    #
    # ⚠️ ENTRY CONDITION, enforced by `launch_batch_backward`: ``s % SPATIAL_ELEM_GRAIN
    # == 0`` and ``s >= SPATIAL_ELEM_GRAIN``.  That is what makes the ragged chunk here
    # SAFE as a plain `T.copy`: `tile` is then also a multiple of 16, so `tail = s %
    # tile` is a multiple of 16 and `tail * itemsize` is a multiple of 32 for both fp16
    # (2 B) and fp32 (4 B).  TRAP 1 of the RAGGED-CHUNK block above -- a partial-extent
    # GM<->UB copy moving whole 32-byte blocks and reading/writing past the extent --
    # cannot fire on a 32-byte-aligned length.  A shape that fails the entry condition
    # keeps the channel-major `_compile_batch_bwd` path, permutes and all.
    tile = min(_spatial_tile(s, dtype_name, cell_extra=36), s)
    chunks = math.ceil(s / tile)
    tail = s % tile
    need_cast = dtype_name != "float32"
    # Pass 2 can read pass 1's leftovers out of UB only when the whole channel was ONE
    # chunk, which now also requires a single plane.
    single_chunk = n_batch == 1 and chunks == 1 and tail == 0
    reduction = n_batch * s

    @tilelang.jit(
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            GradOut: T.Tensor((n_batch * c * s,), dtype_name),
            X: T.Tensor((n_batch * c * s,), dtype_name),
            W: T.Tensor((c,), "float32"),
            Mean: T.Tensor((c,), "float32"),
            Rstd: T.Tensor((c,), "float32"),
            GradX: T.Tensor((n_batch * c * s,), dtype_name),
            GradW: T.Tensor((c,), "float32"),
            GradB: T.Tensor((c,), "float32"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                go = T.alloc_ub((1, tile), dtype_name)
                xv = T.alloc_ub((1, tile), dtype_name)
                go32 = T.alloc_ub((1, tile), "float32")
                x32 = T.alloc_ub((1, tile), "float32")
                work = T.alloc_ub((1, tile), "float32")
                xhat = T.alloc_ub((1, tile), "float32")
                prod = T.alloc_ub((1, tile), "float32")
                sum_do_tile = T.alloc_ub((1, tile), "float32")
                sum_dw_tile = T.alloc_ub((1, tile), "float32")
                scale_tile = T.alloc_ub((1, tile), "float32")
                length_tile = T.alloc_ub((1, tile), "float32")
                sum_do = T.alloc_ub((1,), "float32")
                sum_dw = T.alloc_ub((1,), "float32")
                part = T.alloc_ub((1,), "float32")
                part2 = T.alloc_ub((1,), "float32")
                with T.Scope("V"):
                    for repeat in T.serial(grid_repeats):
                        block = cid + repeat * launch_blocks
                        channel = block * _VEC + vid
                        if channel < c:
                            T.tile.fill(sum_do, 0.0)
                            T.tile.fill(sum_dw, 0.0)
                            mean_v = T.alloc_ub((1,), "float32")
                            rstd_v = T.alloc_ub((1,), "float32")
                            w_v = T.alloc_ub((1,), "float32")
                            T.copy(Mean[channel], mean_v)
                            T.copy(Rstd[channel], rstd_v)
                            T.copy(W[channel], w_v)
                            T.tile.fill(length_tile, float(reduction))
                            # R357: `mean_v` / `rstd_v` are loop-invariant in (batch,
                            # chunk), but `_compile_batch_bwd` broadcast them into `work`
                            # INSIDE the chunk loop -- two full (1, tile) vector ops per
                            # chunk, 2048 chunks per channel on `large-spatial`.  Pass 2
                            # below already hoists its own pair; this is the same move for
                            # pass 1, and it is bit-exact (same value in every lane).
                            # `sum_do_tile` / `sum_dw_tile` are dead until pass 2
                            # re-broadcasts them, so no buffer is added.
                            T.tile.broadcast(sum_do_tile, mean_v)
                            T.tile.broadcast(sum_dw_tile, rstd_v)
                            for batch in T.serial(n_batch):
                                for chunk in T.serial(chunks):
                                    start = chunk * tile
                                    base = (batch * c + channel) * s + start
                                    if tail == 0 or chunk < chunks - 1:
                                        T.copy(GradOut[base], go)
                                        T.copy(X[base], xv)
                                    else:
                                        # A full chunk overwrites all `tile` lanes, so
                                        # the zero-fill `_compile_batch_bwd` did on every
                                        # chunk belongs to the ragged branch only -- that
                                        # is the only place lanes past `tail` can reach
                                        # `T.reduce_sum`.
                                        T.tile.fill(go, 0.0)
                                        T.tile.fill(xv, 0.0)
                                        T.copy(GradOut[base: base + tail], go[0, 0:tail])
                                        T.copy(X[base: base + tail], xv[0, 0:tail])
                                    if need_cast:
                                        T.tile.cast(go32, go, "CAST_NONE", tile)
                                        T.tile.cast(x32, xv, "CAST_NONE", tile)
                                    else:
                                        T.copy(go, go32)
                                        T.copy(xv, x32)
                                    T.tile.sub(xhat, x32, sum_do_tile)   # - mean
                                    T.tile.mul(xhat, xhat, sum_dw_tile)  # * rstd
                                    T.tile.mul(prod, go32, xhat)
                                    T.reduce_sum(go32, part, dim=-1)
                                    T.reduce_sum(prod, part2, dim=-1)
                                    T.tile.add(sum_do, sum_do, part)
                                    T.tile.add(sum_dw, sum_dw, part2)
                            T.copy(sum_do, GradB[channel])
                            T.copy(sum_dw, GradW[channel])
                            # --- pass 2: grad_x ------------------------------
                            T.tile.broadcast(sum_do_tile, sum_do)
                            T.tile.broadcast(sum_dw_tile, sum_dw)
                            part[0] = w_v[0] * rstd_v[0] / float(reduction)
                            # Scalar write -> vector read of the same UB slot; the
                            # in-tree precedent (kernels/scan.py) puts an explicit
                            # barrier between them.
                            T.barrier_all()
                            T.tile.broadcast(scale_tile, part)
                            if single_chunk:
                                T.tile.mul(xhat, xhat, sum_dw_tile)
                                T.tile.mul(work, go32, length_tile)
                                T.tile.sub(work, work, sum_do_tile)
                                T.tile.sub(work, work, xhat)
                                T.tile.mul(work, work, scale_tile)
                                if need_cast:
                                    T.tile.cast(go, work, "CAST_RINT", tile)
                                else:
                                    T.copy(work, go)
                                T.copy(go[0, :], GradX[channel * s])
                                _drain_out_dma(False)
                            else:
                                T.tile.broadcast(xhat, mean_v)
                                T.tile.broadcast(prod, rstd_v)
                                for batch in T.serial(n_batch):
                                    for chunk in T.serial(chunks):
                                        start = chunk * tile
                                        base = (batch * c + channel) * s + start
                                        if tail == 0 or chunk < chunks - 1:
                                            T.copy(GradOut[base], go)
                                            T.copy(X[base], xv)
                                        else:
                                            T.tile.fill(go, 0.0)
                                            T.tile.fill(xv, 0.0)
                                            T.copy(GradOut[base: base + tail], go[0, 0:tail])
                                            T.copy(X[base: base + tail], xv[0, 0:tail])
                                        if need_cast:
                                            T.tile.cast(go32, go, "CAST_NONE", tile)
                                            T.tile.cast(x32, xv, "CAST_NONE", tile)
                                        else:
                                            T.copy(go, go32)
                                            T.copy(xv, x32)
                                        # x32 <- xhat * sum_dw
                                        T.tile.sub(x32, x32, xhat)
                                        T.tile.mul(x32, x32, prod)
                                        T.tile.mul(x32, x32, sum_dw_tile)
                                        # work <- scale * (RS*go - sum_do - xhat*sum_dw)
                                        T.tile.mul(work, go32, length_tile)
                                        T.tile.sub(work, work, sum_do_tile)
                                        T.tile.sub(work, work, x32)
                                        T.tile.mul(work, work, scale_tile)
                                        if need_cast:
                                            T.tile.cast(go, work, "CAST_RINT", tile)
                                        else:
                                            T.copy(work, go)
                                        if tail == 0 or chunk < chunks - 1:
                                            T.copy(go[0, :], GradX[base])
                                        else:
                                            # 32-byte-aligned length (see the ENTRY
                                            # CONDITION note), so this partial-extent
                                            # store cannot spill into the next plane.
                                            T.copy(go[0, 0:tail], GradX[base: base + tail])
                                        # `go` is both the MTE2 landing buffer and the
                                        # MTE3 source, so the next iteration's reload is
                                        # a write-after-read against a store that may
                                        # still be in flight.
                                        _drain_out_dma(False)

        return main

    return factory()


def launch_batch_backward(grad_out, x, weight, mean, rstd):
    shape = tuple(int(v) for v in x.shape)
    c = shape[1]
    if _bn_bwd_arm() != "base":
        # R357: no channel-major copy in, no permute out.  See
        # `_compile_batch_bwd_nchw`.  `.contiguous()` is a no-op on a contiguous
        # tensor and `.reshape(-1)` is a view, so nothing is materialised here.
        n_batch = shape[0]
        s = math.prod(shape[2:]) if len(shape) > 2 else 1
        # ENTRY CONDITION for `_compile_batch_bwd_nchw`: one (n, c) plane must be a whole
        # number of 16-element groups, so that every GM<->UB extent the kernel uses is
        # 32-byte aligned for fp16 AND fp32.  The manifest's rank-2 `[32, 64]` workload
        # (s == 1) fails it and keeps the channel-major path.
        if s >= SPATIAL_ELEM_GRAIN and s % SPATIAL_ELEM_GRAIN == 0:
            kernel = _compile_batch_bwd_nchw(n_batch, c, s, _dtype_name(x.dtype))
            gx = torch.empty((n_batch * c * s,), device=x.device, dtype=x.dtype)
            gw = torch.empty((c,), device=x.device, dtype=torch.float32)
            gb = torch.empty((c,), device=x.device, dtype=torch.float32)
            kernel(grad_out.contiguous().reshape(-1), x.contiguous().reshape(-1),
                   weight.contiguous(), mean.contiguous(), rstd.contiguous(), gx, gw, gb)
            return gx.reshape(shape), gw, gb
    spatial = math.prod(shape) // c
    dtype_name = _dtype_name(x.dtype)
    tile = _spatial_tile(spatial, dtype_name, cell_extra=36)
    padded_spatial = math.ceil(spatial / tile) * tile
    kernel = _compile_batch_bwd(c, padded_spatial, dtype_name, spatial, tile=tile)
    gx = torch.empty((c * padded_spatial,), device=x.device, dtype=x.dtype)
    gw = torch.empty((c,), device=x.device, dtype=torch.float32)
    gb = torch.empty((c,), device=x.device, dtype=torch.float32)
    # The kernel consumes channel-major [C, N*spatial] storage, matching the
    # in-tree BatchNorm kernel's _to_cl layout conversion.
    axes = (1, 0, *range(2, len(shape)))
    if padded_spatial == spatial:
        # R347: `padded_spatial = ceil(spatial / tile) * tile` and the kernel is
        # compiled with that padded extent, so on this call path the two are
        # ALWAYS equal and the zero-fill + slice-assign below were a pure extra
        # `ZerosLike` + full-size `TensorMove` per input (4 device kernels).
        # Kept as the `else` branch for any caller that does pad.
        flat_go = grad_out.contiguous().permute(axes).contiguous().reshape(-1)
        flat_x = x.contiguous().permute(axes).contiguous().reshape(-1)
    else:
        flat_go = torch.zeros((c, padded_spatial), device=x.device, dtype=x.dtype)
        flat_x = torch.zeros_like(flat_go)
        flat_go[:, :spatial] = grad_out.contiguous().permute(axes).contiguous().reshape(c, spatial)
        flat_x[:, :spatial] = x.contiguous().permute(axes).contiguous().reshape(c, spatial)
        flat_go = flat_go.reshape(-1)
        flat_x = flat_x.reshape(-1)
    kernel(flat_go, flat_x, weight.contiguous(), mean.contiguous(), rstd.contiguous(), gx, gw, gb)
    gx = gx.reshape((c, shape[0], *shape[2:]) if padded_spatial == spatial else (c, padded_spatial))
    if padded_spatial != spatial:
        gx = gx[:, :spatial].reshape((c, shape[0], *shape[2:]))
    gx = gx.permute((1, 0, *range(2, len(shape)))).contiguous()
    return gx, gw, gb
