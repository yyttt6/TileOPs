"""NCHW spatial normalization kernels.

The reduction is deliberately expressed over the physical channel/spatial
coordinates.  Flattening an NCHW tensor to rows would change the reduction
domain for GroupNorm and InstanceNorm, so each logical group/channel owns one
tile task and walks its strided channel slices.  Mean and M2 use fp32 tile
Welford state throughout; no second-moment subtraction is used.
"""

from functools import lru_cache
import math

import tilelang
import tilelang.language as T
import torch

from .common import (
    SPATIAL_ELEM_GRAIN,
    TILE_WIDTH_CAP,
    UB_BUDGET_BYTES,
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


def launch_ada_norm(x, scale, shift, gate, *, eps: float):
    shape = tuple(int(v) for v in x.shape)
    n = shape[-1]
    m = math.prod(shape) // n
    kernel = _compile_ada_norm(m, n, _dtype_name(x.dtype), gate is not None, float(eps))
    gate_arg = shift if gate is None else gate
    y = kernel(
        x.contiguous().reshape(m, n),
        scale.contiguous().reshape(m, n),
        shift.contiguous().reshape(m, n),
        gate_arg.contiguous().reshape(m, n),
    )
    return y[:m, :n].reshape(shape)


def launch_group(x, weight, bias, *, groups: int, eps: float):
    shape = tuple(int(v) for v in x.shape)
    n, c, spatial_shape = shape[0], shape[1], shape[2:]
    spatial = math.prod(spatial_shape)
    cpg = c // groups
    tile = _spatial_tile(spatial, _dtype_name(x.dtype), cell_extra=8)
    padded = math.ceil(spatial / tile) * tile
    kernel = _compile_group(n, c, spatial, groups, cpg, _dtype_name(x.dtype), weight is not None, float(eps))
    out = torch.empty((n * c * padded,), device=x.device, dtype=x.dtype)
    if weight is None:
        weight = torch.ones((c,), device=x.device, dtype=x.dtype)
        bias = torch.zeros((c,), device=x.device, dtype=x.dtype)
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
        weight = torch.ones((c,), device=x.device, dtype=x.dtype)
        bias = torch.zeros((c,), device=x.device, dtype=x.dtype)
    kernel(
        x.contiguous().reshape(-1),
        out,
        running_mean.contiguous(),
        running_var.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
    )
    return out.reshape(n, c, padded)[:, :, :spatial].reshape(shape)


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
                            full_chunks = chunks if tail == 0 else chunks - 1
                            for chunk in T.serial(full_chunks):
                                start = chunk * tile
                                for j in T.serial(tile):
                                    idx = channel * spatial + start + j
                                    go_val = T.cast(GradOut[idx], "float32")
                                    x_val = T.cast(X[idx], "float32")
                                    xh_val = (x_val - mean_v[0]) * rstd_v[0]
                                    gx_val = (w_v[0] * rstd_v[0] / float(reduction_spatial)) * (
                                        float(reduction_spatial) * go_val - sum_do[0] - xh_val * sum_dw[0]
                                    )
                                    GradX[idx] = T.cast(gx_val, dtype_name)
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


def launch_batch_backward(grad_out, x, weight, mean, rstd):
    shape = tuple(int(v) for v in x.shape)
    c = shape[1]
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
