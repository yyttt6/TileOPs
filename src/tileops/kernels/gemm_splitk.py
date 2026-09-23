"""Split-K dense GEMM (T267, op-list-150.pdf #113).

The Cube half is the pipelined kernel from :mod:`tileops.kernels.gemm`
(``_build_gemm_pipelined``) with the block index extended from
``(m_tile, n_tile)`` to ``(m_tile, n_tile, k_chunk)``: every chunk accumulates
its own slice of the contraction in fp32 L0C and the fixpipe writes that
partial straight out to an fp32 GM workspace ``[split_k, m, n]``.  A second,
vector-only launch reduces the workspace down the ``split_k`` axis and casts
back to the operand dtype.

WHY THE PLANNER GATES ON BLOCK COUNT, NOT ON K.

Split-K buys occupancy and pays HBM traffic, so it is only ever a win when the
Cube is starved for blocks.  The geometry of the twelve GemmFwdOp workloads
(docs/reports/R264-data/01-shipped-geometry.txt) says which those are:

  * ``ds-v3-decode-gate-up`` plans NINE logical blocks against ~24 AI cores --
    55% of the machine is idle for the whole kernel.  ``mid-m16-attn`` and
    ``mid-m32-attn`` plan sixteen.  There, ``split_k=2`` costs 1-2 MB of extra
    traffic, which is nothing.
  * ``k-dominant-7168x16384`` already plans 896 blocks.  It is the most
    K-dominant workload in the set and therefore the one a "split when K is
    large" rule would pick first -- and ``split_k=2`` there would write and
    re-read ``2 * 4096 * 7168 * 4 = 235 MB``, about +220 us at 1.07 TB/s.  A
    guaranteed loss.

So :func:`_split_plan` splits only while ``logical_blocks * split_k`` stays at
or under ``2 * AI_CORES``, and returns ``split_k == 1`` -- i.e. "just call the
ordinary GEMM, one launch, no workspace" -- for everything else.

⚠️ ``AI_CORES`` is an ESTIMATE.  hw/ascend910b1.yaml derives its fp16 peak from
25 cores (R013), tilelang-ascend's own examples use 20 and 24, and CANN's
MatMulV3 runs at ``Block Dim = 24`` (R264 section 1).  24 is used here.  The
gate is a threshold on a quantity that is only known to +-20%, and the report
for this round lists which workloads land on each side of it.
"""

from functools import lru_cache
from typing import Callable

import tilelang
import tilelang.language as T

from .common import (
    LAUNCH_BLOCK_CAP,
    TILE_GRAIN,
    UB_BUDGET_BYTES,
    grid_repeat_count,
    launch_block_count,
)
from .gemm import (
    L0_STAGES,
    L1_STAGES,
    _dtype_name,
    _pipelined_plan,
    build_gemm_kernel,
)

#: Cube count used by the split-K gate.  See the module docstring: an estimate.
AI_CORES = 24

#: Number of AIV lanes the reduction kernel splits each block across.
_REDUCE_VEC = 2


def _split_plan(m: int, n: int, k: int, itemsize: int = 2):
    """Return ``(split_k, block_m, block_n, block_k, kl0)``, or ``None``.

    ``None`` means "this shape has no pipelined plan at all"; the caller falls
    back to :func:`tileops.kernels.gemm.build_gemm_kernel`, which is always
    correct.  ``split_k == 1`` means "a plan exists but splitting K is not
    worth it", and the caller does the same thing.

    Every condition is a capacity or divisibility fact:

    * ``k_tiles % split_k`` -- each chunk owns ``k_tiles // split_k`` whole K
      tiles, so the split has to divide the tile count evenly.  A ragged last
      chunk would need a second code path for no measurable gain on any
      workload in the set.
    * ``logical_blocks * split_k <= 2 * AI_CORES`` -- the occupancy gate.
    """
    plan = _pipelined_plan(m, n, k, itemsize)
    if plan is None:
        return None
    block_m, block_n, block_k, kl0 = plan
    k_tiles = k // block_k
    logical_blocks = -(-m // block_m) * -(-n // block_n)
    split_k = 1
    while (
        split_k * 2 <= k_tiles
        and k_tiles % (split_k * 2) == 0
        and logical_blocks * split_k * 2 <= 2 * AI_CORES
    ):
        split_k *= 2
    return split_k, block_m, block_n, block_k, kl0


@lru_cache(maxsize=32)
def _build_splitk_partials(m: int, n: int, k: int, dtype_name: str, trans_b: bool,
                           split_k: int) -> Callable:
    """The Cube launch: one fp32 partial product per ``(m_tile, n_tile, chunk)``.

    Identical to :func:`tileops.kernels.gemm._build_gemm_pipelined` except that
    the K loop covers one chunk instead of the whole contraction and the
    readout target is an fp32 workspace slice rather than the output.  All three
    codegen constraints that function documents apply here unchanged:

    * the flag initialisation is emitted AFTER the prologue copy, because a
      ``T.set_flag`` written as the first statement of a ``T.Scope("C")`` body
      is dropped by the compiler and the resulting kernel hangs at AICore=100%
      with no diagnostic;
    * the prologue is ``L1_STAGES - 1`` deep and does not wait, so "exactly one
      set between two waits on an id" holds by construction for a one-shot
      hardware event flag;
    * the ``M -> FIX`` pair before the readout is not optional; a
      ``PipeBarrier<ALL>`` in its place produced "FIXP instruction error: ECC
      verification failed when the l0c is read".

    (R264 section 3; raw data in R264-data/08-hs-deadlock.txt.)
    """
    plan = _pipelined_plan(m, n, k, 2)
    if plan is None:  # pragma: no cover - build_gemm_splitk_kernel checks first
        raise ValueError(f"no pipelined plan for m={m} n={n} k={k}")
    block_m, block_n, block_k, kl0 = plan
    s1, s2 = L1_STAGES, L0_STAGES
    b_shape = (n, k) if trans_b else (k, n)
    b_l1_shape = (block_n, block_k) if trans_b else (block_k, block_n)
    m_tiles = (m + block_m - 1) // block_m
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = k // block_k
    chunk_tiles = k_tiles // split_k
    kk_steps = block_k // kl0
    mn_tiles = m_tiles * n_tiles
    logical_blocks = mn_tiles * split_k
    launch_blocks = launch_block_count(logical_blocks)
    grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _factory(dtype: str = dtype_name) -> Callable:

        @T.prim_func
        def main(
            a: T.Tensor((m, k), dtype),
            b: T.Tensor(b_shape, dtype),
            partials: T.Tensor((split_k, m, n), "float"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, _):
                for grid_repeat in T.serial(grid_repeats):
                    logical_cid = cid + grid_repeat * launch_blocks
                    if logical_cid < logical_blocks:
                        # k_chunk is the SLOWEST-moving index so that the
                        # m_tile/n_tile order inside one chunk matches the dense
                        # kernel's, and consecutive blocks still walk n first.
                        chunk = logical_cid // mn_tiles
                        mn_idx = logical_cid % mn_tiles
                        m0 = (mn_idx // n_tiles) * block_m
                        n0 = (mn_idx % n_tiles) * block_n
                        k_origin = chunk * chunk_tiles * block_k

                        a_l1 = T.alloc_L1((s1, block_m, block_k), dtype)
                        b_l1 = T.alloc_L1((s1,) + b_l1_shape, dtype)
                        a_l0 = T.alloc_L0A((s2, block_m, kl0), dtype)
                        b_l0 = T.alloc_L0B((s2, kl0, block_n), dtype)
                        c_l0 = T.alloc_L0C((block_m, block_n), "float")

                        with T.Scope("C"):
                            T.copy(a[m0:m0 + block_m,
                                     k_origin:k_origin + block_k], a_l1[0, :, :])
                            if trans_b:
                                T.copy(b[n0:n0 + block_n,
                                         k_origin:k_origin + block_k],
                                       b_l1[0, :, :])
                            else:
                                T.copy(b[k_origin:k_origin + block_k,
                                         n0:n0 + block_n], b_l1[0, :, :])
                            T.set_flag("mte2", "mte1", 0)
                            T.set_flag("mte1", "mte2", 1)
                            T.set_flag("m", "mte1", 0)
                            T.set_flag("m", "mte1", 1)

                            for kt in T.serial(chunk_tiles):
                                nxt = kt + 1
                                if nxt < chunk_tiles:
                                    T.wait_flag("mte1", "mte2", nxt % s1)
                                    k1 = k_origin + nxt * block_k
                                    T.copy(a[m0:m0 + block_m, k1:k1 + block_k],
                                           a_l1[nxt % s1, :, :])
                                    if trans_b:
                                        T.copy(b[n0:n0 + block_n, k1:k1 + block_k],
                                               b_l1[nxt % s1, :, :])
                                    else:
                                        T.copy(b[k1:k1 + block_k, n0:n0 + block_n],
                                               b_l1[nxt % s1, :, :])
                                    T.set_flag("mte2", "mte1", nxt % s1)
                                T.wait_flag("mte2", "mte1", kt % s1)

                                for kk in T.serial(kk_steps):
                                    T.wait_flag("m", "mte1", kk % s2)
                                    T.copy(
                                        a_l1[kt % s1, :, kk * kl0:kk * kl0 + kl0],
                                        a_l0[kk % s2, :, :],
                                    )
                                    if trans_b:
                                        T.copy(
                                            b_l1[kt % s1, :, kk * kl0:kk * kl0 + kl0],
                                            b_l0[kk % s2, :, :],
                                            transpose=True,
                                        )
                                    else:
                                        T.copy(
                                            b_l1[kt % s1, kk * kl0:kk * kl0 + kl0, :],
                                            b_l0[kk % s2, :, :],
                                        )
                                    T.set_flag("mte1", "m", kk % s2)
                                    T.wait_flag("mte1", "m", kk % s2)
                                    T.mma(
                                        a_l0[kk % s2, :, :],
                                        b_l0[kk % s2, :, :],
                                        c_l0,
                                        init=T.And(kt == 0, kk == 0),
                                    )
                                    T.set_flag("m", "mte1", kk % s2)

                                T.set_flag("mte1", "mte2", kt % s1)

                            T.wait_flag("mte1", "mte2", 0)
                            T.wait_flag("mte1", "mte2", 1)
                            T.wait_flag("m", "mte1", 0)
                            T.wait_flag("m", "mte1", 1)
                            T.set_flag("m", "fix", 0)
                            T.wait_flag("m", "fix", 0)
                            T.copy(c_l0, partials[chunk, m0:m0 + block_m,
                                                  n0:n0 + block_n])
                            T.set_flag("fix", "m", 0)
                            T.wait_flag("fix", "m", 0)

        return main

    return _factory(dtype_name)


@lru_cache(maxsize=32)
def _build_splitk_reduce(m: int, n: int, dtype_name: str, split_k: int) -> Callable:
    """The vector launch: sum the ``split_k`` fp32 partials, cast to ``dtype``.

    The output is walked as a flat ``m * n`` vector, which is why the partials
    workspace is exactly ``[split_k, m, n]`` and not a tile-padded shape: a flat
    index ``i`` into one partial is ``chunk * m * n + i`` with no per-row stride
    correction.  A ragged trailing tile is handled by ``T.copy``'s own bounds
    handling, the same way the dense kernel's ragged n tile is
    (``ds-v3-decode-gate-up`` has ``n = 2112`` against ``block_n = 256``).
    """
    total = m * n
    # Three UB buffers are live per lane: the fp32 accumulator, the fp32 tile
    # being read, and the dtype tile being written.
    per_element = 4 + 4 + 2
    tile = min(total, (UB_BUDGET_BYTES // per_element) // TILE_GRAIN * TILE_GRAIN)
    tile = max(TILE_GRAIN, tile)
    logical_blocks = max(1, -(-total // (tile * _REDUCE_VEC)))
    # Capped like every other vector template in this tree: blockDim above the
    # 48 vector contexts is time-sliced by the runtime (PROJECT_STATE 13.57), so
    # the extra blocks buy nothing and each one re-pays its own prologue.  The
    # epilogue kernel measured 61.88 us uncapped against a 21 us GEMM on
    # square-1k-nn (R264-data/29-epi-launch-cap.txt).
    launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
    grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _factory(dtype: str = dtype_name) -> Callable:

        @T.prim_func
        def main(
            partials: T.Tensor((split_k, m, n), "float"),
            c: T.Tensor((m, n), dtype),
        ):
            flat_partials = T.Tensor((split_k * total,), "float", partials.data)
            flat_out = T.Tensor((total,), dtype, c.data)
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                accum = T.alloc_ub((tile,), "float")
                staged = T.alloc_ub((tile,), "float")
                result = T.alloc_ub((tile,), dtype)
                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            start = (logical_cid * _REDUCE_VEC + vid) * tile
                            if start < total:
                                T.copy(flat_partials[start:start + tile], accum)
                                for chunk in T.serial(split_k - 1):
                                    base = (chunk + 1) * total + start
                                    T.copy(flat_partials[base:base + tile], staged)
                                    T.tile.add(accum, accum, staged)
                                if dtype == "float32":
                                    T.copy(accum, flat_out[start:start + tile])
                                else:
                                    # dav-2201 has no scalar bf16 cast, so the
                                    # narrowing is done a whole tile at a time
                                    # (same constraint as convolution.py).
                                    T.tile.cast(result, accum, "CAST_RINT", tile)
                                    T.copy(result, flat_out[start:start + tile])

        return main

    return _factory(dtype_name)


def build_gemm_splitk_kernel(
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    dtype,
    trans_a: bool,
    trans_b: bool,
) -> Callable:
    """Build a ``gemm_splitk`` callable for one shape.

    Returns a two-launch callable when :func:`_split_plan` decides to split, and
    ``build_gemm_kernel``'s single launch otherwise.  ``launch.split_k`` records
    which happened, so a benchmark can report how many workloads actually took
    the split path rather than assuming they all did.
    """
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError(f"gemm_splitk expects 2D inputs, got a={a_shape}, b={b_shape}")
    if trans_a:
        raise ValueError(
            "gemm_splitk does not support trans_a=True: the split-K partial "
            "kernel is the non-transposed-A pipelined path only, and trans_a is "
            "False in all twelve GemmFwdOp manifest workloads"
        )
    m, k_a = a_shape
    n = b_shape[0] if trans_b else b_shape[1]
    k_b = b_shape[1] if trans_b else b_shape[0]
    if k_a != k_b:
        raise ValueError(f"gemm_splitk contraction mismatch: a={a_shape}, b={b_shape}")
    name = _dtype_name(dtype)
    plan = _split_plan(m, n, k_a, 2)
    split_k = 1 if plan is None else plan[0]

    if split_k == 1:
        dense = build_gemm_kernel(a_shape, b_shape, dtype, trans_a, trans_b)

        def launch(a, b):
            return dense(a, b)

        launch.split_k = 1
        launch.path_kind = "dense_no_split"
        launch.launches_per_call = 1
        launch.reason = (
            "not split: no pipelined plan" if plan is None else
            f"not split: {-(-m // plan[1]) * -(-n // plan[2])} logical blocks "
            f"already at or above 2 * {AI_CORES} cores"
        )
        launch.compiled = dense
        return launch

    _, block_m, block_n, block_k, _ = plan
    partials = _build_splitk_partials(m, n, k_a, name, bool(trans_b), split_k)
    reduce_kernel = _build_splitk_reduce(m, n, name, split_k)

    def launch(a, b):
        return reduce_kernel(partials(a, b))

    launch.split_k = split_k
    launch.path_kind = "splitk_two_launch"
    launch.launches_per_call = 2
    launch.logical_blocks = -(-m // block_m) * -(-n // block_n) * split_k
    launch.dense_logical_blocks = -(-m // block_m) * -(-n // block_n)
    launch.workspace_shape = (split_k, m, n)
    launch.workspace_bytes = split_k * m * n * 4
    launch.reason = (
        f"split: {-(-m // block_m) * -(-n // block_n)} dense logical blocks is "
        f"below 2 * {AI_CORES} cores; split_k={split_k} raises it to "
        f"{-(-m // block_m) * -(-n // block_n) * split_k}"
    )
    launch.compiled = (partials, reduce_kernel)
    return launch


__all__ = ["build_gemm_splitk_kernel"]
