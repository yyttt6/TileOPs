"""GEMM with a fused bias / bias+ReLU / bias+GELU epilogue (T264, ops 110-112).

The Cube half is the same pipelined kernel as :mod:`tileops.kernels.gemm`
(``_build_gemm_pipelined``): an asynchronous, 2-deep GM->L1 stage in front of
``T.mma``, because ``T.gemm_v0`` blocks the MTE2 pipe until its own MMADs
retire and therefore cannot overlap a prefetch (see that module's docstring and
docs/reports/R264.md section 3).  The epilogue rides on the vector cores of the
SECOND launch: the Cube kernel writes the product, then a vector-only pass adds
the bias and applies the activation.

Why not one fused launch: see ``build_gemm_epilogue_kernel``'s docstring and
docs/reports/R264-data/12-fused-epilogue.txt.  This backend's ``T.copy`` also
has no ``wmma.accumulator -> shared.ub`` lowering (only ``-> global``, see
tilelang-ascend/src/op/ascend.cc's scope dispatch), so even a fused version has
to go through GM.

THE BIAS BROADCAST DOES NOT GO THROUGH THE ELEMENTWISE GATHER PATH.
R263/PROJECT_STATE 13.115 measured ``elementwise_binary.py``'s generic mode --
a per-lane ``_offset_expr`` gather -- running at 6.3-18.4 GB/s against
355-1010 GB/s for the vector path, a 30-150x gap, and that is what a 1-D
last-axis bias used to hit.  This kernel never reaches that module.  It uses
R263's fix directly: the bias row is staged into UB once per block, replicated
down the chunk, and the add is a plain full-tile ``T.tile.add`` over two
same-shaped UB buffers.  No lane-indexed read of ``bias`` is emitted at all.
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
from .gemm import _dtype_name, build_gemm_kernel

#: The epilogues this module builds.  ``bias`` is op 110 of op-list-150.pdf,
#: ``bias_relu`` 111, ``bias_gelu`` 112.
EPILOGUE_KINDS = ("bias", "bias_relu", "bias_gelu")

#: Vector contexts per launch block, as the elementwise templates in this tree
#: use it: one AIV pair per block, each taking half of ``block_total``.
VEC_PER_BLOCK = 2

#: Reserved on top of the declared buffers.  R194/R200 both measured that
#: "the declared buffers summing under budget" is not the same as fitting:
#: ``T.tile.*`` carries scratch that never appears in the allocation list.  A
#: first attempt here without this margin died at
#: ``ascend_memory_planning.cc:767 Memory allocation failed for: y32 required:
#: 49152, new memory available: 48896`` -- 256 bytes short.
UB_SCRATCH_RESERVE_BYTES = 8192


def _staging_arm() -> str:
    """Which bias staging this process builds.  ``r264`` is the control arm.

    R356 measured that the T264 staging -- ``rows`` separate GM reads of the SAME
    ``n``-element bias row, once per VECTOR CONTEXT -- is what this kernel's time
    actually goes on, not the output's GM round trip:

      square-1k-nn bf16, 48 blocks (96 vector contexts), rows = 8
      => 96 * 8 = 768 GM reads of 2 KiB = 1.5 MB, 25.4 us at 62 GB/s,
         against 4.2 MB of real y/out traffic that costs 6.8 us at 616 GB/s.

      variant                                     eager     aiv_mte2_ratio
      rows GM reads (T264, this arm's ``r264``)   32.30 us  0.871
      1 GM read + rows-1 UB->UB copies (R356)      7.85 us  0.355
      read+write with no bias at all (the floor)   6.81 us  0.451
      torch ``y + bias`` on the same card          8.31 us  --

    The two arms' outputs are bit-identical (md5, 1024x1024 bf16), which is what
    ``docs/reports/R356-data/p5-epifix-1024-bias.log`` records: the copies move the
    same bytes to the same place, only fewer times and not across the GM boundary.
    Set ``TILEOPS_GEMM_EPILOGUE_ARM=r264`` to build the control arm.
    """
    import os

    return os.environ.get("TILEOPS_GEMM_EPILOGUE_ARM", "r356").strip().lower()


#: Launch width cap for the epilogue's VECTOR kernel, separate from
#: ``common.LAUNCH_BLOCK_CAP`` (48) which the Cube and elementwise templates use.
#:
#: R356 fixed the bias staging and then measured what was left (R356 section 10
#: clue 4): with the staging gone, the epilogue's own blockDim is worth ~2.25 us
#: (~5%) on square-1k-nn, and it refused to pick a constant off ONE shape.
#: T359 measured the surface over every manifest (m, n, dtype) and all three
#: kinds -- R359-data/p1-epigeom-ALL.txt -- and the two forces separate cleanly:
#:
#:   * EAGER time is flat once the launch is wide enough to saturate GM: the
#:     bandwidth this kernel can pull stops improving at 24 launch blocks
#:     (48 vector contexts) and adding more buys nothing.
#:   * GRAPH time carries a per-block ACL-replay cost on top, and it is a
#:     function of blockDim ALONE (R356's two forced-tile groups give the same
#:     graph-minus-eager curve at equal blockDim and different grid_repeats).
#:
#: So the right launch is the NARROWEST one that still saturates.  That is a
#: measured physical threshold, not a constant fitted to a ratio.
EPILOGUE_LAUNCH_BLOCK_CAP = 24


def _geometry_arm() -> str:
    """Which epilogue launch geometry this process builds.

    ``r356`` is the control arm: ``common.LAUNCH_BLOCK_CAP`` (48), i.e. exactly
    what shipped after R356.  Set ``TILEOPS_GEMM_EPILOGUE_GEOM=r356`` to build it.
    """
    import os

    return os.environ.get("TILEOPS_GEMM_EPILOGUE_GEOM", "r359").strip().lower()


def _launch_cap() -> int:
    return LAUNCH_BLOCK_CAP if _geometry_arm() == "r356" else EPILOGUE_LAUNCH_BLOCK_CAP


def _tile_budget(itemsize: int) -> int:
    """Elements per epilogue tile.

    The working set is three fp32 buffers (staged bias, fp32 input, fp32
    result) plus TWO tiles in the operand dtype (the I/O tile and the staged
    bias in its original dtype), so the per-element cost is
    ``3 * 4 + 2 * itemsize``.
    """
    per_element = 3 * 4 + 2 * itemsize
    budget = UB_BUDGET_BYTES - UB_SCRATCH_RESERVE_BYTES
    return max(TILE_GRAIN, (budget // per_element) // TILE_GRAIN * TILE_GRAIN)


def _epilogue_tile(m: int, n: int, itemsize: int) -> tuple[int, int]:
    """Return ``(rows_per_tile, tile_elements)`` for an ``[M, N]`` output.

    Rows-per-tile geometry, taken from R263's ``row_broadcast`` fix in
    ``elementwise_binary.py``: a tile holds a WHOLE number of output rows and
    every tile start is a multiple of the row width, so the staged bias tile is
    identical for every tile a block visits and can be staged ONCE per block.
    That amortisation is the entire point -- R263 measured the alternative (a
    per-lane gather of the bias) at 6.3-18.4 GB/s against 355-1010 GB/s for the
    vector path, and ``bias_add`` went 0.0068 -> 0.976 when it was fixed
    (PROJECT_STATE 13.115).

    When one row does not fit the budget the tile becomes a divisor of the row
    width instead, one partial row per tile; the bias tile then holds that
    column window, which is still start-aligned because every tile start is a
    multiple of the window.
    """
    limit = _tile_budget(itemsize)
    if n <= limit:
        # `rows` must divide M so that `tile` divides `m * n`; the clamped-start
        # scheme in _build_epilogue relies on that.
        rows = max(1, min(limit // n, m))
        while rows > 1 and m % rows:
            rows -= 1
        return rows, rows * n
    divisor = n
    while divisor > limit:
        divisor //= 2
    divisor = max(1, divisor)
    while n % divisor:
        divisor -= 1
    return 0, max(1, divisor)


@lru_cache(maxsize=64)
def _build_epilogue(m: int, n: int, dtype_name: str, kind: str) -> Callable:
    """Vector-only pass: ``c = act(y + bias)`` over an ``[M, N]`` GEMM output."""
    itemsize = 2
    rows, tile = _epilogue_tile(m, n, itemsize)
    numel = m * n
    if tile > numel or numel % tile:
        raise ValueError(
            f"gemm epilogue: tile {tile} does not divide m*n={numel} for "
            f"m={m} n={n}"
        )
    block_total = tile * VEC_PER_BLOCK
    block_count = max(1, (numel + block_total - 1) // block_total)
    # Cap the launch width so the grid-stride loop actually strides.  Without the
    # cap `launch_blocks == block_count` and `grid_repeats == 1`
    # (PROJECT_STATE 13.57), which means each block visits exactly ONE tile and
    # the once-per-block bias staging is never amortised -- it is then a third of
    # the block's traffic.  Measured on square-1k-nn: 61.88 us with no cap
    # against a 21 us GEMM (ratio 0.3596); see R264-data/29-epi-launch-cap.txt.
    launch_blocks = launch_block_count(min(block_count, _launch_cap()))
    grid_repeats = grid_repeat_count(block_count, launch_blocks)
    # Number of whole bias rows (or column windows) the staged tile holds.
    stage_slots = rows if rows else 1
    stage_width = n if rows else tile
    # A python BOOL, not the int: TVMScript lowers ``if <captured int>:`` to a
    # TIR condition instead of folding it, which failed as
    # "Expected boolean argument for ! operator (logical NOT), but received 3
    # of type int32" (R264-data/08-hs-deadlock.txt).  Only a bool is folded.
    stage_once = bool(rows)
    # A python BOOL for the same reason ``stage_once`` is one: only a bool is folded.
    stage_r264 = bool(_staging_arm() == "r264")
    want_relu = kind == "bias_relu"
    want_gelu = kind == "bias_gelu"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            # A plain vector kernel, so synchronisation is left to the pass --
            # unlike the Cube kernel in kernels/gemm.py, there is no
            # multi-buffer schedule here for it to fight with.
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _factory(dtype: str = dtype_name) -> Callable:

        @T.prim_func
        def main(
            y: T.Tensor((numel,), dtype),
            bias: T.Tensor((n,), dtype),
            out: T.Tensor((numel,), dtype),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                # Allocated OUTSIDE the T.Scope("V") below, as
                # kernels/elementwise_binary.py does: names first bound inside
                # a scope are treated as branch-local by the frontend.
                io_ub = T.alloc_ub((tile,), dtype)
                bias_ub = T.alloc_ub((tile,), dtype)
                x32 = T.alloc_ub((tile,), "float")
                b32 = T.alloc_ub((tile,), "float")
                y32 = T.alloc_ub((tile,), "float")

                with T.Scope("V"):
                    # Stage the bias so the add below is a full-tile vector op over
                    # two same-shaped UB buffers.  No lane-indexed read of `bias` is
                    # emitted on either path.
                    #
                    # ``rows >= 1``: the tile holds whole output rows and every tile
                    # start is a multiple of the row width, so ONE staging before
                    # the grid-stride loop is correct for every tile the block
                    # visits.  That amortisation is R263's row_broadcast fix.
                    # ``rows == 0`` (a row wider than the UB budget, e.g. the
                    # wide-n-24576 workload): the tile is a divisor of the row, so
                    # which COLUMN WINDOW of the bias applies depends on the tile,
                    # and the staging moves inside the loop -- one extra DMA of
                    # `tile` elements per tile, still a contiguous copy and not a
                    # gather.
                    if stage_once:
                        if stage_r264:
                            # T264's staging: one GM read per replicated row.
                            for slot in range(stage_slots):
                                T.copy(bias[0:stage_width],
                                       bias_ub[slot * stage_width:
                                               (slot + 1) * stage_width])
                        else:
                            # R356: read the row from GM ONCE, replicate inside UB.
                            # A UB->UB copy does not touch MTE2 at all, and the
                            # staged tile is byte-identical either way (measured:
                            # R356-data/p5-epifix-1024-bias.log, md5 equal).
                            T.copy(bias[0:stage_width], bias_ub[0:stage_width])
                            for slot in range(1, stage_slots):
                                T.copy(bias_ub[0:stage_width],
                                       bias_ub[slot * stage_width:
                                               (slot + 1) * stage_width])
                        T.copy(bias_ub, b32)

                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < block_count:
                            # Clamp the last tile back inside the operand rather
                            # than running a per-lane ragged tail.  Legal because
                            # every quantity here is a multiple of `tile`:
                            # `tile` divides `numel` (tile is rows*n or a divisor
                            # of n, and numel = m*n), so `numel - tile` is both
                            # tile-aligned and row-aligned and the staged bias
                            # still lines up.  The overlap recomputes some
                            # elements, which is idempotent for an elementwise
                            # epilogue -- and it costs one partial tile against
                            # the scalar tail loop that the generic elementwise
                            # path pays (R194 measured that class at 15.7 GB/s).
                            start = T.min(logical_cid * block_total + vid * tile,
                                          numel - tile)
                            if not stage_once:
                                T.copy(bias[start % n], bias_ub)
                                T.copy(bias_ub, b32)
                            T.copy(y[start], io_ub)
                            T.copy(io_ub, x32)
                            T.tile.add(y32, x32, b32)
                            if want_relu:
                                T.tile.max(y32, y32, 0.0)
                            if want_gelu:
                                # tanh(z) = 2*sigmoid(2z) - 1, fp32 throughout.
                                # Same chain as kernels/elementwise_activation.py
                                # (this vector backend has no erf on dav-2201).
                                T.copy(y32, x32)
                                T.tile.mul(b32, x32, x32)
                                T.tile.mul(b32, b32, x32)
                                T.tile.mul(b32, b32, 0.044715)
                                T.tile.add(b32, b32, x32)
                                T.tile.mul(b32, b32, 1.5957691216057308)
                                T.tile.sigmoid(b32, b32)
                                T.tile.mul(b32, b32, 2.0)
                                T.tile.sub(b32, b32, 1.0)
                                T.tile.add(b32, b32, 1.0)
                                T.tile.mul(y32, x32, b32)
                                T.tile.mul(y32, y32, 0.5)
                                # b32 was scratch; restage for the next tile.
                                T.copy(bias_ub, b32)
                            T.copy(y32, io_ub)
                            T.copy(io_ub, out[start])

        return main

    return _factory(dtype_name)


def build_gemm_epilogue_kernel(
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    bias_shape: tuple[int, ...],
    dtype,
    trans_a: bool,
    trans_b: bool,
    kind: str,
) -> Callable:
    """Validate the operands and return the two-launch composition for *kind*.

    Launch 1 is ``kernels/gemm.py``'s pipelined Cube kernel, unchanged -- which
    is why T264's step-1 speedup carries over to these three ops for free.
    Launch 2 is :func:`_build_epilogue`.

    Two launches, not one.  The fused single-launch version was written first
    (Cube -> GM workspace -> ``T.set_cross_flag`` -> the two AIV contexts) and
    is NOT what ships; it failed twice and the budget went to finishing part 1.
    Both failures are recorded in docs/reports/R264-data/12-fused-epilogue.txt:
    ``T.tile.add`` on bf16 UB buffers does not compile on dav-2201
    ("the 1st parameter maybe need a type '__ubuf__ half *'", i.e. the fp32
    detour the elementwise templates take is mandatory), and the fp16 build
    that did compile returned NaN, so the cross-core handshake with
    ``TL_ASCEND_AUTO_CV_SYNC`` off is not yet right.

    The measured cost of the split is one extra GM round trip of the output,
    ``2 * m * n`` bytes of the operand dtype.  ``devicetimer``'s per-kernel
    interval union counts both launches (PHASE_B_CONTRACT section 4), so the
    reported ratio pays for it honestly.
    """
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError(f"gemm_{kind} expects 2D a/b, got a={a_shape}, b={b_shape}")
    if len(bias_shape) != 1:
        raise ValueError(f"gemm_{kind} expects a 1D bias, got {bias_shape}")
    if kind not in EPILOGUE_KINDS:
        raise ValueError(f"unknown epilogue {kind!r}; expected one of {EPILOGUE_KINDS}")
    if trans_a:
        raise ValueError(f"gemm_{kind} does not support trans_a=True")
    m, k_a = a_shape
    n = b_shape[0] if trans_b else b_shape[1]
    k_b = b_shape[1] if trans_b else b_shape[0]
    if k_a != k_b:
        raise ValueError(f"gemm_{kind} contraction mismatch: a={a_shape}, b={b_shape}")
    if bias_shape[0] != n:
        raise ValueError(
            f"gemm_{kind} bias must have N={n} elements, got {bias_shape[0]}"
        )
    name = _dtype_name(dtype)
    gemm = build_gemm_kernel(a_shape, b_shape, dtype, False, bool(trans_b))
    epilogue = _build_epilogue(m, n, name, kind)

    def launch(a, b, bias):
        y = gemm(a, b)
        return epilogue(y.reshape(-1), bias).reshape(m, n)

    launch.__name__ = f"gemm_{kind}_launch"
    return launch
