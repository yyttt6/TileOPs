"""Shared AIV RoPE kernels for the position-encoding family.

The public TileOPs ops prepare cosine/sine tables on the input device.  This
module only owns the shape-specialized rotation and keeps all arithmetic in an
fp32 UB scratch buffer so BF16 follows the dav-2201 vector restrictions.

R196 rewrite -- why the shape of this file changed
--------------------------------------------------
The previous version was correct but ran the *main path* through three
``for lane in T.serial(tile)`` scalar loops per tile (pair fetch from GM, the
sign flip, and a scalar ``Y[idx] = ...`` GM store).  Scalar cost is proportional
to the total element count and is independent of tile/grid geometry, so the
kernel sat at 2.7-3.8 GB/s on every measured shape while the vendor baseline ran
at 47-145 GB/s (``docs/reports/R196-data/step0_bandwidth.md``).  That is the
"約 3 GB/s" signature from T195 step 0: the disease is the scalar loop, not the
geometry -- this file already had grid-stride.

Both RoPE rotations are the same algebra::

    y[r, c] = x[r, c] * cos[s(r), f(c)] + sign(c) * x[r, p(c)] * sin[s(r), f(c)]

with three *shape-only* permutations of the head row:

    neox      p(c) = (c + half) % head_dim   f(c) = c % half   sign = -1 iff c < half
    non_neox  p(c) = c ^ 1                   f(c) = c // 2     sign = -1 iff c even
    (position-ids adds a pass-through region for c >= rotary_dim)

``p`` and ``f`` do not depend on the row, only on the column, so all three
patterns are built **once per launch** as int32 index vectors (a ``head_dim``
long scalar loop, then vector replication) and applied per tile with
``T.tile.gather`` -- the same byte-offset idiom as
``attention_indexing.py:250``.  The main path is now vector-only: three
``T.copy`` in, three gathers, five vector ops, one ``T.copy`` out.

``T.tile.gather`` needs an explicit ``T.barrier_all()`` between the MTE2 load
that fills its source and the gather itself; without it the gather reads stale
UB and silently returns garbage on ~90% of lanes
(``R196-data/probe_gather_sync.{py,log}``).

R339 -- the neox rotation no longer goes through ``T.tile.gather`` at all
-----------------------------------------------------------------------
R196's three gathers are correct and they did remove the scalar loop, but R339
priced them.  Ablations of this exact kernel on ``2d-b1-s8k-h32-d128`` (identical
geometry, buffers, launch config and GM traffic; only the named instruction
dropped -- ``docs/reports/R339-data/probe/p7_2db1.log``):

    full (shipped)                              392.5 us
    pair gather -> contiguous copy              335.1   (-57)
    cos+sin gathers -> contiguous copies        264.5   (-128)
    all three -> contiguous copies              214.8   (-178)

i.e. **one gather over a 4096-element tile costs ~59 us, about 20x a contiguous
copy of the same tile**, and the three of them were 45% of the kernel.
``PipeUtilization`` says the same thing from the other side: we sat at
``aiv_vec_ratio`` 0.78 with 290 us of vector busy, while the D036 pool winner
(``ops-transformer-b1:aclnnRotaryPositionEmbedding``) sat at ``aiv_mte2_ratio``
0.91 with only 67 us of vector busy -- 4.3x less vector work for the same
arithmetic (``R339-data/prof/extract.json``).

The neox rotation is ``y[:, :h] = x[:, :h]*cos - x[:, h:]*sin`` and
``y[:, h:] = x[:, h:]*cos + x[:, :h]*sin``, i.e. six ordinary vector ops on the
two *contiguous halves* of the head row, with no permutation and no sign mask.
The fp32 expression is bit-identical to the old one (IEEE: ``a + (-(b*c))`` is
``a - b*c``), and the manifest shapes stay at ``max_abs_err == 0.0``.

Getting the halves into UB is the whole design problem, because
``tilelang-ascend/src/tl_templates/ascend/common.h:305`` lowers a NARROW-WINDOW
``T.copy`` between two UB buffers to a **scalar loop with one Cast/DataCopy per
row** (the fast path needs ``src_cols == src_stride && dst_cols == dst_stride``).
Splitting the head row inside UB therefore costs one instruction per row and is
slower than the gathers it replaces -- measured, three different ways, in
``R339-data/probe/p3_sweep.log``, ``p4_*.log`` and ``p5b_*.log``.
``copy_gm_to_ub`` / ``copy_ub_to_gm`` in the same file are the opposite: a single
``AscendC::DataCopyPad`` with blockCount/blockLen/srcStride, one instruction at
any row count.

So ``_compile_rope_neox`` splits the head row **in GM**: four strided
``DataCopyPad``s per tile (x-left, x-right in; y-left, y-right out) over the flat
``(batch*seq*num_heads, head_dim)`` view, and every UB operand the vector unit
sees is a full contiguous buffer.  One work unit is ``rows_per_tile`` CONSECUTIVE
rows of that view, so the DMA stride is ``half*itemsize`` (it walks one
contiguous region and takes every other half-row) rather than a 128-byte burst
every ``num_heads*head_dim`` bytes.

In that row order the cos/sin rows have to be repeated ``num_heads`` times, and
the repetition is over WHOLE ROWS -- a chain of ``log2(num_heads)`` fully
contiguous UB->UB copies, i.e. the fast path above, one instruction each.  For
the 1d layout ``num_heads == 1`` and the chain is empty.

``non_neox`` keeps the R196 gather kernel: its pair is ``c ^ 1``, an interleave,
whose halves are not contiguous windows of anything.
"""

import math
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# --- Geometry (R196) ---
# Same water line as elementwise_binary.py / elementwise_unary.py (R194).
UB_BUDGET_BYTES = 180224
# Beyond this the measured bandwidth stops rising (R194 probe).
TILE_HARD_CAP = 16384
# 48-192 launch blocks is the plateau; blockDim == logical_blocks costs 13x (R194 2.1).
LAUNCH_BLOCK_CAP = 48
# T.tile.compare writes a packed mask with 128-lane native repeat on dav-2201.
MASK_GRAIN = 128
# ... and CompareScalar additionally rejects an operand whose byte size is not a
# multiple of 256 ("CompareScalar alignment error: ... is not 256-byte aligned"),
# i.e. the fp32 lane count must be a multiple of 64.  This is a hard constraint
# on the tile, not a preference.
COMPARE_GRAIN = 64

_ELEM_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype not in _FLOAT_DTYPES:
        raise TypeError(f"position encoding supports float16, bfloat16, and float32; got {dtype}")
    return str(dtype).replace("torch.", "")


def _align32(count: int, width: int) -> int:
    return ((count * width + 31) // 32) * 32


def _ub_bytes(rows: int, head_dim: int, half: int, group: int, elem: int, passthru: bool) -> int:
    """Exact declared-UB footprint of one geometry.

    Counted buffer by buffer rather than with a bytes-per-element constant,
    because the cos/sin staging block is ``groups_per_tile * half`` long and that
    ranges from 64 elements (2d layout, one seq row per tile) to half a tile
    (1d layout, every row needs its own table row).
    """
    tile = rows * head_dim
    gpt = max(1, rows // group)
    blk = gpt * half
    total = 0
    total += _align32(tile, 4) * 4          # x32, pair32, cos32, sin32
    total += _align32(tile, 4) * 5          # perm_i32/u32, cidx_i32/u32, off_i32
    total += _align32(tile, elem) * 2       # x_ub, out_ub
    total += _align32(max(MASK_GRAIN, tile) // 8, 1)   # negmask
    total += _align32(blk, elem) * 2        # cos_half_ub, sin_half_ub
    total += _align32(blk, 4) * 2           # cos_half32, sin_half32
    total += _align32(head_dim, 4)          # base_i32
    total += _align32(head_dim, 4)          # base_f32
    if passthru:
        total += _align32(tile, 4)          # keep x32 alive for the pass-through select
        total += _align32(max(MASK_GRAIN, tile) // 8, 1)  # passmask
    return total


def _plan(total_rows: int, outer: int, head_dim: int, half: int, group: int,
          elem: int, passthru: bool, max_groups: int = 64):
    """Pick rows-per-tile.

    ``outer`` is the number of distinct table rows in the leading dimension
    (``seq_len`` for RoPE, the token count for the position-ids variant); a tile
    may cover ``gpt`` of them and ``gpt`` must divide ``outer`` so no tile
    straddles the wrap.  Candidates are tried largest first and the first one
    that fits the UB budget wins.
    """
    row_cap = max(1, TILE_HARD_CAP // head_dim)
    if group <= row_cap:
        candidates = [g * group for g in range(min(row_cap // group, outer, max_groups), 0, -1)
                      if outer % g == 0]
    else:
        candidates = [r for r in range(min(row_cap, group), 0, -1) if group % r == 0]
    candidates = [r for r in candidates if r <= total_rows and total_rows % r == 0]
    if not candidates:
        candidates = [1]
    fitting = [r for r in candidates
               if _ub_bytes(r, head_dim, half, group, elem, passthru) <= UB_BUDGET_BYTES]
    if not fitting:
        fitting = [candidates[-1]]
    # The sign mask is built with T.tile.compare over a tile-long fp32 buffer, so
    # the tile has to be a whole number of 64-lane compare grains.
    usable = [r for r in fitting if (r * head_dim) % COMPARE_GRAIN == 0]
    if not usable:
        raise NotImplementedError(
            "position_encoding: no tile with (rows * head_dim) %% %d == 0 fits "
            "(head_dim=%d, group=%d, outer=%d, total_rows=%d, candidates=%r). "
            "T.tile.compare needs 256-byte operands." %
            (COMPARE_GRAIN, head_dim, group, outer, total_rows, candidates)
        )
    # Prefer a whole number of packed-mask grains on top of that.
    aligned = [r for r in usable if (r * head_dim) % MASK_GRAIN == 0]
    rows = (aligned or usable)[0]
    return rows, max(1, rows // group)


# --- Geometry for the R339 gather-free neox path ---------------------------
# One tile holds `rows` consecutive rows of the flat (batch*seq*heads, head_dim)
# view.  Declared UB per tile, counted buffer by buffer:
#   xl_ub, xr_ub, yl_ub, yr_ub   (rows, half) dtype
#   cos_ub, sin_ub               (rows_per_seq_block, half) dtype
#   cs                           (rows_per_seq_block, half) fp32  (broadcast seed)
#   a, b, c, d, e                (rows, half) fp32
# Five fp32 tile buffers is the minimum for this expression: after
#   e = xl*cos ; c = xr*cos ; a = xl*sin ; b = xr*sin
# both products of each operand are live at once, and the two results are then
# `e -= b` and `c += a`.
#: Keep at least this many work units so the 96 vector lanes stay balanced.
NEOX_MIN_UNITS = 192
#: ... unless respecting it would make a vector operand shorter than this.
NEOX_MIN_VEC = 256


def _neox_ub_bytes(rows: int, seq_rows: int, half: int, elem: int) -> int:
    total = _align32(rows * half, elem) * 4      # xl_ub, xr_ub, yl_ub, yr_ub
    total += _align32(seq_rows * half, elem) * 2  # cos_ub, sin_ub
    total += _align32(seq_rows * half, 4)         # cs
    total += _align32(rows * half, 4) * 5         # a, b, c, d, e
    return total


#: Smallest half-row the GM split pays for.  The fast path buys ~59 us/gather of vector
#: time and pays for it in MTE, because the four half-row DMAs move the same bytes in
#: ``half*elem``-byte blocks instead of one contiguous burst.  Measured at 64 bytes
#: (head_dim 64, fp16, ``neox-1d-2k-d64``): ``aiv_mte3_time`` 0.45 -> 3.37 us on a 17 us
#: kernel, and the case goes 1.2373 -> 1.0154 at N=5 (R339 section 6c).  At 128 bytes --
#: every other manifest RoPE case -- the same swap is a 1.6-2.0x win.  ⚠️ 64 B is the ONLY
#: width measured below this line, so the constant is "not 64", not a located knee.
NEOX_MIN_DMA_BYTES = 128


def _neox_supported(head_dim: int, half: int, num_heads: int, elem: int) -> bool:
    """The four things the gather-free path needs from the shape.

    * ``num_heads`` a power of two -- the cos/sin repetition is a doubling chain.
    * ``half * elem`` a whole number of 32-byte blocks -- the four half-row DMAs
      are ``DataCopyPad`` with ``blockLen = half*elem``; a non-multiple would take
      ``copy_gm_to_ub``'s padding branch, whose destination stride is computed as
      ``(dstN - maskShapeN) * sizeof(T) / 32`` and is only exact when aligned.
    * ``half * elem`` at least NEOX_MIN_DMA_BYTES -- below that the extra MTE costs more
      than the three gathers it removes.
    * an even ``head_dim`` (already guaranteed by the callers).
    """
    return (
        num_heads >= 1
        and (num_heads & (num_heads - 1)) == 0
        and head_dim % 2 == 0
        and (half * elem) % 32 == 0
        and half * elem >= NEOX_MIN_DMA_BYTES
    )


def _neox_plan(seq_len: int, outer: int, head_dim: int, half: int,
               num_heads: int, elem: int) -> int:
    """Pick the number of seq rows per tile.

    Must divide ``seq_len`` so a tile never straddles the cos/sin wrap or a batch
    boundary.  Largest candidate that fits UB and still leaves NEOX_MIN_UNITS work
    units; if that would make the vector operands shorter than NEOX_MIN_VEC, the
    balance floor is dropped instead (the small shapes are launch-bound anyway).
    """
    divisors = [d for d in range(1, seq_len + 1) if seq_len % d == 0]
    fits = [d for d in divisors
            if _neox_ub_bytes(d * num_heads, d, half, elem) <= UB_BUDGET_BYTES]
    if not fits:
        return 0
    balanced = [d for d in fits if outer // d >= NEOX_MIN_UNITS]
    if balanced:
        best = max(balanced)
        if best * num_heads * half >= NEOX_MIN_VEC:
            return best
    return max(fits)


def _compile_rope_neox(
    seq_len: int,
    head_dim: int,
    half: int,
    dtype_name: str,
    batch: int,
    num_heads: int,
    seq_rows: int,
):
    """Gather-free neox RoPE.  See the R339 section of the module docstring."""
    elem = _ELEM_BYTES[dtype_name]
    outer = batch * seq_len
    total_rows = outer * num_heads
    rows = seq_rows * num_heads
    steps = int(math.log2(num_heads))
    units = outer // seq_rows
    logical_blocks = max(1, math.ceil(units / 2))
    launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
    repeats = grid_repeat_count(logical_blocks, launch_blocks)
    calc_dtype = "float32"
    nh = num_heads

    @tilelang.jit(out_idx=[-1])
    def kernel():
        @T.prim_func
        def main(
            X: T.Tensor((total_rows, head_dim), dtype_name),
            COS: T.Tensor((seq_len, half), dtype_name),
            SIN: T.Tensor((seq_len, half), dtype_name),
            Y: T.Tensor((total_rows, head_dim), dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                xl_ub = T.alloc_ub((rows, half), dtype_name)
                xr_ub = T.alloc_ub((rows, half), dtype_name)
                yl_ub = T.alloc_ub((rows, half), dtype_name)
                yr_ub = T.alloc_ub((rows, half), dtype_name)
                cos_ub = T.alloc_ub((seq_rows, half), dtype_name)
                sin_ub = T.alloc_ub((seq_rows, half), dtype_name)
                cs = T.alloc_ub((seq_rows, half), calc_dtype)
                a = T.alloc_ub((rows, half), calc_dtype)
                b = T.alloc_ub((rows, half), calc_dtype)
                c = T.alloc_ub((rows, half), calc_dtype)
                d = T.alloc_ub((rows, half), calc_dtype)
                e = T.alloc_ub((rows, half), calc_dtype)

                with T.Scope("V"):
                    for repeat in T.serial(repeats):
                        unit = (cid + repeat * launch_blocks) * 2 + vid
                        if unit < units:
                            r0 = unit * seq_rows
                            s0 = r0 % seq_len
                            rr0 = r0 * nh
                            T.copy(X[rr0:rr0 + rows, 0:half], xl_ub)
                            T.copy(X[rr0:rr0 + rows, half:head_dim], xr_ub)
                            T.copy(COS[s0:s0 + seq_rows, 0:half], cos_ub)
                            T.copy(SIN[s0:s0 + seq_rows, 0:half], sin_ub)
                            T.barrier_all()
                            T.copy(xl_ub, a)
                            T.copy(xr_ub, b)
                            if steps == 0:
                                # 1d layout: one cos/sin row per tile row already.
                                T.copy(cos_ub, c)
                                T.copy(sin_ub, d)
                            else:
                                # Repeat each table row num_heads times.  The copy
                                # LENGTH has to be a compile-time constant (it is a
                                # tilelang template parameter), so the chain is
                                # written out and guarded by `steps`, which is a
                                # closure int and is resolved when the prim_func is
                                # parsed.  A `range()` loop would make the length a
                                # TIR expression and bisheng would reject it.
                                T.copy(cos_ub, cs)
                                for k in T.serial(seq_rows):
                                    T.copy(cs[k, 0:half], c[k * nh, 0:half])
                                    if steps >= 1:
                                        T.copy(c[k * nh:k * nh + 1, 0:half],
                                               c[k * nh + 1:k * nh + 2, 0:half])
                                    if steps >= 2:
                                        T.copy(c[k * nh:k * nh + 2, 0:half],
                                               c[k * nh + 2:k * nh + 4, 0:half])
                                    if steps >= 3:
                                        T.copy(c[k * nh:k * nh + 4, 0:half],
                                               c[k * nh + 4:k * nh + 8, 0:half])
                                    if steps >= 4:
                                        T.copy(c[k * nh:k * nh + 8, 0:half],
                                               c[k * nh + 8:k * nh + 16, 0:half])
                                    if steps >= 5:
                                        T.copy(c[k * nh:k * nh + 16, 0:half],
                                               c[k * nh + 16:k * nh + 32, 0:half])
                                    if steps >= 6:
                                        T.copy(c[k * nh:k * nh + 32, 0:half],
                                               c[k * nh + 32:k * nh + 64, 0:half])
                                T.copy(sin_ub, cs)
                                for k in T.serial(seq_rows):
                                    T.copy(cs[k, 0:half], d[k * nh, 0:half])
                                    if steps >= 1:
                                        T.copy(d[k * nh:k * nh + 1, 0:half],
                                               d[k * nh + 1:k * nh + 2, 0:half])
                                    if steps >= 2:
                                        T.copy(d[k * nh:k * nh + 2, 0:half],
                                               d[k * nh + 2:k * nh + 4, 0:half])
                                    if steps >= 3:
                                        T.copy(d[k * nh:k * nh + 4, 0:half],
                                               d[k * nh + 4:k * nh + 8, 0:half])
                                    if steps >= 4:
                                        T.copy(d[k * nh:k * nh + 8, 0:half],
                                               d[k * nh + 8:k * nh + 16, 0:half])
                                    if steps >= 5:
                                        T.copy(d[k * nh:k * nh + 16, 0:half],
                                               d[k * nh + 16:k * nh + 32, 0:half])
                                    if steps >= 6:
                                        T.copy(d[k * nh:k * nh + 32, 0:half],
                                               d[k * nh + 32:k * nh + 64, 0:half])
                            # e = xl*cos ; c = xr*cos ; a = xl*sin ; b = xr*sin.
                            # Written into the operand buffers as they die so the
                            # whole expression needs five fp32 tiles, not seven.
                            T.tile.mul(e, a, c)
                            T.tile.mul(c, b, c)
                            T.tile.mul(a, a, d)
                            T.tile.mul(b, b, d)
                            # IEEE: x*cos + (-(pair*sin)) == x*cos - pair*sin, so
                            # this is bit-identical to the R196 select path.
                            T.tile.sub(e, e, b)
                            T.tile.add(c, c, a)
                            T.copy(e, yl_ub)
                            T.copy(c, yr_ub)
                            T.barrier_all()
                            T.copy(yl_ub, Y[rr0:rr0 + rows, 0:half])
                            T.copy(yr_ub, Y[rr0:rr0 + rows, half:head_dim])

        return main

    return kernel()


@lru_cache(maxsize=128)
def _compile_rope(
    n_total: int,
    seq_len: int,
    head_dim: int,
    half: int,
    dtype_name: str,
    rotation: str,
    layout: str,
    batch: int,
    num_heads: int,
    seq_rows: int = 0,
):
    """Compile one ordinary RoPE layout.

    ``seq_rows > 0`` selects the R339 gather-free neox kernel (see the module
    docstring); 0 keeps the R196 gather kernel, which is what ``non_neox`` and any
    shape the fast path does not support still use.  Both live behind this one
    ``lru_cache`` on purpose: ``adapters/position_encoding._clear_compile_caches``
    clears ``_compile_rope`` and ``_compile_position_ids`` by name, and a second
    cache here would silently survive it.
    """
    if seq_rows:
        return _compile_rope_neox(
            seq_len, head_dim, half, dtype_name, batch, num_heads, seq_rows
        )
    group = num_heads if layout == "2d" else 1
    total_rows = n_total // head_dim
    elem = _ELEM_BYTES[dtype_name]
    rows, gpt = _plan(total_rows, seq_len, head_dim, half, group, elem, False)
    tile = rows * head_dim
    blk = gpt * half
    mask_bytes = max(MASK_GRAIN, tile) // 8
    block_total = tile * 2
    logical_blocks = max(1, math.ceil(n_total / block_total))
    launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
    repeats = grid_repeat_count(logical_blocks, launch_blocks)
    calc_dtype = "float32"

    @tilelang.jit(out_idx=[-1])
    def kernel():
        @T.prim_func
        def main(
            X: T.Tensor((n_total,), dtype_name),
            COS: T.Tensor((seq_len, half), dtype_name),
            SIN: T.Tensor((seq_len, half), dtype_name),
            Y: T.Tensor((n_total,), dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                x_ub = T.alloc_ub((tile,), dtype_name)
                out_ub = T.alloc_ub((tile,), dtype_name)
                x32 = T.alloc_ub((tile,), calc_dtype)
                pair32 = T.alloc_ub((tile,), calc_dtype)
                cos32 = T.alloc_ub((tile,), calc_dtype)
                sin32 = T.alloc_ub((tile,), calc_dtype)
                perm_i32 = T.alloc_ub((tile,), "int32")
                perm_u32 = T.alloc_ub((tile,), "uint32")
                cidx_i32 = T.alloc_ub((tile,), "int32")
                cidx_u32 = T.alloc_ub((tile,), "uint32")
                off_i32 = T.alloc_ub((tile,), "int32")
                negmask = T.alloc_ub((mask_bytes,), "uint8")
                cos_half_ub = T.alloc_ub((gpt, half), dtype_name)
                sin_half_ub = T.alloc_ub((gpt, half), dtype_name)
                cos_half32 = T.alloc_ub((gpt, half), calc_dtype)
                sin_half32 = T.alloc_ub((gpt, half), calc_dtype)
                base_i32 = T.alloc_ub((head_dim,), "int32")
                base_f32 = T.alloc_ub((head_dim,), calc_dtype)

                with T.Scope("V"):
                    # ---- launch-invariant index vectors (R196 step 2) ----
                    # head_dim scalar iterations, once, instead of `tile` per tile.
                    for c in T.serial(head_dim):
                        if rotation == "neox":
                            base_i32[c] = T.if_then_else(c < half, c + half, c - half)
                            base_f32[c] = T.if_then_else(c < half, T.float32(1), T.float32(0))
                        else:
                            base_i32[c] = T.if_then_else((c % 2) == 0, c + 1, c - 1)
                            base_f32[c] = T.if_then_else((c % 2) == 0, T.float32(1), T.float32(0))
                    for k in range(rows):
                        T.copy(base_i32, perm_i32[k * head_dim:(k + 1) * head_dim])
                        T.tile.fill(off_i32[k * head_dim:(k + 1) * head_dim], k * head_dim)
                        T.copy(base_f32, x32[k * head_dim:(k + 1) * head_dim])
                    T.tile.add(perm_i32, perm_i32, off_i32)
                    T.tile.mul(perm_i32, perm_i32, 4)
                    T.reinterpretcast(perm_u32, perm_i32, "uint32_t")
                    T.tile.compare(negmask, x32, T.float32(0.5), "GT")

                    # cidx[k*head_dim + c] = (k // group) * half + f(c)
                    for c in T.serial(head_dim):
                        base_i32[c] = (c % half) if rotation == "neox" else (c // 2)
                    for k in range(rows):
                        T.copy(base_i32, cidx_i32[k * head_dim:(k + 1) * head_dim])
                        T.tile.fill(off_i32[k * head_dim:(k + 1) * head_dim], (k // group) * half)
                    T.tile.add(cidx_i32, cidx_i32, off_i32)
                    T.tile.mul(cidx_i32, cidx_i32, 4)
                    T.reinterpretcast(cidx_u32, cidx_i32, "uint32_t")

                    for repeat in T.serial(repeats):
                        logical_cid = cid + repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            start = logical_cid * block_total + vid * tile
                            if start + tile <= n_total:
                                seq0 = ((start // head_dim) // group) % seq_len
                                T.copy(X[start:start + tile], x_ub)
                                T.copy(COS[seq0:seq0 + gpt, :], cos_half_ub)
                                T.copy(SIN[seq0:seq0 + gpt, :], sin_half_ub)
                                T.barrier_all()
                                T.copy(x_ub, x32)
                                T.copy(cos_half_ub, cos_half32)
                                T.copy(sin_half_ub, sin_half32)
                                T.tile.gather(pair32, x32, perm_u32, 0)
                                T.tile.gather(cos32, cos_half32, cidx_u32, 0)
                                T.tile.gather(sin32, sin_half32, cidx_u32, 0)
                                T.tile.mul(cos32, x32, cos32)
                                T.tile.mul(pair32, pair32, sin32)
                                # -(p*s) is bit-identical to (-p)*s, so folding the
                                # sign flip behind the multiply keeps the old result.
                                T.tile.mul(sin32, pair32, T.float32(-1))
                                T.tile.select(x32, negmask, sin32, pair32,
                                              "VSEL_TENSOR_TENSOR_MODE")
                                T.tile.add(cos32, cos32, x32)
                                # CAST_RINT is also the stable same-dtype fp32 path
                                # on dav-2201; a same-dtype T.copy here can leave the
                                # destination undefined under the vector lowerer.
                                T.copy(cos32, out_ub)
                                T.barrier_all()
                                T.copy(out_ub, Y[start:start + tile])

        return main

    return kernel()


@lru_cache(maxsize=128)
def _compile_position_ids(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    dtype_name: str,
):
    n_total = num_tokens * num_heads * head_dim
    half = rotary_dim // 2
    total_rows = num_tokens * num_heads
    elem = _ELEM_BYTES[dtype_name]
    passthru = rotary_dim < head_dim
    # One small COS/SIN copy per token in the tile, so keep the token count low.
    rows, gpt = _plan(total_rows, num_tokens, head_dim, half, num_heads, elem, passthru,
                      max_groups=8)
    tile = rows * head_dim
    blk = gpt * half
    mask_bytes = max(MASK_GRAIN, tile) // 8
    block_total = tile * 2
    logical_blocks = max(1, math.ceil(n_total / block_total))
    launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
    repeats = grid_repeat_count(logical_blocks, launch_blocks)

    @tilelang.jit(out_idx=[-1])
    def kernel():
        @T.prim_func
        def main(
            X: T.Tensor((n_total,), dtype_name),
            COS: T.Tensor((max_position, half), dtype_name),
            SIN: T.Tensor((max_position, half), dtype_name),
            POS: T.Tensor((num_tokens,), "int32"),
            Y: T.Tensor((n_total,), dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                x_ub = T.alloc_ub((tile,), dtype_name)
                out_ub = T.alloc_ub((tile,), dtype_name)
                x32 = T.alloc_ub((tile,), "float32")
                pair32 = T.alloc_ub((tile,), "float32")
                cos32 = T.alloc_ub((tile,), "float32")
                sin32 = T.alloc_ub((tile,), "float32")
                sel32 = T.alloc_ub((tile,), "float32")
                perm_i32 = T.alloc_ub((tile,), "int32")
                perm_u32 = T.alloc_ub((tile,), "uint32")
                cidx_i32 = T.alloc_ub((tile,), "int32")
                cidx_u32 = T.alloc_ub((tile,), "uint32")
                off_i32 = T.alloc_ub((tile,), "int32")
                negmask = T.alloc_ub((mask_bytes,), "uint8")
                passmask = T.alloc_ub((mask_bytes,), "uint8")
                cos_half_ub = T.alloc_ub((gpt, half), dtype_name)
                sin_half_ub = T.alloc_ub((gpt, half), dtype_name)
                cos_half32 = T.alloc_ub((gpt, half), "float32")
                sin_half32 = T.alloc_ub((gpt, half), "float32")
                base_i32 = T.alloc_ub((head_dim,), "int32")
                base_f32 = T.alloc_ub((head_dim,), "float32")

                with T.Scope("V"):
                    # perm: rotate-half inside the rotary block, identity outside it.
                    for c in T.serial(head_dim):
                        if c < rotary_dim:
                            base_i32[c] = T.if_then_else(c < half, c + half, c - half)
                            base_f32[c] = T.if_then_else(c < half, T.float32(1), T.float32(0))
                        else:
                            base_i32[c] = c
                            base_f32[c] = T.float32(0)
                    for k in range(rows):
                        T.copy(base_i32, perm_i32[k * head_dim:(k + 1) * head_dim])
                        T.tile.fill(off_i32[k * head_dim:(k + 1) * head_dim], k * head_dim)
                        T.copy(base_f32, x32[k * head_dim:(k + 1) * head_dim])
                    T.tile.add(perm_i32, perm_i32, off_i32)
                    T.tile.mul(perm_i32, perm_i32, 4)
                    T.reinterpretcast(perm_u32, perm_i32, "uint32_t")
                    T.tile.compare(negmask, x32, T.float32(0.5), "GT")

                    for c in T.serial(head_dim):
                        # Python `x if cond else y` cannot be used on a TIR loop
                        # var -- it forces bool(Expr).  Use T.if_then_else.
                        base_i32[c] = T.if_then_else(c < rotary_dim, c % half, 0)
                        base_f32[c] = T.if_then_else(
                            c < rotary_dim, T.float32(0), T.float32(1)
                        )
                    for k in range(rows):
                        T.copy(base_i32, cidx_i32[k * head_dim:(k + 1) * head_dim])
                        T.tile.fill(off_i32[k * head_dim:(k + 1) * head_dim],
                                    (k // num_heads) * half)
                        if passthru:
                            T.copy(base_f32, x32[k * head_dim:(k + 1) * head_dim])
                    T.tile.add(cidx_i32, cidx_i32, off_i32)
                    T.tile.mul(cidx_i32, cidx_i32, 4)
                    T.reinterpretcast(cidx_u32, cidx_i32, "uint32_t")
                    if passthru:
                        T.tile.compare(passmask, x32, T.float32(0.5), "GT")

                    for repeat in T.serial(repeats):
                        logical_cid = cid + repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            start = logical_cid * block_total + vid * tile
                            if start + tile <= n_total:
                                token0 = (start // head_dim) // num_heads
                                T.copy(X[start:start + tile], x_ub)
                                for g in range(gpt):
                                    pos = POS[token0 + g]
                                    T.copy(COS[pos, :], cos_half_ub[g, :])
                                    T.copy(SIN[pos, :], sin_half_ub[g, :])
                                T.barrier_all()
                                T.copy(x_ub, x32)
                                T.copy(cos_half_ub, cos_half32)
                                T.copy(sin_half_ub, sin_half32)
                                T.tile.gather(pair32, x32, perm_u32, 0)
                                T.tile.gather(cos32, cos_half32, cidx_u32, 0)
                                T.tile.gather(sin32, sin_half32, cidx_u32, 0)
                                T.tile.mul(cos32, x32, cos32)
                                T.tile.mul(pair32, pair32, sin32)
                                T.tile.mul(sin32, pair32, T.float32(-1))
                                T.tile.select(sel32, negmask, sin32, pair32,
                                              "VSEL_TENSOR_TENSOR_MODE")
                                T.tile.add(cos32, cos32, sel32)
                                if passthru:
                                    # Columns past rotary_dim are copied through
                                    # unchanged, exactly as the scalar store used to.
                                    T.tile.select(sel32, passmask, x32, cos32,
                                                  "VSEL_TENSOR_TENSOR_MODE")
                                    T.copy(sel32, out_ub)
                                else:
                                    T.copy(cos32, out_ub)
                                T.barrier_all()
                                T.copy(out_ub, Y[start:start + tile])

        return main

    return kernel()


def build_rope_kernel(
    x,
    *,
    layout: str,
    rotation: str,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    batch: int | None = None,
    num_heads: int | None = None,
    **_: object,
):
    """Build an ordinary RoPE callable for a TensorSpec input."""
    if x is None:
        raise ValueError("RoPE requires an input tensor")
    if x.dtype != dtype:
        raise TypeError(f"RoPE dtype mismatch: spec={x.dtype}, op={dtype}")
    if layout == "1d":
        expected = (int(seq_len), int(head_dim))
        actual = tuple(x.shape)
        b, h = 1, 1
    elif layout == "2d":
        b, h = int(batch), int(num_heads)
        expected = (b, int(seq_len), h, int(head_dim))
        actual = tuple(x.shape)
    else:
        raise ValueError(f"unsupported RoPE layout {layout!r}")
    if actual != expected:
        raise ValueError(f"RoPE shape mismatch: expected {expected}, got {actual}")
    dtype_name = _dtype_name(dtype)
    half = int(head_dim) // 2
    elem = _ELEM_BYTES[dtype_name]
    seq_rows = 0
    if rotation == "neox" and _neox_supported(int(head_dim), half, h, elem):
        seq_rows = _neox_plan(int(seq_len), b * int(seq_len), int(head_dim), half, h, elem)
    compiled = _compile_rope(
        math.prod(expected), int(seq_len), int(head_dim), half,
        dtype_name, rotation, layout, b, h, seq_rows,
    )
    in_shape = (b * int(seq_len) * h, int(head_dim)) if seq_rows else (-1,)

    def invoke(inp, cos, sin):
        if tuple(inp.shape) != expected:
            raise ValueError(f"RoPE input shape mismatch: expected {expected}, got {tuple(inp.shape)}")
        flat = inp.contiguous().reshape(*in_shape)
        result = compiled(flat, cos.contiguous(), sin.contiguous())
        return result.reshape(expected)

    return invoke


def build_position_ids_kernel(
    x,
    position_ids,
    *,
    max_position: int,
    base: float = 10000.0,
    rotary_dim: int | None = None,
    **_: object,
):
    """Build the packed THD RoPE callable with explicit positions."""
    del base
    if x is None or position_ids is None:
        raise ValueError("RopeNeoxPositionIdsFwdOp requires x and position_ids")
    tokens, heads, dim = (int(v) for v in x.shape)
    rotary = dim if rotary_dim is None else int(rotary_dim)
    if tuple(position_ids.shape) != (tokens,):
        raise ValueError(f"position_ids shape mismatch: expected {(tokens,)}, got {tuple(position_ids.shape)}")
    if position_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"position_ids must be int32 or int64, got {position_ids.dtype}")
    if rotary <= 0 or rotary % 2 or rotary > dim:
        raise ValueError(f"rotary_dim must be positive, even, and <= head_dim; got {rotary}")
    compiled = _compile_position_ids(tokens, heads, dim, rotary, int(max_position), _dtype_name(x.dtype))

    def invoke(inp, cos, sin, pos):
        pos_i32 = pos.to(dtype=torch.int32).contiguous()
        result = compiled(inp.contiguous().reshape(-1), cos.contiguous(), sin.contiguous(), pos_i32)
        return result.reshape((tokens, heads, dim))

    return invoke


__all__ = ["build_position_ids_kernel", "build_rope_kernel"]
