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
):
    """Compile one ordinary RoPE layout."""
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
    compiled = _compile_rope(
        math.prod(expected), int(seq_len), int(head_dim), int(head_dim) // 2,
        _dtype_name(dtype), rotation, layout, b, h,
    )

    def invoke(inp, cos, sin):
        if tuple(inp.shape) != expected:
            raise ValueError(f"RoPE input shape mismatch: expected {expected}, got {tuple(inp.shape)}")
        flat = inp.contiguous().reshape(-1)
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
