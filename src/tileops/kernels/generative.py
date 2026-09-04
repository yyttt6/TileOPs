"""Vector kernels for parameter-generated positional encodings.

R196 rewrite -- why the shape of this file changed
--------------------------------------------------
Both kernels are pure index generators: nothing is read from GM, every output
element is a closed-form function of its flat index.  The previous version
computed that function with a ``for lane in T.serial(tile)`` scalar loop on the
main path (plus two more scalar loops for the even/odd pick and the alibi
negation), so the cost was proportional to the element count and independent of
geometry.  Measured: 2.18-2.55 GB/s against a 36-225 GB/s vendor baseline
(``docs/reports/R196-data/step0_bandwidth.md``) -- the "about 3 GB/s" step-0
signature from T195.

The fix is to align the tile with the output's row structure so every per-lane
quantity is either

  * a scalar that is constant across the tile (``head`` for alibi, ``row`` for
    sinusoidal) -- emitted with ``T.tile.fill`` / a scalar operand, or
  * an affine ramp in the lane index -- emitted with
    ``T.tile.arith_progression``.

Two things learned the hard way and encoded below:

* The old tile ended on a scalar loop, which implicitly ordered the UB writes
  against the MTE3 store.  A vectorised tile ends on the V pipe, so the
  V->MTE3 and MTE3->V hazards must be named with ``T.barrier_all()``; without
  them the output is *non-deterministically* wrong
  (``R196-data/ab_generative_nobarrier.log``).
* ``T.tile.pow`` at full tile width is 68% of the alibi tile's runtime
  (``R196-data/probe_alibi_cost.json``: store-only 765 us, +5 cheap vector ops
  3833 us, +pow 11828 us on the 4k workload).  Alibi's exponent is one scalar
  per tile, so the transcendental runs on 64 lanes and the result is broadcast
  as a scalar operand.  ``T.tile.pow`` on a *BufferRegion* silently returns the
  wrong answer, so it has to be a real 64-element buffer
  (``R196-data/probe_pow_scalar.log``).
"""

import math
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# --- Geometry (R196) ---
UB_BUDGET_BYTES = 180224
# T.tile.pow carries implicit scratch that does not appear in the Python source
# (T195 acceptance note); reserve it before sizing the tile.
POW_SCRATCH_BYTES = 65536
TILE_HARD_CAP = 16384
LAUNCH_BLOCK_CAP = 48
# T.tile.compare (CompareScalar) rejects operands whose byte size is not a
# multiple of 256:
#   "CompareScalar alignment error: size=32, element_bytes=4, total_bytes=128
#    byte is not 256-byte aligned"
# The tile is a divisor of the inner dimension and can be smaller than that, so
# the UB buffers are rounded up to this grain and only `tile` lanes are stored.
COMPARE_GRAIN = 64
# One vector repeat, the smallest useful width for the hoisted transcendental.
SCALAR_LANES = 64

_ELEM_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}


def _ub_cap(per_elem_bytes: int) -> int:
    return max(1, min(TILE_HARD_CAP, (UB_BUDGET_BYTES - POW_SCRATCH_BYTES) // per_elem_bytes))


def _largest_divisor(n: int, cap: int) -> int:
    t = min(int(n), cap)
    while t > 1 and n % t:
        t -= 1
    return max(1, t)


def _pad(tile: int) -> int:
    return ((tile + COMPARE_GRAIN - 1) // COMPARE_GRAIN) * COMPARE_GRAIN


@lru_cache(maxsize=64)
def _compile_sinusoidal(seq_len: int, d_model: int, dtype: str):
    """out[row, dim] = sin(row / 10000**(2*(dim//2)/d_model)) for even dim, cos otherwise.

    The exponent varies per lane, so `pow`/`sin`/`cos` all run at full tile width
    here; the tile is a divisor of ``d_model`` so ``row`` is one scalar per tile.
    """
    out_numel = seq_len * d_model
    elem = _ELEM_BYTES[dtype]
    # a, b, c, aux (fp32) + idx (int32) + out_tile (dtype) + one packed mask bit
    per_elem = 4 * 4 + 4 + elem + 1
    tile = _largest_divisor(d_model, _ub_cap(per_elem))
    tp = _pad(tile)
    mask_bytes = max(32, tp // 8)
    logical_blocks = max(1, math.ceil(out_numel / (tile * 2)))
    launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
    grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)

    @tilelang.jit(out_idx=[-1])
    def kernel():
        @T.prim_func
        def main(PROBE: T.Tensor((out_numel,), dtype), OUTPUT: T.Tensor((out_numel,), dtype)):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                a = T.alloc_ub((tp,), "float32")
                b = T.alloc_ub((tp,), "float32")
                c = T.alloc_ub((tp,), "float32")
                aux = T.alloc_ub((tp,), "float32")
                out_tile = T.alloc_ub((tp,), dtype)
                idx_i32 = T.alloc_ub((tp,), "int32")
                evenmask = T.alloc_ub((mask_bytes,), "uint8")
                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            start = logical_cid * tile * 2 + vid * tile
                            if start + tile <= out_numel:
                                row = start // d_model
                                dim0 = start % d_model
                                # dim, dim % 2 and 2 * (dim // 2) as vectors.
                                # (T.tile.bitwise_and with a scalar src1 lowers to
                                # tl.ascend_bitwise_ands, which is not registered,
                                # so parity goes through the floor route.)
                                T.tile.arith_progression(
                                    b, T.cast(dim0, "float32"), T.float32(1), tp
                                )
                                T.tile.mul(a, b, T.float32(0.5))
                                T.tile.cast(idx_i32, a, "CAST_FLOOR", tp)
                                T.tile.cast(a, idx_i32, "CAST_NONE", tp)
                                T.tile.mul(a, a, T.float32(2))
                                T.tile.sub(c, b, a)
                                T.tile.compare(evenmask, c, T.float32(0.5), "LT")
                                T.tile.fill(c, T.cast(d_model, "float32"))
                                T.tile.div(b, a, c)
                                T.tile.fill(c, T.float32(10000))
                                T.tile.pow(a, c, b)
                                T.tile.fill(b, T.cast(row, "float32"))
                                T.tile.div(c, b, a)
                                T.tile.sin(b, c)
                                T.tile.cos(aux, c)
                                T.tile.select(a, evenmask, b, aux,
                                              "VSEL_TENSOR_TENSOR_MODE")
                                T.barrier_all()
                                if dtype == "float32":
                                    T.copy(a[0:tile], OUTPUT[start:start + tile])
                                else:
                                    T.tile.cast(out_tile, a, "CAST_RINT", tp)
                                    T.copy(out_tile[0:tile], OUTPUT[start:start + tile])
                                T.barrier_all()

        return main

    return kernel()


@lru_cache(maxsize=64)
def _compile_alibi(seq_len: int, num_heads: int, dtype: str):
    """out[head, row, col] = -2**(-8*(head+1)/num_heads) * |row - col|.

    The slope is one scalar per tile and ``|row - col|`` is a launch-invariant
    lane pattern shifted by the tile's first row, so the per-tile main path is
    four full-width vector ops (add, abs, mul, cast) plus the store.  This kernel
    declares far fewer tiles than the sinusoidal one, which is why it gets its
    own compile entry: it buys a 4x wider tile out of the same UB budget, worth
    1.24x at 2048 -> 4096 and a further 1.14x at 4096 -> 8192
    (``R196-data/probe_alibi_cost.json``).
    """
    inner = seq_len * seq_len
    out_numel = num_heads * inner
    elem = _ELEM_BYTES[dtype]
    # a, q (fp32) + out_tile (dtype)
    per_elem = 2 * 4 + elem
    cap = _ub_cap(per_elem)
    # rows_per_tile must divide seq_len so that a tile never straddles a head.
    # Stacking rows needs a per-row T.tile.fill on a BufferRegion, and those
    # regions have to start on a 32-byte boundary -- a 17-element row makes the
    # kernel die with aicore exception 507015
    # (R196-data/ab_generative_unaligned_rows.log).  Rows that are not a
    # multiple of 8 fp32 lanes therefore fall back to one row per tile.
    if seq_len <= cap and seq_len % 8 == 0:
        rows = _largest_divisor(seq_len, max(1, cap // seq_len))
        tile = rows * seq_len
    else:
        rows = 1
        tile = _largest_divisor(seq_len, cap)
    tp = _pad(tile)
    logical_blocks = max(1, math.ceil(out_numel / (tile * 2)))
    launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
    grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)

    @tilelang.jit(out_idx=[-1])
    def kernel():
        @T.prim_func
        def main(PROBE: T.Tensor((out_numel,), dtype), OUTPUT: T.Tensor((out_numel,), dtype)):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                a = T.alloc_ub((tp,), "float32")
                q = T.alloc_ub((tp,), "float32")
                out_tile = T.alloc_ub((tp,), dtype)
                # T.tile.pow needs a real buffer, not a region, to be correct.
                pbase = T.alloc_ub((SCALAR_LANES,), "float32")
                pexp = T.alloc_ub((SCALAR_LANES,), "float32")
                pres = T.alloc_ub((SCALAR_LANES,), "float32")
                with T.Scope("V"):
                    # q[j * seq_len + i] = j - i, built once per launch:
                    #   -lane  +  j * (seq_len + 1)
                    # Exact in fp32 (every term is a small integer) and needs no
                    # integer division, so it holds for any seq_len.
                    T.tile.arith_progression(q, T.float32(0), T.float32(-1), tp)
                    if rows > 1:
                        T.tile.fill(a, T.float32(0))
                        # j == 0 contributes nothing and would be the one region
                        # whose offset is trivially aligned anyway.
                        for j in range(1, rows):
                            T.tile.fill(a[j * seq_len:(j + 1) * seq_len],
                                        T.float32(j * (seq_len + 1)))
                        T.tile.add(q, q, a)
                    T.tile.fill(pbase, T.float32(2))

                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            start = logical_cid * tile * 2 + vid * tile
                            if start + tile <= out_numel:
                                head = start // inner
                                row0 = (start % inner) // seq_len
                                # col0 is 0 whenever the tile is a whole number of
                                # rows; it only matters on the one-row-per-tile
                                # fallback where the tile is a *piece* of a row.
                                col0 = start % seq_len
                                T.tile.fill(
                                    pexp,
                                    T.cast(-8 * (head + 1), "float32")
                                    / T.cast(num_heads, "float32"),
                                )
                                T.tile.pow(pres, pbase, pexp)
                                # -(p * s) is bit-identical to p * (-s), so the
                                # negation rides on the 64-lane scalar.
                                T.tile.mul(pres, pres, T.float32(-1))
                                T.tile.add(a, q, T.cast(row0 - col0, "float32"))
                                T.tile.abs(a, a)
                                # Scalar read of a UB slot the vector unit just
                                # wrote needs its own barrier.
                                T.barrier_all()
                                T.tile.mul(a, a, pres[0])
                                T.barrier_all()
                                if dtype == "float32":
                                    T.copy(a[0:tile], OUTPUT[start:start + tile])
                                else:
                                    T.tile.cast(out_tile, a, "CAST_RINT", tp)
                                    T.copy(out_tile[0:tile], OUTPUT[start:start + tile])
                                T.barrier_all()

        return main

    return kernel()


def _build(kind: str, shape: tuple[int, ...], dtype: torch.dtype):
    if dtype not in _FLOAT_DTYPES:
        raise TypeError(f"{kind} does not support dtype {dtype}")
    name = str(dtype).replace("torch.", "")
    if kind == "sinusoidal":
        compiled = _compile_sinusoidal(int(shape[0]), int(shape[1]), name)
    else:
        compiled = _compile_alibi(int(shape[1]), int(shape[0]), name)

    def invoke(probe):
        flat = probe.reshape(-1)
        result = compiled(flat)
        return result.reshape(shape)

    return invoke


def build_sinusoidal_kernel(shape, dtype):
    return _build("sinusoidal", tuple(shape), dtype)


def build_alibi_kernel(shape, dtype):
    return _build("alibi", tuple(shape), dtype)
