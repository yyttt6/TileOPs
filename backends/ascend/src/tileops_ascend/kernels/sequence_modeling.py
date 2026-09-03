"""Sequence-modeling kernels."""

from functools import lru_cache
import math

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# --- Geometry (R201) --------------------------------------------------------
#
# Step 0 of T195 on this family (docs/reports/R201-data/step0_discriminant.log)
# measured 0.02 - 1.03 GB/s on every one of the 12 manifest cases -- an order
# of magnitude BELOW the "~3 GB/s = per-lane scalar loop" row of the T195
# discriminant table and below even the pool family's 0.106 - 6.23 GB/s
# (13.52).  The cost model that explains it is 13.52's: ~100 ns per scalar UB
# iteration, independent of what the loop body does.  Counted against the
# manifest shapes (docs/reports/R201-data/scalar_iteration_model.py) the four
# ops' scalar iteration counts predict their runtimes to within 20%.  So the
# fix for all four is 13.52's: delete the loops, do not tune them.

#: UB waterline shared with elementwise_binary.py / elementwise_unary.py
#: (R183 / R194).  NOT the normalization waterline in common.py -- these
#: kernels do not call T.reduce_* on wide 2-D tiles.
UB_BUDGET_BYTES = 180224

#: An Ascend vector instruction addresses whole 32-byte blocks, so every tile
#: width is a multiple of 32 bytes in the WIDEST element it is used with:
#: 32/2 = 16 lanes for fp16/bf16 and 32/4 = 8 for fp32.  16 covers both.
#: (Same reasoning as common.SPATIAL_ELEM_GRAIN; this template has no packed
#: masks, so the 128 packed-mask grain does not apply.)
ELEM_GRAIN = 16

#: ``vector_core_cnt`` is 48 on ascend910b1 and ``T.Kernel(N)`` with a
#: ``(cid, vid)`` binding gives N blocks x 2 vector lanes.  R194 measured
#: 48-192 launch blocks as a plateau; stay at the low end.
LAUNCH_BLOCK_CAP = 48
VECTOR_LANES = 2 * LAUNCH_BLOCK_CAP

#: Below this a work item is all DMA/issue overhead and no payload.
MIN_TILE_ELEMS = 256
#: R194 probed that past ~16384 elements the extra width buys no bandwidth;
#: these kernels hold 6 buffers per tile so the UB budget binds first anyway.
MAX_TILE_ELEMS = 4096


def _round_up(value, grain):
    return ((int(value) + int(grain) - 1) // int(grain)) * int(grain)


def _round_down(value, grain):
    return (int(value) // int(grain)) * int(grain)


def _plan_row_tile(rows, cols, bytes_per_elem, *, grain=ELEM_GRAIN,
                   min_tile=MIN_TILE_ELEMS, max_tile=MAX_TILE_ELEMS,
                   ub_budget=UB_BUDGET_BYTES, lanes=VECTOR_LANES):
    """Split an ``(rows, cols)`` row-major job into ``rows * n_tiles`` work
    items of at most ``tile`` columns each.

    ``bytes_per_elem`` is the UB cost of ONE column element summed over every
    ``(tile,)`` buffer the kernel allocates -- caller must derive it by reading
    its own allocation list (T195 step 1.1; guessing it is a silent overflow).

    The last tile of a row OVERLAPS the one before it (``col = min(t*tile,
    cols-tile)``) instead of running short.  That keeps every copy extent a
    compile-time constant -- no ragged branch, and in particular no
    ``for lane in T.serial(tile): if lane < tail`` scalar tail, which is the
    shape that costs 2.65x in the unary template and much more here.  The
    overlapping lanes recompute the same value from the same inputs and write
    the same bytes, so the duplicate GM write is value-identical.
    """
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    ub_cap = _round_down(max(grain, ub_budget // max(1, int(bytes_per_elem))), grain)
    want_tiles = max(1, -(-int(lanes) // rows))
    tile = _round_up(-(-cols // want_tiles), grain)
    tile = max(tile, int(min_tile))
    tile = min(tile, int(max_tile), ub_cap, _round_up(cols, grain))
    tile = max(tile, grain)
    n_tiles = max(1, -(-cols // tile))
    copy_w = cols if n_tiles == 1 else tile
    items = rows * n_tiles
    logical_blocks = max(1, -(-items // 2))
    launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
    return {
        "tile": tile,
        "n_tiles": n_tiles,
        "copy_w": copy_w,
        "last_col": max(0, cols - copy_w),
        "items": items,
        "logical_blocks": logical_blocks,
        "launch_blocks": launch_blocks,
        "grid_repeats": grid_repeat_count(logical_blocks, launch_blocks),
        "ub_bytes": tile * int(bytes_per_elem),
    }


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise TypeError(f"EngramGateConv supports float16/bfloat16, got {dtype}")


#: A scalar GM store from a vector lane is LOST when another lane stores into
#: the same 32-byte line.  R201 measured it directly: one row per lane, 128 rows,
#: five runs -> 91/101/108/103/106 rows never written and never a wrong value,
#: non-identical run to run; the same rows written as one 8 x fp32 aligned
#: ``T.copy`` per lane -> 0 missing, 0 wrong, identical 5/5
#: (docs/reports/R201-data/probe_scalar_gm_store.log).  This is a FIFTH
#: mechanism next to 13.44's three and 13.54's one, and unlike those it presents
#: as MISSING writes rather than wrong values.
#:
#: So every per-row fp32 statistic this family produces is staged in UB and
#: flushed one 32-byte group at a time.  32 / 4 = 8 rows per group.
STAT_GROUP = 8


def _engram_group_plan(rows: int, group: int = STAT_GROUP):
    """Grid geometry for a kernel whose per-row outputs include fp32 SCALARS.

    One work item owns ``group`` consecutive rows so that its statistic writes
    are a single 32-byte-aligned DMA (see STAT_GROUP).
    """
    rows = max(1, int(rows))
    rows_pad = _round_up(rows, group)
    n_groups = rows_pad // group
    logical_blocks = max(1, -(-n_groups // 2))
    launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
    return {
        "rows": rows,
        "rows_pad": rows_pad,
        "group": group,
        "n_groups": n_groups,
        "logical_blocks": logical_blocks,
        "launch_blocks": launch_blocks,
        "grid_repeats": grid_repeat_count(logical_blocks, launch_blocks),
    }


def _engram_row_plan(rows: int):
    """Grid geometry for the per-row Engram kernels.

    Every Engram row (bid, tid) is independent inside a pass, but the pre-R201
    kernel ran all of them on ``T.Kernel(1)`` under ``if vid == 0`` -- one of the
    96 vector lanes (13.53's "96 cores, 1 used", here squared with 13.52's
    scalar loops).  One work item per row, two per launch block.
    """
    rows = max(1, int(rows))
    logical_blocks = max(1, -(-rows // 2))
    launch_blocks = launch_block_count(min(logical_blocks, LAUNCH_BLOCK_CAP))
    return {
        "rows": rows,
        "logical_blocks": logical_blocks,
        "launch_blocks": launch_blocks,
        "grid_repeats": grid_repeat_count(logical_blocks, launch_blocks),
    }


@lru_cache(maxsize=64)
def _compile_engram_fwd_pass1(M: int, seq_len: int, d: int, eps: float, dtype_name: str):
    """Per-row RMSNorm statistics, gate, and ``vhat``.

    R201: the four ``for j in T.serial(d)`` loops of the pre-R201 pass 1
    (sum_h/sum_k, dot_hk, the gate multiply, sum_v) are 40% of the op's scalar
    iteration count; they become ``T.tile.mul`` + ``T.reduce_sum``, and the rows
    become the grid instead of running on one lane under ``if vid == 0``.

    ⚠️ One work item owns ``STAT_GROUP`` consecutive rows, not one row.  The four
    fp32 statistics are scalars, and a scalar GM store from a vector lane is
    lost when a neighbouring lane stores into the same 32-byte line -- see
    STAT_GROUP.  They are staged in UB and flushed with one aligned ``T.copy``
    per group, which is why ``alpha`` / ``rrms_*`` are FLAT and padded to
    ``rows_pad`` here; ``build_engram_fwd_kernel`` hands the caller a
    ``[:rows].view(M, seq_len)`` of them, which is a free view of a contiguous
    prefix.

    ⚠️ The three square-sum reductions and ``dot_hk`` change summation ORDER
    (serial -> vector tree), so this pass is NOT bit-identical to the pre-R201
    kernel; see R201 for the float64 scoring.  The gate multiply
    ``vhat = gate * v`` and the dtype round trip are unchanged.
    """
    d_pad = math.ceil(d / 32) * 32
    plan = _engram_group_plan(M * seq_len)
    rows, rows_pad = plan["rows"], plan["rows_pad"]
    group, n_groups = plan["group"], plan["n_groups"]
    launch_blocks, grid_repeats = plan["launch_blocks"], plan["grid_repeats"]

    @tilelang.jit(out_idx=[], pass_configs=_PASS_CONFIGS,
                  compile_flags=["-O3", "-DENABLE_BF16"])
    def factory():
        @T.prim_func
        def main(
            H: T.Tensor((M, seq_len, d), dtype_name),
            k: T.Tensor((M, seq_len, d), dtype_name),
            v: T.Tensor((M, seq_len, d), dtype_name),
            rms_w_h: T.Tensor((d,), dtype_name),
            vhat: T.Tensor((M, seq_len, d), dtype_name),
            alpha: T.Tensor((rows_pad,), "float32"),
            rrms_h: T.Tensor((rows_pad,), "float32"),
            rrms_k: T.Tensor((rows_pad,), "float32"),
            rrms_v: T.Tensor((rows_pad,), "float32"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                h_typed = T.alloc_ub((d_pad,), dtype_name)
                k_typed = T.alloc_ub((d_pad,), dtype_name)
                v_typed = T.alloc_ub((d_pad,), dtype_name)
                w_h_typed = T.alloc_ub((d_pad,), dtype_name)
                out_typed = T.alloc_ub((d_pad,), dtype_name)
                h32 = T.alloc_ub((d_pad,), "float32")
                k32 = T.alloc_ub((d_pad,), "float32")
                v32 = T.alloc_ub((d_pad,), "float32")
                w_h32 = T.alloc_ub((d_pad,), "float32")
                vhat32 = T.alloc_ub((d_pad,), "float32")
                tmp32 = T.alloc_ub((d_pad,), "float32")
                sq = T.alloc_ub((1, d_pad), "float32")
                red = T.alloc_ub((1,), "float32")
                denom = T.alloc_ub((32,), "float32")
                inv = T.alloc_ub((32,), "float32")
                refine = T.alloc_ub((32,), "float32")
                exp_in = T.alloc_ub((32,), "float32")
                exp_out = T.alloc_ub((32,), "float32")
                st_a = T.alloc_ub((group,), "float32")
                st_h = T.alloc_ub((group,), "float32")
                st_k = T.alloc_ub((group,), "float32")
                st_v = T.alloc_ub((group,), "float32")
                with T.Scope("V"):
                    # Zero the [d, d_pad) padding lanes once: they feed every
                    # reduction below and 0 is the additive identity.  The loop
                    # body only ever overwrites [0, d).
                    T.tile.fill(h_typed, 0.0)
                    T.tile.fill(k_typed, 0.0)
                    T.tile.fill(v_typed, 0.0)
                    T.tile.fill(w_h_typed, 0.0)
                    T.copy(rms_w_h[0:d], w_h_typed[0:d])
                    T.barrier_all()                      # MTE2 -> V (13.55 #1)
                    T.tile.cast(w_h32, w_h_typed, "CAST_NONE", d_pad)
                    for rep in T.serial(grid_repeats):
                        g = (cid + rep * launch_blocks) * 2 + vid
                        if g < n_groups:
                            # The trailing group may run past ``rows``; fill so
                            # the padded slots are deterministic zeros rather
                            # than whatever the allocator left.
                            T.tile.fill(st_a, 0.0)
                            T.tile.fill(st_h, 0.0)
                            T.tile.fill(st_k, 0.0)
                            T.tile.fill(st_v, 0.0)
                            for i in T.serial(group):
                                row = g * group + i
                                if row < rows:
                                    bid = row // seq_len
                                    tid = row % seq_len
                                    T.copy(H[bid, tid, 0:d], h_typed[0:d])
                                    T.copy(k[bid, tid, 0:d], k_typed[0:d])
                                    T.copy(v[bid, tid, 0:d], v_typed[0:d])
                                    T.barrier_all()      # MTE2 -> V
                                    T.tile.cast(h32, h_typed, "CAST_NONE", d_pad)
                                    T.tile.cast(k32, k_typed, "CAST_NONE", d_pad)
                                    T.tile.cast(v32, v_typed, "CAST_NONE", d_pad)

                                    T.tile.mul(sq, h32, h32)
                                    T.reduce_sum(sq, red, dim=-1, clear=True)
                                    denom[0] = red[0] / T.float32(d) + T.float32(eps)
                                    T.tile.rsqrt(inv, denom)
                                    T.tile.mul(refine, inv, inv)
                                    T.tile.mul(refine, refine, denom)
                                    inv[0] = inv[0] * (1.5 - 0.5 * refine[0])
                                    st_h[i] = inv[0]

                                    T.tile.mul(sq, k32, k32)
                                    T.reduce_sum(sq, red, dim=-1, clear=True)
                                    denom[0] = red[0] / T.float32(d) + T.float32(eps)
                                    T.tile.rsqrt(inv, denom)
                                    T.tile.mul(refine, inv, inv)
                                    T.tile.mul(refine, refine, denom)
                                    inv[0] = inv[0] * (1.5 - 0.5 * refine[0])
                                    st_k[i] = inv[0]

                                    # dot_hk = sum_j (h*rh*w) * (k*rk*w); rh and
                                    # rk are row scalars, so they factor out of
                                    # the reduction.
                                    T.tile.mul(tmp32, h32, w_h32)
                                    T.tile.mul(sq, k32, w_h32)
                                    T.tile.mul(sq, sq, tmp32)
                                    T.reduce_sum(sq, red, dim=-1, clear=True)
                                    exp_in[0] = -(red[0] * st_h[i] * st_k[i]
                                                  / T.float32(math.sqrt(d)))
                                    T.tile.exp(exp_out, exp_in)
                                    st_a[i] = 1.0 / (1.0 + exp_out[0])

                                    T.tile.mul(vhat32, v32, st_a[i])
                                    T.barrier_all()      # V -> MTE3 (13.55 #2)
                                    T.tile.cast(out_typed, vhat32, "CAST_RINT", d_pad)
                                    T.barrier_all()
                                    T.copy(out_typed[0:d], vhat[bid, tid, 0:d])
                                    # Match the reference's v_hat.to(dtype) boundary.
                                    T.tile.cast(vhat32, out_typed, "CAST_NONE", d_pad)
                                    T.tile.mul(sq, vhat32, vhat32)
                                    T.reduce_sum(sq, red, dim=-1, clear=True)
                                    denom[0] = red[0] / T.float32(d) + T.float32(eps)
                                    T.tile.rsqrt(inv, denom)
                                    T.tile.mul(refine, inv, inv)
                                    T.tile.mul(refine, refine, denom)
                                    inv[0] = inv[0] * (1.5 - 0.5 * refine[0])
                                    st_v[i] = inv[0]
                            T.barrier_all()              # V -> MTE3
                            T.copy(st_a, alpha[g * group:g * group + group])
                            T.copy(st_h, rrms_h[g * group:g * group + group])
                            T.copy(st_k, rrms_k[g * group:g * group + group])
                            T.copy(st_v, rrms_v[g * group:g * group + group])

        return main

    compiled = factory()
    compiled.r201_plan = dict(plan, d_pad=d_pad)
    return compiled


@lru_cache(maxsize=64)
def _compile_engram_fwd_pass2(M: int, seq_len: int, d: int, dtype_name: str):
    """Causal depthwise conv over ``vhat``, SiLU, residual.

    Split from pass 1 into its OWN LAUNCH because it reads ``vhat`` rows
    ``t-3 .. t`` that other blocks produced: ``T.barrier_all()`` is an in-core
    pipeline barrier, not a device barrier, so the cross-row dependency has to
    be carried by the kernel boundary.
    """
    kernel_size = 4
    d_pad = math.ceil(d / 32) * 32
    seq_pad = _round_up(seq_len, 8)
    plan = _engram_row_plan(M * seq_len)
    rows = plan["rows"]
    rows_pad = _round_up(rows, STAT_GROUP)
    launch_blocks = plan["launch_blocks"]
    grid_repeats = plan["grid_repeats"]

    @tilelang.jit(out_idx=[], pass_configs=_PASS_CONFIGS,
                  compile_flags=["-O3", "-DENABLE_BF16"])
    def factory():
        @T.prim_func
        def main(
            vhat: T.Tensor((M, seq_len, d), dtype_name),
            rms_w_v: T.Tensor((d,), dtype_name),
            conv_w: T.Tensor((kernel_size, d), dtype_name),
            rrms_v: T.Tensor((rows_pad,), "float32"),
            Y: T.Tensor((M, seq_len, d), dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                v_typed = T.alloc_ub((d_pad,), dtype_name)
                conv_w_typed = T.alloc_ub((d_pad,), dtype_name)
                w_v_typed = T.alloc_ub((d_pad,), dtype_name)
                out_typed = T.alloc_ub((d_pad,), dtype_name)
                v32 = T.alloc_ub((d_pad,), "float32")
                conv_w32 = T.alloc_ub((d_pad,), "float32")
                w_v32 = T.alloc_ub((d_pad,), "float32")
                conv32 = T.alloc_ub((d_pad,), "float32")
                out32 = T.alloc_ub((d_pad,), "float32")
                tmp32 = T.alloc_ub((d_pad,), "float32")
                ones = T.alloc_ub((d_pad,), "float32")
                rv_row = T.alloc_ub((seq_pad,), "float32")
                with T.Scope("V"):
                    T.tile.fill(v_typed, 0.0)
                    T.tile.fill(conv_w_typed, 0.0)
                    T.tile.fill(w_v_typed, 0.0)
                    T.tile.fill(ones, 1.0)
                    T.copy(rms_w_v[0:d], w_v_typed[0:d])
                    T.barrier_all()                      # MTE2 -> V
                    T.tile.cast(w_v32, w_v_typed, "CAST_NONE", d_pad)
                    for rep in T.serial(grid_repeats):
                        row = (cid + rep * launch_blocks) * 2 + vid
                        if row < rows:
                            bid = row // seq_len
                            tid = row % seq_len
                            # ⚠️ rrms_v must reach the ``muls`` intrinsic from UB.
                            # A GM BufferLoad operand makes binary_op emit
                            # ``muls(dst, src, <GM ptr>, idx)``, and staging it
                            # in a ``T.alloc_var`` inside the nested ``if src_t
                            # >= 0:`` is rejected outright ("Check store buffer:
                            # rv is not a global or shared or local buffer").
                            # Copying the row is the form that works
                            # (R201-data/probe_primitives.log q8).
                            T.copy(rrms_v[bid * seq_len:bid * seq_len + seq_len],
                                   rv_row[0:seq_len])
                            T.barrier_all()
                            T.tile.fill(conv32, 0.0)
                            for p in T.serial(kernel_size):
                                src_t = tid - (kernel_size - 1) + p
                                if src_t >= 0:
                                    T.copy(vhat[bid, src_t, 0:d], v_typed[0:d])
                                    T.copy(conv_w[p, 0:d], conv_w_typed[0:d])
                                    T.barrier_all()      # MTE2 -> V
                                    T.tile.cast(v32, v_typed, "CAST_NONE", d_pad)
                                    T.tile.cast(conv_w32, conv_w_typed, "CAST_NONE", d_pad)
                                    # ⚠️ rrms_v[bid, src_t] must go through an
                                    # alloc_var: binary_op routes a BufferLoad
                                    # operand to ``muls(dst, src, buf_ptr, idx)``
                                    # and that pointer would be GM, not UB.
                                    # Same association as the pre-R201 body:
                                    # ((v * rv) * w_v) * conv_w, then accumulate.
                                    T.tile.mul(tmp32, v32, rv_row[src_t])
                                    T.tile.mul(tmp32, tmp32, w_v32)
                                    T.tile.mul(tmp32, tmp32, conv_w32)
                                    T.tile.add(conv32, conv32, tmp32)
                            # SiLU written as the pre-R201 scalar chain did it:
                            # 1 / (1 + exp(-x)), then x * that.
                            T.tile.mul(tmp32, conv32, -1.0)
                            T.tile.exp(tmp32, tmp32)
                            T.tile.add(tmp32, tmp32, 1.0)
                            T.tile.div(out32, ones, tmp32)
                            T.tile.mul(out32, conv32, out32)
                            T.copy(vhat[bid, tid, 0:d], v_typed[0:d])
                            T.barrier_all()              # MTE2 -> V
                            T.tile.cast(v32, v_typed, "CAST_NONE", d_pad)
                            T.tile.add(out32, out32, v32)
                            T.barrier_all()              # V -> MTE3
                            T.tile.cast(out_typed, out32, "CAST_RINT", d_pad)
                            T.barrier_all()
                            T.copy(out_typed[0:d], Y[bid, tid, 0:d])

        return main

    compiled = factory()
    compiled.r201_plan = dict(plan, d_pad=d_pad)
    return compiled


def build_engram_fwd_kernel(M, seq_len, d, eps, dtype):
    M, seq_len, d = int(M), int(seq_len), int(d)
    if min(M, seq_len, d) <= 0:
        raise ValueError("EngramGateConvFwdOp dimensions must be positive")
    dtype_name = _dtype_name(dtype)
    pass1 = _compile_engram_fwd_pass1(M, seq_len, d, float(eps), dtype_name)
    pass2 = _compile_engram_fwd_pass2(M, seq_len, d, dtype_name)

    rows = M * seq_len
    rows_pad = pass1.r201_plan["rows_pad"]

    def invoke(H, k, v, rms_w_h, rms_w_v, conv_w):
        device = H.device
        Y = torch.full_like(H, float("nan"))
        vhat = torch.full_like(H, float("nan"))
        # Flat and padded to a whole STAT_GROUP so the kernel can flush each
        # group with one 32-byte-aligned DMA.  ``[:rows].view(M, seq_len)`` is a
        # view of a contiguous prefix, so the caller-visible tensors cost
        # nothing.
        def _stat():
            return torch.full((rows_pad,), float("nan"), dtype=torch.float32, device=device)
        alpha_f, rrms_h_f, rrms_k_f, rrms_v_f = _stat(), _stat(), _stat(), _stat()
        pass1(H, k, v, rms_w_h, vhat, alpha_f, rrms_h_f, rrms_k_f, rrms_v_f)
        pass2(vhat, rms_w_v, conv_w, rrms_v_f, Y)
        return (Y, vhat,
                alpha_f[:rows].view(M, seq_len),
                rrms_h_f[:rows].view(M, seq_len),
                rrms_k_f[:rows].view(M, seq_len),
                rrms_v_f[:rows].view(M, seq_len))

    invoke.compiled = pass1
    invoke.compiled_pass2 = pass2
    invoke.launch_blocks = pass1.r201_plan["launch_blocks"]
    invoke.plan = pass1.r201_plan
    invoke.path_kind = "aiv_row_parallel_two_launch_causal_window"
    return invoke


#: Slots of the per-lane partial buffer shared by the backward stages:
#: 0..3 = dconv_w taps, 4 = drms_w_h, 5 = drms_w_v.  Each slot is d_pad wide so
#: every lane's write is 32-byte aligned (see STAT_GROUP).
_BWD_PART_SLOTS = 6


@lru_cache(maxsize=64)
def _compile_engram_bwd_stage1(M: int, seq_len: int, d: int, dtype_name: str):
    """Recompute the causal conv, apply the SiLU derivative, cache ``d_silu``.

    Row-parallel.  Everything the pre-R201 stage 1 did per lane -- two
    ``for j in T.serial(d)`` loops, one of them issuing a ``T.tile.exp`` PER
    LANE -- is a vector chain here.  All outputs are ``d``-element rows written
    by DMA, so this stage needs no STAT_GROUP treatment.
    """
    kernel_size = 4
    d_pad = math.ceil(d / 32) * 32
    seq_pad = _round_up(seq_len, 8)
    plan = _engram_row_plan(M * seq_len)
    rows, launch_blocks, grid_repeats = (
        plan["rows"], plan["launch_blocks"], plan["grid_repeats"])

    @tilelang.jit(out_idx=[], pass_configs=_PASS_CONFIGS,
                  compile_flags=["-O3", "-DENABLE_BF16"])
    def factory():
        @T.prim_func
        def main(
            dY: T.Tensor((M, seq_len, d), dtype_name),
            rms_w_v: T.Tensor((d,), dtype_name),
            conv_w: T.Tensor((kernel_size, d), dtype_name),
            vhat: T.Tensor((M, seq_len, d), dtype_name),
            rrms_v: T.Tensor((M, seq_len), "float32"),
            d_silu: T.Tensor((M, seq_len, d), "float32"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                typed0 = T.alloc_ub((d_pad,), dtype_name)
                typed1 = T.alloc_ub((d_pad,), dtype_name)
                typed2 = T.alloc_ub((d_pad,), dtype_name)
                x0 = T.alloc_ub((d_pad,), "float32")
                x1 = T.alloc_ub((d_pad,), "float32")
                x2 = T.alloc_ub((d_pad,), "float32")
                work0 = T.alloc_ub((d_pad,), "float32")
                sig = T.alloc_ub((d_pad,), "float32")
                tmp = T.alloc_ub((d_pad,), "float32")
                w_v32 = T.alloc_ub((d_pad,), "float32")
                ones = T.alloc_ub((d_pad,), "float32")
                rvb = T.alloc_ub((seq_pad,), "float32")
                with T.Scope("V"):
                    T.tile.fill(typed0, 0.0)
                    T.tile.fill(typed1, 0.0)
                    T.tile.fill(typed2, 0.0)
                    T.tile.fill(ones, 1.0)
                    T.copy(rms_w_v[0:d], typed0[0:d])
                    T.barrier_all()
                    T.tile.cast(w_v32, typed0, "CAST_NONE", d_pad)
                    for rep in T.serial(grid_repeats):
                        row = (cid + rep * launch_blocks) * 2 + vid
                        if row < rows:
                            bid = row // seq_len
                            tid = row % seq_len
                            T.copy(rrms_v[bid, 0:seq_len], rvb[0:seq_len])
                            T.copy(dY[bid, tid, 0:d], typed0[0:d])
                            T.barrier_all()
                            T.tile.cast(x0, typed0, "CAST_NONE", d_pad)
                            T.tile.fill(work0, 0.0)
                            for p in T.serial(kernel_size):
                                src_t = tid - (kernel_size - 1) + p
                                if src_t >= 0:
                                    T.copy(vhat[bid, src_t, 0:d], typed1[0:d])
                                    T.copy(conv_w[p, 0:d], typed2[0:d])
                                    T.barrier_all()
                                    T.tile.cast(x1, typed1, "CAST_NONE", d_pad)
                                    T.tile.cast(x2, typed2, "CAST_NONE", d_pad)
                                    T.tile.mul(tmp, x1, rvb[src_t])
                                    T.tile.mul(tmp, tmp, w_v32)
                                    T.tile.mul(tmp, tmp, x2)
                                    T.tile.add(work0, work0, tmp)
                            # sig = 1/(1+exp(-conv)); dY * (sig + conv*sig*(1-sig))
                            T.tile.mul(tmp, work0, -1.0)
                            T.tile.exp(tmp, tmp)
                            T.tile.add(tmp, tmp, 1.0)
                            T.tile.div(sig, ones, tmp)
                            T.tile.mul(tmp, work0, sig)
                            T.tile.sub(x1, ones, sig)
                            T.tile.mul(tmp, tmp, x1)
                            T.tile.add(tmp, sig, tmp)
                            T.tile.mul(tmp, x0, tmp)
                            T.barrier_all()
                            T.copy(tmp[0:d], d_silu[bid, tid, 0:d])

        return main

    compiled = factory()
    compiled.r201_plan = dict(plan, d_pad=d_pad)
    return compiled


@lru_cache(maxsize=64)
def _compile_engram_bwd_stage23(M: int, seq_len: int, d: int, dtype_name: str):
    """``dconv_w`` partials (stage 2) and the RMSNorm(vhat) backward (stage 3).

    Both read only ``d_silu`` plus the forward saves, so they share one launch
    and one row-parallel loop.  The two cross-row accumulators (``dconv_w`` and
    ``drms_w_v``) are kept per LANE in UB and flushed to a GM partial buffer;
    ``_compile_engram_bwd_reduce`` folds the lanes.  Lane assignment is fixed by
    the geometry, so the fold is deterministic -- deliberately not
    ``T.tile.atomic_add`` (13.44: a non-deterministic reduction is a
    non-deterministic wrong value waiting to happen).
    """
    kernel_size = 4
    d_pad = math.ceil(d / 32) * 32
    seq_pad = _round_up(seq_len, 8)
    plan = _engram_row_plan(M * seq_len)
    rows, launch_blocks, grid_repeats = (
        plan["rows"], plan["launch_blocks"], plan["grid_repeats"])
    lanes = launch_blocks * 2
    part_len = lanes * _BWD_PART_SLOTS * d_pad

    @tilelang.jit(out_idx=[], pass_configs=_PASS_CONFIGS,
                  compile_flags=["-O3", "-DENABLE_BF16"])
    def factory():
        @T.prim_func
        def main(
            dY: T.Tensor((M, seq_len, d), dtype_name),
            rms_w_v: T.Tensor((d,), dtype_name),
            conv_w: T.Tensor((kernel_size, d), dtype_name),
            vhat: T.Tensor((M, seq_len, d), dtype_name),
            rrms_v: T.Tensor((M, seq_len), "float32"),
            d_silu: T.Tensor((M, seq_len, d), "float32"),
            dvhat_tmp: T.Tensor((M, seq_len, d), "float32"),
            part: T.Tensor((part_len,), "float32"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                typed0 = T.alloc_ub((d_pad,), dtype_name)
                typed1 = T.alloc_ub((d_pad,), dtype_name)
                x0 = T.alloc_ub((d_pad,), "float32")
                x1 = T.alloc_ub((d_pad,), "float32")
                x2 = T.alloc_ub((d_pad,), "float32")
                work0 = T.alloc_ub((d_pad,), "float32")
                tmp = T.alloc_ub((d_pad,), "float32")
                w_v32 = T.alloc_ub((d_pad,), "float32")
                dsil = T.alloc_ub((d_pad,), "float32")
                dconv_acc = T.alloc_ub((kernel_size * d_pad,), "float32")
                drms_v_acc = T.alloc_ub((d_pad,), "float32")
                sq = T.alloc_ub((1, d_pad), "float32")
                red = T.alloc_ub((1,), "float32")
                sc = T.alloc_ub((8,), "float32")
                rvb = T.alloc_ub((seq_pad,), "float32")
                with T.Scope("V"):
                    T.tile.fill(typed0, 0.0)
                    T.tile.fill(typed1, 0.0)
                    T.tile.fill(dconv_acc, 0.0)
                    T.tile.fill(drms_v_acc, 0.0)
                    T.tile.fill(dsil, 0.0)
                    T.tile.fill(x0, 0.0)
                    T.tile.fill(tmp, 0.0)
                    T.copy(rms_w_v[0:d], typed0[0:d])
                    T.barrier_all()
                    T.tile.cast(w_v32, typed0, "CAST_NONE", d_pad)
                    for rep in T.serial(grid_repeats):
                        row = (cid + rep * launch_blocks) * 2 + vid
                        if row < rows:
                            bid = row // seq_len
                            tid = row % seq_len
                            T.copy(rrms_v[bid, 0:seq_len], rvb[0:seq_len])
                            T.copy(d_silu[bid, tid, 0:d], dsil[0:d])
                            T.barrier_all()
                            # ---- stage 2: this row's share of dconv_w[p].
                            for p in T.serial(kernel_size):
                                src_t = tid - (kernel_size - 1) + p
                                if src_t >= 0:
                                    T.copy(vhat[bid, src_t, 0:d], typed1[0:d])
                                    T.barrier_all()
                                    T.tile.cast(x1, typed1, "CAST_NONE", d_pad)
                                    T.tile.mul(tmp, x1, rvb[src_t])
                                    T.tile.mul(tmp, tmp, w_v32)
                                    T.tile.mul(tmp, tmp, dsil)
                                    T.tile.add(dconv_acc[p * d_pad:(p + 1) * d_pad],
                                               dconv_acc[p * d_pad:(p + 1) * d_pad], tmp)
                            # ---- stage 3: transpose the causal window.
                            T.tile.fill(work0, 0.0)
                            for p in T.serial(kernel_size):
                                dst_t = tid + (kernel_size - 1) - p
                                if dst_t < seq_len:
                                    T.copy(conv_w[p, 0:d], typed1[0:d])
                                    T.copy(d_silu[bid, dst_t, 0:d], tmp[0:d])
                                    T.barrier_all()
                                    T.tile.cast(x0, typed1, "CAST_NONE", d_pad)
                                    T.tile.mul(tmp, x0, tmp)
                                    T.tile.add(work0, work0, tmp)
                            T.copy(vhat[bid, tid, 0:d], typed0[0:d])
                            T.copy(dY[bid, tid, 0:d], typed1[0:d])
                            T.barrier_all()
                            T.tile.cast(x0, typed0, "CAST_NONE", d_pad)
                            T.tile.cast(x1, typed1, "CAST_NONE", d_pad)
                            # drms_w_v += work0 * vhat * rv
                            T.tile.mul(tmp, work0, x0)
                            T.tile.mul(tmp, tmp, rvb[tid])
                            T.tile.add(drms_v_acc, drms_v_acc, tmp)
                            # dot_v = sum_j vhat * w_v * work0
                            T.tile.mul(sq, x0, w_v32)
                            T.tile.mul(sq, sq, work0)
                            T.reduce_sum(sq, red, dim=-1, clear=True)
                            # ⚠️ materialise the reduction result in UB before
                            # the next reduce_sum overwrites ``red``: a Python
                            # name bound to ``red[0]`` is an ALIAS, re-read at
                            # every use site, not a captured value.
                            sc[0] = red[0]
                            # dvhat = rv*w_v*work0 - rv^3*vhat*dot_v/d + dY
                            T.tile.mul(x2, w_v32, rvb[tid])
                            T.tile.mul(x2, x2, work0)
                            T.tile.mul(tmp, x0,
                                       rvb[tid] * rvb[tid] * rvb[tid] * sc[0]
                                       / T.float32(d))
                            T.tile.sub(x2, x2, tmp)
                            T.tile.add(x2, x2, x1)
                            T.barrier_all()
                            T.copy(x2[0:d], dvhat_tmp[bid, tid, 0:d])
                    lane = cid * 2 + vid
                    T.barrier_all()
                    T.copy(dconv_acc,
                           part[lane * _BWD_PART_SLOTS * d_pad:
                                (lane * _BWD_PART_SLOTS + kernel_size) * d_pad])
                    T.copy(drms_v_acc,
                           part[(lane * _BWD_PART_SLOTS + 5) * d_pad:
                                (lane * _BWD_PART_SLOTS + 6) * d_pad])

        return main

    compiled = factory()
    compiled.r201_plan = dict(plan, d_pad=d_pad, lanes=lanes, part_len=part_len)
    return compiled


@lru_cache(maxsize=64)
def _compile_engram_bwd_stage4(M: int, seq_len: int, d: int, dtype_name: str):
    """Gate backward and both RMSNorm(H) / RMSNorm(k) backward paths."""
    d_pad = math.ceil(d / 32) * 32
    seq_pad = _round_up(seq_len, 8)
    plan = _engram_row_plan(M * seq_len)
    rows, launch_blocks, grid_repeats = (
        plan["rows"], plan["launch_blocks"], plan["grid_repeats"])
    lanes = launch_blocks * 2
    part_len = lanes * _BWD_PART_SLOTS * d_pad

    @tilelang.jit(out_idx=[], pass_configs=_PASS_CONFIGS,
                  compile_flags=["-O3", "-DENABLE_BF16"])
    def factory():
        @T.prim_func
        def main(
            H: T.Tensor((M, seq_len, d), dtype_name),
            k: T.Tensor((M, seq_len, d), dtype_name),
            v: T.Tensor((M, seq_len, d), dtype_name),
            rms_w_h: T.Tensor((d,), dtype_name),
            alpha: T.Tensor((M, seq_len), "float32"),
            rrms_h: T.Tensor((M, seq_len), "float32"),
            rrms_k: T.Tensor((M, seq_len), "float32"),
            dvhat_tmp: T.Tensor((M, seq_len, d), "float32"),
            dH: T.Tensor((M, seq_len, d), dtype_name),
            dk: T.Tensor((M, seq_len, d), dtype_name),
            dv: T.Tensor((M, seq_len, d), dtype_name),
            part: T.Tensor((part_len,), "float32"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                typed0 = T.alloc_ub((d_pad,), dtype_name)
                typed1 = T.alloc_ub((d_pad,), dtype_name)
                typed2 = T.alloc_ub((d_pad,), dtype_name)
                typed_out = T.alloc_ub((d_pad,), dtype_name)
                x0 = T.alloc_ub((d_pad,), "float32")
                x1 = T.alloc_ub((d_pad,), "float32")
                x2 = T.alloc_ub((d_pad,), "float32")
                x3 = T.alloc_ub((d_pad,), "float32")
                dvh = T.alloc_ub((d_pad,), "float32")
                work0 = T.alloc_ub((d_pad,), "float32")
                work1 = T.alloc_ub((d_pad,), "float32")
                work2 = T.alloc_ub((d_pad,), "float32")
                tmp = T.alloc_ub((d_pad,), "float32")
                w_h32 = T.alloc_ub((d_pad,), "float32")
                drms_h_acc = T.alloc_ub((d_pad,), "float32")
                sq = T.alloc_ub((1, d_pad), "float32")
                red = T.alloc_ub((1,), "float32")
                sc = T.alloc_ub((8,), "float32")
                ab = T.alloc_ub((seq_pad,), "float32")
                rhb = T.alloc_ub((seq_pad,), "float32")
                rkb = T.alloc_ub((seq_pad,), "float32")
                with T.Scope("V"):
                    T.tile.fill(typed0, 0.0)
                    T.tile.fill(typed1, 0.0)
                    T.tile.fill(typed2, 0.0)
                    T.tile.fill(drms_h_acc, 0.0)
                    T.tile.fill(dvh, 0.0)
                    T.copy(rms_w_h[0:d], typed0[0:d])
                    T.barrier_all()
                    T.tile.cast(w_h32, typed0, "CAST_NONE", d_pad)
                    for rep in T.serial(grid_repeats):
                        row = (cid + rep * launch_blocks) * 2 + vid
                        if row < rows:
                            bid = row // seq_len
                            tid = row % seq_len
                            T.copy(alpha[bid, 0:seq_len], ab[0:seq_len])
                            T.copy(rrms_h[bid, 0:seq_len], rhb[0:seq_len])
                            T.copy(rrms_k[bid, 0:seq_len], rkb[0:seq_len])
                            T.copy(H[bid, tid, 0:d], typed0[0:d])
                            T.copy(k[bid, tid, 0:d], typed1[0:d])
                            T.copy(v[bid, tid, 0:d], typed2[0:d])
                            T.copy(dvhat_tmp[bid, tid, 0:d], dvh[0:d])
                            T.barrier_all()
                            T.tile.cast(x0, typed0, "CAST_NONE", d_pad)
                            T.tile.cast(x1, typed1, "CAST_NONE", d_pad)
                            T.tile.cast(x2, typed2, "CAST_NONE", d_pad)
                            T.tile.mul(sq, dvh, x2)
                            T.reduce_sum(sq, red, dim=-1, clear=True)
                            # sc[1] = ddot = dalpha * gate * (1-gate) / sqrt(d)
                            sc[1] = (red[0] * ab[tid] * (1.0 - ab[tid])
                                     / T.float32(math.sqrt(d)))
                            T.tile.mul(x3, x0, rhb[tid])
                            T.tile.mul(x3, x3, w_h32)
                            T.tile.mul(work0, x1, rkb[tid])
                            T.tile.mul(work0, work0, w_h32)
                            # dot_h = sum_j H * w_h * ddot * work0
                            T.tile.mul(sq, x0, w_h32)
                            T.tile.mul(sq, sq, sc[1])
                            T.tile.mul(sq, sq, work0)
                            T.reduce_sum(sq, red, dim=-1, clear=True)
                            sc[2] = red[0]
                            # dot_k = sum_j k * w_h * ddot * x3
                            T.tile.mul(sq, x1, w_h32)
                            T.tile.mul(sq, sq, sc[1])
                            T.tile.mul(sq, sq, x3)
                            T.reduce_sum(sq, red, dim=-1, clear=True)
                            sc[3] = red[0]
                            # work1 = rh*w_h*ddot*work0 - rh^3*H*dot_h/d
                            T.tile.mul(work1, w_h32, rhb[tid])
                            T.tile.mul(work1, work1, sc[1])
                            T.tile.mul(work1, work1, work0)
                            T.tile.mul(tmp, x0,
                                       rhb[tid] * rhb[tid] * rhb[tid] * sc[2]
                                       / T.float32(d))
                            T.tile.sub(work1, work1, tmp)
                            # work2 = rk*w_h*ddot*x3 - rk^3*k*dot_k/d
                            T.tile.mul(work2, w_h32, rkb[tid])
                            T.tile.mul(work2, work2, sc[1])
                            T.tile.mul(work2, work2, x3)
                            T.tile.mul(tmp, x1,
                                       rkb[tid] * rkb[tid] * rkb[tid] * sc[3]
                                       / T.float32(d))
                            T.tile.sub(work2, work2, tmp)
                            # dv = gate * dvhat
                            T.tile.mul(x2, dvh, ab[tid])
                            # drms_w_h += ddot*work0*H*rh + ddot*x3*k*rk
                            T.tile.mul(tmp, work0, x0)
                            T.tile.mul(tmp, tmp, sc[1] * rhb[tid])
                            T.tile.add(drms_h_acc, drms_h_acc, tmp)
                            T.tile.mul(tmp, x3, x1)
                            T.tile.mul(tmp, tmp, sc[1] * rkb[tid])
                            T.tile.add(drms_h_acc, drms_h_acc, tmp)
                            T.barrier_all()
                            T.tile.cast(typed_out, work1, "CAST_RINT", d_pad)
                            T.barrier_all()
                            T.copy(typed_out[0:d], dH[bid, tid, 0:d])
                            T.tile.cast(typed_out, work2, "CAST_RINT", d_pad)
                            T.barrier_all()
                            T.copy(typed_out[0:d], dk[bid, tid, 0:d])
                            T.tile.cast(typed_out, x2, "CAST_RINT", d_pad)
                            T.barrier_all()
                            T.copy(typed_out[0:d], dv[bid, tid, 0:d])
                    lane = cid * 2 + vid
                    T.barrier_all()
                    T.copy(drms_h_acc,
                           part[(lane * _BWD_PART_SLOTS + 4) * d_pad:
                                (lane * _BWD_PART_SLOTS + 5) * d_pad])

        return main

    compiled = factory()
    compiled.r201_plan = dict(plan, d_pad=d_pad, lanes=lanes, part_len=part_len)
    return compiled


@lru_cache(maxsize=64)
def _compile_engram_bwd_reduce(d: int, lanes: int):
    """Fold the per-lane partials into the six ``d``-long weight gradients.

    One work item per slot, so the fold is six lanes wide rather than 6 x lanes
    vector ops on one core.  The destination is a single padded flat buffer:
    each slot owns its own 32-byte-aligned d_pad-wide window, so no two lanes
    ever touch the same line (STAT_GROUP again).
    """
    d_pad = math.ceil(d / 32) * 32
    part_len = lanes * _BWD_PART_SLOTS * d_pad
    logical_blocks = max(1, -(-_BWD_PART_SLOTS // 2))
    launch_blocks = launch_block_count(logical_blocks)

    @tilelang.jit(out_idx=[], pass_configs=_PASS_CONFIGS,
                  compile_flags=["-O3", "-DENABLE_BF16"])
    def factory():
        @T.prim_func
        def main(
            part: T.Tensor((part_len,), "float32"),
            dw: T.Tensor((_BWD_PART_SLOTS * d_pad,), "float32"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                acc = T.alloc_ub((d_pad,), "float32")
                buf = T.alloc_ub((d_pad,), "float32")
                with T.Scope("V"):
                    slot = cid * 2 + vid
                    if slot < _BWD_PART_SLOTS:
                        T.tile.fill(acc, 0.0)
                        for l in T.serial(lanes):
                            T.copy(part[(l * _BWD_PART_SLOTS + slot) * d_pad:
                                        (l * _BWD_PART_SLOTS + slot + 1) * d_pad], buf)
                            T.barrier_all()
                            T.tile.add(acc, acc, buf)
                        T.barrier_all()
                        T.copy(acc, dw[slot * d_pad:(slot + 1) * d_pad])

        return main

    compiled = factory()
    compiled.r201_plan = {"lanes": lanes, "d_pad": d_pad,
                          "slots": _BWD_PART_SLOTS, "launch_blocks": launch_blocks}
    return compiled


def build_engram_bwd_kernel(M, seq_len, d, eps, dtype):
    M, seq_len, d = int(M), int(seq_len), int(d)
    if min(M, seq_len, d) <= 0:
        raise ValueError("EngramGateConvBwdOp dimensions must be positive")
    del eps  # the backward consumes the saved rms values, it never re-derives one
    dtype_name = _dtype_name(dtype)
    st1 = _compile_engram_bwd_stage1(M, seq_len, d, dtype_name)
    st23 = _compile_engram_bwd_stage23(M, seq_len, d, dtype_name)
    st4 = _compile_engram_bwd_stage4(M, seq_len, d, dtype_name)
    lanes = st23.r201_plan["lanes"]
    d_pad = st23.r201_plan["d_pad"]
    part_len = st23.r201_plan["part_len"]
    reduce_k = _compile_engram_bwd_reduce(d, lanes)

    def invoke(dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha, rrms_h, rrms_k, rrms_v):
        device = dY.device
        dH = torch.full_like(dY, float("nan"))
        dk = torch.full_like(dY, float("nan"))
        dv = torch.full_like(dY, float("nan"))
        d_silu = torch.empty((M, seq_len, d), dtype=torch.float32, device=device)
        dvhat_tmp = torch.empty((M, seq_len, d), dtype=torch.float32, device=device)
        # Every lane writes its whole slice of ``part``, so it never needs
        # host-side zeroing.
        part = torch.empty((part_len,), dtype=torch.float32, device=device)
        dw = torch.empty((_BWD_PART_SLOTS * d_pad,), dtype=torch.float32, device=device)
        st1(dY, rms_w_v, conv_w, vhat, rrms_v, d_silu)
        st23(dY, rms_w_v, conv_w, vhat, rrms_v, d_silu, dvhat_tmp, part)
        st4(H, k, v, rms_w_h, alpha, rrms_h, rrms_k, dvhat_tmp, dH, dk, dv, part)
        reduce_k(part, dw)
        conv_block = dw[0:4 * d_pad].view(4, d_pad)[:, :d]
        dconv_w = conv_block if d_pad == d else conv_block.contiguous()
        drms_w_h = dw[4 * d_pad:4 * d_pad + d]
        drms_w_v = dw[5 * d_pad:5 * d_pad + d]
        return dH, dk, dv, drms_w_h, drms_w_v, dconv_w

    invoke.compiled = st1
    invoke.compiled_stages = (st1, st23, st4, reduce_k)
    invoke.launch_blocks = st1.r201_plan["launch_blocks"]
    invoke.plan = st23.r201_plan
    invoke.path_kind = "aiv_row_parallel_four_launch_backward"
    return invoke


def _mhc_pre_plan(n_expand: int, c_x: int):
    """Decide whether ``_compile_mhc_pre`` can take the vectorized R201 path.

    Two hard alignment facts decide it (13.46 / 13.55 #4: an Ascend vector
    instruction and a UB BufferRegion both address whole 32-byte blocks):

      * steps 7 and 8 slice ``x32``/``out32`` at multiples of ``c_x`` fp32
        elements, so ``c_x * 4`` must be a multiple of 32, i.e. ``c_x % 8 == 0``;
      * the ``(R, phi_dim)`` phi tile is broadcast along axis 1, so one row
        (``phi_dim * 4`` bytes) must be a multiple of 32, i.e.
        ``phi_dim % 8 == 0``.

    ``phi_dim = n^2 + 2n`` is 24 for the only ``n_expand`` the manifest uses (4),
    and every manifest ``c_x`` (1280 / 1920 / 2560) is a multiple of 8.  When
    either fails we keep the pre-R201 scalar body rather than emit a kernel that
    faults with ``aicore exception 507015``.
    """
    x_dim = n_expand * c_x
    phi_dim = n_expand * n_expand + 2 * n_expand
    if c_x % 8 or phi_dim % 8:
        return None
    #: rows of phi accumulated per pass.  Power of two so the final column sum
    #: is a halving fold of BufferRegions; must divide x_dim so no pass reads
    #: past the end of phi.
    chunk = 128
    while chunk > 8 and x_dim % chunk:
        chunk //= 2
    if x_dim % chunk:
        return None
    #: width of the sum-of-squares reduction pass.
    sq_width = 512
    x_pad = _round_up(x_dim, max(sq_width, chunk))
    c_pad = _round_up(c_x, 8)
    small_pad = _round_up(max(phi_dim, n_expand * n_expand), 32)
    ub = (x_pad * 2                 # typed (bfloat16)
          + x_pad * 4               # x32
          + x_pad * 4               # out32
          + c_pad * 4               # work
          + sq_width * 4            # sq
          + 3 * chunk * phi_dim * 4  # phi_tile / prod / acc
          + 5 * small_pad * 4       # h32 h_pre h_res math_in math_out
          + 2 * 32 * 4              # row_sum col_sum
          + 8 * 4)                  # red
    while ub > UB_BUDGET_BYTES and chunk > 8:
        ub -= 3 * (chunk // 2) * phi_dim * 4
        chunk //= 2
        while chunk > 8 and x_dim % chunk:
            chunk //= 2
        if x_dim % chunk:
            return None
    if ub > UB_BUDGET_BYTES:
        return None
    return {"x_dim": x_dim, "phi_dim": phi_dim, "x_pad": x_pad, "c_pad": c_pad,
            "small_pad": small_pad, "chunk": chunk, "n_chunks": x_dim // chunk,
            "sq_width": sq_width, "n_sq": x_pad // sq_width, "ub_bytes": ub}


@lru_cache(maxsize=64)
def _compile_mhc_pre_scalar(batch: int, n_expand: int, c_x: int):
    """Pre-R201 fully scalar MHCPre.  Kept as the fallback for shapes the
    vectorized path cannot align (see ``_mhc_pre_plan``)."""
    x_dim = n_expand * c_x
    phi_dim = n_expand * n_expand + 2 * n_expand
    x_pad = math.ceil(x_dim / 32) * 32
    small_pad = math.ceil(max(phi_dim, n_expand * n_expand) / 32) * 32

    @tilelang.jit(
        out_idx=[],
        pass_configs=_PASS_CONFIGS,
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def factory():
        @T.prim_func
        def main(
            phi: T.Tensor((x_dim, phi_dim), "float32"),
            x: T.Tensor((batch, x_dim), "bfloat16"),
            b: T.Tensor((phi_dim,), "float32"),
            alpha_pre: T.float32,
            alpha_res: T.float32,
            sinkhorn_repeat: T.int32,
            sinkhorn_eps: T.float32,
            x_res: T.Tensor((batch, x_dim), "bfloat16"),
            x_layer: T.Tensor((batch, c_x), "bfloat16"),
        ):
            with T.Kernel(batch, is_npu=True) as (cid, vid):
                x_typed = T.alloc_ub((x_pad,), "bfloat16")
                out_typed = T.alloc_ub((x_pad,), "bfloat16")
                x32 = T.alloc_ub((x_pad,), "float32")
                out32 = T.alloc_ub((x_pad,), "float32")
                h32 = T.alloc_ub((small_pad,), "float32")
                h_pre = T.alloc_ub((small_pad,), "float32")
                h_res = T.alloc_ub((small_pad,), "float32")
                row_sum = T.alloc_ub((32,), "float32")
                col_sum = T.alloc_ub((32,), "float32")
                math_in = T.alloc_ub((small_pad,), "float32")
                math_out = T.alloc_ub((small_pad,), "float32")
                with T.Scope("V"):
                    if vid == 0:
                        T.copy(x[cid, 0:x_dim], x_typed[0:x_dim])
                        T.tile.cast(x32, x_typed, "CAST_NONE", x_pad)

                        sum_x2 = T.alloc_var("float32")
                        sum_x2 = 0.0
                        for i in T.serial(x_dim):
                            sum_x2 = sum_x2 + x32[i] * x32[i]
                        math_in[0] = sum_x2
                        T.tile.sqrt(math_out, math_in)
                        radius = math_out[0] / T.float32(math.sqrt(x_dim)) + 0.0001

                        # x @ phi is accumulated entirely in fp32.
                        for q in T.serial(phi_dim):
                            acc = T.alloc_var("float32")
                            acc = 0.0
                            for i in T.serial(x_dim):
                                acc = acc + x32[i] * phi[i, q]
                            h32[q] = acc

                        for n in T.serial(n_expand):
                            math_in[n] = -(alpha_pre * h32[n] / radius + b[n])
                        T.tile.exp(math_out, math_in)
                        for n in T.serial(n_expand):
                            h_pre[n] = 1.0 / (1.0 + math_out[n])

                        # Stable row-wise exp followed by fixed-count Sinkhorn updates.
                        for row in T.serial(n_expand):
                            row_max = T.alloc_var("float32")
                            row_max = -3.402823466e38
                            for col in T.serial(n_expand):
                                q = 2 * n_expand + row * n_expand + col
                                value = alpha_res * h32[q] / radius + b[q]
                                h_res[row * n_expand + col] = value
                                if value > row_max:
                                    row_max = value
                            for col in T.serial(n_expand):
                                math_in[row * n_expand + col] = -(
                                    row_max - h_res[row * n_expand + col]
                                )
                        T.tile.exp(math_out, math_in)
                        for q in T.serial(n_expand * n_expand):
                            h_res[q] = math_out[q]

                        for sink_iter in T.serial(sinkhorn_repeat):
                            for row in T.serial(n_expand):
                                row_sum[row] = 0.0
                                for col in T.serial(n_expand):
                                    row_sum[row] = row_sum[row] + h_res[row * n_expand + col]
                            for row in T.serial(n_expand):
                                for col in T.serial(n_expand):
                                    h_res[row * n_expand + col] = h_res[
                                        row * n_expand + col
                                    ] / (row_sum[row] + sinkhorn_eps)
                            for col in T.serial(n_expand):
                                col_sum[col] = 0.0
                                for row in T.serial(n_expand):
                                    col_sum[col] = col_sum[col] + h_res[row * n_expand + col]
                            for row in T.serial(n_expand):
                                for col in T.serial(n_expand):
                                    h_res[row * n_expand + col] = h_res[
                                        row * n_expand + col
                                    ] / (col_sum[col] + sinkhorn_eps)

                        for col_x in T.serial(c_x):
                            layer_acc = T.alloc_var("float32")
                            layer_acc = 0.0
                            for n in T.serial(n_expand):
                                layer_acc = layer_acc + h_pre[n] * x32[n * c_x + col_x]
                            out32[col_x] = layer_acc
                        T.tile.cast(out_typed, out32, "CAST_RINT", x_pad)
                        T.copy(out_typed[0:c_x], x_layer[cid, 0:c_x])

                        for row in T.serial(n_expand):
                            for col_x in T.serial(c_x):
                                res_acc = T.alloc_var("float32")
                                res_acc = 0.0
                                for n in T.serial(n_expand):
                                    res_acc = res_acc + h_res[row * n_expand + n] * x32[
                                        n * c_x + col_x
                                    ]
                                out32[row * c_x + col_x] = res_acc
                        T.tile.cast(out_typed, out32, "CAST_RINT", x_pad)
                        T.copy(out_typed[0:x_dim], x_res[cid, 0:x_dim])
                        T.barrier_all()

        return main

    return factory()


@lru_cache(maxsize=64)
def _compile_mhc_pre(batch: int, n_expand: int, c_x: int):
    plan = _mhc_pre_plan(n_expand, c_x)
    if plan is None:
        return _compile_mhc_pre_scalar(batch, n_expand, c_x)

    x_dim = plan["x_dim"]
    phi_dim = plan["phi_dim"]
    x_pad = plan["x_pad"]
    c_pad = plan["c_pad"]
    small_pad = plan["small_pad"]
    chunk = plan["chunk"]
    n_chunks = plan["n_chunks"]
    sq_width = plan["sq_width"]
    n_sq = plan["n_sq"]

    @tilelang.jit(
        out_idx=[],
        pass_configs=_PASS_CONFIGS,
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def factory():
        @T.prim_func
        def main(
            phi: T.Tensor((x_dim, phi_dim), "float32"),
            x: T.Tensor((batch, x_dim), "bfloat16"),
            b: T.Tensor((phi_dim,), "float32"),
            alpha_pre: T.float32,
            alpha_res: T.float32,
            sinkhorn_repeat: T.int32,
            sinkhorn_eps: T.float32,
            x_res: T.Tensor((batch, x_dim), "bfloat16"),
            x_layer: T.Tensor((batch, c_x), "bfloat16"),
        ):
            with T.Kernel(batch, is_npu=True) as (cid, vid):
                typed = T.alloc_ub((x_pad,), "bfloat16")
                x32 = T.alloc_ub((x_pad,), "float32")
                out32 = T.alloc_ub((x_pad,), "float32")
                work = T.alloc_ub((c_pad,), "float32")
                sq = T.alloc_ub((1, sq_width), "float32")
                red = T.alloc_ub((1,), "float32")
                phi_tile = T.alloc_ub((chunk, phi_dim), "float32")
                prod = T.alloc_ub((chunk, phi_dim), "float32")
                acc = T.alloc_ub((chunk, phi_dim), "float32")
                # h32 is the reduce_sum destination, so its extent must equal
                # the phi tile's column count exactly (not small_pad).
                h32 = T.alloc_ub((phi_dim,), "float32")
                h_pre = T.alloc_ub((small_pad,), "float32")
                h_res = T.alloc_ub((small_pad,), "float32")
                row_sum = T.alloc_ub((32,), "float32")
                col_sum = T.alloc_ub((32,), "float32")
                math_in = T.alloc_ub((small_pad,), "float32")
                math_out = T.alloc_ub((small_pad,), "float32")
                with T.Scope("V"):
                    if vid == 0:
                        # Zero the padding lanes BEFORE the cast: they feed the
                        # sum-of-squares reduction and the phi accumulation, and
                        # 0 is the additive identity for both (13.44 / R090 --
                        # get the neutral element wrong and you get a silent
                        # finite wrong value, not a crash).
                        T.tile.fill(typed, 0.0)
                        T.copy(x[cid, 0:x_dim], typed[0:x_dim])
                        T.barrier_all()          # MTE2 -> V (13.55 #1)
                        T.tile.cast(x32, typed, "CAST_NONE", x_pad)

                        # --- radius: sum of squares, one vector pass per
                        #     sq_width lanes instead of x_dim scalar iterations.
                        sum_x2 = T.alloc_var("float32")
                        sum_x2 = 0.0
                        for s in T.serial(n_sq):
                            T.tile.mul(sq, x32[s * sq_width:(s + 1) * sq_width],
                                       x32[s * sq_width:(s + 1) * sq_width])
                            T.reduce_sum(sq, red, dim=-1, clear=True)
                            sum_x2 = sum_x2 + red[0]
                        math_in[0] = sum_x2
                        T.tile.sqrt(math_out, math_in)
                        radius = math_out[0] / T.float32(math.sqrt(x_dim)) + 0.0001

                        # --- h32 = x @ phi.
                        # phi is (x_dim, phi_dim) row-major, so a COLUMN of it is
                        # strided; reading it column-wise would amplify the DMA
                        # by phi_dim.  Instead accumulate a (chunk, phi_dim)
                        # partial-sum tile -- phi is read exactly once, densely --
                        # and collapse it at the end with a halving fold.
                        T.tile.fill(acc, 0.0)
                        for s in T.serial(n_chunks):
                            T.copy(phi[s * chunk:(s + 1) * chunk, 0:phi_dim], phi_tile)
                            T.tile.broadcast(prod, x32[s * chunk:(s + 1) * chunk], axis=1)
                            T.tile.mul(prod, prod, phi_tile)
                            T.tile.add(acc, acc, prod)
                        # Collapse the (chunk, phi_dim) partial-sum tile along
                        # its ROW axis.  ⚠️ The obvious halving fold
                        # (``while h > 1: T.tile.add(acc[0:h*PD], ...)``) cannot
                        # be written: TVMScript parses a Python ``while`` (and a
                        # Python ``for`` over a list) inside @T.prim_func as a
                        # TIR construct, not as host-side unrolling -- every
                        # individual add parses, the loop does not
                        # (R201-data/probe_primitives.log q2/q6/q7 and the
                        # bisect in R201.md).  ``T.reduce_sum(..., dim=0)`` does
                        # exactly this reduction in one intrinsic.
                        T.reduce_sum(acc, h32, dim=0, clear=True)

                        for n in T.serial(n_expand):
                            math_in[n] = -(alpha_pre * h32[n] / radius + b[n])
                        T.tile.exp(math_out, math_in)
                        for n in T.serial(n_expand):
                            h_pre[n] = 1.0 / (1.0 + math_out[n])

                        # Stable row-wise exp followed by fixed-count Sinkhorn updates.
                        for row in T.serial(n_expand):
                            row_max = T.alloc_var("float32")
                            row_max = -3.402823466e38
                            for col in T.serial(n_expand):
                                q = 2 * n_expand + row * n_expand + col
                                value = alpha_res * h32[q] / radius + b[q]
                                h_res[row * n_expand + col] = value
                                if value > row_max:
                                    row_max = value
                            for col in T.serial(n_expand):
                                math_in[row * n_expand + col] = -(
                                    row_max - h_res[row * n_expand + col]
                                )
                        T.tile.exp(math_out, math_in)
                        for q in T.serial(n_expand * n_expand):
                            h_res[q] = math_out[q]

                        for sink_iter in T.serial(sinkhorn_repeat):
                            for row in T.serial(n_expand):
                                row_sum[row] = 0.0
                                for col in T.serial(n_expand):
                                    row_sum[row] = row_sum[row] + h_res[row * n_expand + col]
                            for row in T.serial(n_expand):
                                for col in T.serial(n_expand):
                                    h_res[row * n_expand + col] = h_res[
                                        row * n_expand + col
                                    ] / (row_sum[row] + sinkhorn_eps)
                            for col in T.serial(n_expand):
                                col_sum[col] = 0.0
                                for row in T.serial(n_expand):
                                    col_sum[col] = col_sum[col] + h_res[row * n_expand + col]
                            for row in T.serial(n_expand):
                                for col in T.serial(n_expand):
                                    h_res[row * n_expand + col] = h_res[
                                        row * n_expand + col
                                    ] / (col_sum[col] + sinkhorn_eps)

                        # --- x_layer[c] = sum_n h_pre[n] * x[n*c_x + c].
                        # Same accumulation ORDER as the pre-R201 scalar loop
                        # (n ascending, starting from the n = 0 term rather than
                        # from a 0.0 seed, and 0.0 + t0 == t0 exactly), so this
                        # rewrite is bit-identical on its own.
                        T.tile.mul(out32[0:c_x], x32[0:c_x], h_pre[0])
                        for n in T.serial(n_expand - 1):
                            T.tile.mul(work[0:c_x],
                                       x32[(n + 1) * c_x:(n + 2) * c_x], h_pre[n + 1])
                            T.tile.add(out32[0:c_x], out32[0:c_x], work[0:c_x])
                        T.barrier_all()          # V -> MTE3 (13.55 #2)
                        T.tile.cast(typed, out32, "CAST_RINT", x_pad)
                        T.barrier_all()
                        T.copy(typed[0:c_x], x_layer[cid, 0:c_x])

                        # --- x_res[row, c] = sum_n h_res[row, n] * x[n*c_x + c].
                        for row in T.serial(n_expand):
                            T.tile.mul(out32[row * c_x:row * c_x + c_x],
                                       x32[0:c_x], h_res[row * n_expand])
                            for n in T.serial(n_expand - 1):
                                T.tile.mul(work[0:c_x],
                                           x32[(n + 1) * c_x:(n + 2) * c_x],
                                           h_res[row * n_expand + n + 1])
                                T.tile.add(out32[row * c_x:row * c_x + c_x],
                                           out32[row * c_x:row * c_x + c_x],
                                           work[0:c_x])
                        T.barrier_all()          # V -> MTE3 (13.55 #2)
                        T.tile.cast(typed, out32, "CAST_RINT", x_pad)
                        T.barrier_all()
                        T.copy(typed[0:x_dim], x_res[cid, 0:x_dim])

        return main

    compiled = factory()
    compiled.r201_plan = plan
    return compiled


def build_mhc_pre_kernel(
    batch, n_expand, c_x, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat, sinkhorn_eps
):
    del alpha_post  # H_post is intentionally not part of this op's public outputs.
    batch, n_expand, c_x = int(batch), int(n_expand), int(c_x)
    sinkhorn_repeat = int(sinkhorn_repeat)
    if min(batch, n_expand, c_x) <= 0 or sinkhorn_repeat < 0:
        raise ValueError("MHCPreFwdOp dimensions/repeat must be non-negative")
    compiled = _compile_mhc_pre(batch, n_expand, c_x)

    def invoke(phi, x, b, runtime_alpha_pre, runtime_alpha_post, runtime_alpha_res,
               runtime_sinkhorn_repeat, runtime_sinkhorn_eps):
        del runtime_alpha_post
        x_res = torch.full_like(x, float("nan"))
        x_layer = torch.full((batch, c_x), float("nan"), dtype=x.dtype, device=x.device)
        compiled(
            phi,
            x,
            b,
            float(runtime_alpha_pre),
            float(runtime_alpha_res),
            int(runtime_sinkhorn_repeat),
            float(runtime_sinkhorn_eps),
            x_res,
            x_layer,
        )
        return x_res, x_layer

    invoke.compiled = compiled
    invoke.launch_blocks = batch
    invoke.path_kind = "aiv_fp32_projection_fixed_sinkhorn"
    return invoke


@lru_cache(maxsize=64)
def _compile_mhc_post(batch: int, n_expand: int, c_x: int, dtype: str):
    """``x_out[b, n, c] = x_res[b, n, c] + x_layer_out[b, c] * h_post[b, n]``.

    R201 rewrite.  The pre-R201 kernel tiled the FLATTENED output and decoded
    ``(b, n, c)`` from the flat index once PER LANE, so every element paid two
    ``for lane in T.serial(tile)`` iterations (~100 ns each, 13.52) plus three
    integer divisions.  ``docs/reports/R201-data/scalar_iteration_model.log``
    counts 1024 such iterations on the critical path of ``post-large`` = 102 us
    of a measured 179 us.

    Tiling ``(b, n)`` x ``c`` instead makes every operand of a work item a
    CONTIGUOUS run:

      * ``x_res`` and ``x_out`` rows are ``(b*n_expand + n, c)`` of the same
        row-major buffer the caller already has -- a free reshape;
      * ``x_layer_out`` is indexed by ``b`` alone, i.e. one row read per work
        item, no per-lane gather;
      * ``h_post[b, n]`` is ONE scalar per work item, staged through UB so the
        ``muls`` intrinsic reads it from UB rather than GM.

    so the whole body is copies + ``mul`` + ``add`` and the scalar loop count
    drops to zero.  Arithmetic order is unchanged (``res + x*h``, fp32,
    ``CAST_RINT`` on the way out), so the result is bit-identical -- except on
    ``float32``, where the pre-R201 code had the 13.54 bug (see below).
    """
    rows = batch * n_expand
    itemsize = 4 if dtype == "float32" else 2
    need_cast = dtype != "float32"
    # UB per column element, read off the allocation list below: x_ub + res_ub
    # + out_ub (dtype) and x32 + res32 + out32 (fp32).  The dtype trio is
    # allocated even on the float32 path where it is dead, because TVMScript
    # scopes ``T.alloc_ub`` to the enclosing ``if`` -- allocating it under
    # ``if need_cast:`` makes the name undefined at every use site
    # ("error: Undefined variable: x_ub").  Paying for the dead buffers keeps
    # the plan honest about what is actually reserved.
    bytes_per_elem = 3 * itemsize + 3 * 4
    plan = _plan_row_tile(rows, c_x, bytes_per_elem)
    tile = plan["tile"]
    n_tiles = plan["n_tiles"]
    copy_w = plan["copy_w"]
    last_col = plan["last_col"]
    items = plan["items"]
    logical_blocks = plan["logical_blocks"]
    launch_blocks = plan["launch_blocks"]
    grid_repeats = plan["grid_repeats"]
    # h_post is read as a UB scalar by ``muls``; a vector instruction addresses
    # whole 32-byte blocks, so give it 8 fp32 lanes of slack.
    hp_pad = _round_up(rows, 8)

    @tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS,
                  compile_flags=["-O3", "-DENABLE_BF16"])
    def kernel():
        @T.prim_func
        def main(
            x_layer_out: T.Tensor((batch, c_x), dtype),
            h_post: T.Tensor((rows,), "float32"),
            x_res: T.Tensor((rows, c_x), dtype),
            x_out: T.Tensor((rows, c_x), dtype),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                x32 = T.alloc_ub((tile,), "float32")
                res32 = T.alloc_ub((tile,), "float32")
                out32 = T.alloc_ub((tile,), "float32")
                hp = T.alloc_ub((hp_pad,), "float32")
                x_ub = T.alloc_ub((tile,), dtype)
                res_ub = T.alloc_ub((tile,), dtype)
                out_ub = T.alloc_ub((tile,), dtype)
                with T.Scope("V"):
                    # ⚠️ 13.55 trap 1: ``muls`` reads hp[row] straight out of UB
                    # while MTE2 is what filled it.  Without the barrier the
                    # scalar operand is whatever was there before.
                    T.copy(h_post[0:rows], hp[0:rows])
                    T.barrier_all()
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        item = logical_cid * 2 + vid
                        if item < items:
                            row = item // n_tiles
                            col = T.min((item % n_tiles) * tile, last_col)
                            b = row // n_expand
                            if need_cast:
                                T.copy(x_layer_out[b, col:col + copy_w], x_ub[0:copy_w])
                                T.copy(x_res[row, col:col + copy_w], res_ub[0:copy_w])
                                T.barrier_all()          # MTE2 -> V (13.55 #1)
                                T.tile.cast(x32, x_ub, "CAST_NONE", tile)
                                T.tile.cast(res32, res_ub, "CAST_NONE", tile)
                            else:
                                # ⚠️ 13.54: a float32 -> float32 CAST_NONE does
                                # not move data, so the pre-R201 kernel read
                                # uninitialized UB on this path.  Copy instead.
                                T.copy(x_layer_out[b, col:col + copy_w], x32[0:copy_w])
                                T.copy(x_res[row, col:col + copy_w], res32[0:copy_w])
                                T.barrier_all()          # MTE2 -> V (13.55 #1)
                            # ⚠️ Two roundings, on purpose.  The pre-R201 body
                            # was the C++ SCALAR expression ``res32[lane] +
                            # x32[lane] * h``, which -O3 contracts into a
                            # single-rounding FMA; a host replay reproduces the
                            # pre-R201 output 10/10 with an FMA and the output
                            # below 10/10 with mul-then-add
                            # (R201-data/check_fma_hypothesis.log).  There is no
                            # fused vector form to match it with: BOTH
                            # ``T.tile.mul_add_dst`` and ``T.tile.axpy`` were
                            # tried and both round twice
                            # (R201-data/probe_fma_forms.log,
                            # diag_mhc_post_bitdiff_fma.log).  Two roundings is
                            # also what the op's own torch reference does
                            # (``h @ x`` then ``+ x_res``), and R201 agrees with
                            # that reference on 78848/78848 elements where the
                            # pre-R201 kernel disagreed on 7.
                            T.tile.mul(out32, x32, hp[row])
                            T.tile.add(out32, out32, res32)
                            T.barrier_all()              # V -> MTE3 (13.55 #2)
                            if need_cast:
                                T.tile.cast(out_ub, out32, "CAST_RINT", tile)
                                T.barrier_all()
                                T.copy(out_ub[0:copy_w], x_out[row, col:col + copy_w])
                            else:
                                T.copy(out32[0:copy_w], x_out[row, col:col + copy_w])

        return main

    compiled = kernel()
    compiled.r201_plan = dict(plan, rows=rows, dtype=dtype, hp_pad=hp_pad,
                              need_cast=need_cast, bytes_per_elem=bytes_per_elem)
    return compiled


def build_mhc_post_kernel(batch, n_expand, c_x, dtype, *, tune=False):
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"MHCPostFwdOp does not support {dtype}")
    dtype_name = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}[dtype]
    batch, n_expand, c_x = int(batch), int(n_expand), int(c_x)
    rows = batch * n_expand
    compiled = _compile_mhc_post(batch, n_expand, c_x, dtype_name)
    plan = compiled.r201_plan

    def launch(x_layer_out, h_post, x_res):
        shape = (batch, n_expand, c_x)
        if tuple(x_layer_out.shape) != shape[:1] + shape[2:] or tuple(h_post.shape) != shape[:2] or tuple(x_res.shape) != (shape[0], shape[1] * shape[2]):
            raise ValueError("MHCPostFwdOp kernel shape mismatch")
        # Both reshapes are views: (B, N*C) and (B*N, C) are the same row-major
        # buffer, and (B, N) -> (B*N,) likewise.
        result = compiled(x_layer_out, h_post.reshape(rows), x_res.reshape(rows, c_x))
        return result.reshape(batch, n_expand * c_x)

    launch.path_kind = "aiv_row_tiled_outer_broadcast"
    launch.output_shape = (batch, n_expand * c_x)
    launch.logical_blocks = plan["logical_blocks"]
    launch.launch_blocks = plan["launch_blocks"]
    launch.grid_repeats = plan["grid_repeats"]
    launch.plan = plan
    launch.compiled = compiled
    return launch


__all__ = [
    "build_engram_bwd_kernel",
    "build_engram_fwd_kernel",
    "build_mhc_post_kernel",
    "build_mhc_pre_kernel",
]
