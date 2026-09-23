"""Ascend lightning-indexer scores (``FP8LightningIndexerFwdOp``).

The operator (manifest ``attention_indexing.yaml:3``, reference semantics
``TileOPs/workloads/fp8_lightning_indexer.py:ref_program``) is

    logits[b, s, n, g] = sum_h relu( q[b, s, g, h, :] . k[b, n, g, :] ) * w[s, g, h]

masked to ``-inf`` outside the per-query window ``[cu_seqlen_ks[s], cu_seqlen_ke[s])``.

Shape of the exam paper: ``B=1, S=8192, H=32, D=64, Skv=32768, G=1``.  The output is
``8192 * 32768 * 4`` bytes = **1 GiB**, and the per-head score volume the vector unit
has to look at before it can reduce is ``S * H * Skv * 4`` = **32 GiB** -- it is the
``relu`` between the contraction and the head sum that forbids folding the head axis
into the Cube accumulator, so every one of those scores must leave L0C.

Layout
------
The Cube tile is ``[hpad * bm, bn]`` with rows in **head-major** order
(``row = h * bm + m``).  Two things fall out of that choice:

* the head reduction is a 5-level binary tree of whole-region ``T.tile.add``s
  (``score[0:16bm] += score[16bm:32bm]``, ...), i.e. ~1 pass over the tile in 5
  vector instructions rather than ``hpad * bm`` per-row instructions;
* the per-head weight is a broadcast tile ``wb[h*bm+m, :] = weights[m0+m, h]`` built
  once per m-tile and reused by all ``skv / bn`` n-tiles, so its ``hpad * bm`` small
  fills cost nothing per tile.

Why this kernel is written with explicit ``T.Scope("C")`` / ``T.Scope("V")``
---------------------------------------------------------------------------
``TL_ASCEND_AUTO_CV_COMBINE`` was tried first and is **measurably unsafe here**
(R352-data/PROGRESS.md F-H): it inserts the cube->vector RAW flag for
``T.copy(l0c, ub)`` but not the vector->cube WAR flag, so with 8 or more blocks the
Cube's next fixpipe overwrote the score tile while the vector was still reducing the
previous one.  The symptom was one wrong ``(block, n-tile)`` per launch, different on
every run, always the FIRST loop iteration; ``T.barrier_all()`` cannot fix it because
it is a within-core pipe barrier, not a cross-core one.  So the handoff is explicit:
a small per-block GM ping-pong buffer plus ``set_cross_flag`` / ``wait_cross_flag``,
the same shape as ``tilelang-ascend/examples/deepseek_v4/lightning_indexer.py``.

Three ``T.barrier_all()`` placements are load-bearing, and the first one cost this
round the most time to find: after ``T.copy(Weights, wu)`` (MTE2 writes UB, the fills
below read ``wu`` as a **scalar**), after ``T.copy(WS, score_ub)`` (MTE2 -> VEC) and
before the writeback ``T.copy`` (VEC -> MTE3).  Dropping the first one made all eight
blocks wrong while the Cube's workspace stayed bit-correct.

Other measured DSL constraints, all in R352-data/PROGRESS.md:

* this module must NOT use ``from __future__ import annotations`` -- TVMScript
  re-evaluates the parameter annotations and they degrade to ``str``;
* ``-inf`` must be written ``T.float32(float("-inf"))``; a bare Python float is a
  float64 ``FloatImm`` and codegen emits the undeclared ``CUDART_INF``;
* no ``T.Parallel`` anywhere -- one ``T.Parallel`` statement in the same kernel as
  ``T.gemm_v0`` corrupted the GEMM result even when the statement wrote a constant;
* ``T.tile.fill(region, runtime_scalar)`` is the working way to broadcast a scalar
  into a UB row; ``T.tile.mul(region, ones_buffer, scalar)`` is not;
* region extents must be Python constants, so the reduction tree is written as
  literal guarded statements rather than a loop (a variable assigned inside the
  prim_func body becomes a TIR var and the extents stop being constant).
"""

import tilelang
import tilelang.language as T
import torch

from functools import lru_cache

#: Cross-core flag ids.  ``F_C2V`` = "chunk ready in the workspace", ``F_V2C`` =
#: "workspace slot consumed".
_F_C2V = 0
_F_V2C = 1

#: n-tiles per handoff chunk.  Two, so the two AIVs of a block can take one n-tile
#: each and both still execute every flag call -- ``set_cross_flag(mode=2)`` is an
#: AIC<->all-AIVs-in-group barrier, so guarding a flag with ``if vid == 0`` deadlocks
#: (measured: the prototype hung until it was killed).
_TILES_PER_CHUNK = 2

#: Workspace slots, i.e. how many chunks the Cube may run ahead.
_SLOTS = 2 * _TILES_PER_CHUNK

#: Query rows per Cube tile.  ``hpad * bm`` rows of ``bn`` float32 must fit L0C
#: (128 KiB) and ``2 * hpad * bm * bn * 4`` bytes must fit the UB waterline.
_BLOCK_M = 8

#: Key columns per Cube tile.
_BLOCK_N = 64


def _pow2_ceil(value):
    out = 1
    while out < value:
        out *= 2
    return out


@lru_cache(maxsize=16)
def _compile(bg, s, skv, heads, dim, bm, bn, dtype, has_scale):
    """One static (batch*group, seq, seq_kv, heads_per_group, dim) signature."""
    hpad = _pow2_ceil(heads)
    rows = hpad * bm
    m_tiles = s // bm
    n_tiles = skv // bn
    n_chunks = n_tiles // _TILES_PER_CHUNK
    blocks = bg * m_tiles

    scale_len = skv if has_scale else 1

    @tilelang.jit(out_idx=[6], workspace_idx=[7],
                  pass_configs={tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True},
                  compile_flags=["-O3"])
    def factory():
        @T.prim_func
        def main(
            IndexQ: T.Tensor((bg, s, heads, dim), dtype),
            IndexK: T.Tensor((bg, skv, dim), dtype),
            Weights: T.Tensor((bg, s, heads), "float"),
            CuKS: T.Tensor((s,), "int32"),
            CuKE: T.Tensor((s,), "int32"),
            Scale: T.Tensor((bg, scale_len), "float"),
            Logits: T.Tensor((bg, s, skv), "float"),
            WS: T.Tensor((blocks, _SLOTS, rows, bn), "float"),
        ):
            with T.Kernel(blocks, is_npu=True) as (cid, vid):
                g = cid // m_tiles
                m0 = (cid % m_tiles) * bm

                with T.Scope("C"):
                    q_l1 = T.alloc_L1((rows, dim), dtype)
                    k_l1 = T.alloc_L1((bn, dim), dtype)
                    acc_l0c = T.alloc_L0C((rows, bn), "float")

                    T.barrier_all()
                    # Head-major rows: q_l1[h*bm + m] = index_q[g, m0+m, h, :].
                    for h in range(heads):
                        T.copy(IndexQ[g, m0:m0 + bm, h, :], q_l1[h * bm:(h + 1) * bm, :])
                    if hpad != heads:
                        # Padded head rows contract to 0 and their broadcast weight is
                        # 0, so they contribute nothing to the tree's result.
                        T.tile.fill(q_l1[heads * bm:rows, 0:dim], 0)
                    T.barrier_all()

                    for ck in T.serial(n_chunks):
                        if ck >= 2:
                            T.wait_cross_flag(_F_V2C)
                        for t in range(_TILES_PER_CHUNK):
                            n0 = (ck * _TILES_PER_CHUNK + t) * bn
                            T.barrier_all()
                            T.copy(IndexK[g, n0:n0 + bn, :], k_l1)
                            T.barrier_all()
                            T.gemm_v0(q_l1, k_l1, acc_l0c, transpose_B=True, init=True)
                            T.barrier_all()
                            # relu is free here: the fixpipe applies it on the way out.
                            T.copy(acc_l0c, WS[cid, (ck % 2) * _TILES_PER_CHUNK + t, :, :],
                                   enable_relu=True)
                            T.barrier_all()
                        T.pipe_barrier("FIX")
                        T.set_cross_flag("FIX", _F_C2V)

                with T.Scope("V"):
                    score_ub = T.alloc_ub((rows, bn), "float")
                    wb_ub = T.alloc_ub((rows, bn), "float")
                    w_ub = T.alloc_ub((bm, heads), "float")
                    pos_ub = T.alloc_ub((bm, bn), "float")
                    posn_ub = T.alloc_ub((bm, bn), "float")
                    lo_ub = T.alloc_ub((bm, bn), "float")
                    hi_ub = T.alloc_ub((bm, bn), "float")
                    keep_ub = T.alloc_ub((bm * bn // 8,), "uint8")
                    bound_ub = T.alloc_ub((bm * bn // 8,), "uint8")
                    posrow_ub = T.alloc_ub((bn,), "float")
                    scale_ub = T.alloc_ub((bn,), "float")

                    T.barrier_all()
                    T.copy(Weights[g, m0:m0 + bm, 0:heads], w_ub)
                    T.barrier_all()
                    if hpad != heads:
                        T.tile.fill(wb_ub[heads * bm:rows, 0:bn], 0.0)
                    for h in range(heads):
                        for mm in range(bm):
                            T.tile.fill(wb_ub[h * bm + mm, 0:bn], w_ub[mm, h])
                    T.tile.arith_progression(posrow_ub, 0.0, 1.0, bn)
                    for mm in range(bm):
                        T.copy(posrow_ub, pos_ub[mm, 0:bn])
                        T.tile.fill(lo_ub[mm, 0:bn], T.Cast("float", CuKS[m0 + mm]))
                        T.tile.fill(hi_ub[mm, 0:bn], T.Cast("float", CuKE[m0 + mm]))
                    T.barrier_all()

                    for ck in T.serial(n_chunks):
                        T.wait_cross_flag(_F_C2V)
                        n0 = (ck * _TILES_PER_CHUNK + vid) * bn
                        T.copy(WS[cid, (ck % 2) * _TILES_PER_CHUNK + vid, :, :], score_ub)
                        T.barrier_all()
                        T.tile.mul(score_ub, score_ub, wb_ub)
                        # Head reduction tree over the head-major row blocks.
                        if hpad >= 64:
                            T.tile.add(score_ub[0:32 * bm, 0:bn], score_ub[0:32 * bm, 0:bn],
                                       score_ub[32 * bm:64 * bm, 0:bn])
                        if hpad >= 32:
                            T.tile.add(score_ub[0:16 * bm, 0:bn], score_ub[0:16 * bm, 0:bn],
                                       score_ub[16 * bm:32 * bm, 0:bn])
                        if hpad >= 16:
                            T.tile.add(score_ub[0:8 * bm, 0:bn], score_ub[0:8 * bm, 0:bn],
                                       score_ub[8 * bm:16 * bm, 0:bn])
                        if hpad >= 8:
                            T.tile.add(score_ub[0:4 * bm, 0:bn], score_ub[0:4 * bm, 0:bn],
                                       score_ub[4 * bm:8 * bm, 0:bn])
                        if hpad >= 4:
                            T.tile.add(score_ub[0:2 * bm, 0:bn], score_ub[0:2 * bm, 0:bn],
                                       score_ub[2 * bm:4 * bm, 0:bn])
                        if hpad >= 2:
                            T.tile.add(score_ub[0:bm, 0:bn], score_ub[0:bm, 0:bn],
                                       score_ub[bm:2 * bm, 0:bn])
                        if has_scale:
                            # index_k_scale dequantizes k per (b, skv, g).  Every scale
                            # the op's own quantizer emits is > 0 (|amax| / 448 with a
                            # 1e-4 floor), and for s > 0
                            #   sum_h relu(q . (k*s)) * w == s * sum_h relu(q . k) * w,
                            # so it is applied to the reduced row instead of inside the
                            # contraction.  Costs bm vector ops, not a second 1 GiB pass.
                            T.copy(Scale[g, n0:n0 + bn], scale_ub)
                            T.barrier_all()
                            for mm in range(bm):
                                T.tile.mul(score_ub[mm, 0:bn], score_ub[mm, 0:bn], scale_ub)
                        # Window mask -> -inf, the reference's ``masked_fill``.
                        T.tile.add(posn_ub, pos_ub, T.Cast("float", n0))
                        T.tile.compare(keep_ub, posn_ub, lo_ub, "GE")
                        T.tile.compare(bound_ub, posn_ub, hi_ub, "LT")
                        T.tile.bitwise_and(keep_ub, keep_ub, bound_ub)
                        T.tile.select(score_ub[0:bm, 0:bn], keep_ub, score_ub[0:bm, 0:bn],
                                      T.float32(float("-inf")), "VSEL_TENSOR_SCALAR_MODE")
                        T.barrier_all()
                        T.copy(score_ub[0:bm, 0:bn], Logits[g, m0:m0 + bm, n0:n0 + bn])
                        T.barrier_all()
                        T.set_cross_flag("MTE3", _F_V2C)

        return main

    return factory()


_DTYPE_NAME = {torch.bfloat16: "bfloat16", torch.float16: "float16"}

#: ``float8_e4m3fn`` -> ``float16`` is exact: three mantissa bits fit in ten, and
#: every e4m3 subnormal is a float16 subnormal.  Moving the 7 exponent+mantissa bits
#: up by 7 and the sign bit up by 8 puts them where float16 wants them and costs a
#: fixed 2**-8, hence the 256.0.
_E4M3_SHIFT = 7
_E4M3_SCALE = 256.0

#: device -> 256-entry float16 decode table.
_E4M3_TABLE = {}


def _e4m3_table_cpu():
    """The 256 decoded values, derived from the bit formula, NOT from ATen.

    Deriving the table from ``codes.view(torch.float8_e4m3fn).float()`` would make
    the decode and the correctness reference the same computation, and the 256-code
    check in ``docs/reports/R352-data/probe/p2_fp8_decode_256.py`` -- which compares
    this formula against exactly that ATen call -- would be checking ATen against
    itself.  So the table is built the long way and the probe is the cross-check:
    measured 254/254 finite codes bit-identical and both NaN codes recovered.

    Multiplication rather than ``<<``: ``torch.Tensor.__lshift__`` on int16 falls off
    the NPU fast path and cost 292 ms for 16 M elements (R352-data/PROGRESS.md);
    ``(x & 0x80) * 256`` wraps to the same int16 bit pattern.  The table is 256
    elements on CPU, so this cost is irrelevant here -- it is documented because the
    same expression is the one an in-kernel decode would need.
    """
    codes = torch.arange(256, dtype=torch.uint8)
    i16 = codes.to(torch.int16)
    bits = (i16 & 0x80) * 256 + (i16 & 0x7F) * (1 << _E4M3_SHIFT)
    val = bits.view(torch.float16).float() * _E4M3_SCALE
    # 0x7F / 0xFF are e4m3fn's two NaN codes; the bit move sends them to +-480,
    # a magnitude the format cannot hold, so they are restored explicitly.
    nan = (i16 & 0x7F) == 0x7F
    return torch.where(nan, torch.full_like(val, float("nan")), val).to(torch.float16)


def decode_fp8_e4m3fn(x):
    """``float8_e4m3fn`` tensor (or its uint8 view) -> ``float16``, bit-exact.

    A 256-entry gather, not arithmetic.  ``Tensor.to(torch.float8_e4m3fn)`` and
    ``fp8.to(torch.float16)`` both abort on this CANN build with ``aclnnInplaceCopy``
    561103, and the device-side bit-arithmetic decode is dominated by an int16 shift
    that leaves the NPU fast path: measured on 16 M elements, shift form 591 ms,
    multiply form 44 ms, **this gather 0.140 ms**.  All three agree on all 256 codes.
    """
    u8 = x if x.dtype == torch.uint8 else x.view(torch.uint8)
    key = (u8.device.type, u8.device.index)
    table = _E4M3_TABLE.get(key)
    if table is None:
        table = _e4m3_table_cpu().to(u8.device)
        _E4M3_TABLE[key] = table
    flat = torch.index_select(table, 0, u8.reshape(-1).to(torch.int32))
    return flat.view(u8.shape)


def build_fp8_lightning_indexer_kernel(
    index_q_shape, index_k_shape, weights_shape, cu_ks_shape, cu_ke_shape,
    index_k_scale_shape, dtype, clean_logits,
):
    """Return ``launch(index_q, index_k, weights, cu_ks, cu_ke, index_k_scale=None)``.

    Args mirror the manifest's ``signature.inputs`` order; ``dtype`` is the shared
    ``index_q`` / ``index_k`` torch dtype.  The returned callable produces
    ``logits`` with shape ``[batch, seq_len, seq_len_kv, kv_group]`` and dtype
    float32, as ``attention_indexing.yaml:16`` declares.
    """
    batch, seq_len, heads, dim = (int(v) for v in index_q_shape)
    k_batch, seq_len_kv, kv_group, k_dim = (int(v) for v in index_k_shape)
    if k_batch != batch or k_dim != dim:
        raise ValueError("index_q and index_k must agree on batch and index_dim")
    if heads % kv_group:
        raise ValueError(f"heads={heads} must be divisible by kv_group={kv_group}")
    if tuple(int(v) for v in weights_shape) != (seq_len, heads):
        raise ValueError(f"weights must be [seq_len, heads] = {(seq_len, heads)}")
    if tuple(int(v) for v in cu_ks_shape) != (seq_len,) or \
            tuple(int(v) for v in cu_ke_shape) != (seq_len,):
        raise ValueError(f"cu_seqlen_ks/ke must be [seq_len] = {(seq_len,)}")
    if not clean_logits:
        raise ValueError(
            "this Ascend kernel implements clean_logits=True (the manifest's default "
            "and the only value any declared workload uses); clean_logits=False would "
            "be a different output contract, not a tuning knob"
        )
    decode_fp8 = dtype == torch.float8_e4m3fn
    cube_dtype = torch.float16 if decode_fp8 else dtype
    if cube_dtype not in _DTYPE_NAME:
        raise TypeError(
            f"FP8LightningIndexerFwdOp Ascend kernel takes bfloat16, float16 or "
            f"float8_e4m3fn index_q/index_k; got {dtype}"
        )
    hpg = heads // kv_group
    bm, bn = _BLOCK_M, _BLOCK_N
    if seq_len % bm:
        raise ValueError(f"seq_len={seq_len} must be a multiple of {bm}")
    if seq_len_kv % (bn * _TILES_PER_CHUNK):
        raise ValueError(f"seq_len_kv={seq_len_kv} must be a multiple of {bn * _TILES_PER_CHUNK}")
    if seq_len_kv // bn < 2 * _TILES_PER_CHUNK:
        raise ValueError(f"seq_len_kv={seq_len_kv} is too short for the ping-pong pipeline")
    if _pow2_ceil(hpg) > 64:
        raise ValueError(f"heads per kv group must be <= 64, got {hpg}")
    if index_k_scale_shape is not None and \
            tuple(int(v) for v in index_k_scale_shape) != (batch, seq_len_kv, kv_group):
        raise ValueError("index_k_scale must be [batch, seq_len_kv, kv_group]")

    has_scale = index_k_scale_shape is not None
    compiled = _compile(batch * kv_group, seq_len, seq_len_kv, hpg, dim, bm, bn,
                        _DTYPE_NAME[cube_dtype], has_scale)
    dummy_scale = {}

    def launch(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, index_k_scale=None):
        if decode_fp8:
            # 910B1 has no fp8 Cube path (tileops/perf/profile.py:73-89), so the
            # soft-fp8 contraction decodes to fp16 and contracts there.  The decode
            # is INSIDE this callable, i.e. inside whatever times it.
            index_q = decode_fp8_e4m3fn(index_q)
            index_k = decode_fp8_e4m3fn(index_k)
        # (b, s, g, hpg, d) -> (b*g, s, hpg, d); a no-op view when kv_group == 1.
        q = index_q.view(batch, seq_len, kv_group, hpg, dim).permute(0, 2, 1, 3, 4)
        q = q.reshape(batch * kv_group, seq_len, hpg, dim).contiguous()
        k = index_k.permute(0, 2, 1, 3).reshape(batch * kv_group, seq_len_kv, dim).contiguous()
        w = weights.view(seq_len, kv_group, hpg).permute(1, 0, 2)
        w = w.unsqueeze(0).expand(batch, kv_group, seq_len, hpg)
        w = w.reshape(batch * kv_group, seq_len, hpg).contiguous()
        if has_scale:
            if index_k_scale is None:
                raise ValueError("this kernel was built for index_k_scale, none was given")
            scale = index_k_scale.permute(0, 2, 1).reshape(batch * kv_group, seq_len_kv)
            scale = scale.contiguous()
        else:
            if index_k_scale is not None:
                raise ValueError("this kernel was built without index_k_scale")
            key = (index_q.device.index,)
            if key not in dummy_scale:
                dummy_scale[key] = torch.zeros(
                    (batch * kv_group, 1), dtype=torch.float32, device=index_q.device)
            scale = dummy_scale[key]
        out = compiled(q, k, w, cu_seqlen_ks, cu_seqlen_ke, scale)
        out = out.view(batch, kv_group, seq_len, seq_len_kv).permute(0, 2, 3, 1)
        return out.contiguous()

    return launch


__all__ = ["build_fp8_lightning_indexer_kernel", "decode_fp8_e4m3fn"]
