"""Ascend chunkwise Gated DeltaNet adapter.

The implementation composes the checked-in TileLang-Ascend GDN stages.  The
public TileOPs contract is token-major (BTHD), while the stages use head-major
(BHTD), so the adapter keeps that conversion at the call boundary and restores
the declared output layouts before returning.
"""

from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

import torch


_EXAMPLES = Path("/home/dyq/workspace/tilelang-ascend/examples/linear_attention_and_rnn")


def _load_stages():
    path = str(_EXAMPLES)
    if path not in sys.path:
        sys.path.insert(0, path)
    from opt_gdn.opt_gdn_chunk_cumsum import chunk_cumsum
    from opt_gdn.opt_gdn_chunk_h import chunk_h
    from opt_gdn.opt_gdn_chunk_o import chunk_o
    from opt_gdn.opt_gdn_chunk_scaled_dot_kkt import kkt
    from opt_gdn.opt_gdn_solve_tril import solve_tril_64_ker
    from opt_gdn.opt_gdn_wy_fast import wy_fast

    return chunk_cumsum, kkt, solve_tril_64_ker, wy_fast, chunk_h, chunk_o


@lru_cache(maxsize=32)
def _stage_builders(batch: int, heads: int, seq_len: int, dim: int, chunk_size: int):
    chunk_cumsum, kkt, solve_tril_64_ker, wy_fast, chunk_h, chunk_o = _load_stages()
    return (
        lambda g: chunk_cumsum(g, chunk_size),
        lambda k, beta, g: kkt(k, beta, g, chunk_size),
        lambda a: solve_tril_64_ker(batch, heads, seq_len)(
            a, torch.eye(chunk_size, dtype=torch.float32, device=a.device)
        ),
        lambda k, v, beta, g, a: wy_fast(k, v, beta, g, a, chunk_size),
        lambda k, w, u, g: chunk_h(k, w, u, g, chunk_size),
        lambda q, k, nv, s, g: chunk_o(q, k, nv, s, g, chunk_size),
    )


def build_gated_deltanet_kernel(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    g_shape: tuple[int, ...],
    beta_shape: tuple[int, ...],
    dtype: torch.dtype,
    chunk_size: int = 64,
):
    """Build a callable matching ``GatedDeltaNetBTHDFwdOp``."""
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
        raise ValueError("GatedDeltaNetBTHDFwdOp expects rank-4 q/k/v")
    if q_shape != k_shape or q_shape[:3] != v_shape[:3]:
        raise ValueError("q/k/v BTHD shapes are incompatible")
    b, seq_len, heads, dim_k = q_shape
    dim_v = v_shape[-1]
    if dim_k != dim_v:
        raise ValueError(f"GatedDeltaNetBTHDFwdOp requires DK == DV, got {dim_k} and {dim_v}")
    if g_shape != (b, seq_len, heads) or beta_shape != g_shape:
        raise ValueError("g/beta must have shape [B, S, H]")
    if chunk_size != 64:
        raise ValueError(f"GatedDeltaNetBTHDFwdOp requires chunk_size=64, got {chunk_size}")
    if seq_len % chunk_size:
        raise ValueError(f"sequence length {seq_len} is not divisible by chunk_size {chunk_size}")
    if dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"GatedDeltaNetBTHDFwdOp supports float16/bfloat16, got {dtype}")

    stages = _stage_builders(b, heads, seq_len, dim_k, chunk_size)
    num_chunks = seq_len // chunk_size

    def invoke(q, k, v, g, beta):
        expected = (q_shape, k_shape, v_shape, g_shape, beta_shape)
        got = (tuple(q.shape), tuple(k.shape), tuple(v.shape), tuple(g.shape), tuple(beta.shape))
        if got != expected:
            raise ValueError(f"GatedDeltaNetBTHDFwdOp kernel shape mismatch: expected {expected}, got {got}")
        # The checked stages consume BHTD.  Permute is a view; contiguous is
        # required because their GM indexing is compile-time contiguous.
        qh = q.permute(0, 2, 1, 3).contiguous()
        kh = k.permute(0, 2, 1, 3).contiguous()
        vh = v.permute(0, 2, 1, 3).contiguous()
        gh = g.permute(0, 2, 1).contiguous().float()
        bh = beta.permute(0, 2, 1).contiguous()
        if dtype == torch.bfloat16:
            qh, kh, vh, bh = (x.to(torch.float16) for x in (qh, kh, vh, bh))

        # TileLang stages use independent asynchronous launch paths.  An
        # output buffer must be complete before it is passed as an input to
        # the next stage; without an explicit fence, larger chunk counts can
        # expose stale/partially written workspace even though small shapes
        # often appear finite.
        g_cum = stages[0](gh)
        torch.npu.synchronize()
        a = stages[1](kh, bh, g_cum)
        torch.npu.synchronize()
        a = stages[2](a)
        torch.npu.synchronize()
        w, u = stages[3](kh, vh, bh, g_cum, a)
        torch.npu.synchronize()
        s, nv, final_state = stages[4](kh, w, u, g_cum)
        torch.npu.synchronize()
        oh = stages[5](qh, kh, nv, s, g_cum)
        torch.npu.synchronize()

        # Manifest S has an initial state slot followed by one state per
        # chunk.  The stage exposes the initial-inclusive chunk slots and the
        # final state separately, so materialize the declared shape here.
        state = torch.zeros(
            (b, heads, num_chunks + 1, dim_k, dim_v), dtype=oh.dtype, device=oh.device
        )
        state[:, :, :num_chunks] = s
        state[:, :, num_chunks] = final_state
        out = oh.permute(0, 2, 1, 3).contiguous()
        aw = a.permute(0, 2, 1, 3).contiguous()
        au = aw.clone()
        if dtype == torch.bfloat16:
            out, state, aw, au = (x.to(dtype) for x in (out, state, aw, au))
        return out, state, aw, au

    return invoke
