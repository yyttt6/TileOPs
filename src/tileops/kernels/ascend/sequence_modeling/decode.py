"""Two-stage Ascend 950 implementation for EngramDecodeFwdOp."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
import torch.nn.functional as F

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.sequence_modeling.engram.call_spec import (
    EngramDecodeCall,
    EngramDecodeFwdInterface,
)
from tileops.kernels.tiling import align_up

__all__ = ["AscendEngramDecodeKernel"]

_UB_BUDGET_BYTES = 216 * 1024
_UB_SCRATCH_RESERVE_BYTES = 32 * 1024
_MAX_THREADS = 2048
# A row of ``d`` elements moves through UB with ``T.copy``; 16 elements is the
# narrowest 32-byte vector a half-precision row can be cut into.
_D_ALIGN = 16
# Cube tile shapes. ``T.gemm`` needs static allocation extents, which the L0
# staging buffers give it; the regions handed to the MAD carry the real
# remainder of each axis.
_BLOCK_M = 16
_BLOCK_N = 64
_BLOCK_K = 256
_NUM_STAGES = 2
# The L0B tile is ``block_n * block_k`` elements per stage and the hardware
# gives that scope 64 KiB; the tiling shrinks ``block_k`` to stay under it.
_L0B_BYTES = 64 * 1024


def _next_pow2(n: int) -> int:
    """The smallest power of two that is at least *n*."""
    p = 1
    while p < n:
        p *= 2
    return p


@functools.lru_cache(maxsize=None)
def _build_project(
    batch: int,
    d_mem: int,
    d_pad: int,
    dtype_str: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
):
    """Both projections of one decode step, tiled over ``d`` and the batch.

    One block owns one ``(batch tile, K-or-V, d tile)`` output tile, so each
    weight tile streams through L1 once and the Cube cores run the whole
    ``e_t @ W`` product. The tile extents are symbolic so the last tile of each
    axis carries the real remainder, which is what keeps an unaligned ``d_mem``
    or a batch that is not a multiple of the Cube's M granularity off the host.
    """
    m_tiles = -(-batch // block_m)
    n_tiles = -(-d_pad // block_n)
    k_tiles = -(-d_mem // block_k)
    grid = m_tiles * 2 * n_tiles

    @T.macro
    def _project(E, weight, out, e_l1, w_l1, e_l0, w_l0, res, m0, n0, actual_m, actual_n):
        """One ``(batch tile, d tile)`` of ``out = E @ weight``, K-blocked."""
        for kt in T.Pipelined(k_tiles, num_stages=num_stages):
            k0 = kt * block_k
            actual_k = T.min(d_mem - k0, block_k)
            T.copy(E[m0 : m0 + actual_m, k0 : k0 + actual_k], e_l1[0:actual_m, 0:actual_k])
            T.copy(
                weight[k0 : k0 + actual_k, n0 : n0 + actual_n],
                w_l1[0:actual_n, 0:actual_k],
                transpose=True,
            )
            T.copy(e_l1[0:actual_m, 0:actual_k], e_l0[0:actual_m, 0:actual_k])
            T.copy(w_l1[0:actual_n, 0:actual_k], w_l0[0:actual_n, 0:actual_k])
            T.gemm(
                e_l0[0:actual_m, 0:actual_k],
                w_l0[0:actual_n, 0:actual_k],
                res[0:actual_m, 0:actual_n],
                transpose_B=True,
                clear_accum=kt == 0,
            )
        T.copy(res[0:actual_m, 0:actual_n], out[m0 : m0 + actual_m, n0 : n0 + actual_n])

    # The two launches of one step must not share a symbol: profilers key their
    # per-kernel records by name, and one name would fold both into one row.
    @T.prim_func
    def engram_decode_project(
        E: T.Tensor((batch, d_mem), dtype_str),
        W_K: T.Tensor((d_mem, d_pad), dtype_str),
        W_V: T.Tensor((d_mem, d_pad), dtype_str),
        K_out: T.Tensor((batch, d_pad), "float32"),
        V_out: T.Tensor((batch, d_pad), "float32"),
    ):
        with T.Kernel(grid) as bx:
            e_l1 = T.alloc_l1((block_m, block_k), dtype_str)
            w_l1 = T.alloc_l1((block_n, block_k), dtype_str)
            e_l0 = T.alloc_l0a((block_m, block_k), dtype_str)
            w_l0 = T.alloc_l0b((block_n, block_k), dtype_str)
            res = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions(
                {e_l1: num_stages, w_l1: num_stages, e_l0: num_stages, w_l0: num_stages}
            )
            m0 = (bx // (2 * n_tiles)) * block_m
            which = (bx // n_tiles) % 2
            n0 = (bx % n_tiles) * block_n
            actual_m = T.min(batch - m0, block_m)
            actual_n = T.min(d_pad - n0, block_n)
            if which == 0:
                _project(E, W_K, K_out, e_l1, w_l1, e_l0, w_l0, res, m0, n0, actual_m, actual_n)
            else:
                _project(E, W_V, V_out, e_l1, w_l1, e_l0, w_l0, res, m0, n0, actual_m, actual_n)

    return tilelang.compile(engram_decode_project, target="ascend", out_idx=[])


@functools.lru_cache(maxsize=None)
def _build_step(
    batch: int,
    d: int,
    d_pad: int,
    max_conv_len: int,
    conv_kernel_size: int,
    dilation: int,
    eps: float,
    dtype_str: str,
    threads: int,
    chunk_rows: int,
):
    """Everything in a decode step that reduces over ``d``, one core per batch row.

    The row stays in registers across the three norms, the gate, the dilated
    causal convolution and the SiLU, so ``d`` is read from global memory once.
    Both norms divide by the model dimension ``d``, not by the padded width:
    the padding lanes hold zeros and so contribute nothing to a sum.
    """
    w = conv_kernel_size
    taps = max(1, w - 1)
    tile = _next_pow2(d_pad)

    @T.prim_func
    def engram_decode_step(
        h_t: T.Tensor((batch, d_pad), dtype_str),
        K_in: T.Tensor((batch, d_pad), "float32"),
        V_in: T.Tensor((batch, d_pad), "float32"),
        conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype_str),
        rms_w_h: T.Tensor((d_pad,), dtype_str),
        rms_w_v: T.Tensor((d_pad,), dtype_str),
        conv_w: T.Tensor((w, d_pad), dtype_str),
        y_t: T.Tensor((batch, d_pad), dtype_str),
        new_conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype_str),
    ):
        with T.Kernel(batch) as bx:
            h_ub = T.alloc_shared((d_pad,), dtype_str)
            wh_ub = T.alloc_shared((d_pad,), dtype_str)
            wv_ub = T.alloc_shared((d_pad,), dtype_str)
            cw_ub = T.alloc_shared((w, d_pad), dtype_str)
            tap_ub = T.alloc_shared((taps, d_pad), dtype_str)
            y_ub = T.alloc_shared((d_pad,), dtype_str)
            vn_ub = T.alloc_shared((d_pad,), dtype_str)
            shift_ub = T.alloc_shared((chunk_rows, d_pad), dtype_str)
            k_ub = T.alloc_shared((d_pad,), "float32")
            v_ub = T.alloc_shared((d_pad,), "float32")

            T.copy(h_t[bx, :], h_ub)
            T.copy(K_in[bx, :], k_ub)
            T.copy(V_in[bx, :], v_ub)
            T.copy(rms_w_h, wh_ub)
            T.copy(rms_w_v, wv_ub)
            T.copy(conv_w, cw_ub)
            for p in range(w - 1):
                # Tap p is the value from (w-1-p)*dilation steps ago; the entry
                # cache's last slot is one step ago.
                T.copy(conv_state[bx, max_conv_len - (w - 1 - p) * dilation, :], tap_ub[p, :])

            with T.SimtVF(threads=threads):
                hf = T.alloc_fragment((tile,), "float32")
                kf = T.alloc_fragment((tile,), "float32")
                vf = T.alloc_fragment((tile,), "float32")
                wf = T.alloc_fragment((tile,), "float32")
                cwf = T.alloc_fragment((tile,), "float32")
                tapf = T.alloc_fragment((tile,), "float32")
                co = T.alloc_fragment((tile,), "float32")

                # Only the fragment needs power-of-two padding; the padded lanes
                # hold zeros, so every reduction below may run over ``tile``.
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    hf[i] = T.if_then_else(i < d_pad, T.cast(h_ub[i], "float32"), T.float32(0))
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    kf[i] = T.if_then_else(i < d_pad, k_ub[i], T.float32(0))
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    vf[i] = T.if_then_else(i < d_pad, v_ub[i], T.float32(0))
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    wf[i] = T.if_then_else(i < d_pad, T.cast(wh_ub[i], "float32"), T.float32(0))

                # h_norm = RMSNorm(h_t, rms_w_h)
                acc = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(acc)
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    T.reducer_update(acc[0], hf[i] * hf[i])
                s = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(acc, s)
                for i in T.Parallel(1):
                    s[i] = T.rsqrt(s[i] / d + eps)
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    hf[i] = hf[i] * s[0] * wf[i]

                # k_norm = RMSNorm(k.to(dtype), rms_w_h): the projection lands
                # back in the input dtype before it is normalised.
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    kf[i] = T.cast(T.cast(kf[i], dtype_str), "float32")
                acc_k = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(acc_k)
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    T.reducer_update(acc_k[0], kf[i] * kf[i])
                sk = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(acc_k, sk)
                for i in T.Parallel(1):
                    sk[i] = T.rsqrt(sk[i] / d + eps)
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    kf[i] = kf[i] * sk[0] * wf[i]

                # alpha = sigmoid(dot(h_norm, k_norm) / sqrt(d))
                acc_d = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(acc_d)
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    T.reducer_update(acc_d[0], hf[i] * kf[i])
                sd = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(acc_d, sd)
                alpha = T.alloc_fragment((1,), "float32")
                for i in T.Parallel(1):
                    alpha[i] = 1.0 / (1.0 + T.exp(-(sd[i] / T.sqrt(T.float32(d)))))

                # v_hat = alpha * v, then vhat_norm = RMSNorm(v_hat.to(dtype), rms_w_v)
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    vf[i] = alpha[0] * vf[i]
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    kf[i] = T.cast(T.cast(vf[i], dtype_str), "float32")
                acc_v = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(acc_v)
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    T.reducer_update(acc_v[0], kf[i] * kf[i])
                sv = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(acc_v, sv)
                for i in T.Parallel(1):
                    sv[i] = T.rsqrt(sv[i] / d + eps)
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    wf[i] = T.if_then_else(i < d_pad, T.cast(wv_ub[i], "float32"), T.float32(0))
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    kf[i] = kf[i] * sv[0] * wf[i]

                # conv_out = conv_w[w-1] * vhat_norm + sum_p conv_w[p] * tap[p]
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    cwf[i] = T.if_then_else(
                        i < d_pad, T.cast(cw_ub[w - 1, i], "float32"), T.float32(0)
                    )
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    co[i] = cwf[i] * kf[i]
                for p in range(w - 1):
                    for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                        tapf[i] = T.if_then_else(
                            i < d_pad, T.cast(tap_ub[p, i], "float32"), T.float32(0)
                        )
                    for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                        cwf[i] = T.if_then_else(
                            i < d_pad, T.cast(cw_ub[p, i], "float32"), T.float32(0)
                        )
                    for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                        co[i] = co[i] + cwf[i] * tapf[i]

                # y_t = SiLU(conv_out) + v_hat
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    if i < d_pad:
                        y_ub[i] = T.cast(co[i] / (1.0 + T.exp(-co[i])) + vf[i], dtype_str)
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                    if i < d_pad:
                        vn_ub[i] = T.cast(kf[i], dtype_str)

            T.copy(y_ub, y_t[bx, :])
            T.copy(vn_ub, new_conv_state[bx, max_conv_len - 1, :])
            # The shift is one contiguous run: entry rows 1.. are output rows 0...
            nshift = max_conv_len - 1
            for c in T.serial(-(-nshift // chunk_rows)):
                r0 = c * chunk_rows
                rows = T.min(chunk_rows, nshift - r0)
                T.copy(conv_state[bx, r0 + 1 : r0 + 1 + rows, :], shift_ub[0:rows, :])
                T.copy(shift_ub[0:rows, :], new_conv_state[bx, r0 : r0 + rows, :])

    return tilelang.compile(engram_decode_step, target="ascend", out_idx=[])


def _pick_block_k(block_n: int, itemsize: int, num_stages: int) -> int:
    """The widest K tile whose L0B staging fits the scope, capped at ``_BLOCK_K``."""
    limit = _L0B_BYTES // max(1, block_n * itemsize * num_stages)
    block_k = 16
    while block_k * 2 <= min(limit, _BLOCK_K):
        block_k *= 2
    return block_k


def _pick_threads(d_pad: int) -> int:
    """One thread per lane where the row is wide enough, capped for small rows."""
    return max(1, min(_MAX_THREADS, _next_pow2(d_pad), 512))


def _pick_chunk_rows(d_pad: int, max_conv_len: int, conv_kernel_size: int, itemsize: int) -> int:
    """Rows of the cache shift that fit UB beside the step's other buffers, at least one."""
    # h_t, the two norm weights, y_t and the new cache slot are rows of the input
    # dtype; the conv weights and taps are ``w`` and ``w - 1`` of them; the two
    # projections arrive as float32 rows.
    fixed = (5 + conv_kernel_size + max(1, conv_kernel_size - 1)) * d_pad * itemsize + 2 * d_pad * 4
    budget = _UB_BUDGET_BYTES - _UB_SCRATCH_RESERVE_BYTES - fixed
    rows = max(1, budget // max(1, d_pad * itemsize))
    return max(1, min(rows, max(1, max_conv_len - 1)))


class AscendEngramDecodeKernel(Kernel, EngramDecodeFwdInterface):
    """Two-stage Engram decode on Ascend 950: Cube projection and Vector epilogue.

    The projections run as two Cube ``NT`` GEMMs tiled over ``d`` and the batch;
    the RMSNorm gating, the dilated causal convolution, the SiLU and the cache
    shift run one core per batch row, keeping the full-row reductions off a
    cross-core barrier.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: EngramDecodeCall) -> Entry:
        return (call, lambda: cls(call))

    def __init__(self, call: EngramDecodeCall) -> None:
        super().__init__(device_index=call.device.index if call.device is not None else None)
        min_cache = call.dilation * (call.conv_kernel_size - 1)
        if call.max_conv_len < min_cache:
            raise ValueError(
                f"max_conv_len ({call.max_conv_len}) must be >= "
                f"dilation * (conv_kernel_size - 1) = {min_cache}"
            )
        self.batch = call.batch
        self.d = call.d
        # Only ``d`` is padded, and only to a 32-byte vector: the Cube tiles carry
        # their own remainders, so ``d_mem`` and ``batch`` reach the kernel as-is.
        self.d_pad = align_up(call.d, _D_ALIGN)
        self.max_conv_len = call.max_conv_len
        self.dtype = call.dtype
        itemsize = call.dtype.itemsize
        self.threads = _pick_threads(self.d_pad)
        self.chunk_rows = _pick_chunk_rows(
            self.d_pad, call.max_conv_len, call.conv_kernel_size, itemsize
        )
        block_n = min(_BLOCK_N, self.d_pad)
        self.project = _build_project(
            call.batch,
            call.d_mem,
            self.d_pad,
            self.dtype_str,
            _BLOCK_M,
            block_n,
            _pick_block_k(block_n, itemsize, _NUM_STAGES),
            _NUM_STAGES,
        )
        self.step = _build_step(
            call.batch,
            call.d,
            self.d_pad,
            call.max_conv_len,
            call.conv_kernel_size,
            call.dilation,
            call.eps,
            self.dtype_str,
            self.threads,
            self.chunk_rows,
        )

    def forward(
        self,
        e_t: torch.Tensor,
        h_t: torch.Tensor,
        conv_state: torch.Tensor,
        W_K: torch.Tensor,
        W_V: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run one decode step in two launches without modifying inputs in place."""
        self._require_devices(e_t=e_t, h_t=h_t, conv_state=conv_state, W_K=W_K, W_V=W_V)
        state_pad = self.max_conv_len - conv_state.shape[1]
        if state_pad > 0:
            conv_state = F.pad(conv_state, (0, 0, state_pad, 0))
        pad = self.d_pad - self.d
        if pad:
            h_t = F.pad(h_t, (0, pad))
            conv_state = F.pad(conv_state, (0, pad))
            W_K = F.pad(W_K, (0, pad))
            W_V = F.pad(W_V, (0, pad))
            rms_w_h = F.pad(rms_w_h, (0, pad))
            rms_w_v = F.pad(rms_w_v, (0, pad))
            conv_w = F.pad(conv_w, (0, pad))
        k_out = torch.empty((self.batch, self.d_pad), dtype=torch.float32, device=e_t.device)
        v_out = torch.empty((self.batch, self.d_pad), dtype=torch.float32, device=e_t.device)
        self.project(e_t, W_K, W_V, k_out, v_out)
        y_t = torch.empty((self.batch, self.d_pad), dtype=e_t.dtype, device=e_t.device)
        new_conv_state = torch.empty(
            (self.batch, self.max_conv_len, self.d_pad), dtype=e_t.dtype, device=e_t.device
        )
        self.step(
            h_t,
            k_out,
            v_out,
            conv_state,
            rms_w_h,
            rms_w_v,
            conv_w,
            y_t,
            new_conv_state,
        )
        if pad:
            return [y_t[:, : self.d], new_conv_state[:, :, : self.d]]
        return [y_t, new_conv_state]


register_kernel_type("EngramDecodeFwdOp", "ascend", AscendEngramDecodeKernel)
