"""M128 SIMD producer with two M64 Cube consumers and two L0A slots."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.mamba.call_spec import SSDChunkStateCall, SSDChunkStateFwdInterface

__all__ = ["AscendSSDChunkStateFwdKernel"]


@functools.lru_cache(maxsize=64)
def _build(B, NC, Q, H, P, G, N, dtype, has_seq_idx):
    K, KH, M, CM = 256, 128, 128, 64
    PAIRS = H // 2
    JOBS = B * NC * 2
    HEAD_TURNS = PAIRS // 2
    GRID = min(32, JOBS)
    ROUNDS = (JOBS + GRID - 1) // GRID

    @T.macro
    def metadata_load(b, c, h, sid, dt, dA, dt_ub, da_ub, end_da_ub):
        for head in T.serial(2):
            T.copy(dt[b, h + head, c, sid * KH : (sid + 1) * KH], dt_ub[head, 0:KH])
            T.copy(dA[b, h + head, c, sid * KH : (sid + 1) * KH], da_ub[head, 0:KH])
            # Each endpoint destination starts at a separate aligned 32-byte row.
            T.copy(dA[b, h + head, c, K - 1 : K], end_da_ub[head, 0:1])

    @T.macro
    def load_chunk_mask(b, c, sid, seq_idx, seq_ub, end_seq_ub):
        # Chunk metadata is invariant across every head pair.
        if has_seq_idx:
            T.copy(seq_idx[b, c * K + sid * KH : c * K + (sid + 1) * KH], seq_ub)
            T.copy(seq_idx[b, (c + 1) * K - 1 : (c + 1) * K], end_seq_ub[0:1])

    @T.macro
    def metadata_weights(dt_ub, da_ub, end_da_ub, seq_ub, end_seq_ub, weights):
        with T.SimdVF():
            full = S.pset(32)
            zero = S.vdup(T.float32(0), "float32", full)
            if has_seq_idx:
                end_id = S.vld(end_seq_ub[0], "BRC_B32")
                nonnegative = S.vcmps(end_id, T.int32(0), full, "ge")
            for head in range(2):
                end_da = S.vld(end_da_ub[head, 0], "BRC_B32")
                for block in range(KH // 64):
                    delta = S.vsub(end_da, S.vld(da_ub[head, block * 64]), full)
                    weight = S.vmul(
                        S.vexp(S.vmin(delta, zero, full), full),
                        S.vld(dt_ub[head, block * 64]),
                        full,
                    )
                    if has_seq_idx:
                        same = S.vcmp(S.vld(seq_ub[block * 64]), end_id, full, "eq")
                        weight = S.vsel(weight, zero, S.pand(same, nonnegative, full))
                    S.vsts(weights[head, block * 64], weight, full)
            # Explicit SIMD store-to-load visibility, including the next VF.
            S.mem_bar("VST_VLD")

    @T.macro
    def load_x(b, c, h, sid, x_flat, raw_x):
        T.copy(x_flat[b, c * K + sid * KH : c * K + (sid + 1) * KH, h * P : (h + 2) * P], raw_x)

    @T.macro
    def pack_weighted_nz(raw_x, weights, scaled_nz):
        # Final ROUND_R values go directly to the compact-NZ transfer layout.
        with T.SimdVF():
            full = S.pset(32)
            packed_mask = S.pset(16)
            nz_ptr = S.make_ubuf_ptr(T.access_ptr(scaled_nz[0, 0], "w", KH + 1, M), dtype)
            for k in range(KH):
                x0 = S.vcvt(S.vld(raw_x[k, 0], "UNPK_B16"), "float32", full)
                x1 = S.vcvt(S.vld(raw_x[k, P], "UNPK_B16"), "float32", full)
                product0 = S.vmul(x0, S.vld(weights[0, k], "BRC_B32"), full)
                product1 = S.vmul(x1, S.vld(weights[1, k], "BRC_B32"), full)
                rounded0 = S.vcvt(product0, dtype, full, round="ROUND_R", sat=False, part=0)
                rounded1 = S.vcvt(product1, dtype, full, round="ROUND_R", sat=False, part=0)
                # PART_EVEN puts 64 values at even B16 lanes. Join both heads.
                packed, unused = S.vdintlv(rounded0, rounded1)
                nz_ptr = S.vsstb(
                    packed, nz_ptr, T.int32(((KH + 1) << 16) | 1), packed_mask, update=True
                )
            S.mem_bar("VST_VLD")

    @T.macro
    def load_resident_b(b, c, Bmat, b_l1, b_l0):
        # One B transfer for every head pair in this chunk stripe.
        T.copy(Bmat[b, c * K : (c + 1) * K, 0, 0:N], b_l1, transpose=True)
        T.copy(b_l1, b_l0)

    @T.macro
    def publish_a(scaled_nz, a_l1):
        T.dual_copy(scaled_nz[0:KH, 0:M], a_l1)

    @T.macro
    def write_result(job, subhead, acc, out_flat):
        T.copy(acc, out_flat[job * M + subhead * CM : job * M + (subhead + 1) * CM, 0:N])

    @T.prim_func
    def ssd_chunk_state_fwd(
        x: T.Tensor((B, NC * Q, H, P), dtype),
        Bmat: T.Tensor((B, NC * Q, G, N), dtype),
        dt: T.Tensor((B, H, NC, Q), "float32"),
        dA: T.Tensor((B, H, NC, Q), "float32"),
        seq_idx: T.Tensor((B, NC * Q), "int32"),
        states: T.Tensor((B, NC, H, P, N), "float32"),
    ):
        x_flat = T.decl_buffer((B, NC * Q, H * P), dtype, data=x.data)
        out_flat = T.decl_buffer((B * NC * H * P, N), "float32", data=states.data)
        with T.MixedKernel(GRID, sids=2) as (bx, sid):
            raw_x = T.alloc_shared((KH, M), dtype)
            scaled_nz = T.alloc_shared((KH + 1, M), dtype)
            T.annotate_layout({scaled_nz: make_ascend_compact_nz_layout(scaled_nz)})
            dt_ub = T.alloc_shared((2, KH), "float32")
            da_ub = T.alloc_shared((2, KH), "float32")
            end_da_ub = T.alloc_shared((2, 8), "float32")
            weights = T.alloc_shared((2, KH), "float32")
            seq_ub = T.alloc_shared((KH,), "int32")
            end_seq_ub = T.alloc_shared((8,), "int32")
            a_l1 = T.alloc_l1((K, M), dtype)
            b_l1 = T.alloc_l1((N, K), dtype)
            a_l0 = T.alloc_l0a((CM, K), dtype)
            b_l0 = T.alloc_l0b((N, K), dtype)
            acc = T.alloc_l0c((CM, N), "float32")
            # B is invariant throughout the head pipeline; L0B fills its 64KiB pool.
            T.annotate_buffer_versions(
                {
                    b_l1: 1,
                    b_l0: 1,
                    # The inner subhead clock owns two 32KiB L0A epochs.
                    a_l0: (2, "iteration"),
                    # Full metadata epochs can be transferred ahead of their SIMD reader.
                    dt_ub: (2, "iteration"),
                    da_ub: (2, "iteration"),
                    end_da_ub: (2, "iteration"),
                    # Both weights accesses are ordered tasks on PIPE_V at stage 1.
                    weights: 1,
                    # Per-chunk resident data, not head-turn scratch.
                    seq_ub: 1,
                    end_seq_ub: 1,
                    raw_x: (2, "iteration"),
                    scaled_nz: (2, "iteration"),
                    a_l1: (2, "iteration"),
                    acc: (2, "iteration"),
                }
            )
            # A second chunk cannot coexist with the resident full-size L0B.
            for turn in T.serial(ROUNDS):
                job = turn * GRID + bx
                if job < JOBS:
                    stripe = job % 2
                    c = job // 2 % NC
                    b = job // (2 * NC)
                    load_resident_b(b, c, Bmat, b_l1, b_l0)
                    load_chunk_mask(b, c, sid, seq_idx, seq_ub, end_seq_ub)
                    for head_turn in T.Pipelined(
                        HEAD_TURNS, num_stages=2, annotations={"enable_offset": True}
                    ):
                        pair = head_turn * 2 + stripe
                        h = pair * 2
                        output_job = (b * NC + c) * PAIRS + pair
                        # One lookahead epoch: DMA may prepare i+1 while SIMD consumes i.
                        with T.Stage(0):
                            load_x(b, c, h, sid, x_flat, raw_x)
                            metadata_load(b, c, h, sid, dt, dA, dt_ub, da_ub, end_da_ub)
                        with T.Stage(1):
                            metadata_weights(dt_ub, da_ub, end_da_ub, seq_ub, end_seq_ub, weights)
                            pack_weighted_nz(raw_x, weights, scaled_nz)
                            publish_a(scaled_nz, a_l1)
                        with T.Stage(2):
                            # A-L1 remains owned by the complete pair, covering both readers.
                            for subhead in T.Pipelined(
                                2, num_stages=2, annotations={"enable_offset": True}
                            ):
                                with T.Stage(0):
                                    T.copy(
                                        a_l1[0:K, subhead * CM : (subhead + 1) * CM],
                                        a_l0,
                                        transpose=True,
                                    )
                                with T.Stage(1):
                                    T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
                                    write_result(output_job, subhead, acc, out_flat)

    return tilelang.compile(ssd_chunk_state_fwd, target="ascend", out_idx=[5])


class AscendSSDChunkStateFwdKernel(Kernel, SSDChunkStateFwdInterface):
    """Compute FP32 chunk-end states for the Ascend 950 SSD tile geometry."""

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: SSDChunkStateCall) -> str | None:
        if (call.chunk_len, call.d_head, call.d_state, call.n_groups) != (256, 64, 128, 1):
            return "requires chunk_len=256, d_head=64, d_state=128 and n_groups=1"
        if call.n_heads <= 0 or call.n_heads % 4:
            return "requires a positive head count divisible by four"
        if call.dtype not in (torch.float16, torch.bfloat16) or call.dt_dtype != torch.float32:
            return "requires FP16 or BF16 activations and FP32 dt"
        return None

    @classmethod
    def entry_for(cls, call: SSDChunkStateCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: SSDChunkStateCall) -> None:
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.dtype = call.dtype
        self.kernel = _build(
            call.batch,
            call.num_chunks,
            call.chunk_len,
            call.n_heads,
            call.d_head,
            call.n_groups,
            call.d_state,
            self.dtype_str,
            call.has_seq_idx,
        )

    def forward(
        self,
        x: torch.Tensor,
        Bmat: torch.Tensor,
        dt: torch.Tensor,
        dA_cumsum: torch.Tensor,
        seq_idx: torch.Tensor,
    ) -> torch.Tensor:
        # DMA operates on 32-byte blocks. Handle contiguous views with an unaligned offset.
        x, Bmat, dt, dA_cumsum, seq_idx = (
            tensor.clone() if tensor.data_ptr() % 32 else tensor
            for tensor in (x, Bmat, dt, dA_cumsum, seq_idx)
        )
        return self.kernel(x, Bmat, dt, dA_cumsum, seq_idx)


register_kernel_type("SSDChunkStateFwdOp", "ascend", AscendSSDChunkStateFwdKernel)
