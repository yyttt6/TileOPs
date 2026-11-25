"""Pool closing windows against the pre-call rings, then commit only surviving ring rows."""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.attention.gated_pool_compress import GatedPoolCompressFwdInterface
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type


@lru_cache(maxsize=64)
def _programs(tokens, batch, slots, ratio, overlap, dim, dtype):
    span = ratio * (2 if overlap else 1)
    width = dim * (2 if overlap else 1)

    @T.macro
    def load_value(kv, score, ape, kv_state, score_state, tok, pos, first, slot, j, channel, v, s):
        p = pos - span + 1 + j
        col = channel + T.if_then_else(overlap and j >= ratio, dim, 0)
        if p < 0:
            v[0] = 0
            s[0] = -T.infinity("float32")
        elif p < first:
            v[0] = kv_state[slot, p % span, col]
            s[0] = score_state[slot, p % span, col]
        else:
            v[0] = T.cast(kv[tok + p - pos, col], "float32")
            s[0] = T.cast(score[tok + p - pos, col], "float32") + T.cast(
                ape[p % ratio, col], "float32"
            )

    @T.prim_func
    def pool(
        kv: T.Tensor((tokens, width), dtype),
        score: T.Tensor((tokens, width), dtype),
        ape: T.Tensor((ratio, width), dtype),
        kv_state: T.Tensor((slots, span, width), "float32"),
        score_state: T.Tensor((slots, span, width), "float32"),
        seq: T.Tensor((tokens,), "int32"),
        cu: T.Tensor((batch + 1,), "int32"),
        starts: T.Tensor((batch,), "int32"),
        slot_ids: T.Tensor((batch,), "int32"),
        result: T.Tensor((tokens, dim), "bfloat16"),
        valid: T.Tensor((tokens,), "bool"),
    ):
        with T.Kernel(tokens) as tok:
            request = seq[tok]
            first = starts[request]
            pos = first + tok - cu[request]
            slot = slot_ids[request]
            closing = (pos + 1) % ratio == 0
            valid[tok] = closing
            if closing:
                with T.SimtVF(threads=min(128, dim)):
                    for col in T.Parallel(dim):
                        v = T.alloc_local((1,), "float32")
                        s = T.alloc_local((1,), "float32")
                        peak = T.alloc_local((1,), "float32")
                        total = T.alloc_local((1,), "float32")
                        value = T.alloc_local((1,), "float32")
                        peak[0] = -T.infinity("float32")
                        total[0] = 0
                        value[0] = 0
                        for j in T.serial(span):
                            load_value(
                                kv,
                                score,
                                ape,
                                kv_state,
                                score_state,
                                tok,
                                pos,
                                first,
                                slot,
                                j,
                                col,
                                v,
                                s,
                            )
                            peak[0] = T.max(peak[0], s[0])
                        for j in T.serial(span):
                            load_value(
                                kv,
                                score,
                                ape,
                                kv_state,
                                score_state,
                                tok,
                                pos,
                                first,
                                slot,
                                j,
                                col,
                                v,
                                s,
                            )
                            total[0] += T.exp(s[0] - peak[0])
                        for j in T.serial(span):
                            load_value(
                                kv,
                                score,
                                ape,
                                kv_state,
                                score_state,
                                tok,
                                pos,
                                first,
                                slot,
                                j,
                                col,
                                v,
                                s,
                            )
                            weight = T.cast(T.exp(s[0] - peak[0]) / total[0], dtype)
                            value[0] += T.cast(
                                T.cast(weight * T.cast(v[0], dtype), dtype), "float32"
                            )
                        result[tok, col] = T.cast(value[0], "bfloat16")

    @T.prim_func
    def commit(
        kv: T.Tensor((tokens, width), dtype),
        score: T.Tensor((tokens, width), dtype),
        ape: T.Tensor((ratio, width), dtype),
        kv_state: T.Tensor((slots, span, width), "float32"),
        score_state: T.Tensor((slots, span, width), "float32"),
        seq: T.Tensor((tokens,), "int32"),
        cu: T.Tensor((batch + 1,), "int32"),
        starts: T.Tensor((batch,), "int32"),
        slot_ids: T.Tensor((batch,), "int32"),
    ):
        with T.Kernel(tokens) as tok:
            req = seq[tok]
            if tok >= cu[req + 1] - span:
                pos = starts[req] + tok - cu[req]
                slot = slot_ids[req]
                with T.SimtVF(threads=min(128, width)):
                    for col in T.Parallel(width):
                        kv_state[slot, pos % span, col] = T.cast(kv[tok, col], "float32")
                        score_state[slot, pos % span, col] = T.cast(
                            score[tok, col], "float32"
                        ) + T.cast(ape[pos % ratio, col], "float32")

    return tilelang.compile(pool, target="ascend", out_idx=[9, 10]), tilelang.compile(
        commit, target="ascend", out_idx=[]
    )


class AscendGatedPoolCompressKernel(Kernel, GatedPoolCompressFwdInterface):
    devices = frozenset({"npu"})
    supported_archs = None

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(self, kv, score, ape, kv_state, score_state, cu_seqlens, start_pos, state_slots):
        tokens = kv.shape[0]
        if not tokens:
            return kv.new_empty((0, self.call.dim), dtype=torch.bfloat16), kv.new_empty(
                (0,), dtype=torch.bool
            )
        args = (
            kv,
            score,
            ape,
            kv_state,
            score_state,
            torch.searchsorted(
                cu_seqlens[1:],
                torch.arange(tokens, device=kv.device, dtype=torch.int32),
                right=True,
                out_int32=True,
            ),
            cu_seqlens,
            start_pos,
            state_slots,
        )
        pool, commit = _programs(
            tokens,
            start_pos.numel(),
            kv_state.shape[0],
            self.call.ratio,
            self.call.overlap,
            self.call.dim,
            self.dtype_to_str(kv.dtype),
        )
        entries, valid = pool(*args)
        commit(*args)
        return entries, valid


register_kernel_type("GatedPoolCompressFwdOp", "ascend", AscendGatedPoolCompressKernel)
