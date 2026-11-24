"""Stable tight routing in bounded workspaces for large expert batches."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.ascend.moe.pre_permute import route_pass_configs
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.moe.call_spec import PrePermuteCall, PrePermuteFwdInterface
from tileops.backend._registry import register_kernel_type


@functools.lru_cache(maxsize=32)
def _stages(tokens, top_k, experts, hidden, dtype):
    routes = tokens * top_k
    tile = 256
    chunks = (routes + tile - 1) // tile

    @T.prim_func
    def rank(
        ids: T.Tensor((routes,), "int32"),
        ranks: T.Tensor((routes,), "int32"),
        counts: T.Tensor((chunks, experts), "int32"),
    ):
        with T.Kernel(chunks) as bx:
            src = T.alloc_shared((tile,), "int32")
            local = T.alloc_shared((tile,), "int32")
            hist = T.alloc_shared((experts,), "int32")
            first = bx * tile
            used = T.min(routes - first, tile)
            T.copy(ids[first : first + used], src[:used])
            T.ascend_set_flag("MTE2_S", 0)
            T.ascend_wait_flag("MTE2_S", 0)
            for g in T.serial(experts):
                hist[g] = 0
            for j in T.serial(used):
                g = src[j]
                local[j] = hist[g]
                hist[g] += 1
            T.ascend_set_flag("S_MTE3", 0)
            T.ascend_wait_flag("S_MTE3", 0)
            T.copy(hist[:experts], counts[bx, :])
            T.copy(local[:used], ranks[first : first + used])

    @T.prim_func
    def prefix(
        counts: T.Tensor((chunks, experts), "int32"),
        offsets: T.Tensor((chunks, experts), "int32"),
        ends: T.Tensor((experts,), "int32"),
    ):
        with T.Kernel(1):
            totals = T.alloc_shared((experts,), "int32")
            with T.SimtVF(threads=256):
                for g in T.Parallel(experts):
                    total = T.alloc_local((1,), "int32")
                    total[0] = 0
                    for c in T.serial(chunks):
                        total[0] += counts[c, g]
                    totals[g] = total[0]
            with T.SimtVF(threads=256):
                for g in T.Parallel(experts):
                    pos = T.alloc_local((1,), "int32")
                    pos[0] = 0
                    for previous in T.serial(g):
                        pos[0] += totals[previous]
                    for c in T.serial(chunks):
                        offsets[c, g] = pos[0]
                        pos[0] += counts[c, g]
                    ends[g] = pos[0]

    @T.prim_func
    def scatter(
        x: T.Tensor((tokens, hidden), dtype),
        ids: T.Tensor((routes,), "int32"),
        ranks: T.Tensor((routes,), "int32"),
        offsets: T.Tensor((chunks, experts), "int32"),
        output: T.Tensor((routes, hidden), dtype),
        inverse: T.Tensor((routes,), "int32"),
    ):
        with T.Kernel(tokens) as token:
            row = T.alloc_shared((hidden,), dtype)
            T.copy(x[token, :], row)
            T.ascend_set_flag("MTE2_MTE3", 0)
            T.ascend_wait_flag("MTE2_MTE3", 0)
            for j in T.serial(top_k):
                index = token * top_k + j
                g = ids[index]
                dst = offsets[index // tile, g] + ranks[index]
                inverse[index] = dst
                T.copy(row, output[dst, :])

    return (
        tilelang.compile(rank, target="ascend", out_idx=[1, 2], pass_configs=route_pass_configs()),
        tilelang.compile(prefix, target="ascend", out_idx=[1, 2]),
        tilelang.compile(
            scatter, target="ascend", out_idx=[4, 5], pass_configs=route_pass_configs()
        ),
    )


class AscendLargePrePermuteKernel(Kernel, PrePermuteFwdInterface):
    """Rank routes in fixed-size chunks and prefix their counts in expert order."""

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: PrePermuteCall):
        if call.layout.selection_key != "tight_physical_psum":
            return "requires tight physical prefix sums"
        if call.input_dtype not in (torch.float16, torch.bfloat16):
            return "requires float16 or bfloat16"
        if call.num_tokens * call.top_k <= 4096:
            return "the small-route implementation serves up to 4096 routes"
        if not 0 < call.num_experts <= 1024:
            return "requires 1–1024 local experts"
        if call.hidden_size % 16 or not 0 < call.hidden_size <= 16384:
            return "the row DMA requires a width divisible by 16 and at most 16384"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: PrePermuteCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: PrePermuteCall):
        super().__init__(device_index=call.device.index)
        self.dtype = call.input_dtype
        self.rank, self.prefix, self.scatter = _stages(
            call.num_tokens,
            call.top_k,
            call.num_experts,
            call.hidden_size,
            self.dtype_str,
        )

    def forward(self, hidden_states, local_expert_ids):
        flat = local_expert_ids.reshape(-1)
        ranks, counts = self.rank(flat)
        offsets, ends = self.prefix(counts)
        out, inverse = self.scatter(hidden_states, flat, ranks, offsets)
        return out, ends, inverse


register_kernel_type("MoEPrePermuteFwdOp", "ascend_large", AscendLargePrePermuteKernel)
