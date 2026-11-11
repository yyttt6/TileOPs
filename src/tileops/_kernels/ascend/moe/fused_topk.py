"""Fused MoE top-k routing on the Ascend 950 AI core.

Scores each token's router logits, selects its ``top_k`` experts, and returns
their weights and ids. Supports softmax and sigmoid scoring, an optional
selection-only correction bias, grouped routing, renormalization and output
scaling. It serves ``num_experts % 32 == 0``.
"""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.moe.call_spec import FusedTopKCall, FusedTopKFwdInterface
from tileops.backend._registry import register_kernel_type

__all__ = ["AscendFusedTopKKernel"]

_WARP_LANES = 32
_NEG = -3.0e38
_NUM_CORES = 64


def _pick_tokens_per_block(tokens: int) -> int:
    """The tokens per block for a call of *tokens* tokens.

    A block holds one 32-lane group per token, and a block covers at least four tokens
    where the token count allows it.
    """
    target = max(4, tokens // _NUM_CORES)
    for tpb in (16, 8, 4, 2, 1):
        if tpb <= target and tokens % tpb == 0:
            return tpb
    return 1


def make_fused_topk(
    tokens: int,
    experts: int,
    top_k: int,
    dtype: str = "bfloat16",
    scoring_func: str = "softmax",
    renormalize: bool = False,
    with_bias: bool = False,
    num_expert_group: int = 1,
    topk_group: int = 1,
    routed_scaling_factor: float = 1.0,
):
    """Build ``main(logits_gm, bias_gm, weights_gm, ids_gm)`` for one routing call."""
    if scoring_func not in ("softmax", "sigmoid"):
        raise ValueError(f"unknown scoring_func: {scoring_func!r}")
    if with_bias and scoring_func != "sigmoid":
        raise ValueError("with_bias requires scoring_func='sigmoid'")
    if dtype not in ("float16", "bfloat16", "float32"):
        raise TypeError(f"FusedTopK supports float16/bfloat16/float32, got {dtype}")
    if top_k > experts:
        raise ValueError(f"top_k={top_k} cannot exceed num_experts={experts}")
    if experts % 32 != 0:
        raise ValueError(f"num_experts must be a multiple of 32, got {experts}")
    if num_expert_group < 1 or experts % num_expert_group != 0:
        raise ValueError(
            f"num_expert_group ({num_expert_group}) must be >= 1 and divide num_experts ({experts})"
        )
    group_size = experts // num_expert_group
    if not 1 <= topk_group <= num_expert_group or top_k > topk_group * group_size:
        raise ValueError(
            f"topk_group ({topk_group}) must lie in [1, {num_expert_group}] and keep "
            f"at least top_k ({top_k}) experts"
        )
    if num_expert_group > 1 and with_bias and group_size < 2:
        raise ValueError("a biased group score sums two experts; each group needs two")

    tokens_per_block = _pick_tokens_per_block(tokens)
    elems_per_thread = experts // _WARP_LANES  # exact: E % 32 == 0
    num_blocks = tokens // tokens_per_block
    threads = tokens_per_block * _WARP_LANES
    launch = max(d for d in range(1, min(num_blocks, _NUM_CORES) + 1) if num_blocks % d == 0)
    grid_repeats = num_blocks // launch
    grouped = num_expert_group > 1
    scaled = routed_scaling_factor != 1.0
    group_size = experts // num_expert_group

    def group_of(j, lane_id):
        """The group of slot ``j``'s expert, ``j * 32 + lane_id``."""
        if group_size % _WARP_LANES == 0:
            return (j * _WARP_LANES) // group_size
        return (j * _WARP_LANES + lane_id) // group_size

    @T.prim_func
    def main(
        logits_gm: T.Tensor((tokens, experts), dtype),
        bias_gm: T.Tensor((experts,), "float32"),
        weights_gm: T.Tensor((tokens, top_k), "float32"),
        ids_gm: T.Tensor((tokens, top_k), "int32"),
    ):
        with T.Kernel(launch) as bx:
            x_ub = T.alloc_shared((tokens_per_block, experts), dtype)
            w_ub = T.alloc_shared((tokens_per_block, top_k), "float32")
            i_ub = T.alloc_shared((tokens_per_block, top_k), "int32")
            versions = {x_ub: 2, w_ub: 1, i_ub: 1}
            if with_bias:
                bias_ub = T.alloc_shared((experts,), "float32")
                versions[bias_ub] = 1
                T.copy(bias_gm, bias_ub)
            T.annotate_buffer_versions(versions)
            # A single pass over the grid has nothing to overlap.
            for g in T.Pipelined(grid_repeats, num_stages=1 if grid_repeats == 1 else 2):
                block_id = bx + g * launch
                token_base = block_id * tokens_per_block
                T.copy(logits_gm[token_base : token_base + tokens_per_block, :], x_ub)

                with T.SimtVF(threads=threads):
                    tx = T.get_thread_binding()
                    warp_id = tx // _WARP_LANES
                    lane_id = tx % _WARP_LANES

                    my_scores = T.alloc_local([elems_per_thread], "float32")
                    sel = T.alloc_local([elems_per_thread], "float32")

                    for j in T.serial(elems_per_thread):
                        my_scores[j] = T.cast(x_ub[warp_id, j * _WARP_LANES + lane_id], "float32")

                    if scoring_func == "softmax":
                        l_max = T.alloc_var(T.float32)
                        l_max = _NEG
                        for j in T.serial(elems_per_thread):
                            l_max = T.max(l_max, my_scores[j])
                        l_max = T.warp_reduce_max(l_max)
                        if renormalize:
                            for j in T.serial(elems_per_thread):
                                my_scores[j] = T.exp(my_scores[j] - l_max)
                        else:
                            l_sum = T.alloc_var(T.float32)
                            l_sum = T.float32(0)
                            for j in T.serial(elems_per_thread):
                                val = T.exp(my_scores[j] - l_max)
                                my_scores[j] = val
                                l_sum = l_sum + val
                            l_sum = T.warp_reduce_sum(l_sum)
                            if scaled:
                                inv_row_sum = T.alloc_var(T.float32)
                                inv_row_sum = T.float32(routed_scaling_factor) / l_sum
                            else:
                                inv_row_sum = T.alloc_var(T.float32)
                                inv_row_sum = T.float32(1) / l_sum
                    else:  # sigmoid, element-wise, no row reduction
                        for j in T.serial(elems_per_thread):
                            my_scores[j] = T.float32(1) / (T.float32(1) + T.exp(-my_scores[j]))
                        if not renormalize:
                            inv_row_sum = T.alloc_var(T.float32)
                            inv_row_sum = T.float32(routed_scaling_factor if scaled else 1.0)

                    if with_bias:
                        for j in T.serial(elems_per_thread):
                            sel[j] = my_scores[j] + bias_ub[j * _WARP_LANES + lane_id]
                    else:
                        for j in T.serial(elems_per_thread):
                            sel[j] = my_scores[j]

                    if grouped:
                        group_score = T.alloc_local([num_expert_group], "float32")
                        if group_size == _WARP_LANES:
                            # A group is one slot, so a lane holds one score per group.
                            for gidx in T.unroll(num_expert_group):
                                gbest1 = T.warp_reduce_max(sel[gidx])
                                if with_bias:
                                    dup = T.warp_reduce_sum(
                                        T.if_then_else(sel[gidx] == gbest1, 1, 0)
                                    )
                                    cand = T.if_then_else(sel[gidx] < gbest1, sel[gidx], _NEG)
                                    second = T.if_then_else(dup >= 2, gbest1, T.warp_reduce_max(cand))
                                    group_score[gidx] = gbest1 + second
                                else:
                                    group_score[gidx] = gbest1
                        else:
                            local_best1 = T.alloc_local([num_expert_group], "float32")
                            local_best2 = T.alloc_local([num_expert_group], "float32")
                            for gidx in T.unroll(num_expert_group):
                                local_best1[gidx] = _NEG
                                local_best2[gidx] = _NEG
                            for j in T.unroll(elems_per_thread):
                                for gidx in T.unroll(num_expert_group):
                                    if group_of(j, lane_id) == gidx:
                                        if with_bias:
                                            local_best2[gidx] = T.max(
                                                local_best2[gidx],
                                                T.min(local_best1[gidx], sel[j]),
                                            )
                                        local_best1[gidx] = T.max(local_best1[gidx], sel[j])
                            for gidx in T.unroll(num_expert_group):
                                gbest1 = T.warp_reduce_max(local_best1[gidx])
                                if with_bias:
                                    dup = T.warp_reduce_sum(
                                        T.if_then_else(local_best1[gidx] == gbest1, 1, 0)
                                    )
                                    cand = T.if_then_else(
                                        local_best1[gidx] < gbest1,
                                        local_best1[gidx],
                                        local_best2[gidx],
                                    )
                                    second = T.if_then_else(dup >= 2, gbest1, T.warp_reduce_max(cand))
                                    group_score[gidx] = gbest1 + second
                                else:
                                    group_score[gidx] = gbest1
                        kept = T.alloc_local([num_expert_group], "int32")
                        for gidx in T.unroll(num_expert_group):
                            kept[gidx] = 0
                        for _ in T.unroll(topk_group):
                            pick = T.alloc_var(T.int32)
                            pick_val = T.alloc_var(T.float32)
                            pick = T.int32(-1)
                            pick_val = _NEG
                            for gidx in T.unroll(num_expert_group):
                                if kept[gidx] == 0 and group_score[gidx] > pick_val:
                                    pick = gidx
                                    pick_val = group_score[gidx]
                            for gidx in T.unroll(num_expert_group):
                                if gidx == pick:
                                    kept[gidx] = 1
                        for j in T.unroll(elems_per_thread):
                            for gidx in T.unroll(num_expert_group):
                                if group_of(j, lane_id) == gidx and kept[gidx] == 0:
                                    sel[j] = _NEG

                    local_max = T.alloc_var(T.float32)
                    local_idx = T.alloc_var(T.int32)

                    # The winner's own score is its biased value minus the bias.
                    sel_vals = T.alloc_local([top_k], "float32")
                    idx_vals = T.alloc_local([top_k], "int32")
                    orig = T.alloc_var(T.float32)
                    if renormalize:
                        sel_sum = T.alloc_var(T.float32)
                        sel_sum = T.float32(0)

                    for k in T.serial(top_k):
                        local_max = _NEG
                        local_idx = T.int32(-1)
                        for j in T.serial(elems_per_thread):
                            if sel[j] > local_max:
                                local_max = sel[j]
                                local_idx = j * _WARP_LANES + lane_id

                        l_best_val = T.warp_reduce_max(local_max)
                        idx_cand = T.if_then_else(local_max == l_best_val, local_idx, experts)
                        l_best_idx = T.warp_reduce_min(idx_cand)

                        if lane_id == 0:
                            i_ub[warp_id, k] = l_best_idx
                        sel_vals[k] = l_best_val
                        idx_vals[k] = l_best_idx

                        for j in T.serial(elems_per_thread):
                            if j * _WARP_LANES + lane_id == l_best_idx:
                                sel[j] = _NEG

                    if lane_id == 0:
                        for k in T.serial(top_k):
                            orig = sel_vals[k]
                            if with_bias:
                                orig = orig - bias_ub[idx_vals[k]]
                            if renormalize:
                                sel_vals[k] = orig
                                sel_sum = sel_sum + orig
                            else:
                                w_ub[warp_id, k] = orig * inv_row_sum

                    if renormalize:
                        inv_sel_sum = T.alloc_var(T.float32)
                        if scaled:
                            inv_sel_sum = T.float32(routed_scaling_factor) / sel_sum
                        else:
                            inv_sel_sum = T.float32(1) / sel_sum
                        if lane_id == 0:
                            for k in T.serial(top_k):
                                w_ub[warp_id, k] = sel_vals[k] * inv_sel_sum

                T.copy(w_ub, weights_gm[token_base : token_base + tokens_per_block, :])
                T.copy(i_ub, ids_gm[token_base : token_base + tokens_per_block, :])

    return main


@functools.lru_cache(maxsize=None)
def _compile(
    tokens: int,
    experts: int,
    top_k: int,
    dtype: str,
    scoring_func: str,
    renormalize: bool,
    with_bias: bool,
    num_expert_group: int,
    topk_group: int,
    routed_scaling_factor: float,
):
    return tilelang.compile(
        make_fused_topk(
            tokens,
            experts,
            top_k,
            dtype,
            scoring_func,
            renormalize,
            with_bias,
            num_expert_group,
            topk_group,
            routed_scaling_factor,
        ),
        target="ascend",
        out_idx=[2, 3],
    )


class AscendFusedTopKKernel(Kernel, FusedTopKFwdInterface):
    """Per-token top-k routing with one 32-lane SIMT group per token."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: FusedTopKCall) -> str | None:
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return f"needs float16, bfloat16 or float32 logits, got {call.dtype}"
        if min(call.num_tokens, call.num_experts, call.top_k) <= 0:
            return "requires positive token, expert and top-k counts"
        if call.top_k > call.num_experts:
            return f"needs top_k <= num_experts, got {call.top_k} > {call.num_experts}"
        if call.with_correction_bias and call.scoring_func != "sigmoid":
            return "a correction bias requires sigmoid scoring"
        if call.num_experts % _WARP_LANES:
            return f"needs num_experts divisible by {_WARP_LANES}, got {call.num_experts}"
        if call.num_expert_group < 1 or call.num_experts % call.num_expert_group:
            return (
                f"needs num_expert_group >= 1 dividing num_experts, got "
                f"{call.num_expert_group} and {call.num_experts}"
            )
        group_size = call.num_experts // call.num_expert_group
        if not 1 <= call.topk_group <= call.num_expert_group:
            return f"needs topk_group in [1, {call.num_expert_group}], got {call.topk_group}"
        if call.top_k > call.topk_group * group_size:
            return (
                f"needs the kept groups to hold top_k, got {call.top_k} > "
                f"{call.topk_group * group_size}"
            )
        if call.num_expert_group > 1 and call.with_correction_bias and group_size < 2:
            return "a biased group score sums two experts; each group needs two"
        return super().refusal(call)

    def __init__(self, call: FusedTopKCall) -> None:
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        # The kernel takes a bias tensor, which it ignores without a correction bias.
        self._zero_bias: Optional[torch.Tensor] = None
        self._kernel = _compile(
            call.num_tokens,
            call.num_experts,
            call.top_k,
            self.dtype_str,
            call.scoring_func,
            call.renormalize,
            call.with_correction_bias,
            call.num_expert_group,
            call.topk_group,
            call.routed_scaling_factor,
        )

    def forward(
        self, gating_output: torch.Tensor, correction_bias: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.call
        self._require_devices(gating_output=gating_output, correction_bias=correction_bias)
        if gating_output.shape != (c.num_tokens, c.num_experts) or gating_output.dtype != c.dtype:
            raise ValueError("gating_output does not match the routing specialization")
        if c.with_correction_bias:
            if correction_bias is None or correction_bias.shape != (c.num_experts,):
                raise ValueError("correction_bias must be one float32 value per expert")
        elif correction_bias is not None:
            raise ValueError("this specialization was built without a correction bias")
        bias = correction_bias
        if bias is None:
            if self._zero_bias is None:
                self._zero_bias = torch.zeros(
                    (c.num_experts,), dtype=torch.float32, device=gating_output.device
                )
            bias = self._zero_bias
        weights, ids = self._kernel(gating_output, bias)
        return weights, ids


register_kernel_type("FusedTopKFwdOp", "ascend", AscendFusedTopKKernel)
