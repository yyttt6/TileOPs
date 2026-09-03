"""MoE routing family (archetype 10)."""

from __future__ import annotations

import torch

from .._registry import register
from ..kernels.moe import (
    build_moe_permute_align_kernel,
    build_moe_permute_kernel,
    build_moe_unpermute_kernel,
)
from ..kernels.gemm import build_gemm_kernel
from ..kernels.elementwise_activation import build_activation_kernel


@register("MoePermuteNopadFwdOp")
def build_moe_permute(hidden_states, topk_ids, expert_map=None, *, num_experts, num_experts_local):
    if hidden_states is None or topk_ids is None:
        raise ValueError("MoePermuteNopadFwdOp requires hidden_states and topk_ids")
    if hidden_states.device != topk_ids.device:
        raise ValueError("hidden_states and topk_ids must share a device")
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("hidden_states must be float16 or bfloat16")
    if topk_ids.dtype != torch.int32:
        raise TypeError("topk_ids must be int32")
    if expert_map is not None:
        raise NotImplementedError("expert_map EP routing is reserved for the follow-up")
    return build_moe_permute_kernel(
        tuple(hidden_states.shape),
        tuple(topk_ids.shape),
        hidden_states.dtype,
        int(num_experts),
        int(num_experts_local),
        False,
    )


@register("MoePermuteAlignFwdOp")
def build_moe_permute_align(topk_ids, *, total_tokens, top_k, num_experts, block_size=64):
    if topk_ids is None:
        raise ValueError("MoePermuteAlignFwdOp requires topk_ids")
    if topk_ids.dtype != torch.int32:
        raise TypeError("MoePermuteAlignFwdOp requires int32 topk_ids")
    return build_moe_permute_align_kernel(total_tokens, top_k, num_experts, block_size)


@register("MoeUnpermuteFwdOp")
def build_moe_unpermute(mm2_pad, fwd_idx, topk_weights, *, total_tokens, top_k, hidden_size, out=None):
    if mm2_pad is None or fwd_idx is None or topk_weights is None:
        raise ValueError("MoeUnpermuteFwdOp requires mm2_pad, fwd_idx, and topk_weights")
    if mm2_pad.device != fwd_idx.device or mm2_pad.device != topk_weights.device:
        raise ValueError("MoeUnpermuteFwdOp inputs must share a device")
    if mm2_pad.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("MoeUnpermuteFwdOp mm2_pad must use float16 or bfloat16")
    if fwd_idx.dtype != torch.int32 or topk_weights.dtype != torch.float32:
        raise TypeError("MoeUnpermuteFwdOp requires int32 indices and float32 weights")
    routed_pairs = int(total_tokens) * int(top_k)
    if len(mm2_pad.shape) != 2 or int(mm2_pad.shape[0]) < routed_pairs or int(mm2_pad.shape[1]) != int(hidden_size):
        raise ValueError(
            "MoeUnpermuteFwdOp mm2_pad must be [padded_batch_sum, hidden_size] "
            "with padded_batch_sum >= total_tokens * top_k"
        )
    if tuple(fwd_idx.shape) != (routed_pairs,):
        raise ValueError("MoeUnpermuteFwdOp fwd_idx shape mismatch")
    if tuple(topk_weights.shape) != (int(total_tokens), int(top_k)):
        raise ValueError("MoeUnpermuteFwdOp topk_weights shape mismatch")
    return build_moe_unpermute_kernel(
        tuple(mm2_pad.shape),
        tuple(fwd_idx.shape),
        tuple(topk_weights.shape),
        mm2_pad.dtype,
    )


def _unsupported(op_name):
    def builder(*args, **kwargs):
        raise NotImplementedError(
            f"{op_name} is blocked in T069: Cube/grouped-GEMM or fused routing template is not proven on 910B1"
        )

    builder.__name__ = f"build_{op_name}"
    return builder


def _build_moe_grouped_route3(a, b, true_sizes, true_offsets, *, output_n, activation=None):
    """Build the correct route-3 MoE grouped path.

    Runtime metadata is read only for scheduling.  Each non-empty expert gets
    one dense GEMM launch; no host-side matmul is used.  ``activation`` adds a
    proven TileLang activation launch to each expert result.
    """
    if a.ndim != 2 or b.ndim != 3:
        raise ValueError("MoE grouped operands must be [numel,K] and [E,N,K]")
    if a.shape[1] != b.shape[2] or b.shape[1] != output_n:
        raise ValueError("MoE grouped operand shapes do not match requested GEMM")
    if true_sizes.ndim != 1 or true_offsets.shape != true_sizes.shape:
        raise ValueError("MoE metadata must be matching 1D tensors")
    if true_sizes.dtype != torch.int32 or true_offsets.dtype != torch.int32:
        raise TypeError("MoE metadata tensors must use int32")
    if a.device != b.device or a.device != true_sizes.device or a.device != true_offsets.device:
        raise ValueError("MoE grouped inputs must share a device")
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("MoE grouped operands must share float16/bfloat16 dtype")

    numel, k = map(int, a.shape)
    experts = int(b.shape[0])
    if int(true_sizes.numel()) != experts:
        raise ValueError("MoE metadata length must equal expert count")
    dtype = a.dtype
    activation_kind = activation

    def invoke(a_in, b_in, sizes_in, offsets_in):
        sizes = sizes_in.detach().cpu().tolist()
        offsets = offsets_in.detach().cpu().tolist()
        output_width = output_n // 2 if activation_kind else output_n
        output = torch.empty((numel, output_width), dtype=dtype, device=a_in.device)
        for expert, (offset, size) in enumerate(zip(offsets, sizes)):
            offset, size = int(offset), int(size)
            if offset < 0 or size < 0 or offset + size > numel:
                raise ValueError("MoE metadata has an out-of-range expert segment")
            if size == 0:
                continue
            a_group = a_in[offset : offset + size, :]
            b_group = b_in[expert]
            gemm = build_gemm_kernel(
                tuple(a_group.shape), tuple(b_group.shape), dtype, False, True
            )
            gemm_out = gemm(a_group, b_group)
            if activation_kind:
                act = build_activation_kernel(
                    (tuple(gemm_out.shape),),
                    dtype,
                    op_kind=activation_kind,
                    output_shape=(size, output_width),
                    op_name="MoeGateUpFwdOp",
                )
                gemm_out = act(gemm_out)
            output[offset : offset + size].copy_(gemm_out)
        return output

    return invoke


@register("MoeGroupedGemmNopadFwdOp")
def build_moe_grouped_gemm(a, b, true_sizes, true_offsets, *, numel, num_experts, n, k):
    if a is None or b is None or true_sizes is None or true_offsets is None:
        raise ValueError("MoeGroupedGemmNopadFwdOp requires four tensors")
    if tuple(a.shape) != (int(numel), int(k)) or tuple(b.shape) != (int(num_experts), int(n), int(k)):
        raise ValueError("MoeGroupedGemmNopadFwdOp shape mismatch")
    return _build_moe_grouped_route3(
        a, b, true_sizes, true_offsets, output_n=int(n)
    )


@register("MoeGateUpFwdOp")
def build_moe_gate_up(a, b, true_sizes, true_offsets, *, numel, num_experts, ffn, k, activation="silu_and_mul"):
    if activation not in {"silu_and_mul", "gelu_and_mul"}:
        raise ValueError(f"unsupported MoE gate/up activation: {activation}")
    if a is None or b is None:
        raise ValueError("MoeGateUpFwdOp requires four tensors")
    if tuple(a.shape) != (int(numel), int(k)) or tuple(b.shape) != (int(num_experts), 2 * int(ffn), int(k)):
        raise ValueError("MoeGateUpFwdOp shape mismatch")
    return _build_moe_grouped_route3(
        a,
        b,
        true_sizes,
        true_offsets,
        output_n=2 * int(ffn),
        activation=activation,
    )


@register("FusedMoEExpertsNopadPersistent3WGFwdOp")
def build_fused_moe_experts(
    output, hidden_states, w_gate_up, w_down, topk_weights, topk_ids, workspace1, workspace2,
    expert_map=None, *, num_tokens, num_experts, num_experts_local, top_k, hidden_size, ffn_size,
    routed_scaling_factor=1.0, activation="silu_and_mul",
):
    return _unsupported("FusedMoEExpertsNopadPersistent3WGFwdOp")(
        output, hidden_states, w_gate_up, w_down, topk_weights, topk_ids, workspace1, workspace2,
        expert_map, num_tokens=num_tokens, num_experts=num_experts, num_experts_local=num_experts_local,
        top_k=top_k, hidden_size=hidden_size, ffn_size=ffn_size,
        routed_scaling_factor=routed_scaling_factor, activation=activation,
    )


@register("FusedMoeFwdOp")
def build_fused_moe(
    hidden_states, gating_output, w_gate_up, w_down, correction_bias=None, *, num_tokens, num_experts,
    top_k, hidden_size, ffn_size, scoring_func="softmax", renormalize=False,
    routed_scaling_factor=1.0, activation="silu_and_mul",
):
    return _unsupported("FusedMoeFwdOp")(
        hidden_states, gating_output, w_gate_up, w_down, correction_bias,
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k, hidden_size=hidden_size,
        ffn_size=ffn_size, scoring_func=scoring_func, renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor, activation=activation,
    )


__all__ = [
    "build_moe_permute",
    "build_moe_permute_align",
    "build_moe_unpermute",
    "build_moe_grouped_gemm",
    "build_moe_gate_up",
    "build_fused_moe_experts",
    "build_fused_moe",
]
