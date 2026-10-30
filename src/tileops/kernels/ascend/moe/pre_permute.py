"""Stable, tight MoE routing for Ascend 950, using TileLang's Ascend dialect."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.call_spec import PrePermuteCall, PrePermuteFwdInterface

__all__ = ["AscendMoEPrePermuteKernel"]


def route_pass_configs():
    """Keep the explicit DMA/vector/scalar dependencies and separate UB buffers."""
    return {
        tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }


def make_route_fused(num_tokens: int, hidden_size: int, top_k: int, num_experts: int, dtype: str):
    """Compute stable route destinations locally, then read each token only once.

    Each block counts all routes, avoiding a global barrier or a second launch.
    Only block zero writes the shared metadata outputs. Other blocks use their
    own UB inverse indices to scatter disjoint tokens' routes.
    """
    num_routes = num_tokens * top_k
    padded_routes = (num_routes + 63) // 64 * 64
    blocks = min(num_tokens, 64)

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, hidden_size), dtype),
        ids: T.Tensor((num_routes,), "int32"),
        y: T.Tensor((num_routes, hidden_size), dtype),
        ends: T.Tensor((num_experts,), "int32"),
        inverse: T.Tensor((num_routes,), "int32"),
    ):
        with T.Kernel(blocks) as bx:
            ids_ub = T.alloc_shared((padded_routes,), "int32")
            inverse_ub = T.alloc_shared((padded_routes,), "int32")
            ends_ub = T.alloc_shared((num_experts,), "int32")
            row_ub = T.alloc_shared((hidden_size,), dtype)
            T.copy(ids, ids_ub[:num_routes])
            T.ascend_set_flag("MTE2_V", 0)
            T.ascend_wait_flag("MTE2_V", 0)
            with T.SimdVF():
                full = T.simd.pset(32)
                offset = T.simd.alloc_var("int32")
                offset = T.simd.vdup(0, "int32", full)
                for expert in T.serial(num_experts):
                    for chunk in T.serial(padded_routes // 64):
                        valid = T.simd.update_mask(T.min(64, num_routes - chunk * 64), 32)
                        values = T.simd.vld(ids_ub[chunk * 64])
                        selected = T.simd.vcmps(values, expert, valid, "eq")
                        prefix = T.simd.vusqz(selected, "int32")
                        rank = T.simd.vadd(prefix, offset, full)
                        T.simd.vsts(inverse_ub[chunk * 64], rank, selected)
                        flags = T.simd.vdup(1, "int32", selected)
                        count = T.simd.vcadd(flags, full)
                        offset = T.simd.vadd(offset, T.simd.vdupv(count, full), full)
                    T.simd.vsts(ends_ub[expert], offset, full, dist="ONEPT_B32")
            # Scalar destination loads must observe the vector prefix results.
            T.ascend_set_flag("V_S", 0)
            T.ascend_wait_flag("V_S", 0)
            if bx == 0:
                T.ascend_set_flag("V_MTE3", 0)
                T.ascend_wait_flag("V_MTE3", 0)
                T.copy(ends_ub, ends)
                T.copy(inverse_ub[:num_routes], inverse)
            for step in T.serial(T.ceildiv(num_tokens, blocks)):
                token = step * blocks + bx
                if token < num_tokens:
                    # Finish all K writes before reusing the row buffer.
                    T.ascend_set_flag("MTE3_MTE2", 0)
                    T.ascend_wait_flag("MTE3_MTE2", 0)
                    T.copy(x[token, :], row_ub)
                    T.ascend_set_flag("MTE2_MTE3", 0)
                    T.ascend_wait_flag("MTE2_MTE3", 0)
                    for route in T.serial(top_k):
                        dest = inverse_ub[token * top_k + route]
                        T.copy(row_ub, y[dest, :])
            T.ascend_set_flag("MTE3_S", 0)
            T.ascend_wait_flag("MTE3_S", 0)

    return main


@functools.lru_cache(maxsize=None)
def _compile(num_tokens: int, hidden_size: int, top_k: int, num_experts: int, dtype: str):
    return tilelang.compile(
        make_route_fused(num_tokens, hidden_size, top_k, num_experts, dtype),
        target="ascend",
        out_idx=[],
        pass_configs=route_pass_configs(),
    )


class AscendMoEPrePermuteKernel(Kernel, PrePermuteFwdInterface):
    """Tight physical-psum routing; equal expert IDs retain input route order."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: PrePermuteCall) -> str | None:
        if getattr(call.layout, "selection_key", None) != "tight_physical_psum":
            return "requires tight_physical_psum layout"
        if call.input_dtype not in (torch.float16, torch.bfloat16):
            return "requires float16 or bfloat16"
        if min(call.num_tokens, call.top_k, call.num_experts, call.hidden_size) <= 0:
            return "requires positive token, route, expert and hidden dimensions"
        if call.num_tokens * call.top_k > 1024 or call.num_experts > 1024:
            return "supports at most 1024 routes and 1024 experts"
        if call.hidden_size % 16 or call.hidden_size > 16384:
            return "requires hidden_size divisible by 16 and at most 16384"
        return super().refusal(call)

    def __init__(self, call: PrePermuteCall) -> None:
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.input_dtype
        self._route = _compile(
            call.num_tokens, call.hidden_size, call.top_k, call.num_experts, self.dtype_str
        )

    def forward(self, hidden_states: torch.Tensor, local_expert_ids: torch.Tensor):
        self._require_devices(hidden_states=hidden_states, local_expert_ids=local_expert_ids)
        c = self.call
        if (
            hidden_states.shape != (c.num_tokens, c.hidden_size)
            or hidden_states.dtype != c.input_dtype
        ):
            raise ValueError("hidden_states does not match the pre-permute specialization")
        if (
            local_expert_ids.shape != (c.num_tokens, c.top_k)
            or local_expert_ids.dtype != torch.int32
        ):
            raise ValueError("local_expert_ids must match the specialization and have dtype int32")
        if not hidden_states.is_contiguous() or not local_expert_ids.is_contiguous():
            raise ValueError("pre-permute inputs must be contiguous")
        device = hidden_states.device
        routes = c.num_tokens * c.top_k
        output = torch.empty((routes, c.hidden_size), dtype=self.dtype, device=device)
        ends = torch.empty((c.num_experts,), dtype=torch.int32, device=device)
        inverse = torch.empty((routes,), dtype=torch.int32, device=device)
        self._route(hidden_states, local_expert_ids.view(-1), output, ends, inverse)
        return output, ends, inverse


register_kernel_type("MoEPrePermuteFwdOp", "ascend", AscendMoEPrePermuteKernel)
