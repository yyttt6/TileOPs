"""SIMD weighted inverse-permute with a two-stage token pipeline.

Each core owns a contiguous token block.  For every hidden span, the K expert
rows of the next token are copied into the alternate UB stage while the
current token's K reduction runs in SimdVF.  Routing metadata is staged once
per block; the inner loop is only MTE copies, vector arithmetic and one output
copy.  The pipeline owns all MTE/V/S dependencies explicitly.
"""

import functools

import torch

from tileops._kernels.moe.ascend.optimized.common import (
    AscendMoEKernel,
    check_dtype,
    check_out,
    check_scale,
    check_tensor,
)

__all__ = ["AscendMoEPostPermuteKernel"]

VREG_LANES = 128
PIPELINE_STAGES = 2

# T.copy only issues the transfer under these configs; the flags below publish
# each transfer to its consumer.
PASS_CONFIGS = {
    "tl.enable_auto_schedule": False,
    "tl.disable_shared_memory_reuse": True,
    "tl.disable_thread_storage_sync": True,
}


@functools.lru_cache(maxsize=32)
def _make_simd_unpermute(
    num_tokens,
    top_k,
    hidden_size,
    materialized_rows,
    dtype,
    scaling=1.0,
    *,
    out_dtype=None,
    num_cores=72,
):
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    out_dtype = dtype if out_dtype is None else out_dtype
    cores = min(num_tokens, num_cores)
    base = num_tokens // cores
    rem = num_tokens % cores
    max_tokens = base + (1 if rem else 0)

    h_pad = -(-hidden_size // VREG_LANES) * VREG_LANES
    # Two source stages plus two output stages share the UB budget.
    budget = 168 * 1024
    unit = PIPELINE_STAGES * (top_k + 1) * 2
    h_span = min(h_pad, max(VREG_LANES, budget // unit // VREG_LANES * VREG_LANES))
    n_spans = -(-h_pad // h_span)
    h_span // VREG_LANES
    meta_len = max_tokens * top_k

    @T.prim_func
    def _simd_unpermute_main(
        expert_output: T.Tensor((materialized_rows, hidden_size), dtype),
        inverse_indices: T.Tensor((num_tokens * top_k,), "int32"),
        topk_weights: T.Tensor((num_tokens * top_k,), "float32"),
        output: T.Tensor((num_tokens, hidden_size), out_dtype),
    ):
        with T.Kernel(cores) as core:
            inv_ub = T.alloc_shared((meta_len,), "int32")
            # vld broadcasts one fp32 element; the tail pad keeps the register
            # load inside the allocation for the last route.
            weight_ub = T.alloc_shared((meta_len + VREG_LANES,), "float32")
            src = T.alloc_shared((PIPELINE_STAGES, top_k, h_span), dtype)
            out_ub = T.alloc_shared((PIPELINE_STAGES, h_span), out_dtype)

            start = core * base + T.min(core, rem)
            count = base + T.min(T.max(rem - core, 0), 1)
            meta_begin = start * top_k
            meta_count = count * top_k
            T.copy(inverse_indices[meta_begin : meta_begin + meta_count], inv_ub[0:meta_count])
            T.copy(topk_weights[meta_begin : meta_begin + meta_count], weight_ub[0:meta_count])
            T.ascend_set_flag("MTE2_S", 0)
            T.ascend_wait_flag("MTE2_S", 0)

            for span in T.serial(n_spans):
                s0 = span * h_span
                span_valid = T.min(hidden_size - s0, h_span)

                # Prologue: stage token 0 into buffer 0.
                for k in T.serial(top_k):
                    slot = T.cast(inv_ub[k], "int64")
                    T.copy(expert_output[slot, s0 : s0 + span_valid], src[0, k, 0:span_valid])
                T.ascend_set_flag("MTE2_V", 0)

                for local in T.serial(count):
                    stage = local % PIPELINE_STAGES
                    next_stage = 1 - stage
                    token_idx = T.cast(start + local, "int64")

                    # Stage the next token before consuming the current one.
                    # Its buffer was released by the store two iterations ago.
                    if local + 1 < count:
                        if local > 0:
                            T.ascend_wait_flag("MTE3_MTE2", next_stage)
                        for k in T.serial(top_k):
                            slot = T.cast(inv_ub[(local + 1) * top_k + k], "int64")
                            T.copy(
                                expert_output[slot, s0 : s0 + span_valid],
                                src[next_stage, k, 0:span_valid],
                            )
                        T.ascend_set_flag("MTE2_V", next_stage)

                    T.ascend_wait_flag("MTE2_V", stage)
                    with T.SimdVF():
                        m32 = S.pset(32)
                        m16 = S.pset(16)
                        acc_e = S.alloc_local((1,), "float32")
                        acc_o = S.alloc_local((1,), "float32")
                        for c in T.serial(T.ceildiv(span_valid, VREG_LANES)):
                            base_col = c * VREG_LANES
                            acc_e[0] = S.vdup(T.float32(0.0), "float32", m32)
                            acc_o[0] = S.vdup(T.float32(0.0), "float32", m32)
                            for k in T.serial(top_k):
                                s_vec = S.vld(src[stage, k, base_col])
                                w_vec = S.vld(weight_ub[local * top_k + k], dist="BRC_B32")
                                S.vmula(acc_e[0], S.vcvt(s_vec, "float32", part=0), w_vec, m32)
                                S.vmula(acc_o[0], S.vcvt(s_vec, "float32", part=1), w_vec, m32)
                            if scaling != 1.0:
                                acc_e[0] = S.vmuls(acc_e[0], T.float32(scaling), m32)
                                acc_o[0] = S.vmuls(acc_o[0], T.float32(scaling), m32)
                            even_bits = T.reinterpret(
                                S.vcvt(acc_e[0], out_dtype, part=0), "uint16x128"
                            )
                            odd_bits = T.reinterpret(
                                S.vcvt(acc_o[0], out_dtype, part=1), "uint16x128"
                            )
                            merged = T.reinterpret(
                                S.vor(even_bits, odd_bits, m16), out_dtype + "x128"
                            )
                            valid = T.min(hidden_size - s0 - base_col, VREG_LANES)
                            S.vsts(out_ub[stage, base_col], merged, S.update_mask(valid, 16))
                    T.ascend_set_flag("V_MTE3", stage)
                    T.ascend_wait_flag("V_MTE3", stage)
                    T.copy(out_ub[stage, 0:span_valid], output[token_idx, s0 : s0 + span_valid])
                    T.ascend_set_flag("MTE3_MTE2", stage)

                # Drain this span's last stores before the next span reuses UB.
                if count > 1:
                    T.ascend_wait_flag("MTE3_MTE2", 0)
                    T.ascend_wait_flag("MTE3_MTE2", 1)
                else:
                    T.ascend_wait_flag("MTE3_MTE2", 0)

    return _simd_unpermute_main


class AscendMoEPostPermuteKernel(AscendMoEKernel):
    """SIMD weighted inverse-permute with a two-stage token pipeline."""

    def __init__(
        self,
        num_tokens,
        top_k,
        hidden_size,
        dtype=torch.bfloat16,
        *,
        routed_scaling_factor=1.0,
        out_dtype=None,
        materialized_rows=None,
        num_cores=72,
    ):
        super().__init__(num_cores)
        check_dtype(dtype)
        check_scale(routed_scaling_factor)
        if min(num_tokens, top_k, hidden_size) <= 0:
            raise ValueError("positive dimensions are required")
        if out_dtype not in (None, torch.float16, torch.bfloat16):
            raise ValueError("post-permute out_dtype must be float16 or bfloat16")
        self.num_tokens, self.top_k, self.hidden_size = num_tokens, top_k, hidden_size
        self.materialized_rows = (
            num_tokens * top_k if materialized_rows is None else materialized_rows
        )
        if self.materialized_rows <= 0:
            raise ValueError("materialized_rows must be positive")
        self.dtype, self.out_dtype = dtype, dtype if out_dtype is None else out_dtype
        self.routed_scaling_factor = routed_scaling_factor

    def compile(self, device=None):
        if device is None or device.type not in ("npu", "privateuseone"):
            raise ValueError("an explicit NPU device is required")
        if device not in self._compiled:
            import tilelang

            with torch.npu.device(device):
                self._compiled[device] = tilelang.compile(
                    _make_simd_unpermute(
                        self.num_tokens,
                        self.top_k,
                        self.hidden_size,
                        self.materialized_rows,
                        str(self.dtype).removeprefix("torch."),
                        scaling=self.routed_scaling_factor,
                        out_dtype=str(self.out_dtype).removeprefix("torch."),
                        num_cores=self.config["num_cores"],
                    ),
                    target="ascend",
                    out_idx=None,
                    pass_configs=PASS_CONFIGS,
                )
        return self._compiled[device]

    def __call__(self, expert_output, topk_weights, inverse_indices, *, out=None):
        rows = self.num_tokens * self.top_k
        check_tensor(
            expert_output, "expert_output", (self.materialized_rows, self.hidden_size), self.dtype
        )
        check_tensor(inverse_indices, "inverse_indices", (rows,), torch.int32, expert_output.device)
        check_tensor(
            topk_weights,
            "topk_weights",
            (self.num_tokens, self.top_k),
            torch.float32,
            expert_output.device,
        )
        shape = (self.num_tokens, self.hidden_size)
        check_out(
            out,
            shape,
            self.out_dtype,
            expert_output.device,
            (expert_output, inverse_indices, topk_weights),
        )
        program = self.compile(expert_output.device)
        output = (
            torch.empty(shape, dtype=self.out_dtype, device=expert_output.device)
            if out is None
            else out
        )
        program(
            expert_output,
            inverse_indices,
            topk_weights.reshape(-1),
            output,
        )
        return output
