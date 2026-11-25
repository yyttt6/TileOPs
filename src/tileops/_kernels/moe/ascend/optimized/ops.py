"""Host-level Ascend MoE entry points for the six delivered boundaries.

The device work is supplied by the kernels in this package.  These classes keep
the public operator names, constructor parameters and ``forward`` signatures of
``tileops._ops.moe`` so a caller can switch to the Ascend host path without
rewriting the model code.  Each class specializes its kernels on first use and
then calls them directly; it deliberately does not go through ``Op._execute``,
``kernel_for`` or a per-call ``CallSpec`` build.

The supported layout is ``tight_physical_psum``.  No operator fusion is done
here: 124 is pre -> expert MLP -> post, 129 is two grouped GEMMs, and 125 is
TopK -> experts, exactly as their public definitions.
"""

from __future__ import annotations

from typing import Optional

from tileops._kernels.moe.ascend.optimized.common import check_scale
from tileops._kernels.moe.ascend.optimized.grouped_gemm_optimized import (
    AscendMoEGroupedGEMMOptimizedKernel,
)
from tileops._kernels.moe.ascend.optimized.post_permute_optimized import (
    AscendMoEPostPermuteKernel,
)
from tileops._kernels.moe.ascend.optimized.pre_permute_optimized import (
    AscendMoEPrePermuteKernel,
)
from tileops._kernels.moe.call_spec import FusedTopKCall
from tileops._ops.moe.contracts import ContiguousLayoutSpec, RoutingEpilogueSpec

__all__ = [
    "FusedMoEExpertsFwdOp",
    "FusedMoEFwdOp",
    "FusedTopKFwdOp",
    "MoEExpertMLPFwdOp",
    "MoEPostPermuteFwdOp",
    "MoEPrePermuteFwdOp",
]

_TIGHT = ContiguousLayoutSpec.tight_physical_psum()


def _tight(layout) -> None:
    if getattr(layout, "selection_key", None) != "tight_physical_psum":
        raise NotImplementedError("the Ascend host path serves tight_physical_psum only")


def _cached(cache, key, build):
    value = cache.get(key)
    if value is None:
        value = build()
        cache[key] = value
    return value


class FusedTopKFwdOp:
    """MoE top-k routing: ``gating_output[T,E] -> (weights[T,K], ids[T,K])``."""

    def __init__(
        self,
        top_k: int,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        num_expert_group: int = 1,
        topk_group: int = 1,
        routed_scaling_factor: float = 1.0,
        *,
        target=None,
    ) -> None:
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.target = target
        self._kernels = {}

    def __call__(self, gating_output, correction_bias=None):
        return self.forward(gating_output, correction_bias)

    def forward(self, gating_output, correction_bias=None):
        """Return ``(topk_weights, topk_ids)`` for one ``[T,E]`` logits tensor."""
        if not gating_output.is_contiguous():
            raise ValueError("gating_output must be contiguous")
        tokens, experts = gating_output.shape
        has_bias = correction_bias is not None
        key = (gating_output.device, gating_output.dtype, tokens, experts, has_bias)

        def build():
            from tileops._kernels.ascend.moe.fused_topk import AscendFusedTopKKernel

            return AscendFusedTopKKernel(
                FusedTopKCall(
                    num_tokens=tokens,
                    num_experts=experts,
                    top_k=self.top_k,
                    scoring_func=self.scoring_func,
                    renormalize=self.renormalize,
                    with_correction_bias=has_bias,
                    num_expert_group=self.num_expert_group,
                    topk_group=self.topk_group,
                    routed_scaling_factor=self.routed_scaling_factor,
                    dtype=gating_output.dtype,
                    device=gating_output.device,
                )
            )

        return _cached(self._kernels, key, build)(gating_output, correction_bias)


class MoEPrePermuteFwdOp:
    """Tight expert packing: ``hidden[T,H], ids[T,K] -> packed[P,H], ends[E], inverse[P]``."""

    def __init__(self, layout, num_local_experts: int, *, target=None) -> None:
        _tight(layout)
        self.layout = layout
        self.num_local_experts = num_local_experts
        self.target = target
        self._kernels = {}

    def __call__(self, hidden_states, local_expert_ids):
        return self.forward(hidden_states, local_expert_ids)

    def forward(self, hidden_states, local_expert_ids):
        tokens, hidden = hidden_states.shape
        top_k = local_expert_ids.shape[1]
        key = (
            hidden_states.device,
            hidden_states.dtype,
            tokens,
            hidden,
            top_k,
            self.num_local_experts,
        )
        kernel = _cached(
            self._kernels,
            key,
            lambda: AscendMoEPrePermuteKernel(
                tokens,
                hidden,
                top_k,
                self.num_local_experts,
                hidden_states.dtype,
            ),
        )
        return kernel(hidden_states, local_expert_ids)


class MoEGroupedGEMMFwdOp:
    """Internal grouped GEMM leaf used by 129 and 124; not one of the six boundaries."""

    def __init__(
        self,
        layout,
        *,
        activation: Optional[str] = None,
        out_dtype=None,
        target=None,
        num_cores: int = 36,
    ) -> None:
        _tight(layout)
        self.layout = layout
        self.activation = activation
        self.out_dtype = out_dtype
        self.target = target
        self.num_cores = num_cores
        self._kernels = {}

    def __call__(self, a, b, layout_metadata, *, out=None, bias=None):
        return self.forward(a, b, layout_metadata, out=out, bias=bias)

    def forward(self, a, b, layout_metadata, *, out=None, bias=None):
        experts, n, k = b.shape
        m = a.numel() // k
        key = (
            a.device,
            a.dtype,
            m,
            experts,
            n,
            k,
            self.activation,
            self.out_dtype,
            bias is not None,
            self.num_cores,
        )

        def build():
            return AscendMoEGroupedGEMMOptimizedKernel(
                m,
                experts,
                n,
                k,
                a.dtype,
                activation=self.activation,
                cd_dtype=self.out_dtype,
                has_bias=bias is not None,
                num_cores=self.num_cores,
            )

        return _cached(self._kernels, key, build)(a, b, layout_metadata, bias=bias, out=out)


class MoEExpertMLPFwdOp:
    """Two grouped GEMMs per expert, with the gated activation fused in the first."""

    def __init__(
        self,
        layout,
        activation: str = "silu_and_mul",
        *,
        target=None,
        num_cores: int = 36,
    ) -> None:
        _tight(layout)
        self.layout = layout
        self.activation = activation
        self.target = target
        self.num_cores = num_cores
        self._pipelines = {}

    def __call__(
        self,
        expert_input,
        w_gate_up,
        w_down,
        layout_metadata,
        *,
        out=None,
        b_gate_up=None,
        b_down=None,
    ):
        return self.forward(
            expert_input,
            w_gate_up,
            w_down,
            layout_metadata,
            out=out,
            b_gate_up=b_gate_up,
            b_down=b_down,
        )

    def forward(
        self,
        expert_input,
        w_gate_up,
        w_down,
        layout_metadata,
        *,
        out=None,
        b_gate_up=None,
        b_down=None,
    ):
        experts, twice_ffn, hidden = w_gate_up.shape
        ffn = twice_ffn // 2
        m = expert_input.numel() // hidden
        key = (
            expert_input.device,
            expert_input.dtype,
            m,
            experts,
            ffn,
            hidden,
            b_gate_up is not None,
            b_down is not None,
            self.num_cores,
        )

        def build():
            gate_up = MoEGroupedGEMMFwdOp(
                self.layout, activation=self.activation, num_cores=self.num_cores
            )
            down = MoEGroupedGEMMFwdOp(self.layout, num_cores=self.num_cores)
            return gate_up, down

        gate_up, down = _cached(self._pipelines, key, build)
        activated = gate_up(expert_input, w_gate_up, layout_metadata, bias=b_gate_up)
        return down(activated, w_down, layout_metadata, bias=b_down, out=out)


class MoEPostPermuteFwdOp:
    """Weighted inverse permutation: ``Z[P,H], weights[T,K], inverse[T*K] -> Y[T,H]``."""

    def __init__(
        self,
        layout,
        epilogue=None,
        out_dtype=None,
        *,
        target=None,
    ) -> None:
        _tight(layout)
        self.layout = layout
        self.epilogue = epilogue
        self.out_dtype = out_dtype
        self.target = target
        self._kernels = {}
        self.scale = 1.0 if epilogue is None else epilogue.routed_scaling_factor
        check_scale(self.scale)

    def __call__(self, expert_output, topk_weights, inverse_indices, *, out=None):
        return self.forward(expert_output, topk_weights, inverse_indices, out=out)

    def forward(self, expert_output, topk_weights, inverse_indices, *, out=None):
        tokens, top_k = topk_weights.shape
        rows, hidden = expert_output.shape
        key = (
            expert_output.device,
            expert_output.dtype,
            self.out_dtype,
            tokens,
            top_k,
            hidden,
            rows,
            self.scale,
        )
        kernel = _cached(
            self._kernels,
            key,
            lambda: AscendMoEPostPermuteKernel(
                tokens,
                top_k,
                hidden,
                expert_output.dtype,
                routed_scaling_factor=self.scale,
                out_dtype=self.out_dtype,
                materialized_rows=rows,
            ),
        )
        return kernel(expert_output, topk_weights, inverse_indices, out=out)


class FusedMoEExpertsFwdOp:
    """Expert MLP with routing: 134 -> 129 -> 133, no operator fusion."""

    def __init__(
        self,
        routed_scaling_factor: float = 1.0,
        *,
        activation: str = "silu_and_mul",
        target=None,
        num_cores: int = 36,
    ) -> None:
        check_scale(routed_scaling_factor)
        self.routed_scaling_factor = routed_scaling_factor
        self.activation = activation
        self.target = target
        self.num_cores = num_cores
        self._pipelines = {}

    def __call__(
        self,
        hidden_states,
        w_gate_up,
        w_down,
        topk_weights,
        topk_ids,
        b_gate_up=None,
        b_down=None,
        *,
        out=None,
    ):
        return self.forward(
            hidden_states,
            w_gate_up,
            w_down,
            topk_weights,
            topk_ids,
            b_gate_up,
            b_down,
            out=out,
        )

    def forward(
        self,
        hidden_states,
        w_gate_up,
        w_down,
        topk_weights,
        topk_ids,
        b_gate_up=None,
        b_down=None,
        *,
        out=None,
    ):
        experts = w_gate_up.shape[0]
        key = (hidden_states.device, hidden_states.dtype, experts, self.num_cores)

        def build():
            pre = MoEPrePermuteFwdOp(_TIGHT, experts)
            mlp = MoEExpertMLPFwdOp(_TIGHT, self.activation, num_cores=self.num_cores)
            post = MoEPostPermuteFwdOp(
                _TIGHT,
                RoutingEpilogueSpec(routed_scaling_factor=self.routed_scaling_factor),
            )
            return pre, mlp, post

        pre, mlp, post = _cached(self._pipelines, key, build)
        expert_input, physical_ends, inverse_indices = pre(hidden_states, topk_ids)
        expert_output = mlp(
            expert_input,
            w_gate_up,
            w_down,
            physical_ends,
            b_gate_up=b_gate_up,
            b_down=b_down,
        )
        return post(expert_output, topk_weights, inverse_indices, out=out)


class FusedMoEFwdOp:
    """Routed MoE: 127 -> 124.  The TopK weight keeps scaling 1; 124 applies r once."""

    def __init__(
        self,
        top_k: int,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        routed_scaling_factor: float = 1.0,
        num_expert_group: int = 1,
        topk_group: int = 1,
        *,
        activation: str = "silu_and_mul",
        target=None,
        experts: Optional[FusedMoEExpertsFwdOp] = None,
        num_cores: int = 36,
    ) -> None:
        if (
            experts is not None
            and activation != "silu_and_mul"
            and activation != experts.activation
        ):
            raise ValueError("activation conflicts with the injected experts instance")
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.activation = experts.activation if experts is not None else activation
        self.target = target
        self.num_cores = experts.num_cores if experts is not None else num_cores
        self._topk = FusedTopKFwdOp(
            top_k,
            scoring_func,
            renormalize,
            num_expert_group,
            topk_group,
            routed_scaling_factor=1.0,
        )
        self._experts = (
            experts
            if experts is not None
            else FusedMoEExpertsFwdOp(
                routed_scaling_factor,
                activation=self.activation,
                num_cores=self.num_cores,
            )
        )

    def __call__(
        self,
        hidden_states,
        gating_output,
        w_gate_up,
        w_down,
        correction_bias=None,
        b_gate_up=None,
        b_down=None,
    ):
        return self.forward(
            hidden_states,
            gating_output,
            w_gate_up,
            w_down,
            correction_bias,
            b_gate_up,
            b_down,
        )

    def forward(
        self,
        hidden_states,
        gating_output,
        w_gate_up,
        w_down,
        correction_bias=None,
        b_gate_up=None,
        b_down=None,
    ):
        topk_weights, topk_ids = self._topk(gating_output, correction_bias)
        return self._experts(
            hidden_states,
            w_gate_up,
            w_down,
            topk_weights,
            topk_ids,
            b_gate_up,
            b_down,
        )
