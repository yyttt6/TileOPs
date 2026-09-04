"""The gate/up stage of the MoE expert pipeline, activation included."""

from typing import ClassVar, Dict, Tuple

import torch

from tileops.perf.profile import cube_roof

from ...compile_boundary import get_instance
from ...op_base import Op
from ._common import GroupedOperandEagerForward
from tileops.backend import Kernel

__all__ = ["MoeGateUpFwdOp"]

#: The implementations of this role; each states its own region.
#: The slot this op asks its target for.
_GATE_UP_SLOT = "moe_gate_up"

#: The grouped GEMM the separate-activation implementation composes with.
_GEMM_KEYS = ("moe_grouped_gemm_kernel", "moe_grouped_gemm_persistent_kernel")


class MoeGateUpFwdOp(GroupedOperandEagerForward, Op):
    """Gate/up GEMM and its gated activation.

    Takes the tight permuted rows and the stacked gate||up weights, and returns
    activated rows of width ``ffn``.

    Example:
        ```python linenums="1"
        op = MoeGateUpFwdOp(numel=4096, num_experts=128, ffn=2048, k=7168)
        act = op(a, b, true_sizes, true_offsets)  # [4096, 2048]
        ```
    """

    #: The operator this op registers; a test asserts the graph holds nothing else.
    compile_op_names: ClassVar[Tuple[str, ...]] = ("tileops::moe_gate_up_fwd",)

    def __init__(
        self,
        numel: int,
        num_experts: int,
        ffn: int,
        k: int,
        activation: str = "silu_and_mul",
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            numel: T * top_k tight row count.
            num_experts: Number of local experts E.
            ffn: FFN width; ``b`` holds 2*ffn rows (gate||up).
            k: Hidden size K.
            activation: 'silu_and_mul' or 'gelu_and_mul'.
            tune: Whether to autotune.
        """
        self.numel = numel
        self.num_experts = num_experts
        self.ffn = ffn
        self.k = k
        self.activation = activation
        self.tune = tune

        self.dispatch_kernel()

    def _get_kernel(self, inputs: tuple, dtype: torch.dtype) -> Kernel:
        """The fused gate/up GEMM for this call, built once per input signature."""
        del dtype  # read off the tensors by the base class, which keys on them
        return self.get_or_build_kernel(_GATE_UP_SLOT, inputs)

    def _infer_output_shapes(
        self,
        a_shape: tuple,
        b_shape: tuple,
        true_sizes_shape: tuple,
        true_offsets_shape: tuple,
    ) -> Dict[str, tuple]:
        # b is [num_experts, 2 * ffn, K]; the gated activation halves that width.
        return {"c": (a_shape[0], b_shape[1] // 2)}


    def forward(
        self,
        a: torch.Tensor,  # [numel, K]
        b: torch.Tensor,  # [num_experts, 2*ffn, K]
        true_sizes: torch.Tensor,  # [E] int32
        true_offsets: torch.Tensor,  # [E] int32
    ) -> torch.Tensor:
        """Run the gate/up GEMM and apply the gated activation.

        Args:
            a: [numel, K] tight permuted activations.
            b: [num_experts, 2*ffn, K] gate||up expert weights.
            true_sizes: [E] int32 token count per expert.
            true_offsets: [E] int32 tight start offset per expert in a.

        Returns:
            [numel, ffn] activated output.
        """
        return _moe_gate_up_fwd(a, b, true_sizes, true_offsets, self._instance_key)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on the Cube unit."""
        return cube_roof(self.dtype)


@torch.library.custom_op("tileops::moe_gate_up_fwd", mutates_args=())
def _moe_gate_up_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    true_sizes: torch.Tensor,
    true_offsets: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(a, b, true_sizes, true_offsets)


@_moe_gate_up_fwd.register_fake
def _moe_gate_up_fwd_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    true_sizes: torch.Tensor,
    true_offsets: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(
        tuple(a.shape), tuple(b.shape), tuple(true_sizes.shape), tuple(true_offsets.shape)
    )
    # ``new_empty``, not ``empty_like``: a non-contiguous input's strides must not reach the fake.
    return a.new_empty(shapes["c"])
