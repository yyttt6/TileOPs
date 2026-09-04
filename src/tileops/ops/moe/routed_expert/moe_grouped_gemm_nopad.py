"""MoE grouped GEMM op (no-pad variant): NT GEMM with precomputed tile scheduling."""

from typing import ClassVar, Dict, Tuple

import torch

from tileops.perf.profile import cube_roof

from ...compile_boundary import get_instance
from ...op_base import Op
from ._common import GroupedOperandEagerForward
from tileops.backend import Kernel

__all__ = ["MoeGroupedGemmNopadFwdOp"]

#: The implementations of this role; each states its own region.
#: The slot this op asks its target for.
_GEMM_SLOT = "moe_grouped_gemm"


class MoeGroupedGemmNopadFwdOp(GroupedOperandEagerForward, Op):
    """NT grouped GEMM for MoE without block_m-aligned padding.

    Uses a GPU tile scheduler to map each CTA to its (expert, row_offset) in O(1),
    eliminating the O(E) per-CTA expert scan in standard grouped GEMM.

    Accepts tight A[T*K, K] inputs (no padding between experts) from
    MoePermuteNoPadOp, producing tight C[T*K, N] outputs.

    Example:
        ```python linenums="1"
        op = MoeGroupedGemmNopadFwdOp(numel=16384, num_experts=256, n=4096, k=2048,
        )
        C = op(A, B, true_sizes, true_offsets)  # [numel, N]
        ```
    """

    #: The operator this op registers; a test asserts the graph holds nothing else.
    compile_op_names: ClassVar[Tuple[str, ...]] = ("tileops::moe_grouped_gemm_nopad_fwd",)

    def __init__(
        self,
        numel: int,
        num_experts: int,
        n: int,
        k: int,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            numel: T * top_k, total (token, expert) pairs = tight row count.
            num_experts: Total number of experts E.
            n: Output feature dimension N (e.g. 2*ffn_size or hidden_size).
            k: Input feature dimension K (hidden_size or ffn_size).
            tune: Whether to autotune.
        """
        self.numel = numel
        self.num_experts = num_experts
        self.n = n
        self.k = k
        self.tune = tune

        self.dispatch_kernel()

    def _get_kernel(self, inputs: tuple, dtype: torch.dtype) -> Kernel:
        del dtype  # on the tensors the target is handed
        return self.get_or_build_kernel(_GEMM_SLOT, inputs)

    def _infer_output_shapes(
        self,
        a_shape: tuple,
        b_shape: tuple,
        true_sizes_shape: tuple,
        true_offsets_shape: tuple,
    ) -> Dict[str, tuple]:
        # b is [num_experts, N, K]; the tight output keeps a's row count.
        return {"c": (a_shape[0], b_shape[1])}


    def forward(
        self,
        a: torch.Tensor,  # [numel, K]
        b: torch.Tensor,  # [num_experts, N, K]
        true_sizes: torch.Tensor,  # [E] int32
        true_offsets: torch.Tensor,  # [E] int32
    ) -> torch.Tensor:
        """Run tile-scheduled NT GEMM.

        Args:
            a: [numel, K] tight permuted activations.
            b: [num_experts, N, K] expert weights (NT: B^T applied).
            true_sizes: [E] int32 true token count per expert.
            true_offsets: [E] int32 tight start offset per expert in a.

        Returns:
            C: [numel, N] GEMM output.
        """
        return _moe_grouped_gemm_nopad_fwd(a, b, true_sizes, true_offsets, self._instance_key)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on the Cube unit."""
        return cube_roof(self.dtype)


@torch.library.custom_op("tileops::moe_grouped_gemm_nopad_fwd", mutates_args=())
def _moe_grouped_gemm_nopad_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    true_sizes: torch.Tensor,
    true_offsets: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(a, b, true_sizes, true_offsets)


@_moe_grouped_gemm_nopad_fwd.register_fake
def _moe_grouped_gemm_nopad_fwd_fake(
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
