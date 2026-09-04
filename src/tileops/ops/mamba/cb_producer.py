"""
CB Producer Op - High-level interface for CB matrix computation.
"""


import torch

from tileops.perf.profile import cube_roof

from .._validation import check_tensor_shape
from ..op_base import Op
from tileops.backend import Kernel

__all__ = ["CBProducerFwdOp"]


class CBProducerFwdOp(Op):
    """CB (C@B) matrix producer operator.

    Computes cb[b,c,g,l,s] = sum_n C[b,c,g,l,n] * B[b,c,g,s,n]
    with causal masking (cb[l,s] = 0 if s > l).

    """

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        n_groups: int,
        chunk_len: int,
        d_state: int,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            batch: Batch size
            num_chunks: Number of chunks
            n_groups: Number of groups
            chunk_len: Chunk length (Q)
            d_state: State dimension (N)
            tune: Whether to autotune
        """
        self.batch = batch
        self.num_chunks = num_chunks
        self.n_groups = n_groups
        self.chunk_len = chunk_len
        self.d_state = d_state
        self.tune = tune

        # Use standard Op dispatch pattern
        self.dispatch_kernel()

    def _get_kernel(self, inputs: "tuple[torch.Tensor | None, ...]", dtype: torch.dtype) -> Kernel:
        return self.get_or_build_kernel("cb_producer", inputs)


    def _infer_output_shapes(
        self,
        C_mat_shape: tuple[int, ...],
        B_mat_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: one causal ``(Q, Q)`` block per batch, chunk and group."""
        batch, _, groups, _ = C_mat_shape
        return {"cb": (batch, self.num_chunks, groups, self.chunk_len, self.chunk_len)}

    def forward(
        self,
        C_mat: torch.Tensor,
        B_mat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            C_mat: [B, S, G, N]  dtype (contiguous)
            B_mat: [B, S, G, N]  dtype (contiguous)

        Returns:
            cb: [B, C, G, Q, Q]  dtype
        """
        self._validate_dtypes(C_mat, B_mat)
        S = self.num_chunks * self.chunk_len
        expected_shape = (self.batch, S, self.n_groups, self.d_state)
        self.dtype = C_mat.dtype
        check_tensor_shape("C_mat", C_mat, expected_shape)
        check_tensor_shape("B_mat", B_mat, expected_shape)
        C_mat = C_mat.contiguous()
        B_mat = B_mat.contiguous()
        return self._get_kernel((C_mat, B_mat), C_mat.dtype)(C_mat, B_mat)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on the Cube unit."""
        return cube_roof(self.dtype)
