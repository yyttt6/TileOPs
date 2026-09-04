
import torch

from tileops.perf.profile import cube_roof

from ..op_base import Op
from tileops.backend import Kernel

__all__ = ["MultiHeadLatentAttentionDecodeWithKVCacheFwdOp"]


class MultiHeadLatentAttentionDecodeWithKVCacheFwdOp(Op):
    """Layout: BSHD"""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seqlen_kv: int,
        dim: int,
        pe_dim: int,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            pe_dim: Manifest ``params.pe_dim``, ``int``.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.pe_dim = pe_dim

        self.tune = tune
        self.dispatch_kernel()

    def _get_kernel(self, inputs: "tuple[torch.Tensor | None, ...]", dtype: torch.dtype) -> Kernel:
        return self.get_or_build_kernel("mla_decode_kernel", inputs)


    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        q_pe_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        k_pe_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``shape_rules``: ``o.shape == q.shape``."""
        return {"o": tuple(q_shape)}

    def forward(
        self, q: torch.Tensor, q_pe: torch.Tensor, k: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            q: Input tensor, dtype ``float16 | bfloat16``.
            q_pe: Input tensor, dtype ``same_as(q)``.
            k: Input tensor, dtype ``same_as(q)``.
            k_pe: Input tensor, dtype ``same_as(q)``.

        Returns:
            ``o``, as the manifest declares. Shape rules: ``o.shape == (B, H, D)``.
        """
        self._validate_dtypes(q, q_pe, k, k_pe)
        self.dtype = q.dtype
        return self._get_kernel((q, q_pe, k, k_pe), q.dtype)(q, q_pe, k, k_pe)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on the Cube unit."""
        return cube_roof(self.dtype)
