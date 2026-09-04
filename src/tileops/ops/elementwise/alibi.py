"""ALiBi position-encoding generative op."""


import torch

from tileops.backend import Target

from ..op_base import Op


class AlibiFwdOp(Op):
    """ALiBi position encoding: bias[h, i, j] = -slope_h * |i - j|.

    Generates the full (num_heads, seq_len, seq_len) bias tensor.

    Note:
        Eager-only. Unlike the other elementwise ops in this package,
        ``AlibiFwdOp`` is not registered as a ``torch.library.custom_op``,
        so ``torch.compile`` graph capture is not supported. The op has
        zero tensor inputs and constructs its output entirely from
        ``__init__`` parameters; no compile-time wrapping is needed.

    """

    _op_name = "alibi"

    def __init__(
        self,
        *,
        seq_len: int,
        num_heads: int,
        dtype: torch.dtype,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            seq_len: Sequence length.
            num_heads: Number of attention heads.
            dtype: Torch dtype.
            target: Which set of kernels serves this op — a target name, or ``None`` to
                decide from the input device.
        """
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dtype = dtype
        self.target = target
        self.dispatch_kernel()


    def _infer_output_shapes(self) -> dict[str, tuple[int, ...]]:
        return {"output": (self.num_heads, self.seq_len, self.seq_len)}

    def _validate_dtypes(self) -> None:
        return None

    @property
    def total_memory(self) -> int:
        return self.num_heads * self.seq_len * self.seq_len * self.dtype.itemsize

    def eval_roofline(self) -> tuple[int, int]:
        n_elem = self.num_heads * self.seq_len * self.seq_len
        return 3 * n_elem, self.total_memory


    def forward(self) -> torch.Tensor:
        # The op promised ``self.dtype``; whichever storage the backend chose to
        # compute in is its own business and does not reach the caller.
        """Run the op on the inputs the manifest declares.

        Returns:
            ``output``, as the manifest declares.
        """
        kernel = self.get_or_build_kernel(self._op_name, ())
        out = kernel().reshape(self.num_heads, self.seq_len, self.seq_len)
        return out if out.dtype == self.dtype else out.to(self.dtype)
