"""Sinusoidal positional encoding generative op."""


import torch

from tileops.backend import Target

from ..op_base import Op


class SinusoidalFwdOp(Op):
    """Sinusoidal positional encoding from "Attention Is All You Need".

    Generates the full (seq_len, d_model) encoding tensor.

    Note:
        Eager-only. Unlike the other elementwise ops in this package,
        ``SinusoidalFwdOp`` is not registered as a ``torch.library.custom_op``,
        so ``torch.compile`` graph capture is not supported. The op has
        zero tensor inputs and constructs its output entirely from
        ``__init__`` parameters; no compile-time wrapping is needed.

    """

    _op_name = "sinusoidal"

    def __init__(
        self,
        *,
        seq_len: int,
        d_model: int,
        dtype: torch.dtype,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            seq_len: Sequence length.
            d_model: Model dimension.
            dtype: Torch dtype.
            target: Which set of kernels serves this op — a target name, or ``None`` to
                decide from the input device.
        """
        self.seq_len = seq_len
        self.d_model = d_model
        self.dtype = dtype
        self.target = target
        self.dispatch_kernel()


    def _infer_output_shapes(self) -> dict[str, tuple[int, ...]]:
        return {"output": (self.seq_len, self.d_model)}

    def _validate_dtypes(self) -> None:
        return None

    @property
    def total_memory(self) -> int:
        return self.seq_len * self.d_model * self.dtype.itemsize

    def eval_roofline(self) -> tuple[int, int]:
        n_elem = self.seq_len * self.d_model
        return 6 * n_elem, self.total_memory


    def forward(self) -> torch.Tensor:
        # The op promised ``self.dtype``; whichever storage the backend chose to
        # compute in is its own business and does not reach the caller.
        """Run the op on the inputs the manifest declares.

        Returns:
            ``output``, as the manifest declares.
        """
        kernel = self.get_or_build_kernel(self._op_name, ())
        out = kernel().reshape(self.seq_len, self.d_model)
        return out if out.dtype == self.dtype else out.to(self.dtype)
