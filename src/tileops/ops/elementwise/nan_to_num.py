"""NanToNum op: replace NaN, +Inf, -Inf with specified values."""

from math import prod
from typing import Dict, Optional

import torch

from tileops.backend import Target

from ..op_base import Op
from ._base import _PerDtypeKernels


class NanToNumFwdOp(_PerDtypeKernels, Op):
    """NanToNum: replace NaN, +Inf, -Inf with specified values."""

    _op_name = "nan_to_num"
    _wrapped = None

    def __init__(
        self,
        *,
        nan: float = 0.0,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
        target: Target = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            nan: Replacement for NaN (default 0.0).
            posinf: Replacement for +Inf. Manifest default ``None`` resolves
                to the largest finite value representable in the element type of the
                call (matches ``torch.nan_to_num``). Explicit values
                must also be representable in that dtype end-to-end; values
                that fit only in the kernel's intermediate dtype (e.g. fp16
                for fp8_e5m2) are rejected so the post-cast cannot resurface
                them as Inf.
            neginf: Replacement for -Inf. Manifest default ``None`` resolves
                to the smallest (most negative) finite value representable
                in the element type of the call.
            target: Which set of kernels serves this op — a target name, or ``None`` to
                decide from the input device.
            tune: Whether to autotune the kernel.
        """
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf
        self.target = target
        self.tune = tune
        # Manifest input binding for the synthesized eval_roofline
        # (docs/design/roofline.md §4.4.3); bound by the first forward.
        self.input_shape: Optional[tuple] = None
        self.dispatch_kernel()


    def _infer_output_shapes(self, input_shape: tuple) -> Dict[str, tuple]:
        """Manifest ``shape_rules``: ``output.shape == input.shape``."""
        return {"output": tuple(input_shape)}

    @property
    def N_total(self) -> int:
        """Element count of the most recent forward."""
        if self.input_shape is None:
            raise RuntimeError(
                "NanToNumFwdOp needs a prior forward() call: the element count arrives "
                "with the tensor"
            )
        return prod(self.input_shape)

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        self._validate_dtypes(input)
        input = input.contiguous()
        result = self._kernel((input,), input.dtype, input.numel())(input)
        self._note_call(input.dtype, input_shape=tuple(input.shape))
        return result

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            input: Input tensor, dtype ``float16 | bfloat16 | float32``.

        Returns:
            ``output``, as the manifest declares. Shape rules: ``output.shape == input.shape``.
        """
        return type(self)._wrapped(input, self._instance_key)
