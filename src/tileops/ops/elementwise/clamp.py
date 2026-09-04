"""Clamp ops: Tensor-bound bounds, and the scalar-bound form."""

from math import prod
from typing import Dict, Optional

import torch

from tileops.backend import Target

from ..compile_boundary import get_instance
from ..op_base import Op
from ._base import (
    _PerDtypeKernels,
    _require_one_device,
    _require_shape_inference,
    broadcast_or_raise,
    resolve_output_dtype,
)


class ClampFwdOp(_PerDtypeKernels, Op):
    """Clamp with Tensor lower and/or upper bounds (broadcasting).

    Conforms to ``torch.clamp(input, min, max)`` where ``min`` and ``max``
    are each either a Tensor or ``None``. At least one of the two bounds
    must be a Tensor. All Tensor operands broadcast together. A single bound
    is ``torch.clamp_min`` / ``torch.clamp_max``.

    Which bounds this call carries is read off the call, not settled at
    construction: the manifest declares both as ``optional: true``, so presence is a
    fact of the call, and it reaches the kernel's cache key because it changes what
    gets built. One instance therefore serves ``clamp``, ``clamp_min`` and
    ``clamp_max``, one specialization each.

    """

    _op_name = "clamp"
    _wrapped = None

    def __init__(
        self,
        *,
        target: Target = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Which set of kernels serves this op — a target name, or ``None`` to
                decide from the input device.
            tune: Whether to autotune.
        """
        self.target = target
        self.tune = tune
        self.input_shape: Optional[tuple] = None
        self.min_shape: Optional[tuple] = None
        self.max_shape: Optional[tuple] = None
        self.dispatch_kernel()


    def _infer_output_shapes(
        self,
        input_shape: tuple,
        min_shape: Optional[tuple],
        max_shape: Optional[tuple],
    ) -> Dict[str, tuple]:
        """Manifest ``shape_rules``: the broadcast of the operands this call passed.

        An absent bound arrives as ``None`` — the value ``forward`` was handed — so
        this reads presence off the slot rather than off how many slots there are.
        """
        return {
            "output": broadcast_or_raise(
                "ClampFwdOp", input=input_shape, min=min_shape, max=max_shape
            )
        }

    @property
    def out_shape(self) -> tuple:
        """Broadcast output shape of the most recent forward."""
        if self.input_shape is None:
            raise RuntimeError(
                "ClampFwdOp needs a prior forward() call: which bounds it serves, and "
                "their shapes, arrive with the call"
            )
        return self._infer_output_shapes(self.input_shape, self.min_shape, self.max_shape)["output"]

    @property
    def N_total(self) -> int:
        """Output element count of the most recent forward."""
        return prod(self.out_shape)

    def _eager_forward(
        self,
        input: torch.Tensor,
        min: Optional[torch.Tensor] = None,
        max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if min is None and max is None:
            raise ValueError(
                "ClampFwdOp requires at least one of `min` or `max` to be a Tensor; "
                "both None is not a valid clamp."
            )
        _require_one_device("ClampFwdOp", input=input, min=min, max=max)
        # Optional inputs are keyword-only on the generated validator.
        self._validate_dtypes(input, min=min, max=max)
        for name, bound in (("min", min), ("max", max)):
            if bound is not None and bound.dtype != input.dtype:
                raise ValueError(f"Expected {name}.dtype {input.dtype}, got {bound.dtype}")
        shapes = dict(
            input_shape=tuple(input.shape),
            min_shape=None if min is None else tuple(min.shape),
            max_shape=None if max is None else tuple(max.shape),
        )
        n_total = prod(self._infer_output_shapes(*shapes.values())["output"])
        input = input.contiguous()
        min = None if min is None else min.contiguous()
        max = None if max is None else max.contiguous()
        kernel = self._kernel(
            (input, min, max),
            input.dtype,
            n_total,
            min is not None,
            max is not None,
        )
        result = kernel(input, min, max)
        self._note_call(input.dtype, **shapes)
        return result

    def forward(
        self,
        input: torch.Tensor,
        min: Optional[torch.Tensor] = None,
        max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            input: Input tensor, dtype ``float16 | bfloat16 | float32``.
            min: Input tensor, dtype ``same_as(input)``. Optional.
            max: Input tensor, dtype ``same_as(input)``. Optional.

        Returns:
            ``output``, as the manifest declares. Shape rules: ``min is None or max is None or output.shape == broadcast_shapes(input.shape, min.shape, max.shape)``; ``min is None or max is not None or output.shape == broadcast_shapes(input.shape, min.shape)``; ``max is None or min is not None or output.shape == broadcast_shapes(input.shape, max.shape)``.
        """
        return type(self)._wrapped(input, min, max, self._instance_key)


class ClampScalarFwdOp(_PerDtypeKernels, Op):
    """Scalar-bound clamp (``torch.clamp(input, min: Number|None, max: Number|None)``)."""

    _op_name = "clamp"
    _wrapped = None

    def __init__(
        self,
        *,
        min: Optional[float] = None,
        max: Optional[float] = None,
        target: Target = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            min: Lower bound (Number or None).
            max: Upper bound (Number or None).
            target: Which set of kernels serves this op — a target name, or ``None`` to
                decide from the input device.
            tune: Whether to autotune.
        """
        if min is None and max is None:
            raise ValueError(
                "ClampScalarFwdOp requires at least one of `min` or `max` to be a "
                "Number; both None is not a valid clamp."
            )
        self.min = min
        self.max = max
        self.target = target
        self.tune = tune
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
                "ClampScalarFwdOp needs a prior forward() call: the element count "
                "arrives with the tensor"
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


# The compile boundary: one operator for this op, registered at import time. The op's
# key crosses it, and the body trades the key back for the instance — see
# src/tileops/ops/compile_boundary.py.

# The bounds are annotated ``Optional[torch.Tensor]``, so the schema reads
# ``Tensor? min, Tensor? max`` and one registration serves clamp, clamp_min and clamp_max.
_require_shape_inference(ClampFwdOp)


@torch.library.custom_op("tileops::elementwise_clamp_tensor", mutates_args=())
def _clamp_tensor_fwd(
    input: torch.Tensor,
    min: Optional[torch.Tensor],
    max: Optional[torch.Tensor],
    instance_key: str,
) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(input, min, max)


@_clamp_tensor_fwd.register_fake
def _clamp_tensor_fwd_fake(
    input: torch.Tensor,
    min: Optional[torch.Tensor],
    max: Optional[torch.Tensor],
    instance_key: str,
) -> torch.Tensor:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(
        tuple(input.shape),
        None if min is None else tuple(min.shape),
        None if max is None else tuple(max.shape),
    )
    return input.new_empty(
        shapes["output"], dtype=resolve_output_dtype(ClampFwdOp.__name__, input.dtype)
    )


ClampFwdOp._wrapped = _clamp_tensor_fwd
ClampFwdOp.compile_op_names = ("tileops::elementwise_clamp_tensor",)
