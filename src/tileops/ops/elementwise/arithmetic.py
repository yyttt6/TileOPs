"""Binary arithmetic elementwise ops with broadcasting."""

from math import prod
from typing import Dict, Optional

import torch

from tileops.backend import Target

from ..compile_boundary import get_instance
from ..op_base import Op
from ._base import (
    BinaryOp,
    _AlphaScaledBinaryOp,
    _PerDtypeKernels,
    _require_one_device,
    _require_shape_inference,
    broadcast_or_raise,
    resolve_output_dtype,
)


class AddFwdOp(_AlphaScaledBinaryOp):
    """Element-wise addition with broadcast: y = input + alpha * other.

    Conforms to ``torch.add(input, other, *, alpha=1)``. ``alpha`` is baked
    into the kernel, so non-default ``alpha`` runs through the same fast
    kernel as the default.
    """

    _op_name = "add"


class SubFwdOp(_AlphaScaledBinaryOp):
    """Element-wise subtraction with broadcast: y = input - alpha * other.

    Conforms to ``torch.sub(input, other, *, alpha=1)``. ``alpha`` is baked
    into the kernel, so non-default ``alpha`` runs through the same fast
    kernel as the default.
    """

    _op_name = "sub"


class BiasAddFwdOp(BinaryOp):
    """T263: PDF entry 49 -- ``bias_add(x[M, N], bias[N]) -> out[M, N]``.

    An add whose second operand is rank 1 and aligns with the last axis of
    ``input``, so ``BinaryOp``'s broadcast machinery and ``AddFwdOp``'s
    roofline both apply unchanged. The rank-1 constraint from the PDF
    signature is enforced by the kernel builder; broadcasting itself follows
    PyTorch, i.e. it is exactly ``torch.add(x, bias)``.
    """

    _op_name = "bias_add"
    _other_name = "bias"

    def _infer_output_shapes(self, input_shape: tuple, bias_shape: tuple) -> Dict[str, tuple]:
        """Manifest ``shape_rules``: ``output.shape == broadcast_shapes(...)``.

        Overridden for the same reason ``PowFwdOp`` overrides it: the manifest
        names the second operand ``bias``, and ``_other_name`` only rebinds
        ``forward``'s signature, not this one.
        """
        return {
            "output": broadcast_or_raise("BiasAddFwdOp", input=input_shape, bias=bias_shape)
        }


class MulFwdOp(BinaryOp):
    """Element-wise multiplication with broadcast: y = input * other."""

    _op_name = "mul"


#: The rounding modes ``torch.div`` defines, which is what the manifest declares.
#: ``rounding_mode`` is a manifest param, so it reaches a target's ``build_kernel``
#: by name; this is only the input check.
_DIV_ROUNDING_MODES = (None, "trunc", "floor")


class DivFwdOp(BinaryOp):
    """Element-wise division with broadcast: y = input / other.

    Conforms to ``torch.div(input, other, *, rounding_mode=None)``.
    ``rounding_mode`` accepts ``None`` (true division), ``"trunc"``
    (truncation toward zero), or ``"floor"`` (floor division); each
    value selects a dedicated kernel specialization. It is fixed for the
    instance, which is why it is not part of the memory key.
    """

    _op_name = "div"

    def __init__(
        self,
        *,
        rounding_mode: Optional[str] = None,
        target: Target = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            rounding_mode: Manifest ``params.rounding_mode``, ``str | None``, default ``None``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        if rounding_mode not in _DIV_ROUNDING_MODES:
            raise ValueError(
                f"DivFwdOp received rounding_mode={rounding_mode!r}; "
                "manifest allows None, 'trunc', or 'floor'"
            )
        self.rounding_mode = rounding_mode
        super().__init__(target=target, tune=tune)


class RemainderFwdOp(BinaryOp):
    """Element-wise remainder with broadcast: y = a % b."""

    _op_name = "remainder"


class PowFwdOp(BinaryOp):
    """Element-wise power with broadcast: y = input ** exponent.

    Conforms to ``torch.pow(input, exponent)``: the second operand carries
    the manifest-declared name ``exponent`` rather than the generic
    ``other`` so the L1 signature check matches the manifest.
    """

    _op_name = "pow"
    _other_name = "exponent"

    def _infer_output_shapes(self, input_shape: tuple, exponent_shape: tuple) -> Dict[str, tuple]:
        """Manifest ``shape_rules``: ``output.shape == broadcast_shapes(...)``."""
        return {
            "output": broadcast_or_raise("PowFwdOp", input=input_shape, exponent=exponent_shape)
        }


class FloorDivideFwdOp(BinaryOp):
    """Element-wise floor division with broadcast: y = floor(a / b)."""

    _op_name = "floor_divide"


class LerpFwdOp(BinaryOp):
    """Element-wise lerp with broadcast: y = a + weight * (b - a).

    Unlike ``torch.lerp(a, b, weight)`` where weight is a runtime parameter,
    here weight is a **construction-time constant** baked into the compiled
    kernel. This enables compile-time folding but means a new Op instance is
    needed for each distinct weight value.

    """

    _op_name = "lerp"
    _other_name = "end"

    def _infer_output_shapes(self, input_shape: tuple, end_shape: tuple) -> Dict[str, tuple]:
        """Manifest ``shape_rules``: ``output.shape == broadcast_shapes(...)``."""
        return {"output": broadcast_or_raise("LerpFwdOp", input=input_shape, end=end_shape)}

    def __init__(
        self,
        *,
        weight: float = 0.5,
        target: Target = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            weight: Scalar interpolation weight, fixed at construction (manifest
                ``params.weight``, default 0.5).
            target: Which set of kernels serves this op.
            tune: Whether to autotune.
        """
        self.weight = weight
        super().__init__(target=target, tune=tune)


class Atan2FwdOp(BinaryOp):
    """T263: PDF entry 24 -- ``atan2(y[*S], x[*S]) -> out[*S]``.

    Argument order is ``torch.atan2(input, other)``: ``input`` is the
    numerator (y) and ``other`` the denominator (x). Derived from
    ``MaximumFwdOp`` -- same two-tensor broadcasting surface, same workloads.
    """

    _op_name = "atan2"

    def _infer_output_shapes(self, input_shape: tuple, other_shape: tuple) -> Dict[str, tuple]:
        """Manifest ``shape_rules``: ``output.shape == broadcast_shapes(...)``.

        Spelled out rather than inherited so the shape rule can be checked
        against the validator's ``SimpleNamespace`` mock, which has no
        ``_other_name`` class attribute for the inherited body to read.
        """
        return {
            "output": broadcast_or_raise("Atan2FwdOp", input=input_shape, other=other_shape)
        }


class MaximumFwdOp(BinaryOp):
    """Element-wise maximum with broadcast: y = max(a, b)."""

    _op_name = "maximum"


class MinimumFwdOp(BinaryOp):
    """Element-wise minimum with broadcast: y = min(a, b)."""

    _op_name = "minimum"


class LerpTensorFwdOp(_PerDtypeKernels, Op):
    """Tensor-weight lerp: out = input + weight * (end - input).

    Conforms to the Tensor-weight overload of ``torch.lerp`` —
    ``torch.lerp(input, end, weight: Tensor)`` where ``weight`` is a Tensor that
    broadcasts together with ``input`` and ``end`` to the output shape. The scalar-weight
    overload is handled separately by ``LerpFwdOp``.

    """

    _op_name = "lerp_tensor"
    _wrapped = None

    # Manifest declares all three operands as ``float16 | bfloat16 | float32``;
    # fp8 dtypes are rejected at the op-layer signature so the impl matches
    # the manifest contract (the kernel also rejects fp8 independently).
    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        *,
        target: Target = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Which set of kernels serves this op.
            tune: Whether to autotune.
        """
        self.target = target
        self.tune = tune
        self.input_shape: Optional[tuple] = None
        self.end_shape: Optional[tuple] = None
        self.weight_shape: Optional[tuple] = None
        self.dispatch_kernel()

    def _infer_output_shapes(
        self,
        input_shape: tuple,
        end_shape: tuple,
        weight_shape: tuple,
    ) -> Dict[str, tuple]:
        """Manifest ``shape_rules``: ``output.shape == broadcast_shapes(...)``."""
        return {
            "output": broadcast_or_raise(
                "LerpTensorFwdOp", input=input_shape, end=end_shape, weight=weight_shape
            )
        }


    @property
    def out_shape(self) -> tuple:
        """Broadcast output shape of the most recent forward."""
        if self.input_shape is None:
            raise RuntimeError(
                "LerpTensorFwdOp needs a prior forward() call: the operand shapes "
                "arrive with the tensors"
            )
        return self._infer_output_shapes(self.input_shape, self.end_shape, self.weight_shape)[
            "output"
        ]

    @property
    def N_total(self) -> int:
        """Output element count of the most recent forward."""
        return prod(self.out_shape)

    def _eager_forward(
        self,
        input: torch.Tensor,
        end: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        _require_one_device("LerpTensorFwdOp", input=input, end=end, weight=weight)
        for name, t in (("end", end), ("weight", weight)):
            if t.dtype != input.dtype:
                raise ValueError(f"Expected {name}.dtype {input.dtype}, got {t.dtype}")
        self._validate_dtypes(input, end, weight)
        shapes = dict(
            input_shape=tuple(input.shape),
            end_shape=tuple(end.shape),
            weight_shape=tuple(weight.shape),
        )
        n_total = prod(self._infer_output_shapes(*shapes.values())["output"])
        input = input.contiguous()
        end = end.contiguous()
        weight = weight.contiguous()
        result = self._kernel((input, end, weight), input.dtype, n_total)(input, end, weight)
        self._note_call(input.dtype, **shapes)
        return result

    def forward(
        self,
        input: torch.Tensor,
        end: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            input: Input tensor, dtype ``float16 | bfloat16 | float32``.
            end: Input tensor, dtype ``same_as(input)``.
            weight: Input tensor, dtype ``same_as(input)``.

        Returns:
            ``output``, as the manifest declares. Shape rules: ``output.shape == broadcast_shapes(input.shape, end.shape, weight.shape)``.
        """
        return type(self)._wrapped(input, end, weight, self._instance_key)


# The compile boundary: one operator for this op, registered at import time. The op's
# key crosses it, and the body trades the key back for the instance — see
# src/tileops/ops/compile_boundary.py.

# Its own name, not the scalar ``LerpFwdOp``'s: the weight is a third tensor here.
_require_shape_inference(LerpTensorFwdOp)


@torch.library.custom_op("tileops::elementwise_lerp_tensor", mutates_args=())
def _lerp_tensor_fwd(
    input: torch.Tensor,
    end: torch.Tensor,
    weight: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(input, end, weight)


@_lerp_tensor_fwd.register_fake
def _lerp_tensor_fwd_fake(
    input: torch.Tensor,
    end: torch.Tensor,
    weight: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(tuple(input.shape), tuple(end.shape), tuple(weight.shape))
    return input.new_empty(
        shapes["output"], dtype=resolve_output_dtype(LerpTensorFwdOp.__name__, input.dtype)
    )


LerpTensorFwdOp._wrapped = _lerp_tensor_fwd
LerpTensorFwdOp.compile_op_names = ("tileops::elementwise_lerp_tensor",)
