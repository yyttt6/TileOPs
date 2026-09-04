"""Unary math elementwise ops (exp/log/sqrt/abs/neg/round/etc.)."""


import torch

from tileops.backend import Target

from ._base import (
    _MANIFEST_INT_DTYPES,
    UnaryOp,
    _IntIdentityUnaryOp,
)


class ExpFwdOp(UnaryOp):
    """Element-wise exp(x)."""

    _op_name = "exp"


class LogFwdOp(UnaryOp):
    """Element-wise log(x)."""

    _op_name = "log"


class SqrtFwdOp(UnaryOp):
    """Element-wise sqrt(x)."""

    _op_name = "sqrt"


class RsqrtFwdOp(UnaryOp):
    """Element-wise 1/sqrt(x)."""

    _op_name = "rsqrt"


class AbsFwdOp(_IntIdentityUnaryOp):
    """Element-wise |x|."""

    _op_name = "abs"
    _int_handler = staticmethod(torch.abs)


class NegFwdOp(_IntIdentityUnaryOp):
    """Element-wise -x."""

    _op_name = "neg"
    _int_handler = staticmethod(torch.neg)


class ReciprocalFwdOp(UnaryOp):
    """Element-wise 1/x.

    Mirrors ``torch.reciprocal`` int-input promotion: the manifest declares the
    output as ``promote_int_to_float(input)``, and ``ReciprocalFwdKernel.specialize``
    names float32 as the compute type for an integral input. The semantic dtype
    keys the specialization and drives roofline accounting — integer input bytes,
    float32 output bytes — while the kernel is built for the type it computes in.
    Floating inputs follow the standard same-dtype path.
    """

    _op_name = "reciprocal"


class SignFwdOp(_IntIdentityUnaryOp):
    """Element-wise sign(x): -1, 0, or +1."""

    _op_name = "sign"
    # Manifest: flops = "2 * N" (two compares + selects per element).
    FLOPS_PER_ELEM = 2
    _int_handler = staticmethod(torch.sign)


class SinFwdOp(UnaryOp):
    """Element-wise sin(x)."""

    _op_name = "sin"


class CosFwdOp(UnaryOp):
    """Element-wise cos(x)."""

    _op_name = "cos"


class FloorFwdOp(_IntIdentityUnaryOp):
    """Element-wise floor(x)."""

    _op_name = "floor"


class CeilFwdOp(_IntIdentityUnaryOp):
    """Element-wise ceil(x)."""

    _op_name = "ceil"


class _RoundDecimalsCall:
    """In-tree stand-in for ``round(x, decimals=k)`` with ``k != 0``.

    ``round(x, decimals=k) == round(x * 10**k) / 10**k``, which the shipped
    round-to-nearest-integer kernel does not do. Only the in-tree path builds one:
    with a target selected, ``decimals`` is handed over as the manifest param it is
    and the backend serves every value of it. Not a ``Kernel``, so ``autotune`` walks
    past it — there is nothing to tune.
    """

    def __init__(self, decimals: int):
        """Build the op. Shapes and dtype are taken from the first call."""
        self._decimals = decimals

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        # Integer dtypes are no-ops regardless of decimals (rounding an int
        # produces the same int). Match the float-path identity contract.
        if input.dtype in _MANIFEST_INT_DTYPES:
            return input.clone()
        # Run through fp32 so low-precision inputs (fp16/bf16) cannot overflow
        # when ``torch.round`` internally scales by ``10**decimals`` — e.g.
        # ``100 * 10**4 = 1e6`` exceeds fp16 max (~65504). The single down-cast
        # at the end restores the op's contract dtype.
        return torch.round(input.float(), decimals=self._decimals).to(input.dtype)


class RoundFwdOp(_IntIdentityUnaryOp):
    """Element-wise round(x) to ``decimals`` decimal places.

    The shipped kernel performs banker's round-to-nearest-integer, matching
    ``torch.round`` for ``decimals=0``. ``decimals`` is a manifest param, so it is
    fixed for the instance and handed to whichever kernel serves the op; in-tree, a
    non-zero value selects ``_RoundDecimalsCall``.

    """

    _op_name = "round"

    def __init__(
        self,
        *,
        decimals: int = 0,
        target: Target = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            decimals: Number of decimal places to round to (manifest
                ``params.decimals``, default 0).
            target: Which set of kernels serves this op.
            tune: Whether to autotune.
        """
        self.decimals = int(decimals)
        super().__init__(target=target, tune=tune)


class TruncFwdOp(_IntIdentityUnaryOp):
    """Element-wise trunc(x)."""

    _op_name = "trunc"


class ErfFwdOp(UnaryOp):
    """Element-wise erf(x)."""

    _op_name = "erf"


class Log1pFwdOp(UnaryOp):
    """Element-wise log(1 + x)."""

    _op_name = "log1p"
    # Manifest: flops = "2 * N" (1 add + 1 log).
    FLOPS_PER_ELEM = 2


class Expm1FwdOp(UnaryOp):
    """Element-wise exp(x) - 1."""

    _op_name = "expm1"
    # Manifest: flops = "2 * N" (1 exp + 1 sub).
    FLOPS_PER_ELEM = 2
