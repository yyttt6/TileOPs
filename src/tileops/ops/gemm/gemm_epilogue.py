"""Dense GEMM with a fused bias / bias+ReLU / bias+GELU epilogue (T264).

Ops 110-112 of ``op-list-150.pdf``.  The PDF signature is
``gemm_bias(a: T[M,K], b: T[K,N], bias: T[N]) -> c: T[M,N]``, i.e. the NN
layout.  These ops keep ``GemmFwdOp``'s ``trans_a`` / ``trans_b`` parameters
(default NT, ``trans_b=True``) so that the twelve ``GemmFwdOp`` manifest
workloads apply to them verbatim -- ``trans_b=False`` is exactly the PDF's
``b: T[K,N]``, and the NT case is the superset.  ``trans_a=True`` is rejected:
the fused Cube path is non-transposed-A only, and it is False in all twelve
workloads.
"""

from typing import Hashable, Optional, Tuple

import torch

from tileops.perf.profile import cube_roof

from ..op_base import Op
from tileops.backend import Kernel, Target

__all__ = ["GemmBiasFwdOp", "GemmBiasGeluFwdOp", "GemmBiasReluFwdOp"]


class _GemmBiasBase(Op):
    """Shared plumbing: the three ops differ only in which builder they ask for.

    Modelled on :class:`tileops.ops.gemm.gemm.GemmFwdOp` -- input-inferred
    ``m``/``n``/``k``/dtype bound on the first ``forward()`` so the manifest's
    func-mode roofline can read them, and a per-``(m, n, k, dtype)`` kernel
    cache.
    """

    #: Which of this op's kernels ``forward`` asks the backend for.
    _kernel_role: str = ""

    def __init__(
        self,
        trans_a: bool = False,
        trans_b: bool = True,
        tune: bool = False,
        target: "Target" = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            trans_a: Must be ``False``; the fused epilogue path has no
                transposed-A kernel.
            trans_b: Whether ``b`` is stored $[N \\times K]$. Default ``True``
                (NT). ``False`` is the PDF's $[K \\times N]$.
            tune: Whether to autotune (applied when a kernel is first built).
            target: Which set of kernels serves this op, or ``None`` to decide
                from the input device.  ``tileops.backend.dispatch.detect_target``
                returns ``None`` for an ``npu`` device in this distribution, so
                callers on Ascend pass ``target="ascend"`` explicitly (which is
                what the norm ops' call sites do).
        """
        if trans_a:
            raise ValueError(
                f"{type(self).__name__} does not support trans_a=True: the fused "
                "Cube path is non-transposed-A only (see "
                "tileops/kernels/gemm_epilogue.py)"
            )
        self.trans_a = False
        self.trans_b = trans_b
        self.tune = tune
        self.target = target
        self.dispatch_kernel()
        self._active_sig: Optional[tuple] = None
        self._active: Optional[tuple] = None
        self.m: Optional[int] = None
        self.n: Optional[int] = None
        self.k: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None

    def _infer_mnk(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[int, int, int]:
        m, k_a = a.shape[0], a.shape[1]
        n, k_b = (b.shape[0], b.shape[1]) if self.trans_b else (b.shape[1], b.shape[0])
        if k_a != k_b:
            raise ValueError(
                f"GEMM contraction dim mismatch: a contributes K={k_a}, b contributes "
                f"K={k_b} (a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}, "
                f"trans_b={self.trans_b})"
            )
        return m, n, k_a

    def _cache_key(self, *input_shapes: Tuple[int, ...]) -> Hashable:
        return (
            self.m,
            self.n,
            self.k,
            self.trans_a,
            self.trans_b,
            None if self.dtype is None else str(self.dtype),
        )

    def _infer_output_shapes(
        self,
        a_shape: tuple[int, ...],
        b_shape: tuple[int, ...],
        bias_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        del bias_shape
        n = b_shape[0] if self.trans_b else b_shape[1]
        return {"c": (a_shape[0], n)}

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """Multiply, then add ``bias`` (and apply this op's activation).

        Args:
            a: Left operand, $[M \\times K]$.
            b: Right operand, $[N \\times K]$ under the default NT layout, or
                $[K \\times N]$ when ``trans_b`` is false.
            bias: $[N]$, broadcast over the $M$ axis.

        Returns:
            $[M \\times N]$, in the dtype of the inputs.
        """
        sig = (a.shape, b.shape, bias.shape, a.dtype)
        if sig != self._active_sig:
            self._validate_dtypes(a, b, bias)
            m, n, k = self._infer_mnk(a, b)
            if bias.shape != (n,):
                raise ValueError(
                    f"{type(self).__name__} expects bias of shape ({n},), got "
                    f"{tuple(bias.shape)}"
                )
            self.m, self.n, self.k = m, n, k
            self.dtype = a.dtype
            kernel: Kernel = self.get_or_build_kernel(self._kernel_role, (a, b, bias))
            self.kernel = kernel
            self._active = (kernel,)
            self._active_sig = sig
        (kernel,) = self._active
        return kernel(a, b, bias)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on the Cube unit.

        The bias add and the activation run on the vector cores of the same mix
        launch and are $O(MN)$ against the GEMM's $O(MNK)$, so the Cube roof is
        still the binding one -- which is also why these ops reuse
        ``GemmFwdOp``'s roofline formula.
        """
        return cube_roof(self.dtype)


class GemmBiasFwdOp(_GemmBiasBase):
    """``c = a @ b + bias`` with the bias add fused into the GEMM launch."""

    _kernel_role = "gemm_bias_kernel"


class GemmBiasReluFwdOp(_GemmBiasBase):
    """``c = relu(a @ b + bias)``, fused."""

    _kernel_role = "gemm_bias_relu_kernel"


class GemmBiasGeluFwdOp(_GemmBiasBase):
    """``c = gelu(a @ b + bias)``, fused.

    GELU uses the tanh approximation, computed in fp32 as
    ``2 * sigmoid(2z) - 1``.  This backend's vector unit has no ``erf``
    intrinsic on dav-2201, so ``kernels/elementwise_activation.py`` uses the
    same chain for exact GELU as well; the reference this is checked against is
    ``torch.nn.functional.gelu(..., approximate="tanh")``.
    """

    _kernel_role = "gemm_bias_gelu_kernel"
