"""Split-K dense GEMM (T267, op-list-150.pdf #113).

The PDF signature is ``gemm_splitk(a: T[M,K], b: T[K,N]) -> c: T[M,N]``: the
same product as ``gemm``, computed with a K-partitioned schedule.  Split-K is a
SCHEDULE, not a different function, so this op keeps ``GemmFwdOp``'s
``trans_b`` parameter (default NT) and reuses that op's twelve manifest
workloads verbatim -- ``trans_b=False`` is exactly the PDF's ``b: T[K,N]``.

``split_k`` is chosen by the kernel's planner from the tile geometry, not
passed in by the caller: whether splitting K helps is a fact about how many
logical blocks the shape already has against the machine's core count, and a
caller-supplied value would let a benchmark pick a split for shapes where it is
a guaranteed loss.  See ``tileops/kernels/gemm_splitk.py`` for the gate and
``op.split_k`` for what it decided.  ``trans_a=True`` is rejected: the split-K
partial kernel is the non-transposed-A pipelined path only, and ``trans_a`` is
False in all twelve workloads.
"""

from typing import Hashable, Optional, Tuple

import torch

from tileops.perf.profile import cube_roof

from ..op_base import Op
from tileops.backend import Kernel, Target

__all__ = ["GemmSplitKFwdOp"]


class GemmSplitKFwdOp(Op):
    """``c = a @ b`` computed with a K-partitioned schedule.

    Input-inferred exactly like :class:`tileops.ops.gemm.gemm.GemmFwdOp`:
    ``m``/``n``/``k`` and the dtype are bound on the first ``forward()`` so the
    manifest's func-mode roofline can read them, and the kernel is built and
    cached per ``(m, n, k, dtype, trans_b)``.
    """

    def __init__(
        self,
        trans_a: bool = False,
        trans_b: bool = True,
        tune: bool = False,
        target: "Target" = None,
    ) -> None:
        """Build the op.  Shapes and dtype are taken from the first call.

        Args:
            trans_a: Must be ``False``; the split-K partial kernel has no
                transposed-A path.
            trans_b: Whether ``b`` is stored $[N \\times K]$.  Default ``True``
                (NT).  ``False`` is the PDF's $[K \\times N]$.
            tune: Whether to autotune (applied when the kernel is first built).
            target: Which set of kernels serves this op, or ``None`` to decide
                from the input device.  ``detect_target`` returns ``None`` for
                an ``npu`` device in this distribution, so callers on Ascend
                pass ``target="ascend"`` explicitly.
        """
        if trans_a:
            raise ValueError(
                "GemmSplitKFwdOp does not support trans_a=True: the split-K "
                "partial kernel is the non-transposed-A pipelined path only "
                "(see tileops/kernels/gemm_splitk.py)"
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

    #: The manifest's declared dtype union for ``a``.  Checked at the op layer,
    #: not left to the kernel: ``scripts/validate_manifest.py``'s STRICT-PARITY
    #: pass flags an op whose ``_validate_dtypes`` accepts an out-of-union dtype
    #: (it probes with uint8), and "the kernel would have raised later" is not
    #: the same contract.
    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    def _validate_dtypes(self, a: torch.Tensor, b: torch.Tensor) -> None:
        if a.dtype not in self._SUPPORTED_DTYPES:
            raise TypeError(
                f"GemmSplitKFwdOp supports torch.float16/torch.bfloat16, got "
                f"{a.dtype}"
            )
        if a.dtype != b.dtype:
            raise TypeError(
                f"GemmSplitKFwdOp operands must share a dtype, got {a.dtype} "
                f"and {b.dtype}"
            )

    def _infer_mnk(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[int, int, int]:
        m, k_a = a.shape[0], a.shape[1]
        n, k_b = (b.shape[0], b.shape[1]) if self.trans_b else (b.shape[1], b.shape[0])
        if k_a != k_b:
            raise ValueError(
                f"GEMM contraction dim mismatch: a contributes K={k_a}, b "
                f"contributes K={k_b} (a.shape={tuple(a.shape)}, "
                f"b.shape={tuple(b.shape)}, trans_b={self.trans_b})"
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
    ) -> dict[str, tuple[int, ...]]:
        n = b_shape[0] if self.trans_b else b_shape[1]
        return {"c": (a_shape[0], n)}

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply the two matrices with a K-partitioned schedule.

        Args:
            a: Left operand, $[M \\times K]$.
            b: Right operand, $[N \\times K]$ under the default NT layout, or
                $[K \\times N]$ when ``trans_b`` is false.

        Returns:
            The product, $[M \\times N]$, in the dtype of the inputs.

        Example:
            ```python linenums="1"
            op = GemmSplitKFwdOp(target="ascend")
            c = op(a, b)
            op.split_k          # what the planner chose for this shape
            ```
        """
        sig = (a.shape, b.shape, a.dtype)
        if sig != self._active_sig:
            self._validate_dtypes(a, b)
            m, n, k = self._infer_mnk(a, b)
            self.m, self.n, self.k = m, n, k
            self.dtype = a.dtype
            kernel: Kernel = self.get_or_build_kernel("gemm_splitk_kernel", (a, b))
            self.kernel = kernel
            # The planner's decision, exposed so a benchmark can report how many
            # workloads actually took the split path instead of assuming.
            self.split_k = getattr(kernel, "split_k", None)
            self.split_reason = getattr(kernel, "reason", None)
            self._active = (kernel,)
            self._active_sig = sig
        (kernel,) = self._active
        return kernel(a, b)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on the Cube unit.

        Split-K does not change the contraction count, so this is
        ``GemmFwdOp``'s roof -- which is also why the two ops share a roofline
        formula and are comparable case by case.  What split-K does change is
        the BYTES: the fp32 partials workspace is written once and read once
        more, and that traffic is deliberately not in the roofline, because the
        roofline prices the op's semantics and not one schedule's scratch.
        """
        return cube_roof(self.dtype)
