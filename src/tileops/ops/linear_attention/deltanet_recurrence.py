from typing import Optional, Tuple

import torch


from ..op_base import Op
from tileops.backend import Kernel

__all__ = ["DeltaNetDecodeFwdOp"]

#: The slot this op asks its target for.
DELTANET_DECODE_SLOT = "deltanet_decode"


class DeltaNetDecodeFwdOp(Op):
    """DeltaNet decode (single-step recurrence, ungated).

    Computes one step of the delta rule (no gate):
        v_new = beta * (v - S @ k)
        o     = S @ q + (q . k) * v_new
        S_new = S + outer(k, v_new)

    Layout: BHD (batch, head, dim).
    Supports float32, float16, and bfloat16 with fp32 accumulation.

    For fp32 dtype, dispatches to a dedicated FP32 kernel that uses
    element-wise matvec instead of T.gemm to avoid TF32 mantissa truncation.
    """

    def __init__(
        self,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.batch = None
        self.heads = None
        self.dim_k = None
        self.dim_v = None
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel()
        self._active_sig: Optional[tuple] = None
        self.kernel = None


    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        del batch, heads, dim_k, dim_v, dtype, device_index  # all on the tensors
        return self.get_or_build_kernel(DELTANET_DECODE_SLOT, inputs)

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        beta_shape: tuple[int, ...],
        state_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        del k_shape, beta_shape
        return {
            "o": (q_shape[0], q_shape[1], v_shape[-1]),
            "new_state": state_shape,
        }

    def _validate_dtypes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> None:
        dtype = q.dtype
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported dtype: {dtype}")
        for name, tensor in (("q", q), ("k", k), ("v", v), ("beta", beta), ("state", state)):
            if tensor.dtype != dtype:
                raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")

    def _validate_shapes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> None:
        if q.ndim != 3:
            raise ValueError("q must have shape [batch, heads, dim_k]")
        batch, heads, dim_k = q.shape
        if v.ndim != 3 or v.shape[:2] != (batch, heads):
            raise ValueError("v must have shape [batch, heads, dim_v]")
        dim_v = v.shape[2]
        q_shape = (batch, heads, dim_k)
        v_shape = (batch, heads, dim_v)
        beta_shape = (batch, heads)
        state_shape = (batch, heads, dim_k, dim_v)
        expected_shapes = (
            ("q", q, q_shape),
            ("k", k, q_shape),
            ("v", v, v_shape),
            ("beta", beta, beta_shape),
            ("state", state, state_shape),
        )
        for name, tensor, expected in expected_shapes:
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
        if not all(tensor.device.type == "npu" for tensor in (q, k, v, beta, state)):
            raise ValueError("q, k, v, beta, and state must be NPU tensors")
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = q.dtype
        self.kernel = self._get_kernel(
            (q, k, v, beta, state), batch, heads, dim_k, dim_v, q.dtype, q.device.index
        )

    def _validate_output_shapes(
        self,
        o: torch.Tensor,
        new_state: torch.Tensor,
    ) -> None:
        o_shape = (self.batch, self.heads, self.dim_v)
        state_shape = (self.batch, self.heads, self.dim_k, self.dim_v)
        if tuple(o.shape) != o_shape:
            raise ValueError(f"o must have shape {o_shape}, got {tuple(o.shape)}")
        if tuple(new_state.shape) != state_shape:
            raise ValueError(
                f"new_state must have shape {state_shape}, got {tuple(new_state.shape)}"
            )

    def eval_roofline(self) -> tuple[int, int]:
        from tileops.perf.formulas import deltanet_decode_roofline

        return deltanet_decode_roofline(self)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            q: Input tensor, dtype ``float16 | bfloat16 | float32``.
            k: Input tensor, dtype ``same_as(q)``.
            v: Input tensor, dtype ``same_as(q)``.
            beta: Input tensor, dtype ``same_as(q)``.
            state: Input tensor, dtype ``same_as(q)``.

        Returns:
            ``o``, ``new_state``, as the manifest declares. Shape rules: ``o.shape == (B, H, DV)``; ``new_state.shape == (B, H, DK, DV)``.
        """
        sig = (
            q.shape,
            k.shape,
            v.shape,
            beta.shape,
            state.shape,
            q.dtype,
            k.dtype,
            v.dtype,
            beta.dtype,
            state.dtype,
            q.device,
            k.device,
            v.device,
            beta.device,
            state.device,
            getattr(self, "tune", None),
        )
        if sig != getattr(self, "_active_sig", None):
            self._validate_dtypes(q, k, v, beta, state)
            self._validate_shapes(q, k, v, beta, state)
            self._active_sig = sig
        o, new_state = self.kernel(q, k, v, beta, state)
        self._validate_output_shapes(o, new_state)
        return o, new_state
