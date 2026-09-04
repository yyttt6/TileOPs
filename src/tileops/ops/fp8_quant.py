from typing import Tuple

import torch


from .op_base import Op
from tileops.backend import Kernel

__all__ = ["FP8QuantFwdOp"]


class FP8QuantFwdOp(Op):
    def __init__(self, tune: bool = False):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.batch = None
        self.seq_len_kv = None
        self.kv_group = None
        self.index_dim = None
        self.in_dtype = None
        self.tune = tune
        self.dispatch_kernel()
        self.kernel = None


    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        batch: int,
        seq_len_kv: int,
        kv_group: int,
        index_dim: int,
        in_dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        return self.get_or_build_kernel("fp8_quant_kernel", inputs)

    def _infer_output_shapes(
        self,
        input_tensor_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: one scale per row, and the quantized tensor itself."""
        return {
            "scale_tensor": tuple(input_tensor_shape[:-1]),
            "output_tensor": tuple(input_tensor_shape),
        }

    def forward(self, input_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            input_tensor: Input tensor, dtype ``float16 | bfloat16 | float32``.

        Returns:
            ``scale_tensor``, ``output_tensor``, as the manifest declares.
        """
        if input_tensor.device.type != "npu":
            raise ValueError("FP8QuantFwdOp expects an NPU input tensor")
        if input_tensor.ndim != 4:
            raise ValueError("FP8QuantFwdOp expects input_tensor shape [B, S, G, D]")
        if input_tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"FP8QuantFwdOp does not support dtype {input_tensor.dtype}")

        batch, seq_len_kv, kv_group, index_dim = input_tensor.shape
        if min(batch, seq_len_kv, kv_group, index_dim) <= 0:
            raise ValueError("FP8QuantFwdOp input dimensions must be positive")

        self.batch = batch
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.index_dim = index_dim
        self.in_dtype = input_tensor.dtype
        self.kernel = self._get_kernel(
            (input_tensor),
            batch,
            seq_len_kv,
            kv_group,
            index_dim,
            input_tensor.dtype,
            input_tensor.device.index,
        )
        return self.kernel(input_tensor)
