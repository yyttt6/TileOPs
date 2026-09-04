
import torch


from .op_base import Op
from tileops.backend import Kernel

__all__ = ["TopkSelectorFwdOp"]


class TopkSelectorFwdOp(Op):
    def __init__(self, topk: int, tune: bool = False) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            topk: Manifest ``params.topk``, ``int``.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.batch = None
        self.seq_len = None
        self.seq_len_kv = None
        self.kv_group = None
        self.topk = topk
        self.in_dtype = None
        self.out_dtype = torch.int32
        self.tune = tune

        self.dispatch_kernel()
        self.kernel = None


    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        batch: int,
        seq_len: int,
        seq_len_kv: int,
        kv_group: int,
        in_dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        return self.get_or_build_kernel("topk_selector_kernel", inputs)

    def _infer_output_shapes(
        self,
        index_score_shape: tuple[int, ...],
        starts_shape: tuple[int, ...],
        ends_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: $[batch \\times seq\\_len \\times kv\\_group \\times topk]$."""
        batch, seq_len, _, kv_group = index_score_shape
        return {"indexes": (batch, seq_len, kv_group, self.topk)}

    def forward(self, index_score, starts, ends) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            index_score: Input tensor, dtype ``float32``.
            starts: Input tensor, dtype ``int32``.
            ends: Input tensor, dtype ``int32``.

        Returns:
            ``indexes``, as the manifest declares.
        """
        if index_score.device.type != "npu":
            raise ValueError("TopkSelectorFwdOp expects NPU inputs")
        if index_score.ndim != 4:
            raise ValueError("TopkSelectorFwdOp expects index_score shape [B, S, S_kv, G]")
        if starts.ndim != 2 or ends.ndim != 2:
            raise ValueError("TopkSelectorFwdOp expects starts/ends shape [B, S]")
        if starts.device.type != "npu" or ends.device.type != "npu":
            raise ValueError("starts and ends must be NPU tensors")
        if starts.dtype != torch.int32 or ends.dtype != torch.int32:
            raise ValueError("TopkSelectorFwdOp expects int32 starts/ends tensors")

        batch, seq_len, seq_len_kv, kv_group = index_score.shape
        if starts.shape != (batch, seq_len) or ends.shape != (batch, seq_len):
            raise ValueError("TopkSelectorFwdOp starts/ends must match index_score batch/seq_len")
        if not 0 < self.topk <= seq_len_kv:
            raise ValueError(f"topk must satisfy 0 < topk <= seq_len_kv={seq_len_kv}")

        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.in_dtype = index_score.dtype
        self.kernel = self._get_kernel(
            (index_score, starts, ends),
            batch,
            seq_len,
            seq_len_kv,
            kv_group,
            index_score.dtype,
            index_score.device.index,
        )

        return self.kernel(index_score, starts, ends)
