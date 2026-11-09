"""Windowed exact radix selector through the public Ascend interface."""

import torch

import tileops.backend
from tileops.kernels.ascend.attention.radix_select import topk_select_ascend
from tileops.kernels.ascend.attention.simd_select import (
    SIMD_WORKSPACE_LIMIT,
    simd_workspace_bytes,
    topk_select_simd,
)
from tileops.kernels.ascend.attention.streaming_select import (
    STREAMING_WORKSPACE_LIMIT,
    streaming_workspace_bytes,
    topk_select_streaming,
)
from tileops.kernels.attention.topk_select import TopKSelectCall, TopKSelectFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel


class AscendTopKSelectKernel(Kernel, TopKSelectFwdInterface):
    """Exact FP32 selection for the public batch/query/group input contract."""

    devices = frozenset({"npu"})
    general = True

    @classmethod
    def refusal(cls, call: TopKSelectCall) -> str | None:
        if call.dtype != torch.float32 or call.out_dtype != torch.int32:
            return "requires float32 scores and int32 output"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: TopKSelectCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: TopKSelectCall):
        super().__init__()
        self.call = call
        self.rows = call.batch * call.seq_len * call.kv_group
        # Current development uses the 28-core PR card; workers stride over all rows.
        self.program = (
            topk_select_ascend(
                self.rows, call.seq_len_kv, call.topk, min(self.rows, 28), call.kv_group
            )
            if self.rows
            else None
        )

    def forward(self, index_score, starts, ends):
        call = self.call
        output = torch.empty(
            (call.batch, call.seq_len, call.kv_group, call.topk),
            dtype=torch.int32,
            device=index_score.device,
        )
        if self.program is not None:
            self.program(
                index_score.reshape(call.batch * call.seq_len, call.seq_len_kv, call.kv_group),
                starts.reshape(call.batch * call.seq_len),
                ends.reshape(call.batch * call.seq_len),
                output.reshape(self.rows, call.topk),
            )
        return output


class _AscendTopKSelectSIMDKernel(Kernel, TopKSelectFwdInterface):
    """Bind the public score, window and index tensors to a SIMD kernel."""

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: TopKSelectCall) -> str | None:
        if call.dtype != torch.float32 or call.out_dtype != torch.int32:
            return "requires float32 scores and int32 output"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: TopKSelectCall) -> Entry:
        return call, lambda: cls(call)

    def forward(self, index_score, starts, ends):
        call = self.call
        output = torch.empty(
            (call.batch, call.seq_len, call.kv_group, call.topk),
            dtype=torch.int32,
            device=index_score.device,
        )
        if self.program is not None:
            ramp = torch.arange(64, dtype=torch.int32, device=index_score.device)
            self.program(
                index_score.view(torch.uint32).reshape(
                    call.batch * call.seq_len, call.seq_len_kv * call.kv_group
                ),
                starts.reshape(call.batch * call.seq_len),
                ends.reshape(call.batch * call.seq_len),
                ramp.view(torch.uint32),
                output.view(torch.uint32).reshape(self.rows, call.topk),
            )
        return output


class AscendTopKSelectResidentKernel(_AscendTopKSelectSIMDKernel):
    """Select from rows held in the unified buffer."""

    preferred_over = frozenset({"ascend_streaming_early_emit"})

    @classmethod
    def refusal(cls, call: TopKSelectCall) -> str | None:
        if not 0 < call.topk <= call.seq_len_kv < 65536:
            return "resident uint16 histogram requires 0 < K <= N < 65536"
        if simd_workspace_bytes(call.seq_len_kv, call.topk, call.kv_group) > SIMD_WORKSPACE_LIMIT:
            return "resident SIMD plan exceeds its 96 KiB workspace budget"
        return super().refusal(call)

    def __init__(self, call: TopKSelectCall):
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__()
        self.call = call
        self.rows = call.batch * call.seq_len * call.kv_group
        self.program = (
            topk_select_simd(
                self.rows,
                call.seq_len_kv,
                call.topk,
                call.kv_group,
                min(self.rows, 28),
            )
            if self.rows
            else None
        )


class AscendTopKSelectEarlyEmitKernel(_AscendTopKSelectSIMDKernel):
    """Stream long rows, emitting definitive indices during collection."""

    @classmethod
    def refusal(cls, call: TopKSelectCall) -> str | None:
        if not 0 < call.topk <= call.seq_len_kv < 2147483647:
            return "requires positive K <= N within int32 count range"
        if call.kv_group != 1:
            return "early output requires a contiguous single-group row"
        if streaming_workspace_bytes(call.topk) > STREAMING_WORKSPACE_LIMIT:
            return "early output buffers exceed 96 KiB"
        return super().refusal(call)

    def __init__(self, call: TopKSelectCall):
        from tilelang.carver.arch.driver import get_num_cube_cores

        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index)
        self.call = call
        self.rows = call.batch * call.seq_len
        self.workers = 0
        self.program = None
        if self.rows:
            device_id = self.device_index
            if device_id is None:
                device_id = torch.npu.current_device()
            self.workers = min(self.rows, 32, int(get_num_cube_cores(device_id) or 28))
            self.program = topk_select_streaming(
                self.rows,
                call.seq_len_kv,
                call.topk,
                self.workers,
            )


tileops.backend.register_kernel_type("TopKSelectFwdOp", "ascend", AscendTopKSelectKernel)
tileops.backend.register_kernel_type(
    "TopKSelectFwdOp", "ascend_resident", AscendTopKSelectResidentKernel
)
tileops.backend.register_kernel_type(
    "TopKSelectFwdOp", "ascend_streaming_early_emit", AscendTopKSelectEarlyEmitKernel
)
