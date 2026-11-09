"""Ascend selective State Space Model (SSM) recurrence with packed-sequence resets."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.mamba.call_spec import SelectiveScanFwdInterface

__all__ = ["AscendSelectiveScanKernel"]


@functools.lru_cache(maxsize=64)
def _scan(dim, length, states, groups, dtype, skip, gate, bias, segmented, initial, softplus):
    batch = T.dynamic("batch")
    # Each lane owns a channel's state, avoiding global intermediate state histories.
    blocks = 64

    @T.prim_func
    def main(
        U: T.Tensor((batch, dim, length), dtype),
        Delta: T.Tensor((batch, dim, length), dtype),
        A: T.Tensor((dim, states), "float32"),
        B: T.Tensor((batch, groups, states, length), dtype),
        C: T.Tensor((batch, groups, states, length), dtype),
        D: T.Tensor((dim,), "float32"),
        Z: T.Tensor((batch, dim, length), dtype),
        Bias: T.Tensor((dim,), "float32"),
        Seq: T.Tensor((batch, length), "int32"),
        Init: T.Tensor((batch, dim, states), "float32"),
        Y: T.Tensor((batch, dim, length), dtype),
        Final: T.Tensor((batch, dim, states), "float32"),
    ):
        with T.Kernel(blocks) as bx, T.SimtVF(threads=256):
            for step in T.serial(T.ceildiv(batch * dim, blocks * 256)):
                for j in T.Parallel(256, coalesced_width=T.int32(1)):
                    channel = (step * blocks + bx) * 256 + j
                    if channel < batch * dim:
                        b, d = channel // dim, channel % dim
                        g = d // (dim // groups)
                        state = T.alloc_local((states,), "float32")
                        decay = T.alloc_local((states,), "float32")
                        for n in T.unroll(states):
                            state[n] = 0
                            if initial:
                                state[n] = Init[b, d, n]
                            decay[n] = A[d, n]
                        for t in T.serial(length):
                            if segmented and t > 0 and Seq[b, t] != Seq[b, T.max(0, t - 1)]:
                                for n in T.unroll(states):
                                    state[n] = 0
                            dt = T.cast(Delta[b, d, t], "float32")
                            if bias:
                                dt = dt + Bias[d]
                            if softplus:
                                dt = T.if_then_else(dt > 20, dt, T.log(1 + T.exp(T.min(dt, 20))))
                            u = T.cast(U[b, d, t], "float32")
                            acc = T.alloc_local((1,), "float32")
                            acc[0] = 0
                            for n in T.unroll(states):
                                state[n] = T.exp(dt * decay[n]) * state[n] + dt * u * T.cast(
                                    B[b, g, n, t], "float32"
                                )
                                acc[0] = acc[0] + state[n] * T.cast(C[b, g, n, t], "float32")
                            if skip:
                                acc[0] = acc[0] + D[d] * u
                            if gate:
                                z = T.cast(Z[b, d, t], "float32")
                                acc[0] = acc[0] * z / (1 + T.exp(-z))
                            Y[b, d, t] = T.cast(acc[0], dtype)
                        for n in T.unroll(states):
                            Final[b, d, n] = state[n]

    return tilelang.compile(main, target="ascend", out_idx=[10, 11])


class AscendSelectiveScanKernel(Kernel, SelectiveScanFwdInterface):
    """Scan recurrent state without materializing an entire time-by-state tensor."""

    devices = frozenset({"npu"})

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self._run = _scan(
            call.dim,
            call.seqlen,
            call.dstate,
            call.groups,
            self.dtype_to_str(call.dtype),
            call.has_skip,
            call.has_gate,
            call.has_bias,
            call.has_seq_idx,
            call.has_init,
            call.softplus,
        )
        self._dim, self._states = call.dim, call.dstate
        self._float = torch.empty(call.dim, dtype=torch.float32, device=call.device)

    def forward(
        self,
        u,
        delta,
        A,
        B_in,
        C_in,
        D_in=None,
        z=None,
        delta_bias=None,
        seq_idx=None,
        initial_states=None,
    ):
        if seq_idx is None:
            seq_idx = torch.empty((u.shape[0], u.shape[2]), dtype=torch.int32, device=u.device)
        if initial_states is None:
            initial_states = torch.empty(
                (u.shape[0], self._dim, self._states), dtype=torch.float32, device=u.device
            )
        return self._run(
            u,
            delta,
            A,
            B_in,
            C_in,
            self._float if D_in is None else D_in,
            u if z is None else z,
            self._float if delta_bias is None else delta_bias,
            seq_idx,
            initial_states,
        )


register_kernel_type("SelectiveScanFwdOp", "ascend", AscendSelectiveScanKernel)
