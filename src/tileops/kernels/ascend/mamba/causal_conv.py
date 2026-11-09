"""Ascend causal depthwise convolution over packed sequences and decode states."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.mamba.call_spec import CausalConv1dFwdInterface, CausalConv1dUpdateFwdInterface

__all__ = ["AscendCausalConv1dKernel", "AscendCausalConv1dUpdateKernel"]


@functools.lru_cache(maxsize=64)
def _prefill_gather(dim, width, dtype, bias, initial, silu):
    total, batch = T.dynamic("total"), T.dynamic("batch")
    # Bound the per-core output tile; sequences and their offsets stay on the device.
    tile = 2048
    blocks = 64

    @T.prim_func
    def main(
        X: T.Tensor((total, dim), dtype),
        W: T.Tensor((dim, width), dtype),
        State: T.Tensor((batch, dim, width - 1), dtype),
        Offsets: T.Tensor((batch + 1,), "int32"),
        Bias: T.Tensor((dim,), dtype),
        Initial: T.Tensor((batch,), "bool"),
        Y: T.Tensor((total * dim,), dtype),
    ):
        with T.Kernel(blocks) as bx:
            out = T.alloc_shared((tile,), dtype)
            for step in T.serial(T.ceildiv(total * dim, blocks * tile)):
                start = (step * blocks + bx) * tile
                if start < total * dim:
                    with T.SimtVF(threads=256):
                        for j in T.Parallel(tile, coalesced_width=T.int32(1)):
                            index = start + j
                            if index < total * dim:
                                t, d = index // dim, index % dim
                                sequence = T.alloc_local((1,), "int32")
                                sequence[0] = 0
                                for s in T.serial(batch):
                                    if t >= Offsets[s] and t < Offsets[s + 1]:
                                        sequence[0] = s
                                begin = Offsets[sequence[0]]
                                acc = T.alloc_local((1,), "float32")
                                acc[0] = 0
                                for k in T.unroll(width):
                                    pos = t - width + 1 + k
                                    value = T.alloc_local((1,), "float32")
                                    value[0] = 0
                                    if pos >= begin:
                                        value[0] = T.cast(X[pos, d], "float32")
                                    elif initial and Initial[sequence[0]]:
                                        value[0] = T.cast(
                                            State[sequence[0], d, width - 1 + pos - begin],
                                            "float32",
                                        )
                                    acc[0] = acc[0] + value[0] * T.cast(W[d, k], "float32")
                                if bias:
                                    acc[0] = acc[0] + T.cast(Bias[d], "float32")
                                if silu:
                                    acc[0] = acc[0] / (1 + T.exp(-acc[0]))
                                out[j] = T.cast(acc[0], dtype)
                    count = T.min(tile, total * dim - start)
                    T.copy(out[:count], Y[start : start + count])

    return tilelang.compile(main, target="ascend", out_idx=[6])


@functools.lru_cache(maxsize=64)
def _prefill(dim, width, dtype, bias, initial, silu):
    total, batch = T.dynamic("total"), T.dynamic("batch")
    # Reuse each input halo and weight tile in UB instead of issuing scalar GM
    # reads for every tap of every output. Thirty-two tokens amortize DMA and VF setup.
    tokens, channels = 32, 256
    channel_tiles = (dim + channels - 1) // channels
    blocks = 64

    @T.prim_func
    def main(
        X: T.Tensor((total, dim), dtype),
        W: T.Tensor((dim, width), dtype),
        State: T.Tensor((batch, dim, width - 1), dtype),
        Offsets: T.Tensor((batch + 1,), "int32"),
        Bias: T.Tensor((dim,), dtype),
        Initial: T.Tensor((batch,), "bool"),
        Y: T.Tensor((total, dim), dtype),
    ):
        with T.Kernel(blocks) as bx:
            data = T.alloc_shared((tokens + width - 1, channels), dtype)
            weights = T.alloc_shared((channels, width), dtype)
            out = T.alloc_shared((tokens, channels), dtype)
            metadata = T.alloc_shared((tokens, 3), "int32")
            bias_values = T.alloc_shared((channels,), dtype)
            for step in T.serial(T.ceildiv(T.ceildiv(total, tokens) * channel_tiles, blocks)):
                work = step * blocks + bx
                if work < T.ceildiv(total, tokens) * channel_tiles:
                    first = work // channel_tiles * tokens
                    channel = work % channel_tiles * channels
                    input_first = T.max(0, first - width + 1)
                    input_count = T.min(total, first + tokens) - input_first
                    channel_count = T.min(channels, dim - channel)
                    # Every channel of a token shares its sequence boundary.
                    with T.SimtVF(threads=32):
                        for i in T.Parallel(tokens):
                            # Initialize every lane, including the unused tail, so shared
                            # metadata synchronization never sits in a divergent branch.
                            t = T.min(first + i, total - 1)
                            sequence = T.alloc_local((1,), "int32")
                            sequence[0] = 0
                            for s in T.serial(batch):
                                if t >= Offsets[s] and t < Offsets[s + 1]:
                                    sequence[0] = s
                            metadata[i, 0] = sequence[0]
                            metadata[i, 1] = Offsets[sequence[0]]
                            metadata[i, 2] = T.cast(Initial[sequence[0]], "int32") if initial else 0
                    if bias:
                        T.copy(Bias[channel : channel + channel_count], bias_values[:channel_count])
                    T.copy(
                        X[
                            input_first : input_first + input_count,
                            channel : channel + channel_count,
                        ],
                        data[:input_count, :channel_count],
                    )
                    T.copy(W[channel : channel + channel_count, :], weights[:channel_count, :])
                    with T.SimtVF(threads=256):
                        for i, j in T.Parallel(tokens, channels, coalesced_width=T.int32(1)):
                            t, d = first + i, channel + j
                            if t < total and d < dim:
                                sequence = metadata[i, 0]
                                begin = metadata[i, 1]
                                acc = T.alloc_local((1,), "float32")
                                acc[0] = 0
                                for k in T.unroll(width):
                                    pos = t - width + 1 + k
                                    value = T.alloc_local((1,), "float32")
                                    value[0] = 0
                                    if pos >= begin:
                                        value[0] = T.cast(data[pos - input_first, j], "float32")
                                    elif initial and metadata[i, 2] != 0:
                                        value[0] = T.cast(
                                            State[sequence, d, width - 1 + pos - begin],
                                            "float32",
                                        )
                                    acc[0] = acc[0] + value[0] * T.cast(weights[j, k], "float32")
                                if bias:
                                    acc[0] = acc[0] + T.cast(bias_values[j], "float32")
                                if silu:
                                    acc[0] = acc[0] / (1 + T.exp(-acc[0]))
                                out[i, j] = T.cast(acc[0], dtype)
                    count = T.min(tokens, total - first)
                    T.copy(
                        out[:count, :channel_count],
                        Y[first : first + count, channel : channel + channel_count],
                    )

    return tilelang.compile(main, target="ascend", out_idx=[6])


@functools.lru_cache(maxsize=64)
def _finish_state(dim, width, dtype, initial):
    total, batch = T.dynamic("total"), T.dynamic("batch")

    @T.prim_func
    def main(
        X: T.Tensor((total, dim), dtype),
        State: T.Tensor((batch, dim, width - 1), dtype),
        Offsets: T.Tensor((batch + 1,), "int32"),
        Initial: T.Tensor((batch,), "bool"),
    ):
        with T.Kernel(64) as bx, T.SimtVF(threads=256):
            for step in T.serial(T.ceildiv(batch * dim, 64 * 256)):
                for j in T.Parallel(256, coalesced_width=T.int32(1)):
                    i = (step * 64 + bx) * 256 + j
                    if i < batch * dim:
                        s, d = i // dim, i % dim
                        begin, end = Offsets[s], Offsets[s + 1]
                        if end > begin:
                            history = T.alloc_local((width - 1,), dtype)
                            for k in T.unroll(width - 1):
                                pos = end - width + 1 + k
                                value = T.alloc_local((1,), dtype)
                                value[0] = T.cast(0, dtype)
                                if pos >= begin:
                                    value[0] = X[pos, d]
                                elif initial and Initial[s]:
                                    value[0] = State[s, d, width - 1 + pos - begin]
                                history[k] = value[0]
                            for k in T.unroll(width - 1):
                                State[s, d, k] = history[k]

    return tilelang.compile(main, target="ascend", out_idx=[])


@functools.lru_cache(maxsize=64)
def _update(dim, width, dtype, bias, silu):
    batch = T.dynamic("batch")

    @T.prim_func
    def main(
        X: T.Tensor((batch, dim), dtype),
        W: T.Tensor((dim, width), dtype),
        State: T.Tensor((batch, dim, width - 1), dtype),
        Bias: T.Tensor((dim,), dtype),
        Y: T.Tensor((batch, dim), dtype),
    ):
        with T.Kernel(64) as bx, T.SimtVF(threads=256):
            for step in T.serial(T.ceildiv(batch * dim, 64 * 256)):
                for j in T.Parallel(256, coalesced_width=T.int32(1)):
                    i = (step * 64 + bx) * 256 + j
                    if i < batch * dim:
                        b, d = i // dim, i % dim
                        history = T.alloc_local((width,), dtype)
                        acc = T.alloc_local((1,), "float32")
                        acc[0] = 0
                        for k in T.unroll(width - 1):
                            history[k] = State[b, d, k]
                        history[width - 1] = X[b, d]
                        for k in T.unroll(width):
                            acc[0] = acc[0] + T.cast(history[k], "float32") * T.cast(
                                W[d, k], "float32"
                            )
                        if bias:
                            acc[0] = acc[0] + T.cast(Bias[d], "float32")
                        if silu:
                            acc[0] = acc[0] / (1 + T.exp(-acc[0]))
                        Y[b, d] = T.cast(acc[0], dtype)
                        for k in T.unroll(width - 1):
                            State[b, d, k] = history[k + 1]

    return tilelang.compile(main, target="ascend", out_idx=[4])


class AscendCausalConv1dKernel(Kernel, CausalConv1dFwdInterface):
    """Convolve packed sequences before advancing their caller-owned states."""

    devices = frozenset({"npu"})

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        dtype = self.dtype_to_str(call.dtype)
        # DMA rows require aligned half-precision strides; retain scalar gathers
        # for irregular channel counts covered by the public contract.
        prefill = _prefill if call.dim % 16 == 0 else _prefill_gather
        self._run = prefill(call.dim, call.width, dtype, call.has_bias, call.has_init, call.silu)
        self._finish = _finish_state(call.dim, call.width, dtype, call.has_init)
        self._bias = torch.empty(call.dim, dtype=call.dtype, device=call.device)

    def forward(self, x, weight, conv_state, cu_seqlens, bias=None, has_initial_state=None):
        flags = has_initial_state
        if flags is None:
            flags = torch.empty(conv_state.shape[0], dtype=torch.bool, device=x.device)
        y = self._run(
            x, weight, conv_state, cu_seqlens, self._bias if bias is None else bias, flags
        )
        self._finish(x, conv_state, cu_seqlens, flags)
        return y.view_as(x)


class AscendCausalConv1dUpdateKernel(Kernel, CausalConv1dUpdateFwdInterface):
    """Advance one causal convolution token per sequence in place."""

    devices = frozenset({"npu"})

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self._run = _update(
            call.dim, call.width, self.dtype_to_str(call.dtype), call.has_bias, call.silu
        )
        self._bias = torch.empty(call.dim, dtype=call.dtype, device=call.device)

    def forward(self, x, weight, conv_state, bias=None):
        return self._run(x, weight, conv_state, self._bias if bias is None else bias)


register_kernel_type("CausalConv1dFwdOp", "ascend", AscendCausalConv1dKernel)
register_kernel_type("CausalConv1dUpdateFwdOp", "ascend", AscendCausalConv1dUpdateKernel)
