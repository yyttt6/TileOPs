"""Gather routed rows and reduce weighted expert outputs on Ascend vector cores."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.ascend.moe.pre_permute import route_pass_configs
from tileops._kernels.buffer_utils import tensors_overlap
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.moe.call_spec import PostPermuteCall, PostPermuteFwdInterface
from tileops.backend._registry import register_kernel_type


@functools.lru_cache(maxsize=32)
def _build(tokens, top_k, hidden, rows, dtype, out_dtype, scale):
    # Bound tile storage even for large top-k. Split single-token decode across cores.
    channels = min(1024, max(64, ((hidden + 63) // 64) * 64))
    if tokens < 32:
        channels = min(channels, max(64, ((hidden * tokens + 4095) // 4096) * 64))
    channels = min(channels, max(64, (32768 // max(1, top_k) // 64) * 64))
    tiles = (hidden + channels - 1) // channels
    routes = min(top_k, 32)
    blocks = (top_k + routes - 1) // routes

    @T.prim_func
    def main(
        experts: T.Tensor((rows, hidden), dtype),
        inverse: T.Tensor((tokens * top_k,), "int32"),
        weights: T.Tensor((tokens, top_k), "float32"),
        output: T.Tensor((tokens, hidden), out_dtype),
    ):
        with T.Kernel(tokens * tiles) as bx:
            token = bx // tiles
            start = bx % tiles * channels
            count = T.min(hidden - start, channels)
            ids = T.alloc_shared((routes,), "int32")
            w = T.alloc_shared((routes,), "float32")
            x = T.alloc_shared((routes, channels), dtype)
            sums = T.alloc_shared((channels,), "float32")
            y = T.alloc_shared((channels,), out_dtype)
            for block in T.serial(blocks):
                first = block * routes
                nr = T.min(routes, top_k - first)
                T.copy(inverse[token * top_k + first : token * top_k + first + nr], ids[:nr])
                T.copy(weights[token, first : first + nr], w[:nr])
                # The scalar address calculation consumes the just-loaded row IDs.
                T.ascend_set_flag("MTE2_S", 0)
                T.ascend_wait_flag("MTE2_S", 0)
                for route in T.serial(nr):
                    T.copy(experts[ids[route], start : start + count], x[route, :count])
                T.ascend_set_flag("MTE2_V", 0)
                T.ascend_wait_flag("MTE2_V", 0)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for v in T.serial(channels // 64):
                        acc = T.simd.alloc_local((1,), "float32")
                        if block == 0:
                            acc[0] = T.simd.vdup(T.float32(0), "float32", mask)
                        else:
                            acc[0] = T.simd.vld(sums[v * 64])
                        for route in T.serial(nr):
                            value = T.simd.vcvt(
                                T.simd.vld(x[route, v * 64], dist="UNPK_B16"), "float32", part=0
                            )
                            weight = T.simd.vld(w[route], dist="BRC_B32")
                            acc[0] = T.simd.vadd(acc[0], T.simd.vmul(weight, value, mask), mask)
                        if block == blocks - 1:
                            value = T.simd.vmuls(acc[0], T.float32(scale), mask)
                            narrowed = T.simd.vcvt(
                                value, out_dtype, round="ROUND_R", sat=False, part=0
                            )
                            T.simd.vsts(y[v * 64], narrowed, mask, dist="PK_B32")
                        else:
                            T.simd.vsts(sums[v * 64], acc[0], mask)
                if block + 1 < blocks:
                    T.ascend_set_flag("V_MTE2", 0)
                    T.ascend_wait_flag("V_MTE2", 0)
            T.ascend_set_flag("V_MTE3", 0)
            T.ascend_wait_flag("V_MTE3", 0)
            T.copy(y[:count], output[token, start : start + count])

    return tilelang.compile(main, target="ascend", out_idx=[], pass_configs=route_pass_configs())


@functools.lru_cache(maxsize=32)
def _batched(tokens, top_k, hidden, rows, dtype, out_dtype, scale):
    # Gather one block of route metadata per core and write the reduced tokens together.
    batch = 8 if hidden <= 256 else 1
    width = hidden
    cores = tokens // batch

    @T.prim_func
    def main(
        experts: T.Tensor((rows, hidden), dtype),
        inverse: T.Tensor((tokens * top_k,), "int32"),
        weights: T.Tensor((tokens, top_k), "float32"),
        output: T.Tensor((tokens, hidden), out_dtype),
    ):
        with T.Kernel(cores) as bx:
            ids = T.alloc_shared((batch * top_k,), "int32")
            ws = T.alloc_shared((batch, top_k), "float32")
            x = T.alloc_shared((batch, top_k, width), dtype)
            y = T.alloc_shared((batch, width), out_dtype)
            first = bx * batch
            T.copy(inverse[first * top_k : (first + batch) * top_k], ids)
            T.copy(weights[first : first + batch, :], ws)
            T.ascend_set_flag("MTE2_S", 0)
            T.ascend_wait_flag("MTE2_S", 0)
            for i, route in T.grid(batch, top_k):
                row = ids[i * top_k + route]
                T.copy(experts[row, :], x[i, route, :])
            T.ascend_set_flag("MTE2_V", 0)
            T.ascend_wait_flag("MTE2_V", 0)
            with T.SimdVF():
                mask = T.simd.pset(32)
                for i in T.serial(batch):
                    for v in T.serial(width // 64):
                        acc = T.simd.alloc_local((1,), "float32")
                        acc[0] = T.simd.vdup(T.float32(0), "float32", mask)
                        for route in T.unroll(top_k, explicit=True):
                            weight = T.simd.vld(ws[i, route], dist="BRC_B32")
                            value = T.simd.vcvt(
                                T.simd.vld(x[i, route, v * 64], dist="UNPK_B16"), "float32", part=0
                            )
                            acc[0] = T.simd.vadd(acc[0], T.simd.vmul(weight, value, mask), mask)
                        scaled = T.simd.vmuls(acc[0], T.float32(scale), mask)
                        narrowed = T.simd.vcvt(
                            scaled, out_dtype, round="ROUND_R", sat=False, part=0
                        )
                        T.simd.vsts(y[i, v * 64], narrowed, mask, dist="PK_B32")
            T.ascend_set_flag("V_MTE3", 0)
            T.ascend_wait_flag("V_MTE3", 0)
            T.copy(y, output[first : first + batch, :])

    return tilelang.compile(main, target="ascend", out_idx=[], pass_configs=route_pass_configs())


@functools.lru_cache(maxsize=32)
def _unaligned(tokens, top_k, hidden, rows, dtype, out_dtype, scale):
    """Gather individual elements when a row starts between DMA alignment boundaries."""
    tile = 256

    @T.prim_func
    def main(
        experts: T.Tensor((rows, hidden), dtype),
        inverse: T.Tensor((tokens * top_k,), "int32"),
        weights: T.Tensor((tokens, top_k), "float32"),
        output: T.Tensor((tokens, hidden), out_dtype),
    ):
        with T.Kernel(T.ceildiv(tokens * hidden, tile)) as bx:  # noqa: SIM117
            with T.SimtVF(threads=128):
                for j in T.Parallel(tile):
                    idx = bx * tile + j
                    if idx < tokens * hidden:
                        token = idx // hidden
                        col = idx % hidden
                        acc = T.alloc_local((1,), "float32")
                        acc[0] = 0
                        for k in T.serial(top_k):
                            row = inverse[token * top_k + k]
                            acc[0] += weights[token, k] * T.cast(experts[row, col], "float32")
                        output[token, col] = T.cast(acc[0] * scale, out_dtype)

    return tilelang.compile(main, target="ascend", out_idx=[])


class AscendMoEPostPermuteKernel(Kernel, PostPermuteFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: PostPermuteCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: PostPermuteCall):
        super().__init__(device_index=call.device.index)
        self.shape = (call.num_tokens, call.hidden_size)
        self.dtype = call.output_dtype
        self.top_k = call.top_k
        self._args = None
        if call.num_tokens and call.top_k:
            self._args = (
                call.num_tokens,
                call.top_k,
                call.hidden_size,
                call.materialized_rows,
                str(call.input_dtype).removeprefix("torch."),
                self.dtype_str,
                call.epilogue.routed_scaling_factor,
            )

    def forward(self, expert_output, inverse_indices, topk_weights, out=None):
        if out is None:
            out = torch.empty(self.shape, dtype=self.dtype, device=expert_output.device)
        else:
            if tuple(out.shape) != self.shape or out.dtype != self.dtype:
                raise ValueError("out shape or dtype does not match the reduced output")
            if out.device != expert_output.device or not out.is_contiguous():
                raise ValueError("out must be contiguous on the input device")
            for name, tensor in (
                ("expert_output", expert_output),
                ("inverse_indices", inverse_indices),
                ("topk_weights", topk_weights),
            ):
                if tensors_overlap(out, tensor):
                    raise ValueError(f"out must not overlap {name} in memory")
        if self._args is not None:
            unaligned = self.shape[1] % 16 or any(
                t.data_ptr() % 32 for t in (expert_output, inverse_indices, topk_weights, out)
            )
            tokens, hidden = self.shape
            batched = (hidden <= 256 and tokens >= 128 and tokens % 8 == 0) or (
                hidden > 256 and tokens >= 64
            )
            if unaligned:
                build = _unaligned
            elif (
                batched
                and hidden % 64 == 0
                and self.top_k <= 8
                and 2 * (self.top_k + 1) * hidden <= 160 * 1024
            ):
                build = _batched
            else:
                build = _build
            run = build(*self._args)
            run(expert_output.view(-1, self.shape[1]), inverse_indices, topk_weights, out)
        elif self.top_k == 0:
            out.zero_()
        return out


register_kernel_type("MoEPostPermuteFwdOp", "ascend", AscendMoEPostPermuteKernel)
