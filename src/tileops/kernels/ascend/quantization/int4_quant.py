"""Ascend asymmetric INT4 group quantization with high-nibble-first packing."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.ascend.quantization.int4_groups import build_int4_groups
from tileops.kernels.ascend.quantization.int4_stream import build_int4_stream
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.quantization.call_spec import INT4QuantPerGroupFwdInterface, QuantizeCall

__all__ = ["AscendINT4QuantPerGroupKernel"]


@functools.lru_cache(maxsize=64)
def _build(rows: int, k: int, group_size: int):
    groups = rows * k // group_size
    cols = max(64, 1 << (group_size - 1).bit_length())
    blocks = min(64, groups)

    @T.prim_func
    def main(
        W: T.Tensor((rows * k,), "float16"),
        Q: T.Tensor((rows * k // 2,), "int8"),
        Params: T.Tensor((groups, 2), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            values = T.alloc_shared((cols,), "float32")
            packed = T.alloc_shared((cols // 2,), "int8")
            parameters = T.alloc_shared((3,), "float32")
            for step in T.serial(T.ceildiv(groups, blocks)):
                group = step * blocks + bx
                if group < groups:
                    with T.SimtVF(threads=256):
                        for j in T.Parallel(cols, coalesced_width=T.int32(1)):
                            # Repeating the first valid element leaves both extrema unchanged.
                            values[j] = T.cast(
                                W[group * group_size + T.min(j, group_size - 1)], "float32"
                            )
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        lo = T.simd.alloc_local((1,), "float32")
                        hi = T.simd.alloc_local((1,), "float32")
                        lo[0] = T.simd.vcmin(T.simd.vld(values[0]), full)
                        hi[0] = T.simd.vcmax(T.simd.vld(values[0]), full)
                        for v in T.serial(1, cols // 64):
                            lo[0] = T.simd.vmin(
                                lo[0], T.simd.vcmin(T.simd.vld(values[v * 64]), full), full
                            )
                            hi[0] = T.simd.vmax(
                                hi[0], T.simd.vcmax(T.simd.vld(values[v * 64]), full), full
                            )
                        T.simd.vsts(parameters[0], lo[0], one, "ONEPT_B32")
                        T.simd.vsts(parameters[1], hi[0], one, "ONEPT_B32")
                    with T.SimtVF(threads=32):
                        for j in T.Parallel(1):
                            parameters[j + 2] = T.if_then_else(
                                parameters[0] == parameters[1],
                                T.float32(16),
                                parameters[1] - parameters[0],
                            )
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        low = T.simd.vdup(parameters[0], "float32")
                        high = T.simd.vdup(parameters[1], "float32")
                        offset = T.simd.vmul(
                            T.simd.vadd(low, high, full),
                            T.simd.vdup(T.float32(0.5), "float32"),
                            full,
                        )
                        denominator = T.simd.vdup(parameters[2], "float32")
                        multiplier = T.simd.vdiv(
                            T.simd.vdup(T.float32(16), "float32"),
                            denominator,
                            full,
                            precision="exact",
                        )
                        scale = T.simd.vdiv(
                            T.simd.vdup(T.float32(1), "float32"),
                            multiplier,
                            full,
                            precision="exact",
                        )
                        T.simd.vsts(parameters[0], scale, one, "ONEPT_B32")
                        T.simd.vsts(parameters[1], offset, one, "ONEPT_B32")
                        for v in T.serial(cols // 64):
                            scaled = T.simd.vmul(
                                T.simd.vsub(T.simd.vld(values[v * 64]), offset, full),
                                multiplier,
                                full,
                            )
                            T.simd.vsts(values[v * 64], scaled, full)
                    with T.SimtVF(threads=256):
                        for j in T.Parallel(cols, coalesced_width=T.int32(1)):
                            value = values[j]
                            lower = T.floor(value)
                            rem = value - lower
                            increment = (rem > T.float32(0.5)) | (
                                (rem == T.float32(0.5)) & (T.cast(lower, "int32") % 2 != 0)
                            )
                            values[j] = T.max(
                                T.float32(-8),
                                T.min(
                                    T.float32(7),
                                    lower + T.if_then_else(increment, T.float32(1), T.float32(0)),
                                ),
                            )
                    with T.SimtVF(threads=256):
                        for j in T.Parallel(cols // 2, coalesced_width=T.int32(1)):
                            high = T.cast(values[j * 2], "int32") & 15
                            low = T.cast(values[j * 2 + 1], "int32") & 15
                            packed[j] = T.cast((high << 4) | low, "int8")
                    T.copy(
                        packed[: group_size // 2],
                        Q[group * group_size // 2 : (group + 1) * group_size // 2],
                    )
                    T.copy(parameters[:2], Params[group, :])

    return tilelang.compile(main, target="ascend", out_idx=[1, 2])


class AscendINT4QuantPerGroupKernel(Kernel, INT4QuantPerGroupFwdInterface):
    """Quantize groups against their min/max and pack adjacent signed codes."""

    devices = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: QuantizeCall) -> str | None:
        reason = super().refusal(call)
        if reason is None and call.group_size % 2:
            return "Ascend INT4 packing requires an even group_size"
        return reason

    def __init__(self, call: QuantizeCall):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self._run = (
            build_int4_groups(call.rows * call.cols, call.group_size)
            if 64 <= call.group_size <= 8192 and 8192 % call.group_size == 0 and call.rows
            else build_int4_stream(call.rows * call.cols, call.group_size)
            if call.group_size > 8192 and call.rows
            else _build(call.rows, call.cols, call.group_size)
        )

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_devices(w=w)
        codes, parameters = self._run(w.view(-1))
        return codes.view(self.call.rows, self.call.cols // 2), parameters.view(-1, 2)


register_kernel_type("INT4QuantPerGroupFwdOp", "ascend", AscendINT4QuantPerGroupKernel)
