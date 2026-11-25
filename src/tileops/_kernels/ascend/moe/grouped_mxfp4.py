"""Unpack per-32 e2m1 weight blocks on the vector cores, retaining float32 scales for GEMM."""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.moe.call_spec import MGroupedGEMMMXFP4FwdInterface
from tileops.backend._registry import register_kernel_type


@lru_cache(maxsize=32)
def _weights(n, k, encoded_scale):
    elements = n * k
    tile = 2048
    blocks = min(64, (elements + tile - 1) // tile)

    @T.prim_func
    def main(
        packed: T.Tensor((elements // 2,), "uint8"),
        scales: T.Tensor((elements // 32,), "uint8" if encoded_scale else "float32"),
        out: T.Tensor((elements,), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            values = T.alloc_shared((tile,), "float32")
            for step in T.serial(T.ceildiv(elements, blocks * tile)):
                start = (step * blocks + bx) * tile
                if start < elements:
                    count = T.min(tile, elements - start)
                    with T.SimtVF(threads=256):
                        for lane in T.Parallel(tile):
                            if lane < count:
                                i = start + lane
                                byte = T.cast(packed[i // 2], "int32")
                                code = (byte >> ((i % 2) * 4)) & 15
                                m = code & 7
                                mag = T.if_then_else(
                                    m <= 4,
                                    0.5 * T.cast(m, "float32"),
                                    T.if_then_else(m == 5, 3.0, T.if_then_else(m == 6, 4.0, 6.0)),
                                )
                                if encoded_scale:
                                    exponent = T.cast(scales[i // 32], "uint32")
                                    bits = T.if_then_else(
                                        exponent == 0,
                                        T.uint32(0x00400000),
                                        T.if_then_else(
                                            exponent == 255,
                                            T.uint32(0x7FC00000),
                                            exponent << T.uint32(23),
                                        ),
                                    )
                                    scale = T.reinterpret(bits, "float32")
                                else:
                                    scale = scales[i // 32]
                                values[lane] = T.if_then_else(code >= 8, -mag, mag) * scale
                    T.copy(values[:count], out[start : start + count])

    return tilelang.compile(main, target="ascend", out_idx=[2])


def _activation_scales(t):
    if t.dtype == torch.float32:
        return t
    exponent = t.view(torch.uint8).to(torch.int32)
    bits = torch.where(exponent == 0, 0x00400000, exponent << 23)
    return torch.where(exponent == 255, 0x7FC00000, bits).view(torch.float32)


class AscendMoEGroupedMXFP4Kernel(Kernel, MGroupedGEMMMXFP4FwdInterface):
    devices = frozenset({"npu"})
    supported_archs = None

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(self, a, a_scale, b, b_scale, layout_metadata, out=None):
        e, n, half_k = b.shape
        k = 2 * half_k
        if out is None:
            out = torch.empty((*a.shape[:-1], n), dtype=torch.bfloat16, device=a.device)
        if not out.numel() or not e:
            return out
        if not k:
            return out.zero_()
        rows = a.numel() // k
        encoded = b_scale.dtype == torch.float8_e8m0fnu
        weight_kernel = _weights(n, k, encoded)
        lhs = (
            a.float().view(rows, k // 128, 128)
            * _activation_scales(a_scale).view(rows, k // 128, 1)
        ).view(rows, k)
        dst = out.view(rows, n)
        kind = self.call.kind
        per_row = kind != "masked" and self.call.metadata_kind == "per_row"
        bounds = None if per_row else layout_metadata.tolist()
        start = 0
        for expert in range(e):
            if per_row:
                select = torch.where(layout_metadata == expert)[0]
                if select.numel() == 0:
                    continue
                left = lhs.index_select(0, select)
            else:
                if kind == "masked":
                    start = expert * self.call.max_m
                    end = start + bounds[expert]
                else:
                    align = self.call.alignment
                    start = (start + align - 1) // align * align
                    end = bounds[expert]
                left = lhs[start:end]
                if not left.shape[0]:
                    start = end
                    continue
            scales = b_scale[expert].reshape(-1)
            weights = weight_kernel(
                b[expert].reshape(-1), scales.view(torch.uint8) if encoded else scales
            )
            product = torch.mm(left, weights.view(n, k).T).to(torch.bfloat16)
            if per_row:
                dst.index_copy_(0, select, product)
            else:
                dst[start:end].copy_(product)
                start = end
        return out


register_kernel_type("MoEGroupedGEMMMXFP4FwdOp", "ascend", AscendMoEGroupedMXFP4Kernel)
