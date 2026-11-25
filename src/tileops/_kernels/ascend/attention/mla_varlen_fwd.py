"""MLA prefill after decompression, using the ragged FlashAttention core."""

import torch

from tileops._kernels.ascend.attention.gqa_varlen import _attention, _layout, _pack
from tileops._kernels.attention.call_spec import MLAVarlenFwdInterface
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type


class AscendMLAVarlenKernel(Kernel, MLAVarlenFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call
        self.dtype = call.dtype

    def forward(self, q, k_nope, k_pe, v, cu_seqlens):
        t, h, d = q.shape
        dv = v.shape[-1]
        if not t:
            return v.new_empty((0, h, dv)), q.new_empty((h, 0), dtype=torch.float32)
        b = cu_seqlens.numel() - 1
        bm, bn = (64 if 128 < d <= 512 else 32), 128
        off = _layout(b, 1, bm, bn)(cu_seqlens, cu_seqlens)
        key = torch.cat((k_nope, k_pe[:, None, :].expand(-1, h, -1)), dim=-1)
        packed = []
        for mode, x, size, cap in (
            ("q", q, d, bm),
            ("k", key, d, bn),
            ("v", v, dv, bn),
        ):
            fn = _pack(b, h, h, size, cap, self.dtype_str, False, mode, 0, False)
            packed.append(fn(x, cu_seqlens, cu_seqlens, off))
        c = self.call
        core = _attention(
            b,
            h,
            h,
            d,
            self.dtype_str,
            c.is_causal,
            -1,
            -1,
            d**-0.5 if c.sm_scale is None else c.sm_scale,
            0.0,
            False,
            bm,
            bn,
            value_dim=dv,
            with_lse=True,
            packed_kh=h if d > 128 else 0,
        )
        out = q.new_empty((t, h, dv))
        lse = q.new_empty((h, t), dtype=torch.float32)
        core(*packed, cu_seqlens, cu_seqlens, off, out, lse)
        return out, lse


register_kernel_type("MLAVarlenFwdOp", "ascend", AscendMLAVarlenKernel)
