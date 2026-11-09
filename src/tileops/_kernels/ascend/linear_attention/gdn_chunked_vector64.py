"""Compose qualified fused inverse and SIMD state; direct bounded output."""

import functools

import tilelang
import torch

from tileops._kernels.ascend.linear_attention.gdn_inverse64_gram import inverse
from tileops._kernels.ascend.linear_attention.gdn_inverse64_gram_compact import (
    inverse as compact_inverse,
)
from tileops._kernels.ascend.linear_attention.gdn_pack_compact64 import pack as compact_pack
from tileops._kernels.ascend.linear_attention.gdn_pack_group_reuse64 import pack as group_pack
from tileops._kernels.ascend.linear_attention.gdn_pack_vector64 import pack
from tileops._kernels.ascend.linear_attention.gdn_sequential_vector_output import prim
from tileops._kernels.ascend.linear_attention.gdn_weighted_wu_vector64 import weighted_wu


@functools.lru_cache(maxsize=96)
def output_state(B, N, H, K, V, dtype, initial, VT, tokens, exact):
    return tilelang.compile(
        prim(B, N, H, K, V, dtype, initial, VT=VT, C=64, tokens=tokens, exact=exact),
        target="ascend",
        out_idx=[9],
    )


class ChunkedGDNVector64:
    state_vt = 128

    def __init__(
        self, B, S, H, HV, K, V, dtype, scale, initial, norm, raw_gate, beta_sigmoid, negative
    ):
        self.length = S
        N = (S + 63) // 64
        J = B * N * HV
        self.state_vt = 128 if B * HV >= 32 else 64
        pack_builder = pack
        group = 1
        if not norm and HV > H:
            assert HV % H == 0
            group = HV // H
            pack_builder = group_pack if B * N * H >= 64 else compact_pack
        self.pack = tilelang.compile(
            pack_builder(
                B, S, H, HV, K, V, dtype, scale, initial, norm, raw_gate, beta_sigmoid, negative
            ),
            target="ascend",
            out_idx=list(range(8, 16)),
        )
        inverse_prim = (
            inverse(J, K, dtype) if group == 1 else compact_inverse(J, K, dtype, group=group)
        )
        self.inverse = tilelang.compile(inverse_prim, target="ascend", out_idx=[4, 5])
        self.wu = weighted_wu(64, K, V, dtype)
        self.state_args = (B, N, HV, K, V, dtype, initial)

    def __call__(self, q, k, v, g, beta, h0, A_log, dt_bias, output=None, offset=0):
        qp, kp, vp, bp, gp, qg, kg, ke = self.pack(q, k, v, g, beta, h0, A_log, dt_bias)
        inv, att = self.inverse(qp, kp, bp, gp)
        w, u = self.wu(inv, kg, vp, bp)
        if output is None:
            output = torch.empty(
                (q.shape[0], self.length, v.shape[2], v.shape[3]), device=v.device, dtype=v.dtype
            )
        self.states = output_state(
            *self.state_args,
            min(self.state_vt, v.shape[3]),
            self.length,
            output.shape[1] == self.length and offset == 0,
        )
        final = self.states(w, u, ke, gp, h0, qg, att, output, offset)
        return output, final

    def get_kernel_source(self):
        return "\n".join(
            x.get_kernel_source() for x in vars(self).values() if hasattr(x, "get_kernel_source")
        )


build = functools.lru_cache(maxsize=96)(ChunkedGDNVector64)
