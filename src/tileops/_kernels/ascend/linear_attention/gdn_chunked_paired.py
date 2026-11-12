"""Candidate dense GDN pipeline with sequential chunk state propagation."""

import functools

import tilelang

from tileops._kernels.ascend.linear_attention.deltanet_training_fwd import weighted_wu
from tileops._kernels.ascend.linear_attention.gdn_gram_pair import build as gram_pair
from tileops._kernels.ascend.linear_attention.gdn_inverse_branch import inverse
from tileops._kernels.ascend.linear_attention.gdn_pack_hybrid import pack
from tileops._kernels.ascend.linear_attention.gdn_sequential_direct512 import build as sequential


class ChunkedGDNPaired:
    def __init__(
        self, B, S, H, HV, K, V, dtype, scale, initial, norm, raw_gate, beta_sigmoid, negative
    ):
        self.length = S
        self.N = N = (S + 31) // 32
        self.initial = initial
        J = B * N * HV
        C = 32

        def compile(f, idx):
            return tilelang.compile(f, target="ascend", out_idx=idx)

        self.pack = compile(
            pack(B, S, H, HV, K, V, dtype, scale, initial, norm, raw_gate, beta_sigmoid, negative),
            list(range(8, 16)),
        )
        self.gram = gram_pair(J, C, K, dtype)
        self.inverse = compile(inverse(J, dtype), [4, 5])
        self.wu = weighted_wu(C, K, V, dtype)
        self.states = sequential(B, N, HV, K, V, dtype, initial)

    def __call__(self, q, k, v, g, beta, h0, A_log, dt_bias):
        qp, kp, vp, bp, gp, qg, kg, ke = self.pack(q, k, v, g, beta, h0, A_log, dt_bias)
        gram, qk = self.gram(qp, kp)
        b, att = self.inverse(gram, qk, bp, gp)
        w, u = self.wu(b, kg, vp, bp)
        result, final = self.states(w, u, ke, gp, h0, qg, att)
        return result[:, : self.length].contiguous(), final

    def get_kernel_source(self):
        return "\n".join(
            x.get_kernel_source() for x in vars(self).values() if hasattr(x, "get_kernel_source")
        )


build = functools.lru_cache(maxsize=96)(ChunkedGDNPaired)
