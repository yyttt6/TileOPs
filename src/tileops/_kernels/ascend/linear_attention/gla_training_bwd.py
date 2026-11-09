"""GLA gradients with reverse boundary propagation and parallel chunk replay."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.linear_attention.call_spec import GLAChunkBwdInterface
from tileops.backend._registry import register_kernel_type


def boundaries(B, S, H, K, V, C, dtype, scale):
    N = S // C
    kp = tilelang.next_power_of_2(K)
    cols = 16
    tiles = (V + cols - 1) // cols
    jobs = B * H * tiles
    grid = min(64, jobs)

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, K), dtype),
        G: T.Tensor((B, S, H, K), dtype),
        DO: T.Tensor((B, S, H, V), dtype),
        DHT: T.Tensor((B, H, K, V), "float32"),
        A: T.Tensor((B, N, H, K, V), "float32"),
        DI: T.Tensor((B, H, K, V), "float32"),
    ):
        with T.Kernel(grid) as bx:  # noqa: SIM117 - TileLang DSL scopes are distinct
            with T.SimtVF(threads=256):
                adj = T.alloc_fragment((kp, cols), "float32")
                for step in T.serial((jobs + grid - 1) // grid):
                    job = step * grid + bx
                    if job < jobs:
                        b = job // (H * tiles)
                        h = job // tiles % H
                        col = job % tiles * cols
                        for i, j in T.Parallel(kp, cols):
                            if i < K and col + j < V:
                                adj[i, j] = DHT[b, h, i, col + j]
                        for nrev in T.serial(N):
                            n = N - 1 - nrev
                            for i, j in T.Parallel(kp, cols):
                                if i < K and col + j < V:
                                    A[b, n, h, i, col + j] = adj[i, j]
                            for reverse in T.serial(C):
                                t = (n + 1) * C - 1 - reverse
                                for i, j in T.Parallel(kp, cols):
                                    if i < K and col + j < V:
                                        adj[i, j] = (
                                            adj[i, j]
                                            + scale
                                            * T.cast(Q[b, t, h, i], "float32")
                                            * T.cast(DO[b, t, h, col + j], "float32")
                                        ) * T.exp(T.cast(G[b, t, h, i], "float32"))
                        for i, j in T.Parallel(kp, cols):
                            if i < K and col + j < V:
                                DI[b, h, i, col + j] = adj[i, j]

    return main


def local_gradients(B, S, H, K, V, C, dtype, scale, cols):
    N = S // C
    kp = tilelang.next_power_of_2(K)
    tiles = (V + cols - 1) // cols
    jobs = B * N * H * tiles
    grid = min(64, jobs)

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, K), dtype),
        KK: T.Tensor((B, S, H, K), dtype),
        VV: T.Tensor((B, S, H, V), dtype),
        G: T.Tensor((B, S, H, K), dtype),
        HH: T.Tensor((B, N + 1, H, K, V), "float32"),
        DO: T.Tensor((B, S, H, V), dtype),
        A: T.Tensor((B, N, H, K, V), "float32"),
        PART: T.Tensor((3, B, S, H, tiles, K), "float32"),
        DV: T.Tensor((B, S, H, V), "float32"),
    ):
        with T.Kernel(grid) as bx:
            q = T.alloc_shared((C, kp), "float32")
            k = T.alloc_shared((C, kp), "float32")
            gate = T.alloc_shared((C, kp), "float32")
            v = T.alloc_shared((C, cols), "float32")
            dout = T.alloc_shared((C, cols), "float32")
            states = T.alloc_shared((C + 1, kp, cols), "float32")
            adj = T.alloc_shared((kp, cols), "float32")
            for step in T.serial((jobs + grid - 1) // grid):
                job = step * grid + bx
                if job < jobs:
                    tile = job % tiles
                    h = job // tiles % H
                    n = job // (tiles * H) % N
                    b = job // (tiles * H * N)
                    col = tile * cols
                    valid = T.min(cols, V - col)
                    with T.SimtVF(threads=256):
                        for t, i in T.Parallel(C, kp):
                            q[t, i] = T.if_then_else(
                                i < K, T.cast(Q[b, n * C + t, h, i], "float32"), T.float32(0)
                            )
                            k[t, i] = T.if_then_else(
                                i < K, T.cast(KK[b, n * C + t, h, i], "float32"), T.float32(0)
                            )
                            gate[t, i] = T.if_then_else(
                                i < K, T.exp(T.cast(G[b, n * C + t, h, i], "float32")), T.float32(0)
                            )
                        for t, j in T.Parallel(C, cols):
                            v[t, j] = T.if_then_else(
                                j < valid,
                                T.cast(VV[b, n * C + t, h, col + j], "float32"),
                                T.float32(0),
                            )
                            dout[t, j] = T.if_then_else(
                                j < valid,
                                T.cast(DO[b, n * C + t, h, col + j], "float32"),
                                T.float32(0),
                            )
                        for i, j in T.Parallel(kp, cols):
                            states[0, i, j] = T.if_then_else(
                                i < K and j < valid, HH[b, n, h, i, col + j], T.float32(0)
                            )
                            adj[i, j] = T.if_then_else(
                                i < K and j < valid, A[b, n, h, i, col + j], T.float32(0)
                            )
                    with T.SimtVF(threads=256):
                        for t in T.serial(C):
                            for i, j in T.Parallel(kp, cols):
                                states[t + 1, i, j] = (
                                    states[t, i, j] * gate[t, i] + k[t, i] * v[t, j]
                                )
                    for reverse in T.serial(C):
                        t = C - 1 - reverse
                        with T.SimtVF(threads=min(1024, kp * cols)):
                            rq = T.alloc_reducer((kp,), "float32", op="sum")
                            rk = T.alloc_reducer((kp,), "float32", op="sum")
                            rg = T.alloc_reducer((kp,), "float32", op="sum")
                            rv = T.alloc_reducer((cols,), "float32", op="sum")
                            T.reducer_init(rq)
                            T.reducer_init(rk)
                            T.reducer_init(rg)
                            T.reducer_init(rv)
                            for i, j in T.Parallel(kp, cols):
                                current = adj[i, j] + scale * q[t, i] * dout[t, j]
                                T.reducer_update(rq[i], scale * states[t + 1, i, j] * dout[t, j])
                                T.reducer_update(rk[i], current * v[t, j])
                                T.reducer_update(rg[i], current * states[t, i, j] * gate[t, i])
                                T.reducer_update(rv[j], current * k[t, i])
                                adj[i, j] = current * gate[t, i]
                            oq = T.alloc_fragment((kp,), "float32")
                            ok = T.alloc_fragment((kp,), "float32")
                            og = T.alloc_fragment((kp,), "float32")
                            ov = T.alloc_fragment((cols,), "float32")
                            T.finalize_reducer(rq, oq)
                            T.finalize_reducer(rk, ok)
                            T.finalize_reducer(rg, og)
                            T.finalize_reducer(rv, ov)
                            for i in T.Parallel(kp):
                                if i < K:
                                    PART[0, b, n * C + t, h, tile, i] = oq[i]
                                    PART[1, b, n * C + t, h, tile, i] = ok[i]
                                    PART[2, b, n * C + t, h, tile, i] = og[i]
                            for j in T.Parallel(cols):
                                if j < valid:
                                    DV[b, n * C + t, h, col + j] = ov[j]

    return main


def combine(B, S, H, K, tiles):
    total = B * S * H * K
    grid = min(64, (total + 255) // 256)

    @T.prim_func
    def main(
        P: T.Tensor((3, B, S, H, tiles, K), "float32"),
        DQ: T.Tensor((B, S, H, K), "float32"),
        DK: T.Tensor((B, S, H, K), "float32"),
        DG: T.Tensor((B, S, H, K), "float32"),
    ):
        with T.Kernel(grid) as bx:  # noqa: SIM117 - TileLang DSL scopes are distinct
            with T.SimtVF(threads=256):
                aq = T.alloc_fragment((256,), "float32")
                ak = T.alloc_fragment((256,), "float32")
                ag = T.alloc_fragment((256,), "float32")
                for step in T.serial((total + grid * 256 - 1) // (grid * 256)):
                    for lane in T.Parallel(256):
                        x = (step * grid + bx) * 256 + lane
                        if x < total:
                            i = x % K
                            h = x // K % H
                            t = x // (K * H) % S
                            b = x // (K * H * S)
                            aq[lane] = 0.0
                            ak[lane] = 0.0
                            ag[lane] = 0.0
                            for tile in T.serial(tiles):
                                aq[lane] += P[0, b, t, h, tile, i]
                                ak[lane] += P[1, b, t, h, tile, i]
                                ag[lane] += P[2, b, t, h, tile, i]
                            DQ[b, t, h, i] = aq[lane]
                            DK[b, t, h, i] = ak[lane]
                            DG[b, t, h, i] = ag[lane]

    return main


@functools.lru_cache(maxsize=64)
def build(B, S, H, K, V, C, dtype, scale):
    kp = tilelang.next_power_of_2(K)
    cols = 8
    # The 64-token, 64-key tile needs 188416 bytes with eight value columns,
    # including the adjoint buffer. Bound explicit shared storage to 192 KiB.
    budget = 192 * 1024 if C == 64 and K == 64 else 180 * 1024
    while cols > 1 and (C + 2) * kp * cols * 4 + C * kp * 12 + C * cols * 8 > budget:
        cols //= 2
    return (
        tilelang.compile(
            boundaries(B, S, H, K, V, C, dtype, scale), target="ascend", out_idx=[4, 5]
        ),
        tilelang.compile(
            local_gradients(B, S, H, K, V, C, dtype, scale, cols), target="ascend", out_idx=[7, 8]
        ),
        tilelang.compile(
            combine(B, S, H, K, (V + cols - 1) // cols), target="ascend", out_idx=[1, 2, 3]
        ),
    )


class AscendGLAChunkBwdKernel(Kernel, GLAChunkBwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call

    def forward(self, q, k, v, g, h, do, dht, has_initial_state=False):
        c = self.call
        scale = c.scale
        self.compiled = build(
            c.batch,
            c.seq_len,
            c.heads,
            c.dim_k,
            c.dim_v,
            c.chunk_size,
            str(c.dtype).removeprefix("torch."),
            scale,
        )
        q, k, v, g, h, do, dht = (x.contiguous() for x in (q, k, v, g, h, do, dht))
        adjoints, dinitial = self.compiled[0](q, g, do, dht)
        partials, dv = self.compiled[1](q, k, v, g, h, do, adjoints)
        dq, dk, dg = self.compiled[2](partials)
        return dq, dk, dv, dg, dinitial if has_initial_state else None


register_kernel_type("GLAChunkBwdOp", "ascend", AscendGLAChunkBwdKernel)
