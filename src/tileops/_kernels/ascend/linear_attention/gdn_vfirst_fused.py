"""Single-SIMT value-major decode: one warp owns one value channel."""

import functools

import tilelang
import tilelang.ascend.language as T


def prim(
    B, H, HV, K, V, dtype, scale, initial, norm, raw_gate, beta_sigmoid, negative, threads=256
):
    assert K in (64, 128) and threads % 32 == 0
    warps = threads // 32
    tiles = (V + warps - 1) // warps
    jobs = B * HV * tiles
    grid = min(64, jobs)
    F = K // 32

    @T.prim_func
    def main(
        Q: T.Tensor((B, 1, H, K), dtype),
        KK: T.Tensor((B, 1, H, K), dtype),
        VV: T.Tensor((B, 1, HV, V), dtype),
        G: T.Tensor((B, 1, HV), dtype),
        Beta: T.Tensor((B, 1, HV), dtype),
        H0: T.Tensor((B, HV, V, K), "float32"),
        ALog: T.Tensor((HV,), "float32"),
        Bias: T.Tensor((HV,), "float32"),
        O: T.Tensor((B, 1, HV, V), dtype),
        HT: T.Tensor((B, HV, V, K), "float32"),
    ):
        with T.Kernel(grid) as bx:  # noqa: SIM117 - TileLang DSL scopes are distinct
            with T.SimtVF(threads=threads):
                tid = T.get_thread_binding()
                lane = tid % 32
                warp = tid // 32
                q = T.alloc_local((F,), "float32")
                k = T.alloc_local((F,), "float32")
                state = T.alloc_local((F,), "float32")
                sums = T.alloc_local((4,), "float32")
                for step in T.serial((jobs + grid - 1) // grid):
                    job = step * grid + bx
                    col = job % tiles * warps + warp
                    if job < jobs and col < V:
                        b = job // (HV * tiles)
                        h = job // tiles % HV
                        qh = h // (HV // H)
                        sums[0] = 0.0
                        sums[1] = 0.0
                        sums[2] = 0.0
                        sums[3] = 0.0
                        for f in T.unroll(F):
                            q[f] = T.cast(Q[b, 0, qh, f * 32 + lane], "float32")
                            k[f] = T.cast(KK[b, 0, qh, f * 32 + lane], "float32")
                            state[f] = H0[b, h, col, f * 32 + lane] if initial else T.float32(0)
                            if norm:
                                sums[0] = sums[0] + q[f] * q[f]
                                sums[1] = sums[1] + k[f] * k[f]
                        if norm:
                            for shift in T.unroll(5):
                                sums[0] = sums[
                                    0
                                ] + T.call_extern(  # lint: extern-target allow: Ascend shuffle
                                    "float32", "asc_shfl_xor", sums[0], 1 << shift
                                )
                                sums[1] = sums[
                                    1
                                ] + T.call_extern(  # lint: extern-target allow: Ascend shuffle
                                    "float32", "asc_shfl_xor", sums[1], 1 << shift
                                )
                            for f in T.unroll(F):
                                q[f] = q[f] * T.rsqrt(sums[0] + 1e-6)
                                k[f] = k[f] * T.rsqrt(sums[1] + 1e-6)
                        for f in T.unroll(F):
                            sums[2] = sums[2] + k[f] * state[f]
                        for shift in T.unroll(5):
                            sums[2] = sums[
                                2
                            ] + T.call_extern(  # lint: extern-target allow: Ascend shuffle
                                "float32", "asc_shfl_xor", sums[2], 1 << shift
                            )
                        gate = T.cast(G[b, 0, h], "float32")
                        beta = T.cast(Beta[b, 0, h], "float32")
                        if raw_gate:
                            x = gate + Bias[h]
                            gate = -T.exp(ALog[h]) * T.if_then_else(x > 20.0, x, T.log1p(T.exp(x)))
                        if beta_sigmoid:
                            beta = (2.0 if negative else 1.0) / (1.0 + T.exp(-beta))
                        decay = T.exp(gate)
                        residual = beta * (T.cast(VV[b, 0, h, col], "float32") - decay * sums[2])
                        for f in T.unroll(F):
                            state[f] = decay * state[f] + k[f] * residual
                            HT[b, h, col, f * 32 + lane] = state[f]
                            sums[3] = sums[3] + q[f] * scale * state[f]
                        for shift in T.unroll(5):
                            sums[3] = sums[
                                3
                            ] + T.call_extern(  # lint: extern-target allow: Ascend shuffle
                                "float32", "asc_shfl_xor", sums[3], 1 << shift
                            )
                        if lane == 0:
                            O[b, 0, h, col] = sums[3]

    return main


@functools.lru_cache(maxsize=128)
def build(
    B,
    S,
    H,
    HV,
    K,
    V,
    dtype,
    scale,
    initial,
    norm,
    raw_gate,
    beta_sigmoid,
    negative,
    vfirst,
    offsets,
    threads=1024,
):
    assert S == 1 and vfirst and not offsets
    return tilelang.compile(
        prim(
            B, H, HV, K, V, dtype, scale, initial, norm, raw_gate, beta_sigmoid, negative, threads
        ),
        target="ascend",
        out_idx=[8, 9],
    )
