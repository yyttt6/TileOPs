"""Chunked GLA with gate prefixes, block summaries, a state scan, and Cube readout."""

import functools

import tilelang
import tilelang.ascend.language as T


def preprocessing(B, S, H, K, V, dtype, scale, C=32):
    N = (S + C - 1) // C
    J = B * N * H
    grid = min(64, J)

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, K), dtype),
        KK: T.Tensor((B, S, H, K), dtype),
        VV: T.Tensor((B, S, H, V), dtype),
        G: T.Tensor((B, S, H, K), dtype),
        GC: T.Tensor((J, C, K), "float32"),
        QG: T.Tensor((J, C, K), dtype),
        KW: T.Tensor((J, K, C), dtype),
        VB: T.Tensor((J, C, V), dtype),
        D: T.Tensor((J, K), "float32"),
    ):
        with T.Kernel(grid) as bx:
            q = T.alloc_shared((C, K), dtype)
            k = T.alloc_shared((C, K), dtype)
            g = T.alloc_shared((C, K), dtype)
            v = T.alloc_shared((C, V), dtype)
            gc = T.alloc_shared((C, K), "float32")
            qg = T.alloc_shared((C, K), dtype)
            kw = T.alloc_shared((K, C), dtype)
            dd = T.alloc_shared((K,), "float32")
            for step in T.serial((J + grid - 1) // grid):
                job = step * grid + bx
                if job < J:
                    b = job // (N * H)
                    h = job % H
                    start = job // H % N * C
                    nt = T.min(C, S - start)
                    if nt < C:
                        with T.SimtVF(threads=256):
                            for t, i in T.Parallel(C, K):
                                q[t, i] = 0
                                k[t, i] = 0
                                g[t, i] = 0
                            for t, j in T.Parallel(C, V):
                                v[t, j] = 0
                    T.copy(Q[b, start : start + nt, h, :], q[:nt, :])
                    T.copy(KK[b, start : start + nt, h, :], k[:nt, :])
                    T.copy(G[b, start : start + nt, h, :], g[:nt, :])
                    T.copy(VV[b, start : start + nt, h, :], v[:nt, :])
                    with T.SimtVF(threads=128):
                        accum = T.alloc_fragment((K,), "float32")
                        for i in T.Parallel(K):
                            accum[i] = 0
                        for t in T.serial(C):
                            for i in T.Parallel(K):
                                accum[i] = accum[i] + T.cast(g[t, i], "float32")
                                gc[t, i] = accum[i]
                        for i in T.Parallel(K):
                            dd[i] = T.exp(accum[i])
                    with T.SimtVF(threads=256):
                        for t, i in T.Parallel(C, K):
                            qg[t, i] = T.cast(q[t, i], "float32") * T.exp(gc[t, i]) * scale
                            kw[i, t] = T.cast(k[t, i], "float32") * T.exp(gc[C - 1, i] - gc[t, i])
                    T.copy(gc, GC[job, :, :])
                    T.copy(qg, QG[job, :, :])
                    T.copy(kw, KW[job, :, :])
                    T.copy(v, VB[job, :, :])
                    T.copy(dd, D[job, :])

    return main


def summary(J, K, V, dtype, C=32):
    grid = min(32, J)

    @T.prim_func
    def main(
        KW: T.Tensor((J, K, C), dtype),
        VB: T.Tensor((J, C, V), dtype),
        U: T.Tensor((J, K, V), "float32"),
    ):
        with T.Kernel(grid) as bx:
            a = T.alloc_l1((K, C), dtype)
            b = T.alloc_l1((V, C), dtype)
            aa = T.alloc_l0a((K, C), dtype)
            bb = T.alloc_l0b((V, C), dtype)
            cc = T.alloc_l0c((K, V), "float32")
            for step in T.serial((J + grid - 1) // grid):
                job = step * grid + bx
                if job < J:
                    T.copy(KW[job, :, :], a)
                    T.copy(VB[job, :, :], b, transpose=True)
                    T.copy(a, aa)
                    T.copy(b, bb)
                    T.gemm(aa, bb, cc, transpose_B=True, clear_accum=True)
                    T.copy(cc, U[job, :, :])

    return main


def scan(B, N, H, K, V, dtype, initial):
    cols = 8 if B * H < 16 else 32
    tiles = (V + cols - 1) // cols
    jobs = B * H * tiles
    grid = min(64, jobs)

    @T.prim_func
    def main(
        U: T.Tensor((B * N * H, K, V), "float32"),
        D: T.Tensor((B * N * H, K), "float32"),
        H0: T.Tensor((B, H, K, V), "float32"),
        HB: T.Tensor((B * N * H, K, V), dtype),
        HT: T.Tensor((B, H, K, V), "float32"),
    ):
        with T.Kernel(grid) as bx:  # noqa: SIM117 - TileLang DSL scopes are distinct
            with T.SimtVF(threads=256):
                state = T.alloc_fragment((K, cols), "float32")
                for step in T.serial((jobs + grid - 1) // grid):
                    job = step * grid + bx
                    if job < jobs:
                        b = job // (H * tiles)
                        h = job // tiles % H
                        col = job % tiles * cols
                        for i, j in T.Parallel(K, cols):
                            if col + j < V:
                                state[i, j] = H0[b, h, i, col + j] if initial else T.float32(0)
                        for n in T.serial(N):
                            index = (b * N + n) * H + h
                            for i, j in T.Parallel(K, cols):
                                if col + j < V:
                                    HB[index, i, col + j] = state[i, j]
                                    state[i, j] = state[i, j] * D[index, i] + U[index, i, col + j]
                        for i, j in T.Parallel(K, cols):
                            if col + j < V:
                                HT[b, h, i, col + j] = state[i, j]

    return main


def scores(B, S, H, K, dtype, scale, C=32, R=16, workers=1024):
    N = (S + C - 1) // C
    J = B * N * H
    rows = C // R
    jobs = J * rows
    grid = min(64, jobs)

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, K), dtype),
        KK: T.Tensor((B, S, H, K), dtype),
        GC: T.Tensor((J, C, K), "float32"),
        A: T.Tensor((J, C, C), dtype),
    ):
        with T.Kernel(grid) as bx:
            q = T.alloc_shared((R, K), dtype)
            k = T.alloc_shared((C, K), dtype)
            gc = T.alloc_shared((C, K), "float32")
            out = T.alloc_shared((R, C), dtype)
            for step in T.serial((jobs + grid - 1) // grid):
                job = step * grid + bx
                if job < jobs:
                    index = job // rows
                    r = job % rows * R
                    b = index // (N * H)
                    h = index % H
                    start = index // H % N * C
                    nt = T.min(C, S - start)
                    nr = T.max(0, T.min(R, nt - r))
                    if nt < C or nr < R:
                        with T.SimtVF(threads=256):
                            for t, i in T.Parallel(R, K):
                                q[t, i] = 0
                            for t, i in T.Parallel(C, K):
                                k[t, i] = 0
                    if nr > 0:
                        T.copy(Q[b, start + r : start + r + nr, h, :], q[:nr, :])
                    T.copy(KK[b, start : start + nt, h, :], k[:nt, :])
                    T.copy(GC[index, :, :], gc)
                    with T.SimtVF(threads=workers):
                        sums = T.alloc_reducer((R, C), "float32", op="sum")
                        T.reducer_init(sums)
                        for i, j, z in T.Parallel(R, C, K):
                            if j <= r + i:
                                delta = gc[r + i, z] - gc[j, z]
                                T.reducer_update(
                                    sums[i, j],
                                    T.cast(q[i, z], "float32")
                                    * T.cast(k[j, z], "float32")
                                    * T.exp(delta),
                                )
                        values = T.alloc_fragment((R, C), "float32")
                        T.finalize_reducer(sums, values)
                        for i, j in T.Parallel(R, C):
                            out[i, j] = values[i, j] * scale
                    T.copy(out, A[index, r : r + R, :])

    return main


def output(B, S, H, K, V, dtype, C=32):
    N = (S + C - 1) // C
    J = B * N * H
    grid = min(32, J)

    @T.prim_func
    def main(
        QG: T.Tensor((J, C, K), dtype),
        HB: T.Tensor((J, K, V), dtype),
        A: T.Tensor((J, C, C), dtype),
        VB: T.Tensor((J, C, V), dtype),
        O: T.Tensor((B, N * C, H, V), dtype),
    ):
        with T.Kernel(grid) as bx:
            q = T.alloc_l1((C, K), dtype)
            h = T.alloc_l1((V, K), dtype)
            a = T.alloc_l1((C, C), dtype)
            v = T.alloc_l1((V, C), dtype)
            q0 = T.alloc_l0a((C, K), dtype)
            h0 = T.alloc_l0b((V, K), dtype)
            a0 = T.alloc_l0a((C, C), dtype)
            v0 = T.alloc_l0b((V, C), dtype)
            cc = T.alloc_l0c((C, V), "float32")
            for step in T.serial((J + grid - 1) // grid):
                job = step * grid + bx
                if job < J:
                    T.copy(QG[job, :, :], q)
                    T.copy(HB[job, :, :], h, transpose=True)
                    T.copy(q, q0)
                    T.copy(h, h0)
                    T.gemm(q0, h0, cc, transpose_B=True, clear_accum=True)
                    T.copy(A[job, :, :], a)
                    T.copy(VB[job, :, :], v, transpose=True)
                    T.copy(a, a0)
                    T.copy(v, v0)
                    T.gemm(a0, v0, cc, transpose_B=True, clear_accum=False)
                    b = job // (N * H)
                    head = job % H
                    start = job // H % N * C
                    # Preserve the C-row accumulator layout through FIXPIPE; crop after storing.
                    T.copy(cc, O[b, start : start + C, head, :])

    return main


def preprocessing_mixed(B, S, H, K, V, dtype, scale, C=32):
    N = (S + C - 1) // C
    J = B * N * H
    grid = min(64, J)

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, K), dtype),
        KK: T.Tensor((B, S, H, K), dtype),
        VV: T.Tensor((B, S, H, V), dtype),
        G: T.Tensor((B, S, H, K), dtype),
        GC: T.Tensor((J, C, K), "float32"),
        QG: T.Tensor((J, C, K), dtype),
        KW: T.Tensor((J, K, C), dtype),
        VB: T.Tensor((J, C, V), dtype),
        D: T.Tensor((J, K), "float32"),
        QX: T.Tensor((J, 16, K), dtype),
        KX: T.Tensor((J, 16, K), dtype),
    ):
        with T.Kernel(grid) as bx:
            q = T.alloc_shared((C, K), dtype)
            k = T.alloc_shared((C, K), dtype)
            g = T.alloc_shared((C, K), dtype)
            v = T.alloc_shared((C, V), dtype)
            qx = T.alloc_shared((16, K), dtype)
            kx = T.alloc_shared((16, K), dtype)
            gc = T.alloc_shared((C, K), "float32")
            qg = T.alloc_shared((C, K), dtype)
            kw = T.alloc_shared((K, C), dtype)
            dd = T.alloc_shared((K,), "float32")
            for step in T.serial((J + grid - 1) // grid):
                job = step * grid + bx
                if job < J:
                    b = job // (N * H)
                    h = job % H
                    start = job // H % N * C
                    nt = T.min(C, S - start)
                    if nt < C:
                        with T.SimtVF(threads=256):
                            for t, i in T.Parallel(C, K):
                                q[t, i] = 0
                                k[t, i] = 0
                                g[t, i] = 0
                            for t, j in T.Parallel(C, V):
                                v[t, j] = 0
                    T.copy(Q[b, start : start + nt, h, :], q[:nt, :])
                    T.copy(KK[b, start : start + nt, h, :], k[:nt, :])
                    T.copy(G[b, start : start + nt, h, :], g[:nt, :])
                    T.copy(VV[b, start : start + nt, h, :], v[:nt, :])
                    with T.SimtVF(threads=128):
                        accum = T.alloc_fragment((K,), "float32")
                        for i in T.Parallel(K):
                            accum[i] = 0
                        for t in T.serial(C):
                            for i in T.Parallel(K):
                                accum[i] = accum[i] + T.cast(g[t, i], "float32")
                                gc[t, i] = accum[i]
                        for i in T.Parallel(K):
                            dd[i] = T.exp(accum[i])
                    with T.SimtVF(threads=256):
                        for t, i in T.Parallel(C, K):
                            qg[t, i] = T.cast(q[t, i], "float32") * T.exp(gc[t, i]) * scale
                            kw[i, t] = T.cast(k[t, i], "float32") * T.exp(gc[C - 1, i] - gc[t, i])
                    with T.SimtVF(threads=256):
                        for t, i in T.Parallel(16, K):
                            qx[t, i] = (
                                T.cast(q[t + 16, i], "float32")
                                * T.exp(gc[t + 16, i] - gc[15, i])
                                * scale
                            )
                            kx[t, i] = T.cast(k[t, i], "float32") * T.exp(gc[15, i] - gc[t, i])
                    T.copy(qx, QX[job, :, :])
                    T.copy(kx, KX[job, :, :])
                    T.copy(gc, GC[job, :, :])
                    T.copy(qg, QG[job, :, :])
                    T.copy(kw, KW[job, :, :])
                    T.copy(v, VB[job, :, :])
                    T.copy(dd, D[job, :])

    return main


def scores_diagonal(B, S, H, K, dtype, scale):
    C, R = 32, 16
    N = (S + C - 1) // C
    J = B * N * H
    jobs = J * 2
    grid = min(64, jobs)

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, K), dtype),
        KK: T.Tensor((B, S, H, K), dtype),
        GC: T.Tensor((J, C, K), "float32"),
        A: T.Tensor((J, C, C), dtype),
    ):
        with T.Kernel(grid) as bx:
            q = T.alloc_shared((R, K), dtype)
            k = T.alloc_shared((R, K), dtype)
            qf = T.alloc_shared((R, K), "float32")
            kf = T.alloc_shared((R, K), "float32")
            gc = T.alloc_shared((R, K), "float32")
            values = T.alloc_shared((R, R), "float32")
            out = T.alloc_shared((R, R), dtype)
            zero = T.alloc_shared((R, R), dtype)
            with T.SimtVF(threads=256):
                for i, j in T.Parallel(R, R):
                    zero[i, j] = 0
            for step in T.serial((jobs + grid - 1) // grid):
                job = step * grid + bx
                if job < jobs:
                    index = job // 2
                    r = job % 2 * R
                    b = index // (N * H)
                    h = index % H
                    start = index // H % N * C
                    nr = T.max(0, T.min(R, S - start - r))
                    if nr < R:
                        with T.SimtVF(threads=256):
                            for t, i in T.Parallel(R, K):
                                q[t, i] = 0
                                k[t, i] = 0
                    if nr > 0:
                        T.copy(Q[b, start + r : start + r + nr, h, :], q[:nr, :])
                        T.copy(KK[b, start + r : start + r + nr, h, :], k[:nr, :])
                    T.copy(GC[index, r : r + R, :], gc)
                    with T.SimtVF(threads=256):
                        for t, i in T.Parallel(R, K):
                            qf[t, i] = T.cast(q[t, i], "float32")
                            kf[t, i] = T.cast(k[t, i], "float32")
                        for i, j in T.Parallel(R, R):
                            values[i, j] = 0
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        for i in T.serial(R):
                            for j in T.serial(i + 1):
                                total = T.simd.alloc_local(1, "float32")
                                total[0] = T.simd.vdup(T.float32(0), "float32", full)
                                for part in T.serial(K // 64):
                                    qi = T.simd.vld(qf[i, part * 64])
                                    kj = T.simd.vld(kf[j, part * 64])
                                    gi = T.simd.vld(gc[i, part * 64])
                                    gj = T.simd.vld(gc[j, part * 64])
                                    weight = T.simd.vexp(T.simd.vsub(gi, gj, full), full)
                                    product = T.simd.vmul(T.simd.vmul(qi, kj, full), weight, full)
                                    total[0] = T.simd.vadd(total[0], product, full)
                                T.simd.vsts(
                                    values[i, j],
                                    T.simd.vcadd(total[0], full),
                                    one,
                                    dist="ONEPT_B32",
                                )
                    with T.SimtVF(threads=256):
                        for i, j in T.Parallel(R, R):
                            out[i, j] = values[i, j] * scale
                    T.copy(out, A[index, r : r + R, r : r + R])
                    if r == 0:
                        T.copy(zero, A[index, :R, R:C])

    return main


def scores_cross(J, K, dtype):
    grid = min(32, J)

    @T.prim_func
    def main(
        QX: T.Tensor((J, 16, K), dtype),
        KX: T.Tensor((J, 16, K), dtype),
        A: T.Tensor((J, 32, 32), dtype),
    ):
        with T.Kernel(grid) as bx:
            q = T.alloc_l1((16, K), dtype)
            k = T.alloc_l1((16, K), dtype)
            q0 = T.alloc_l0a((16, K), dtype)
            k0 = T.alloc_l0b((16, K), dtype)
            c = T.alloc_l0c((16, 16), "float32")
            for step in T.serial((J + grid - 1) // grid):
                job = step * grid + bx
                if job < J:
                    T.copy(QX[job, :, :], q)
                    T.copy(KX[job, :, :], k)
                    T.copy(q, q0)
                    T.copy(k, k0)
                    T.gemm(q0, k0, c, transpose_B=True, clear_accum=True)
                    T.copy(c, A[job, 16:32, :16])

    return main


class ChunkedGLA:
    def __init__(self, B, S, H, K, V, dtype, scale, initial):
        C = 32
        self.length = S
        N = (S + C - 1) // C
        J = B * N * H
        self.mixed = K == 128
        if self.mixed:
            factories = [
                (preprocessing_mixed(B, S, H, K, V, dtype, scale, C), [4, 5, 6, 7, 8, 9, 10]),
                (summary(J, K, V, dtype, C), [2]),
                (scan(B, N, H, K, V, dtype, initial), [3, 4]),
                (scores_diagonal(B, S, H, K, dtype, scale), [3]),
                (scores_cross(J, K, dtype), []),
                (output(B, S, H, K, V, dtype, C), [4]),
            ]
        else:
            factories = [
                (preprocessing(B, S, H, K, V, dtype, scale, C), [4, 5, 6, 7, 8]),
                (summary(J, K, V, dtype, C), [2]),
                (scan(B, N, H, K, V, dtype, initial), [3, 4]),
                (scores(B, S, H, K, dtype, scale, C), [3]),
                (output(B, S, H, K, V, dtype, C), [4]),
            ]
        self.kernels = [
            tilelang.compile(fn, target="ascend", out_idx=indices) for fn, indices in factories
        ]

    def __call__(self, q, k, v, g, state):
        prepared = self.kernels[0](q, k, v, g)
        gc, qg, kw, vb, decay = prepared[:5]
        contribution = self.kernels[1](kw, vb)
        boundaries, final = self.kernels[2](contribution, decay, state)
        attention = self.kernels[3](q, k, gc)
        if self.mixed:
            self.kernels[4](prepared[5], prepared[6], attention)
            out = self.kernels[5](qg, boundaries, attention, vb)
        else:
            out = self.kernels[4](qg, boundaries, attention, vb)
        return out[:, : self.length].contiguous(), final

    def get_kernel_source(self):
        return "\n\n".join(
            f"// Stage {i}\n" + kernel.get_kernel_source() for i, kernel in enumerate(self.kernels)
        )


build_chunked = functools.lru_cache(maxsize=64)(ChunkedGLA)


class PackedChunkedGLA:
    def __init__(self, lengths, H, K, V, dtype, scale, initial):
        self.lengths = lengths
        self.uniform = len(set(lengths)) == 1
        self.heads, self.keys, self.values = H, K, V
        if self.uniform:
            self.children = [
                build_chunked(len(lengths), lengths[0], H, K, V, dtype, scale, initial)
            ]
        else:
            self.children = [
                build_chunked(1, length, H, K, V, dtype, scale, initial) for length in lengths
            ]

    def __call__(self, q, k, v, g, state):
        import torch

        H, K, V = self.heads, self.keys, self.values
        if self.uniform:
            B, S = len(self.lengths), self.lengths[0]
            out, final = self.children[0](
                q.view(B, S, H, K),
                k.view(B, S, H, K),
                v.view(B, S, H, V),
                g.view(B, S, H, K),
                state,
            )
            return out.view(1, B * S, H, V), final
        outputs, states = [], []
        first = 0
        for seq, (length, child) in enumerate(zip(self.lengths, self.children, strict=True)):
            last = first + length
            out, final = child(
                q[:, first:last],
                k[:, first:last],
                v[:, first:last],
                g[:, first:last],
                state[seq : seq + 1],
            )
            outputs.append(out)
            states.append(final)
            first = last
        return torch.cat(outputs, dim=1), torch.cat(states, dim=0)

    def get_kernel_source(self):
        return "\n\n".join(
            f"// Packed sequence implementation {i}\n" + child.get_kernel_source()
            for i, child in enumerate(self.children)
        )


build_packed_chunked = functools.lru_cache(maxsize=64)(PackedChunkedGLA)
