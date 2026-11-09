"""Single-step recurrence with contiguous state-coordinate parallelism."""

import functools

import tilelang
import tilelang.ascend.language as T


@functools.lru_cache(maxsize=128)
def build_decode(
    batch, dim, seqlen, n, groups, dtype, has_skip, has_gate, has_bias, has_seq, has_init, softplus
):
    assert seqlen == 1 and n == 16
    rows = 1024
    total = batch * dim
    tiles = (total + rows - 1) // rows
    blocks = min(64, tiles)

    @T.prim_func
    def main(
        U: T.Tensor((batch, dim, 1), dtype),
        Delta: T.Tensor((batch, dim, 1), dtype),
        A: T.Tensor((dim, n), "float32"),
        B: T.Tensor((batch, groups, n, 1), dtype),
        C: T.Tensor((batch, groups, n, 1), dtype),
        D: T.Tensor((dim,), "float32"),
        Z: T.Tensor((batch, dim, 1), dtype),
        Bias: T.Tensor((dim,), "float32"),
        Seq: T.Tensor((batch, 1), "int32"),
        Initial: T.Tensor((batch, dim, n), "float32"),
        Y: T.Tensor((batch, dim, 1), "float32"),
        Final: T.Tensor((batch, dim, n), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            projection = T.alloc_shared((rows, n + 1), "float32")
            steps = T.alloc_shared((rows,), "float32")
            drives = T.alloc_shared((rows,), "float32")
            for step in T.serial(T.ceildiv(tiles, blocks)):
                tile = step * blocks + bx
                if tile < tiles:
                    with T.SimtVF(threads=256):
                        for r in T.Parallel(rows):
                            row = tile * rows + r
                            if row < total:
                                bi = row // dim
                                channel = row % dim
                                dt = T.alloc_var("float32")
                                dt = T.cast(Delta[bi, channel, 0], "float32")
                                if has_bias:
                                    dt = dt + Bias[channel]
                                if softplus:  # noqa: SIM102
                                    if dt <= T.float32(20):
                                        ev = T.exp(dt)
                                        if dt < T.float32(-4):
                                            dt = ev * (
                                                T.float32(1)
                                                + ev
                                                * (
                                                    T.float32(-0.5)
                                                    + ev
                                                    * (T.float32(1.0 / 3.0) + ev * T.float32(-0.25))
                                                )
                                            )
                                        else:
                                            dt = T.log(T.float32(1) + ev)
                                uf = T.cast(U[bi, channel, 0], "float32")
                                drive = dt * uf
                                steps[r] = dt
                                drives[r] = drive
                    with T.SimtVF(threads=256):
                        for r, k in T.Parallel(rows, n):
                            row = tile * rows + r
                            if row < total:
                                bi = row // dim
                                channel = row % dim
                                group = channel // (dim // groups)
                                dt_state = steps[r]
                                drive_state = drives[r]
                                old = T.alloc_var("float32")
                                old = T.float32(0)
                                if has_init:
                                    old = Initial[bi, channel, k]
                                state = T.exp(
                                    dt_state * A[channel, k]
                                ) * old + drive_state * T.cast(B[bi, group, k, 0], "float32")
                                Final[bi, channel, k] = state
                                projection[r, k] = state * T.cast(C[bi, group, k, 0], "float32")
                    with T.SimtVF(threads=256):
                        for r in T.Parallel(rows):
                            row = tile * rows + r
                            if row < total:
                                bi = row // dim
                                channel = row % dim
                                acc = T.alloc_var("float32")
                                acc = T.float32(0)
                                for k in T.serial(n):
                                    acc = acc + projection[r, k]
                                if has_skip:
                                    acc = acc + T.cast(U[bi, channel, 0], "float32") * D[channel]
                                if has_gate:
                                    zz = T.cast(Z[bi, channel, 0], "float32")
                                    acc = acc * (zz / (T.float32(1) + T.exp(-zz)))
                                Y[bi, channel, 0] = acc

    return tilelang.compile(main, target="ascend", out_idx=[10, 11])
