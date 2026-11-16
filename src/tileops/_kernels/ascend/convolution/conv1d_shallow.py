"""Pack both operands of a shallow convolution in one Vector launch."""

import functools

import tilelang
import tilelang.ascend.language as T


@functools.lru_cache(maxsize=128)
def build_shallow_pack(c):
    dtype = str(c.dtype).split(".")[-1]
    batch, groups, ci, co = c.n, c.groups, c.c_in_g, c.c_out_g
    length, out, kw = c.l_in, c.out_l, c.kernel_l
    k, kp = ci * kw, 16
    mp, np = (co + 63) // 64 * 64, (out + 127) // 128 * 128
    xtotal, wtotal = batch * groups * kp * np, groups * mp * kp
    width = 2048
    nt = (np + width - 1) // width
    tiles = batch * groups * kp * nt
    dma = c.stride_l == 1 and c.pad_left == 0
    jobs = tiles if dma else (xtotal + 255) // 256
    blocks = min(64, max(jobs, (wtotal + 255) // 256))

    @T.prim_func
    def pack(
        X: T.Tensor((batch, groups * ci, length), dtype),
        W: T.Tensor((groups * co, ci, kw), dtype),
        P: T.Tensor((batch * groups, kp, np), dtype),
        Q: T.Tensor((groups, mp, kp), dtype),
    ):
        with T.Kernel(blocks) as bx:
            if dma:
                raw = T.reshape(X, (batch * groups * ci * length,))
                shared = T.alloc_shared((width,), dtype)
                for step in T.serial(T.ceildiv(tiles, blocks)):
                    tile = step * blocks + bx
                    if tile < tiles:
                        col = tile % nt * width
                        row = tile // nt % kp
                        bg = tile // (nt * kp)
                        valid = T.min(width, np - col)
                        with T.SimdVF():
                            full = T.simd.pset(16)
                            zero = T.simd.vdup(T.cast(0, dtype), dtype, full)
                            for part in T.serial(width // 128):
                                T.simd.vsts(shared[part * 128], zero, full, "NORM_B16")
                        if row < k:
                            pos = col + (row % kw) * c.dilation_l
                            hi = T.min(T.min(valid, out - col), length - pos)
                            if hi > 0:
                                start = (bg * ci + row // kw) * length + pos
                                T.copy(raw[start : start + hi], shared[0:hi])
                        T.copy(shared[0:valid], P[bg, row, col : col + valid])
            else:
                with T.SimtVF(threads=256):
                    for step in T.serial(T.ceildiv(xtotal, blocks * 256)):
                        for lane in T.Parallel(256):
                            index = (step * blocks + bx) * 256 + lane
                            if index < xtotal:
                                j = index % np
                                r = index // np % kp
                                bg = index // (np * kp)
                                pos = j * c.stride_l - c.pad_left + (r % kw) * c.dilation_l
                                value = T.alloc_var(dtype)
                                value = T.cast(0, dtype)
                                if r < k and j < out and pos >= 0 and pos < length:
                                    value = X[bg // groups, bg % groups * ci + r // kw, pos]
                                P[bg, r, j] = value
            with T.SimtVF(threads=256):
                for step in T.serial(T.ceildiv(wtotal, blocks * 256)):
                    for lane in T.Parallel(256):
                        index = (step * blocks + bx) * 256 + lane
                        if index < wtotal:
                            r = index % kp
                            m = index // kp % mp
                            g = index // (mp * kp)
                            value = T.alloc_var(dtype)
                            value = T.cast(0, dtype)
                            if m < co and r < k:
                                value = W[g * co + m, r // kw, r % kw]
                            Q[g, m, r] = value

    return tilelang.compile(pack, target="ascend", out_idx=[2, 3])
