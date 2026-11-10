"""SIMD weight packing for channel-aligned spatial convolutions."""

import functools

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


@functools.lru_cache(maxsize=128)
def _pack_3d_weights(co, ci, kh, kw, dtype, kd=1):
    assert ci % 16 == 0 and co % 8 == 0
    reduction = ci * kd * kh * kw
    pitch = (reduction + 127) // 128 * 128
    assert reduction < 65536
    # The 2D channel/tap permutation repeats every 128 input channels.
    # Keep volume packing on its full index table because depth changes the order.
    period = 128 * kh * kw if kd == 1 and ci % 128 == 0 else pitch
    tile = min(8, max(1, co // 64), max(1, 8192 // reduction))
    cores = min(64, co // tile)

    @T.prim_func
    def main(
        W: T.Tensor((co, reduction), dtype),
        Unused: T.Tensor((co,), dtype),
        P: T.Tensor((co, reduction), dtype),
    ):
        with T.Kernel(cores) as core:
            raw = T.alloc_shared((tile, reduction), dtype)
            packed = T.alloc_shared((tile, pitch), dtype)
            index = T.alloc_shared((period,), "uint16")
            with T.SimtVF(threads=256):
                for j in T.Parallel(period, coalesced_width=T.int32(1)):
                    channel = j // (kh * kw * 16) % (ci // 16) * 16 + j % 16
                    kz = j // (ci * kh * kw)
                    tap = j // 16 % (kh * kw)
                    index[j] = T.cast(
                        T.if_then_else(
                            j < reduction, channel * kd * kh * kw + kz * kh * kw + tap, 0
                        ),
                        "uint16",
                    )
            for step in T.serial(T.ceildiv(co, cores * tile)):
                start = (step * cores + core) * tile
                if start < co:
                    T.copy(W[start : start + tile, :], raw)
                    with T.SimdVF():
                        mask = S.pset(16)
                        for block in T.serial(pitch // period):
                            for part in T.serial(period // 128):
                                offsets = S.vld(index[part * 128])
                                for row in T.Unroll(tile, explicit=True):
                                    value = S.vgather2(raw[row, block * period], offsets, mask)
                                    S.vsts(
                                        packed[row, block * period + part * 128],
                                        value,
                                        mask,
                                        "NORM_B16",
                                    )
                    T.copy(packed[:, :reduction], P[start : start + tile, :])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _simd_3d_supported(ci, co, groups, window):
    return (
        groups == 1
        and window == (3, 3, 3)
        and ci >= 64
        and ci % 16 == 0
        and co >= 128
        and co % 128 == 0
        and ci * 27 <= 8192
    )
