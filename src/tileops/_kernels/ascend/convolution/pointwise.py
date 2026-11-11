"""Pointwise Cube convolution with hardware bias initialization."""

import functools

import tilelang
import tilelang.ascend.language as T
from tilelang.layout import make_ascend_major_k_layout

from tileops._csrc import csrc_include


@functools.lru_cache(maxsize=128)
def _small_contract(batch, groups, m, n, k, dtype, has_bias):
    assert n <= 64 and k <= 512 and k % 16 == 0 and m % 64 == 0
    bm = bn = 64
    tiles = batch * groups * (m // bm)
    cores = min(32, tiles)
    convert = 1 if dtype == "float16" else 0
    quant = 1 if dtype == "float16" else 16

    @T.prim_func
    def main(
        X: T.Tensor((batch, groups, k, n), dtype),
        W: T.Tensor((groups, m, k), dtype),
        Bias: T.Tensor((groups, m), dtype),
        Y: T.Tensor((batch, groups, m, n), dtype),
    ):
        with T.Kernel(cores) as core:
            xl = T.alloc_l1((bn, k), dtype)
            wl = T.alloc_l1((bm, k), dtype)
            bl = T.alloc_l1((bm,), dtype)
            a = T.alloc_l0a((bn, k), dtype)
            b = T.alloc_l0b((bm, k), dtype)
            acc = T.alloc_l0c((bn, bm), "float32")
            T.annotate_layout({b: make_ascend_major_k_layout(b)})
            T.annotate_buffer_versions({xl: 1, wl: 1, bl: 1, a: 1, b: 1, acc: 1})
            with T.Cube():
                for step in T.serial(T.ceildiv(tiles, cores)):
                    idx = step * cores + core
                    if idx < tiles:
                        mi = idx % (m // bm) * bm
                        g = idx // (m // bm) % groups
                        bi = idx // ((m // bm) * groups)
                        T.copy(X[bi, g, :, :bn], xl, transpose=True)
                        T.copy(W[g, mi : mi + bm, :], wl)
                        if has_bias:
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "tileops::conv_bias_gm",
                                    T.access_ptr(bl, "w"),
                                    T.access_ptr(Bias[g, mi], "r"),
                                    bm,
                                )
                            )
                        T.ascend_set_flag("MTE2_MTE1", 0)
                        T.ascend_wait_flag("MTE2_MTE1", 0)
                        T.copy(xl, a)
                        T.copy(wl, b)
                        if has_bias:
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "tileops::conv_bias_bt",
                                    T.access_ptr(bl, "r"),
                                    bm,
                                    convert,
                                )
                            )
                        T.ascend_set_flag("MTE1_M", 0)
                        T.ascend_wait_flag("MTE1_M", 0)
                        if has_bias:
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "tileops::conv_bias_mad",
                                    T.access_ptr(acc, "rw"),
                                    T.access_ptr(a, "r"),
                                    T.access_ptr(b, "r"),
                                    bn,
                                    k,
                                    bm,
                                    T.uint64(0),
                                )
                            )
                        else:
                            T.gemm(a, b, acc, transpose_B=True, clear_accum=True)
                        T.ascend_set_flag("M_FIX", 0)
                        T.ascend_wait_flag("M_FIX", 0)
                        T.evaluate(
                            T.call_extern(
                                "void",
                                "tileops::conv_native_store",
                                T.access_ptr(Y[bi, g, mi, 0], "w"),
                                T.access_ptr(acc, "r"),
                                bm,
                                n,
                                n,
                                bn,
                                quant,
                            )
                        )
                        T.ascend_set_flag("FIX_M", 0)
                        T.ascend_wait_flag("FIX_M", 0)
                        if tiles > cores:
                            T.ascend_set_flag("M_MTE1", 3)
                            T.ascend_wait_flag("M_MTE1", 3)
                            T.ascend_set_flag("MTE1_MTE2", 3)
                            T.ascend_wait_flag("MTE1_MTE2", 3)

    return tilelang.compile(
        main,
        target="ascend",
        out_idx=-1,
        compile_flags=csrc_include("ascend_convolution.h"),
    )


@functools.lru_cache(maxsize=128)
def _pointwise_contract(
    batch, groups, m, n, k, dtype, has_bias, bm_override=0, bn_override=0, bk_override=0
):
    if n <= 64 and k <= 512 and m % 64 == 0 and not (bm_override or bn_override or bk_override):
        return _small_contract(batch, groups, m, n, k, dtype, has_bias)
    assert k % 16 == 0 and m % 16 == 0
    bk = bk_override or min(256, k)
    bm = bm_override or min(128, 16384 // bk, m)
    bn = bn_override or min(256, 16384 // bk, max(16, (n + 15) // 16 * 16))
    assert m % bm == 0 and bm * bk <= 16384 and bn * bk <= 16384 and bm * bn <= 65536
    mt = m // bm
    nt = (n + bn - 1) // bn
    tiles = batch * groups * mt * nt
    cores = min(32, tiles)
    nk = (k + bk - 1) // bk
    convert = 1 if dtype == "float16" else 0
    quant = 1 if dtype == "float16" else 16

    @T.prim_func
    def main(
        X: T.Tensor((batch, groups, k, n), dtype),
        W: T.Tensor((groups, m, k), dtype),
        Bias: T.Tensor((groups, m), dtype),
        Y: T.Tensor((batch, groups, m, n), dtype),
    ):
        with T.Kernel(cores) as core:
            xl = T.alloc_l1((2, bn, bk), dtype)
            wl = T.alloc_l1((2, bm, bk), dtype)
            a = T.alloc_l0a((2, bn, bk), dtype)
            b = T.alloc_l0b((2, bm, bk), dtype)
            acc = T.alloc_l0c((bn, bm), "float32")
            bl = T.alloc_l1((bm,), dtype)
            T.annotate_layout({b: make_ascend_major_k_layout(b)})
            T.annotate_buffer_versions({xl: 1, wl: 1, a: 1, b: 1, acc: 1, bl: 1})
            with T.Cube():
                for step in T.serial(T.ceildiv(tiles, cores)):
                    idx = step * cores + core
                    if idx < tiles:
                        ni = idx % nt * bn
                        mi = idx // nt % mt * bm
                        g = idx // (nt * mt) % groups
                        bi = idx // (nt * mt * groups)
                        if has_bias:
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "tileops::conv_bias_gm",
                                    T.access_ptr(bl, "w"),
                                    T.access_ptr(Bias[g, mi], "r"),
                                    bm,
                                )
                            )
                            T.ascend_set_flag("MTE2_MTE1", 3)
                            T.ascend_wait_flag("MTE2_MTE1", 3)
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "tileops::conv_bias_bt",
                                    T.access_ptr(bl, "r"),
                                    bm,
                                    convert,
                                )
                            )
                        T.ascend_set_flag("MTE1_MTE2", 0)
                        T.ascend_set_flag("MTE1_MTE2", 1)
                        T.ascend_set_flag("M_MTE1", 0)
                        T.ascend_set_flag("M_MTE1", 1)
                        for t in T.serial(nk + 2):
                            if t < nk:
                                slot = t % 2
                                T.ascend_wait_flag("MTE1_MTE2", slot)
                                T.copy(
                                    X[bi, g, t * bk : t * bk + bk, ni : ni + bn],
                                    xl[slot, :, :],
                                    transpose=True,
                                )
                                T.copy(
                                    W[g, mi : mi + bm, t * bk : t * bk + bk],
                                    wl[slot, :, :],
                                )
                                T.ascend_set_flag("MTE2_MTE1", slot)
                            if t > 0 and t <= nk:
                                slot = (t - 1) % 2
                                T.ascend_wait_flag("MTE2_MTE1", slot)
                                T.ascend_wait_flag("M_MTE1", slot)
                                T.copy(xl[slot, :, :], a[slot, :, :])
                                T.copy(wl[slot, :, :], b[slot, :, :])
                                T.ascend_set_flag("MTE1_MTE2", slot)
                                T.ascend_set_flag("MTE1_M", slot)
                            if t > 1:
                                slot = (t - 2) % 2
                                T.ascend_wait_flag("MTE1_M", slot)
                                if has_bias and t == 2:
                                    T.evaluate(
                                        T.call_extern(
                                            "void",
                                            "tileops::conv_bias_mad",
                                            T.access_ptr(acc, "rw"),
                                            T.access_ptr(a[slot, 0, 0], "r"),
                                            T.access_ptr(b[slot, 0, 0], "r"),
                                            bn,
                                            bk,
                                            bm,
                                            T.uint64(0),
                                        )
                                    )
                                else:
                                    T.gemm(
                                        a[slot, :, :],
                                        b[slot, :, :],
                                        acc,
                                        transpose_B=True,
                                        clear_accum=t == 2,
                                    )
                                T.ascend_set_flag("M_MTE1", slot)
                        T.ascend_wait_flag("MTE1_MTE2", 0)
                        T.ascend_wait_flag("MTE1_MTE2", 1)
                        T.ascend_wait_flag("M_MTE1", 0)
                        T.ascend_wait_flag("M_MTE1", 1)
                        T.ascend_set_flag("M_FIX", 0)
                        T.ascend_wait_flag("M_FIX", 0)
                        T.evaluate(
                            T.call_extern(
                                "void",
                                "tileops::conv_native_store",
                                T.access_ptr(Y[bi, g, mi, ni], "w"),
                                T.access_ptr(acc, "r"),
                                bm,
                                T.min(bn, n - ni),
                                n,
                                bn,
                                quant,
                            )
                        )
                        T.ascend_set_flag("FIX_M", 0)
                        T.ascend_wait_flag("FIX_M", 0)

    return tilelang.compile(
        main,
        target="ascend",
        out_idx=-1,
        compile_flags=csrc_include("ascend_convolution.h"),
    )
