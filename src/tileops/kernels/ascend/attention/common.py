"""Compiler settings and SIMD reduction shared by explicit GQA pipelines."""

import functools
import math

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from tileops.kernels.ascend.attention.config import configuration


def pass_configs():
    return {
        tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }


@functools.lru_cache(maxsize=128)
def decode_merge_body(call, splits, rows_per_task=1):
    """Reduce split statistics, optionally batching adjacent heads per DMA."""
    assert rows_per_task > 0
    B, H, HK, D = call.batch, call.heads, call.heads_kv, call.dim
    G, QR, NS = H // HK, configuration(call)[0], splits
    MR, RT = rows_per_task, math.ceil(G / rows_per_task)
    dtype = str(call.dtype).split(".")[-1]

    @T.macro
    def reduce_row(parts, moments, merged, row, r, full):
        maximum = S.alloc_var("float32")
        maximum = S.vld(moments[0, row + r, 0], "BRC_B32")
        for split in range(1, NS):
            maximum = S.vmax(maximum, S.vld(moments[split, row + r, 0], "BRC_B32"), full)
        for d in range(0, D, 64):
            numerator = S.alloc_var("float32")
            denominator = S.alloc_var("float32")
            numerator = S.vdup(T.float32(0), "float32", full)
            denominator = S.vdup(T.float32(0), "float32", full)
            for split in range(NS):
                weight = S.vexpdif(S.vld(moments[split, row + r, 0], "BRC_B32"), maximum, full)
                numerator = S.vadd(
                    numerator, S.vmul(S.vld(parts[split, r, d], "NORM"), weight, full), full
                )
                denominator = S.vadd(
                    denominator,
                    S.vmul(S.vld(moments[split, row + r, 1], "BRC_B32"), weight, full),
                    full,
                )
            if MR == 1:
                S.vsts(
                    merged[d],
                    S.vcvt(S.vdiv(numerator, denominator, full), dtype, full),
                    full,
                    "PK_B32",
                )
            else:
                S.vsts(
                    merged[r, d],
                    S.vcvt(S.vdiv(numerator, denominator, full), dtype, full),
                    full,
                    "PK_B32",
                )

    @T.macro
    def merge(Partial, Stats, O, start, stride):
        parts = T.alloc_shared((NS, MR, D), "float32")
        moments = T.alloc_shared((NS, QR, 2), "float32")
        merged = T.alloc_shared((D,) if MR == 1 else (MR, D), dtype)
        for task in T.serial(start, B * HK * RT, stride):
            if MR == 1:
                batch, h = task // H, task % H
                bh, row = batch * HK + h // G, h % G
                T.copy(Partial[bh, :, row : row + 1, :], parts)
            else:
                bh, row = task // RT, task % RT * MR
                batch, head = bh // HK, bh % HK
                valid = T.min(MR, G - row)
                T.copy(Partial[bh, :, row : row + valid, :], parts[:, :valid, :])
            T.copy(Stats[bh, :, :, :], moments)
            T.ascend_set_flag("MTE2_V", 0)
            T.ascend_wait_flag("MTE2_V", 0)
            with T.SimdVF():
                full = S.pset(32, "PAT_ALL")
                if MR == 1:
                    reduce_row(parts, moments, merged, row, 0, full)
                else:
                    for r in range(valid):
                        reduce_row(parts, moments, merged, row, r, full)
            T.ascend_set_flag("V_MTE3", 0)
            T.ascend_wait_flag("V_MTE3", 0)
            if MR == 1:
                T.copy(merged, O[batch, 0, h, :])
            else:
                T.copy(merged[:valid, :], O[batch, 0, head * G + row : head * G + row + valid, :])
            T.ascend_set_flag("MTE3_V", 0)
            T.ascend_wait_flag("MTE3_V", 0)
            T.ascend_set_flag("V_MTE2", 0)
            T.ascend_wait_flag("V_MTE2", 0)

    return merge
