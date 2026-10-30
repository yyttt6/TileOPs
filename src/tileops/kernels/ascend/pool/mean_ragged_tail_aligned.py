"""Exact ragged-tail vector staging; per-forward gate is owned by mean.py."""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T

from tileops.kernels.ascend.pool.mean_ragged_tail_gate import PARAMS
from tileops.kernels.ascend.pool.mean_ragged_tail_gate import eligible_tensors as eligible_tensors

VARIANT = "mean07-raggedtail-aligned"
ALIGNED = VARIANT + "-vector"
SCALAR = VARIANT + "-scalar19"
SOURCE_SHA = "392a21cc5a88ce21e43d792025b91350a9476344ce3213ab748ab2c0e10634c4"


def supports(*params):
    return tuple(params) == PARAMS


def call_params(call):
    return (
        call.batch_size,
        call.seq_len,
        call.heads * call.dim,
        call.chunk_size,
        call.chunks_per_batch,
        call.seq_num,
        call.use_offsets,
        str(call.dtype).split(".")[-1],
        str(call.accum_dtype).split(".")[-1],
    )


def configuration(path):
    return dict(
        path=path,
        variant=VARIANT,
        feature_tile=512,
        token_tile=32,
        stages=1,
        explicit_ub_bytes=35840,
        dispatch="per-forward-effective-pointer-and-stride",
        x_alignment_bytes=32,
        source_sha256=SOURCE_SHA,
    )


def program(*params):
    assert supports(*params)
    return _general_mean_program(*params).with_attr("tilelang_callee_allocated_outputs", 0)


@lru_cache(maxsize=8)
def _build_cached(path, source_sha, params):
    assert path == ALIGNED and source_sha == SOURCE_SHA and params == PARAMS
    return tilelang.compile(program(*params), target="ascend", out_idx=[])


def build(*params):
    return _build_cached(ALIGNED, SOURCE_SHA, tuple(params))


def mean_tiling(width: int, chunk: int) -> tuple[int, int]:
    return (min(512, max(64, 1 << (width - 1).bit_length())), min(32, chunk))


def _general_mean_program(batch, seq, width, chunk, chunks, seq_num, ragged, dtype, accum_dtype):
    (bw, tokens) = mean_tiling(width, chunk)
    feature_tiles = (width + bw - 1) // bw
    jobs = batch * chunks * feature_tiles
    blocks = min(64, max(1, jobs))

    @T.prim_func
    def main(
        X: T.Tensor((batch, seq, width), dtype),
        Offsets: T.Tensor((seq_num + 1,), "int32"),
        Indices: T.Tensor((chunks, 2), "int32"),
        Y: T.Tensor((batch, chunks, width), dtype),
    ):
        with T.Kernel(blocks) as core:
            staged = T.alloc_shared((tokens, bw), dtype)
            acc = T.alloc_shared((bw,), accum_dtype)
            output = T.alloc_shared((bw,), dtype)
            for iteration in T.serial((jobs + blocks - 1) // blocks):
                job = iteration * blocks + core
                if job < jobs:
                    col = job % feature_tiles * bw
                    row = job // feature_tiles % chunks
                    b = job // (feature_tiles * chunks)
                    start = T.alloc_var("int32")
                    end = T.alloc_var("int32")
                    if ragged:
                        sequence = Indices[row, 0]
                        start = Offsets[sequence] + Indices[row, 1] * chunk
                        end = T.min(start + chunk, Offsets[sequence + 1])
                    else:
                        start = row * chunk
                        end = T.min(start + chunk, seq)
                    with T.SimtVF(threads=128):
                        for f in T.Parallel(bw):
                            acc[f] = T.cast(0, accum_dtype)
                    for part in T.serial((chunk + tokens - 1) // tokens):
                        token = start + part * tokens
                        if token < end:
                            if token + tokens <= end and col + bw <= width:
                                T.copy(X[b, token : token + tokens, col : col + bw], staged)
                            else:
                                with T.SimtVF(threads=256):
                                    for t, f in T.Parallel(tokens, bw):
                                        staged[t, f] = T.cast(0, dtype)
                                        if token + t < end and col + f < width:
                                            staged[t, f] = X[b, token + t, col + f]
                            if accum_dtype == "float32":
                                with T.SimdVF():
                                    for v in T.serial(bw // 64):
                                        running = T.simd.alloc_local((1,), "float32")
                                        running[0] = T.simd.vld(acc[v * 64])
                                        for t in T.serial(tokens):
                                            if dtype == "float32":
                                                value = T.simd.vld(staged[t, v * 64])
                                            else:
                                                value = T.simd.vcvt(
                                                    T.simd.vld(staged[t, v * 64], dist="UNPK_B16"),
                                                    "float32",
                                                    part=0,
                                                )
                                            running[0] = T.simd.vadd(running[0], value)
                                        T.simd.vsts(acc[v * 64], running[0])
                            else:
                                with T.SimtVF(threads=256):
                                    local = T.alloc_fragment((bw,), accum_dtype)
                                    T.copy(acc, local)
                                    for t in T.serial(tokens):
                                        for f in T.Parallel(bw):
                                            local[f] = local[f] + T.cast(staged[t, f], accum_dtype)
                                    T.copy(local, acc)
                    denominator = T.alloc_var(accum_dtype)
                    denominator = T.cast(end - start, accum_dtype)
                    if accum_dtype == "float32":
                        reciprocal = T.alloc_var("float32")
                        reciprocal = T.float32(1.0) / denominator
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            for v in T.serial(bw // 64):
                                value = T.simd.vmuls(T.simd.vld(acc[v * 64]), reciprocal, full)
                                if dtype == "float32":
                                    T.simd.vsts(output[v * 64], value, full)
                                else:
                                    converted = T.simd.vcvt(value, dtype, full, sat=False)
                                    T.simd.vsts(output[v * 64], converted, full, "PK_B32")
                    else:
                        with T.SimtVF(threads=256):
                            for f in T.Parallel(bw):
                                output[f] = T.cast(acc[f] / denominator, dtype)
                    if col + bw <= width:
                        T.copy(output, Y[b, row, col : col + bw])
                    else:
                        with T.SimtVF(threads=128):
                            for f in T.Parallel(bw):
                                if col + f < width:
                                    Y[b, row, col + f] = output[f]

    return main
