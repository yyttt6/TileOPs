"""Ascend sequence-chunk mean, with live uniform/ragged chunk addressing.

Full tiles use 2D DMA and explicit SIMD accumulation over contiguous features.
Only partial token/feature tiles use masked SIMT staging. Accumulation is kept in
the requested dtype; the canonical fp32 path uses 64-lane vector registers.
Uniform full 64-token chunks and three selected ragged metadata shapes use
two-stage SIMD programs. Other calls retain the general program.
"""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.ascend.pool import mean_fullwidth6, mean_fullwidth_stream, mean_uniform_wide
from tileops.kernels.ascend.pool import mean_ragged_tail_aligned as _ragged_tail_aligned
from tileops.kernels.ascend.pool.adaptive_helpers import mean_tiling
from tileops.kernels.ascend.pool.mean_ragged_selected import (
    configuration as _ragged_selected_configuration,
)
from tileops.kernels.ascend.pool.mean_ragged_selected import (
    mean_program as _ragged_selected_program,
)
from tileops.kernels.ascend.pool.mean_ragged_selected import (
    supports as _ragged_selected_supports,
)
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.pool.call_spec import MeanPoolingCall, MeanPoolingFwdInterface

__all__ = ["AscendMeanPoolingFwdKernel"]


def _uniform64_tiling(width, dtype):
    # The promoted variant is fixed at feature_tile=512, stages=2. FP32 inputs
    # reduce the width to 256 to keep the same explicit UB budget as fp16/bf16.
    bw = min(512, max(64, 1 << (width - 1).bit_length()))
    itemsize = 4 if dtype == "float32" else 2
    while 2 * bw * itemsize * 65 > 192 * 1024:
        bw //= 2
    return bw, 2 * bw * itemsize * 65


def mean_configuration(batch, seq, width, chunk, chunks, ragged, dtype, accum_dtype, seq_num=None):
    """Host-only selection metadata; also exposed on the kernel's config."""
    params = (batch, seq, width, chunk, chunks, seq_num, ragged, dtype, accum_dtype)
    if not ragged and mean_fullwidth6.supports("mean17-fullwidth6", *params):
        return mean_fullwidth6.configuration("mean17-fullwidth6", *params)
    if mean_fullwidth_stream.supports("mean08-fullwidth4", *params):
        return mean_fullwidth_stream.configuration("mean08-fullwidth4", *params)
    if mean_uniform_wide.supports("mean03-w768", *params):
        return mean_uniform_wide.configuration("mean03-w768", *params)
    if _ragged_selected_supports(*params):
        return _ragged_selected_configuration(*params)
    if (
        batch > 0
        and seq > 0
        and width > 0
        and not ragged
        and chunk == 64
        and seq % 64 == 0
        and chunks == seq // 64
        and accum_dtype == "float32"
        and dtype in ("float16", "bfloat16", "float32")
    ):
        bw, ub_bytes = _uniform64_tiling(width, dtype)
        if width % bw == 0:
            return {
                "path": "uniform64-simd",
                "feature_tile": bw,
                "token_tile": 64,
                "stages": 2,
                "explicit_ub_bytes": ub_bytes,
            }
    bw, tokens = mean_tiling(width, chunk)
    return {"path": "general", "feature_tile": bw, "token_tile": tokens, "stages": 1}


def _uniform64_simd_program(
    batch,
    seq,
    width,
    chunk,
    chunks,
    seq_num,
    ragged,
    dtype,
):
    """Uniform FP32-accumulation PrimFunc; no compilation or device access."""
    stages = 2
    bw, _ = _uniform64_tiling(width, dtype)
    if ragged or chunk != 64 or seq % 64 or chunks != seq // 64 or width % bw:
        raise ValueError("uniform fast path requires uniform full 64-token and feature tiles")
    jobs = batch * chunks * (width // bw)
    if jobs <= 0:
        raise ValueError("empty calls use the production wrapper")
    blocks = min(64, jobs)
    while jobs % blocks:
        blocks -= 1

    @T.prim_func
    def main(
        X: T.Tensor((batch, seq, width), dtype),
        Offsets: T.Tensor((seq_num + 1,), "int32"),
        Indices: T.Tensor((chunks, 2), "int32"),
        Y: T.Tensor((batch, chunks, width), dtype),
    ):
        with T.Kernel(blocks) as core:
            staged = T.alloc_shared((64, bw), dtype)
            output = T.alloc_shared((bw,), dtype)
            if stages > 1:
                T.annotate_buffer_versions({staged: stages, output: stages})
            # staged[job]: GM->UB producer, sole compute-VF consumer.
            # output[job]: sole compute-VF producer, UB->GM consumer.
            # Different jobs share no state. Alternating raw/output versions
            # permit the next DMA load and previous DMA store to overlap compute;
            # generated MTE2/V/MTE3 events must guard reuse of each version.
            for iteration in T.Pipelined(jobs // blocks, num_stages=stages):
                job = iteration * blocks + core
                col = job % (width // bw) * bw
                row = job // (width // bw) % chunks
                b = job // ((width // bw) * chunks)
                T.copy(X[b, row * 64 : (row + 1) * 64, col : col + bw], staged)
                with T.SimdVF():
                    full = T.simd.pset(32)
                    for v in T.serial(bw // 64):
                        running = T.simd.alloc_local((1,), "float32")
                        running[0] = T.simd.vdup(0.0, "float32", full)
                        for t in T.serial(64):
                            if dtype == "float32":
                                value = T.simd.vld(staged[t, v * 64])
                            else:
                                value = T.simd.vcvt(
                                    T.simd.vld(staged[t, v * 64], dist="UNPK_B16"),
                                    "float32",
                                    part=0,
                                )
                            running[0] = T.simd.vadd(running[0], value, full)
                        average = T.simd.vmuls(running[0], 1.0 / 64, full)
                        if dtype == "float32":
                            T.simd.vsts(output[v * 64], average, full)
                        else:
                            converted = T.simd.vcvt(average, dtype, full, sat=False)
                            T.simd.vsts(output[v * 64], converted, full, "PK_B32")
                T.copy(output, Y[b, row, col : col + bw])

    return main.with_attr("tilelang_callee_allocated_outputs", 0)


def _general_mean_program(batch, seq, width, chunk, chunks, seq_num, ragged, dtype, accum_dtype):
    bw, tokens = mean_tiling(width, chunk)
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
                                    # Offset views need element-aligned GM reads.
                                    for t, f in T.Parallel(tokens, bw, coalesced_width=T.int32(1)):
                                        staged[t, f] = T.cast(0, dtype)
                                        if token + t < end and col + f < width:
                                            staged[t, f] = X[b, token + t, col + f]
                            if accum_dtype == "float32":
                                with T.SimdVF():
                                    for v in T.serial(bw // 64):
                                        # An addressable mutable register carries the
                                        # recurrence across the serial token loop.
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
                    # Keep integer->float count conversion scalar. The SIMT
                    # vectorizer emitted a float2/int2 bitcast for the old
                    # per-lane denominator, turning 64 into ~8.97e-44.
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


def mean_program(batch, seq, width, chunk, chunks, seq_num, ragged, dtype, accum_dtype):
    config = mean_configuration(
        batch, seq, width, chunk, chunks, ragged, dtype, accum_dtype, seq_num=seq_num
    )
    if config["path"] == "mean17-fullwidth6":
        return mean_fullwidth6.mean_program(
            "mean17-fullwidth6",
            batch,
            seq,
            width,
            chunk,
            chunks,
            seq_num,
            ragged,
            dtype,
            accum_dtype,
        )
    if config["path"] == "mean08-fullwidth4":
        return mean_fullwidth_stream.mean_program(
            "mean08-fullwidth4",
            batch,
            seq,
            width,
            chunk,
            chunks,
            seq_num,
            ragged,
            dtype,
            accum_dtype,
        )
    if config["path"] == "mean03-w768":
        return mean_uniform_wide.mean_program(
            "mean03-w768", batch, seq, width, chunk, chunks, seq_num, ragged, dtype, accum_dtype
        )
    if config["path"] == "ragged-live-metadata-simd":
        return _ragged_selected_program(
            batch, seq, width, chunk, chunks, seq_num, ragged, dtype, accum_dtype
        )
    if config["path"] == "uniform64-simd":
        return _uniform64_simd_program(batch, seq, width, chunk, chunks, seq_num, ragged, dtype)
    return _general_mean_program(
        batch, seq, width, chunk, chunks, seq_num, ragged, dtype, accum_dtype
    )


@functools.lru_cache(maxsize=64)
def _build(batch, seq, width, chunk, chunks, seq_num, ragged, dtype, accum_dtype):
    program = mean_program(batch, seq, width, chunk, chunks, seq_num, ragged, dtype, accum_dtype)
    return tilelang.compile(
        program.with_attr("tilelang_callee_allocated_outputs", 0),
        target="ascend",
        out_idx=[],
    )


class AscendMeanPoolingFwdKernel(Kernel, MeanPoolingFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: MeanPoolingCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: MeanPoolingCall):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self.config = mean_configuration(
            call.batch_size,
            call.seq_len,
            call.heads * call.dim,
            call.chunk_size,
            call.chunks_per_batch,
            call.use_offsets,
            str(call.dtype).split(".")[-1],
            str(call.accum_dtype).split(".")[-1],
            seq_num=call.seq_num,
        )
        self.kernel = None
        if call.batch_size * call.chunks_per_batch * call.heads * call.dim:
            self.kernel = _build(
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

        params = _ragged_tail_aligned.call_params(call)
        if _ragged_tail_aligned.supports(*params):
            self._ragged_tail_kernels = {
                _ragged_tail_aligned.SCALAR: self.kernel,
                _ragged_tail_aligned.ALIGNED: _ragged_tail_aligned.build(*params),
            }
            self.config = _ragged_tail_aligned.configuration(_ragged_tail_aligned.SCALAR)

    def forward(
        self, x: torch.Tensor, offsets: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        c = self.call
        y = torch.empty(
            (c.batch_size, c.chunks_per_batch, c.heads, c.dim),
            dtype=x.dtype,
            device=x.device,
        )
        params = _ragged_tail_aligned.call_params(c)
        if _ragged_tail_aligned.supports(*params):
            x3 = x.view(c.batch_size, c.seq_len, c.heads * c.dim)
            y3 = y.view(c.batch_size, c.chunks_per_batch, c.heads * c.dim)
            path = (
                _ragged_tail_aligned.ALIGNED
                if _ragged_tail_aligned.eligible_tensors(params, x3, offsets, indices, y3)
                else _ragged_tail_aligned.SCALAR
            )
            selected = self._ragged_tail_kernels[path]
            self.kernel, self.config = selected, _ragged_tail_aligned.configuration(path)
            selected(x3, offsets, indices, y3)
            return y
        if self.kernel is not None:
            self.kernel(
                x.view(c.batch_size, c.seq_len, c.heads * c.dim),
                offsets,
                indices,
                y.view(c.batch_size, c.chunks_per_batch, c.heads * c.dim),
            )
        return y


register_kernel_type("MeanPoolingFwdOp", "ascend", AscendMeanPoolingFwdKernel)
