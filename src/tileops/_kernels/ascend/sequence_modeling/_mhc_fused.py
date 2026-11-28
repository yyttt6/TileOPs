"""Fused four-stream BF16 mHC for the Ascend manifest workloads."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout


MANIFEST_SHAPES = frozenset({(1, 1280), (2, 1920), (4, 2560)})


def export_launcher(kernel, device):
    adapter = kernel.adapter
    prepare = getattr(adapter, "_prepare_torch_device", None)
    export = getattr(adapter, "get_exportable_executable", None)
    if prepare is None or export is None:
        return kernel
    prepare(device)
    return export()["main"]


@T.macro
def _apply_weights(x_ub, weights_ub, res_ub, layer_ub, tile, weight_stride, weight_offset=0):
    with T.SimdVF():
        for chunk in T.serial(tile // 64):
            col = chunk * 64
            streams = S.alloc_local((4,), T.float32)
            for i in T.unroll(4, explicit=True):
                streams[i] = S.vcvt(S.vld(x_ub[i, col], dist="UNPK_B16"), T.float32, part=0)
            acc = S.alloc_local((1,), T.float32)
            acc[0] = S.vmul(streams[0], S.vld(weights_ub[weight_offset], dist="BRC_B32"))
            for i in T.unroll(1, 4, explicit=True):
                S.vmula(acc[0], streams[i], S.vld(weights_ub[weight_offset + i], dist="BRC_B32"))
            S.vsts(layer_ub[col], S.vcvt(acc[0], T.bfloat16), dist="PK_B32")
            for row in T.unroll(4, explicit=True):
                acc[0] = S.vmul(streams[0], S.vld(weights_ub[weight_offset + (row + 1) * weight_stride], dist="BRC_B32"))
                for i in T.unroll(1, 4, explicit=True):
                    S.vmula(acc[0], streams[i], S.vld(weights_ub[weight_offset + (row + 1) * weight_stride + i], dist="BRC_B32"))
                S.vsts(res_ub[row, col], S.vcvt(acc[0], T.bfloat16), dist="PK_B32")


@T.macro
def _normalize_token(
    logits, logit_offset, square_sum, bias_ub, weights_ub, weight_offset, post_ub, post_offset,
    width, repeat, eps, alpha_pre, alpha_post, alpha_res,
):
    residual = T.alloc_local((16,), T.float32)
    norm = T.sqrt(square_sum / width) + 0.0001
    for i in T.serial(4):
        pre = alpha_pre * logits[logit_offset + i] / norm + bias_ub[i]
        post = alpha_post * logits[logit_offset + 4 + i] / norm + bias_ub[4 + i]
        weights_ub[weight_offset + i] = 1.0 / (1.0 + T.exp(-pre))
        post_ub[post_offset + i] = 2.0 / (1.0 + T.exp(-post))
    for i in T.serial(4):
        row = i * 4
        for j in T.serial(4):
            q = 8 + i * 4 + j
            residual[row + j] = alpha_res * logits[logit_offset + q] / norm + bias_ub[q]
        row_max = T.alloc_var(T.float32, init=residual[row])
        for j in T.serial(1, 4):
            row_max = T.max(row_max, residual[row + j])
        for j in T.serial(4):
            residual[row + j] = T.exp(residual[row + j] - row_max)
    for iteration in T.serial(repeat):
        for i in T.serial(4):
            row_sum = T.alloc_var(T.float32, init=0.0)
            for j in T.serial(4):
                row_sum += residual[i * 4 + j]
            for j in T.serial(4):
                residual[i * 4 + j] /= row_sum + eps
        for j in T.serial(4):
            col_sum = T.alloc_var(T.float32, init=0.0)
            for i in T.serial(4):
                col_sum += residual[i * 4 + j]
            for i in T.serial(4):
                residual[i * 4 + j] /= col_sum + eps
    for i in T.serial(16):
        weights_ub[weight_offset + 4 + i] = residual[i]
    for i in T.serial(20, 24):
        weights_ub[weight_offset + i] = 0.0


@tilelang.jit(target="ascend", execution_backend="tvm_ffi")
def project_cube_fused(
    x, phi, partial, squares, partial_read, squares_read, bias, h_post, x_res, x_layer, *,
    batch: int, hidden: int, cores: int,
    repeat: int, eps: float, alpha_pre: float, alpha_post: float, alpha_res: float,
):
    width, chunk, tile = hidden * 4, 256, hidden
    splits = width // chunk
    rows = 16
    half = rows // 2
    cols = 32
    mad_k = chunk
    row_tiles = (batch + rows - 1) // rows
    jobs = row_tiles * splits
    x: T.Tensor[(batch, width), T.bfloat16]
    phi: T.Tensor[(width, 24), T.float32]
    partial: T.Tensor[(splits, row_tiles * rows, cols), T.float32]
    squares: T.Tensor[(splits, row_tiles * rows, 8), T.float32]

    partial_read: T.Tensor[(splits, row_tiles * rows, cols), T.float32]
    squares_read: T.Tensor[(splits, row_tiles * rows, 8), T.float32]
    bias: T.Tensor[(24,), T.float32]
    h_post: T.Tensor[(batch, 4), T.float32]

    x_res: T.Tensor[(batch, 4, hidden), T.bfloat16]
    x_layer: T.Tensor[(batch, hidden), T.bfloat16]

    with T.MixedKernel(cores) as (core, side):
        x_ub = T.alloc_shared((half, chunk), T.bfloat16)
        nz_ub = T.alloc_shared((half + 1, chunk), T.float32)
        T.annotate_layout({nz_ub: make_ascend_compact_nz_layout(nz_ub)})
        squares_ub = T.alloc_shared((half, 8), T.float32)
        a_l1 = T.alloc_l1((rows, chunk), T.float32)
        b_l1 = T.alloc_l1((chunk, cols), T.float32)
        a_l0 = T.alloc_l0a((rows, mad_k), T.float32)
        b_l0 = T.alloc_l0b((cols, mad_k), T.float32)
        acc = T.alloc_l0c((rows, cols), T.float32)

        T.import_source("#define MHC_SYNC(...) ({ __VA_ARGS__; 0; })\n")
        raw_ub = T.alloc_shared((half, cols), T.float32)
        partial_ub = T.alloc_shared((splits, cols), T.float32)
        split_squares_ub = T.alloc_shared((splits, 8), T.float32)
        bias_ub = T.alloc_shared((24,), T.float32)
        totals = T.alloc_shared((32,), T.float32)
        weights_ub = T.alloc_shared((24,), T.float32)
        post_ub = T.alloc_shared((8,), T.float32)

        streams_ub = T.alloc_shared((4, tile), T.bfloat16)
        res_ub = T.alloc_shared((4, tile), T.bfloat16)
        layer_ub = T.alloc_shared((tile,), T.bfloat16)
        streams = T.reshape(x, (batch, 4, hidden))

        T.set_hf32_mode("nearest_even")
        for step in T.Pipelined(jobs // cores):
            job = step * cores + core
            part = job // row_tiles
            m = job % row_tiles * rows
            vector_m = m + side * half
            k = part * chunk
            # The tail load zero-pads any rows outside the real batch.
            T.copy(x[vector_m:vector_m + half, k:k + chunk], x_ub)
            T.copy(phi[k:k + chunk, 0:cols], b_l1)
            with T.SimdVF():
                full = S.pset(32)
                first = S.pset(32, "PAT_VL1")
                eight = S.pset(32, "PAT_VL8")
                zero = S.vdup(T.float32(0), T.float32, full)
                nz_stride = T.int32((half + 1) << 16)
                for row in T.serial(half):
                    squared = S.alloc_local((1,), T.float32)
                    squared[0] = zero
                    for col_tile in T.unroll(chunk // 64, explicit=True):
                        col = col_tile * 64
                        values = S.vcvt(S.vld(x_ub[row, col], dist="UNPK_B16"), T.float32, part=0)
                        ptr = S.make_ubuf_ptr(T.access_ptr(nz_ub[row, col], "w", 1, 64), "float32")
                        S.vsstb(values, ptr, nz_stride, full)
                        S.vmula(squared[0], values, values, full)
                    total = S.vcadd(squared[0], full)
                    S.vsts(squares_ub[row, 0], S.vsel(total, zero, first), eight)
            # Both Vector halves publish their rows to the same Cube L1 tile.
            T.dual_copy(nz_ub[0:half, 0:chunk], a_l1)
            if step < jobs // cores - 1:
                T.copy(squares_ub, squares[part, vector_m:vector_m + half, 0:8])
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0, transpose=True)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            # Move the Cube result to Vector so both workspaces are published by MTE3.
            T.dual_copy(acc, raw_ub, unit_flag_ctrl=0)
            if step < jobs // cores - 1:
                T.copy(raw_ub, partial[part, vector_m:vector_m + half, 0:cols])

        # Every Vector half participates, including halves with no coefficient work.
        last_job = (jobs // cores - 1) * cores + core
        last_part = last_job // row_tiles
        last_row = last_job % row_tiles * rows + side * half
        with T.PerCoreTask():
            # Anchor publication to the final workspace stores.
            with T.Task():
                T.copy(raw_ub, partial[last_part, last_row:last_row + half, 0:cols])
                T.copy(squares_ub, squares[last_part, last_row:last_row + half, 0:8])
            T.ascend_sync_inter_arrive("PIPE_MTE3", 1)
            T.ascend_sync_inter_wait("PIPE_MTE3", 1)
            _drain = T.call_extern("int32", "MHC_SYNC", T.ascend_pipe_barrier("PIPE_MTE3"))
            _ready = T.call_extern("int32", "MHC_SYNC", T.ascend_set_flag("MTE3_MTE2", 7))
        first = core * 2 + side
        coefficient_jobs = T.ceildiv(T.max(batch - first, 0), cores * 2)
        with T.Stage(0):
            with T.Task():
                T.ascend_wait_flag("MTE3_MTE2", 7)
                T.copy(bias, bias_ub[0:24])
            for step in T.serial(coefficient_jobs):
                token = first + step * cores * 2
                T.copy(partial_read[0:splits, token, 0:cols], partial_ub)
                T.copy(squares_read[0:splits, token, 0:8], split_squares_ub)
                with T.SimtVF(threads=32):
                    for q in T.Parallel(32):
                        value = T.alloc_var(T.float32, init=0.0)
                        if q == 24:
                            for split in T.serial(splits):
                                value += split_squares_ub[split, 0]
                        else:
                            for split in T.serial(splits):
                                value += partial_ub[split, q]
                        totals[q] = value
                with T.SimtVF(threads=32):
                    for lane in T.Parallel(1):
                        _normalize_token(
                            totals, 0, totals[24], bias_ub, weights_ub, 0, post_ub, 0,
                            width, repeat, eps, alpha_pre, alpha_post, alpha_res,
                        )
                # Keep each token's weights on the Vector half that computed them.
                for column_tile in T.serial(hidden // tile):
                    start = column_tile * tile
                    T.copy(streams[token, 0, start], streams_ub)
                    _apply_weights(streams_ub, weights_ub, res_ub, layer_ub, tile, 4)
                    T.copy(res_ub, x_res[token, 0, start])
                    T.copy(layer_ub, x_layer[token, start])
                T.copy(post_ub[0:4], h_post[token, 0])


@tilelang.jit(target="ascend", execution_backend="tvm_ffi")
def post(x_layer, h_post, x_res, out, *, batch: int, hidden: int, cores: int):
    tile, stages, streams, total_streams = 256, 0, 4, 4
    tiles = (hidden + tile - 1) // tile
    groups = total_streams // streams
    x_layer: T.Tensor[(batch, hidden), T.bfloat16]
    h_post: T.Tensor[(batch, total_streams), T.float32]
    x_res: T.Tensor[(batch, total_streams, hidden), T.bfloat16]
    out: T.Tensor[(batch, total_streams, hidden), T.bfloat16]

    with T.Kernel(cores) as core:
        layer_ub = T.alloc_shared((tile,), T.bfloat16)
        post_ub = T.alloc_shared((8,), T.float32)
        res_ub = T.alloc_shared((streams, tile), T.bfloat16)
        out_ub = T.alloc_shared((streams, tile), T.bfloat16)
        for token_group, column_tile in T.Persistent(
            [batch * groups, tiles], cores, core, group_size=1, num_stages=stages,
        ):
            token = token_group // groups
            stream = token_group % groups * streams
            start = column_tile * tile
            count = T.min(tile, hidden - start)
            T.copy(x_layer[token, start:start + count], layer_ub[:count])
            T.copy(h_post[token, stream:stream + streams], post_ub[0:streams])
            T.copy(x_res[token, stream:stream + streams, start:start + count], res_ub[:, :count])
            with T.SimdVF():
                coeff = S.alloc_local((streams,), T.float32)
                for row in T.unroll(streams, explicit=True):
                    coeff[row] = S.vld(post_ub[row], dist="BRC_B32")
                for chunk in T.serial(tile // 64):
                    col = chunk * 64
                    layer = S.vcvt(S.vld(layer_ub[col], dist="UNPK_B16"), T.float32, part=0)
                    for row in T.unroll(streams, explicit=True):
                        residual = S.vcvt(S.vld(res_ub[row, col], dist="UNPK_B16"), T.float32, part=0)
                        result = S.vadd(residual, S.vmul(layer, coeff[row]))
                        S.vsts(out_ub[row, col], S.vcvt(result, T.bfloat16), dist="PK_B32")
            T.copy(out_ub[:, :count], out[token, stream:stream + streams, start:start + count], l2_cache_ctrl="normal_fv")
