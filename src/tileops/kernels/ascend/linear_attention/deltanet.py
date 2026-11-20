"""Ascend DeltaNet forward with chunked prefill and single-token decode."""

import tilelang
import tilelang.ascend.language as T
import torch

import tileops.backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.linear_attention.call_spec import DeltaNetCall, DeltaNetFwdInterface

_CHUNK = 32
_CUBE_BLOCKS = 32
_THREADS = 128
_VECTOR_BLOCKS = 64
_SOLVE_PANEL = 8
_PACK_METADATA_BYTES = 64 * 1024


def _offset_lanes(num_sequences):
    return ((num_sequences + 4) // 4) * 4


def _versions(*buffers):
    return dict.fromkeys(buffers, 1)


def _unpack_token(src):
    return T.simd.vcvt(T.simd.vld(src, dist="UNPK_B16"), "float32", part=0)


def _token_lane(buf, row, col):
    return buf[col] if row is None else buf[row, col]


@T.macro
def _collapse_decode_stripes(acc, c, stride, mask):
    left = T.simd.vadd(acc[c], acc[stride + c], mask)
    right = T.simd.vadd(acc[2 * stride + c], acc[3 * stride + c], mask)
    acc[c] = T.simd.vadd(left, right, mask)


@T.macro
def _prepare_qk(raw, out, row, options, mask):
    vectors, dtype, normalize, scale = options
    qk = T.simd.alloc_local(2 * vectors, "float32")
    for c in T.Unroll(vectors, explicit=True):
        qk[c] = _unpack_token(_token_lane(raw[0], row, c * 64))
        qk[vectors + c] = _unpack_token(_token_lane(raw[1], row, c * 64))
    if normalize:
        norms = T.simd.alloc_local(2, "float32")
        one = T.simd.vdup(T.float32(1), "float32", mask)
        for g in T.Unroll(2, explicit=True):
            norms[g] = T.simd.vdup(T.float32(0), "float32", mask)
        for c in T.Unroll(vectors, explicit=True):
            for g in T.Unroll(2, explicit=True):
                x = qk[g * vectors + c]
                norms[g] = T.simd.vadd(norms[g], T.simd.vmul(x, x, mask), mask)
        for g in T.Unroll(2, explicit=True):
            norm2 = T.simd.vdupv(T.simd.vcadd(norms[g], mask), mask)
            root = T.simd.vsqrt(T.simd.vadds(norm2, T.float32(1e-6), mask), mask)
            norms[g] = T.simd.vdiv(one, root, mask, precision="ftz_true")
        for c in T.Unroll(vectors, explicit=True):
            for g in T.Unroll(2, explicit=True):
                i = g * vectors + c
                normalized = T.simd.vmul(qk[i], norms[g], mask)
                rounded = T.simd.vcvt(normalized, dtype, round="ROUND_R", part=0)
                qk[i] = T.simd.vcvt(rounded, "float32", part=0)
    for c in T.Unroll(vectors, explicit=True):
        scaled = T.simd.vmuls(qk[c], T.float32(scale), mask)
        T.simd.vsts(_token_lane(out[0], row, c * 64), scaled, mask)
        T.simd.vsts(_token_lane(out[1], row, c * 64), qk[vectors + c], mask)


def _build_dense_decode(call, dtype):
    batch, heads, dim_k, dim_v = call.batch, call.heads, call.dim_k, call.dim_v
    scale, normalize, has_initial = call.scale, call.l2norm, call.has_initial_state
    jobs = batch * heads
    blocks = min(_VECTOR_BLOCKS, jobs)
    key_vectors, value_vectors = dim_k // 64, dim_v // 64
    qk_options = (key_vectors, dtype, normalize, scale)

    @T.prim_func
    def main(
        Q: T.Tensor((batch, 1, heads, dim_k), dtype),
        Keys: T.Tensor((batch, 1, heads, dim_k), dtype),
        Values: T.Tensor((batch, 1, heads, dim_v), dtype),
        Beta: T.Tensor((batch, 1, heads), dtype),
        Initial: T.Tensor((batch, heads, dim_k, dim_v), "float32"),
        Output: T.Tensor((batch, 1, heads, dim_v), dtype),
        Final: T.Tensor((batch, heads, dim_k, dim_v), "float32"),
    ):
        with T.Kernel(blocks) as block:
            state_kv = T.alloc_shared((dim_k, dim_v), "float32")
            q_raw = T.alloc_shared((dim_k,), dtype)
            k_raw = T.alloc_shared((dim_k,), dtype)
            v_raw = T.alloc_shared((dim_v,), dtype)
            beta_raw = T.alloc_shared((16,), dtype)
            q_f32 = T.alloc_shared((dim_k,), "float32")
            k_f32 = T.alloc_shared((dim_k,), "float32")
            output_ub = T.alloc_shared((dim_v,), dtype)
            T.annotate_buffer_versions(
                _versions(state_kv, q_raw, k_raw, v_raw, beta_raw, q_f32, k_f32, output_ub)
            )
            for job in T.serial(block, jobs, blocks):
                sequence = job // heads
                head = job % heads
                T.copy(Q[sequence, 0, head, 0:dim_k], q_raw)
                T.copy(Keys[sequence, 0, head, 0:dim_k], k_raw)
                T.copy(Values[sequence, 0, head, 0:dim_v], v_raw)
                T.copy(Beta[sequence, 0, head : head + 1], beta_raw[0:1])
                if has_initial:
                    T.copy(Initial[sequence, head, 0:dim_k, 0:dim_v], state_kv)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    _prepare_qk((q_raw, k_raw), (q_f32, k_f32), None, qk_options, mask)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    delta = T.simd.alloc_local(value_vectors, "float32")
                    if has_initial:
                        prediction = T.simd.alloc_local(4 * value_vectors, "float32")
                        result = T.simd.alloc_local(4 * value_vectors, "float32")
                        zero = T.simd.vdup(T.float32(0), "float32", mask)
                        for i in T.Unroll(4 * value_vectors, explicit=True):
                            prediction[i] = zero
                            result[i] = zero
                        for g in range(dim_k // 4):
                            for r in T.Unroll(4, explicit=True):
                                key_reg = T.simd.vld(k_f32[g * 4 + r], dist="BRC_B32")
                                for c in T.Unroll(value_vectors, explicit=True):
                                    previous = T.simd.vld(state_kv[g * 4 + r, c * 64])
                                    product = T.simd.vmul(previous, key_reg, mask)
                                    prediction[r * value_vectors + c] = T.simd.vadd(
                                        prediction[r * value_vectors + c], product, mask
                                    )
                    beta_reg = T.simd.vcvt(T.simd.vld(beta_raw[0], dist="BRC_B16"), "float32")
                    for c in T.Unroll(value_vectors, explicit=True):
                        value_reg = _unpack_token(v_raw[c * 64])
                        if has_initial:
                            _collapse_decode_stripes(prediction, c, value_vectors, mask)
                            value_reg = T.simd.vsub(value_reg, prediction[c], mask)
                        delta[c] = T.simd.vmul(value_reg, beta_reg, mask)
                    for g in range(dim_k // 4):
                        for r in T.Unroll(4, explicit=True):
                            update_key = T.simd.vld(k_f32[g * 4 + r], dist="BRC_B32")
                            if has_initial:
                                query_reg = T.simd.vld(q_f32[g * 4 + r], dist="BRC_B32")
                            for c in T.Unroll(value_vectors, explicit=True):
                                update_reg = T.simd.vmul(update_key, delta[c], mask)
                                if has_initial:
                                    previous_state = T.simd.vld(state_kv[g * 4 + r, c * 64])
                                    next_state = T.simd.vadd(previous_state, update_reg, mask)
                                else:
                                    next_state = update_reg
                                T.simd.vsts(state_kv[g * 4 + r, c * 64], next_state, mask)
                                if has_initial:
                                    output_product = T.simd.vmul(next_state, query_reg, mask)
                                    result[r * value_vectors + c] = T.simd.vadd(
                                        result[r * value_vectors + c], output_product, mask
                                    )
                    if not has_initial:
                        # For a zero initial state, q @ (k * delta) = (q @ k) * delta.
                        dot = T.simd.alloc_local(1, "float32")
                        dot[0] = T.simd.vdup(T.float32(0), "float32", mask)
                        for c in T.Unroll(key_vectors, explicit=True):
                            qk = T.simd.vmul(
                                T.simd.vld(q_f32[c * 64]), T.simd.vld(k_f32[c * 64]), mask
                            )
                            dot[0] = T.simd.vadd(dot[0], qk, mask)
                        qk_dot = T.simd.vdupv(T.simd.vcadd(dot[0], mask), mask)
                    for c in T.Unroll(value_vectors, explicit=True):
                        if has_initial:
                            _collapse_decode_stripes(result, c, value_vectors, mask)
                            result_sum = result[c]
                        else:
                            result_sum = T.simd.vmul(qk_dot, delta[c], mask)
                        narrowed = T.simd.vcvt(result_sum, dtype, round="ROUND_R", sat=False)
                        T.simd.vsts(output_ub[c * 64], narrowed, mask, dist="PK_B32", extent=64)
                T.copy(state_kv, Final[sequence, head, 0:dim_k, 0:dim_v])
                T.copy(output_ub, Output[sequence, 0, head, 0:dim_v])

    return tilelang.compile(main, target="ascend", out_idx=[5, 6])


@T.macro
def _staged_gemm(a, b, left, right, out, clear, flip):
    m, n = out.shape
    k = a.shape[0 if flip == "a" else 1]
    left_view = T.meta_var(left[0:m, 0:k])
    right_view = T.meta_var(right[0:n, 0:k])
    T.copy(a, left_view, transpose=flip == "a")
    T.copy(b, right_view, transpose=flip == "b")
    T.gemm(left_view, right_view, out, transpose_B=True, clear_accum=clear)


def _build_prefill(call, dtype, slots, value_tile, hf32):
    batch, seq_len, heads, dim = call.batch, call.seq_len, call.heads, call.dim_k
    num_sequences, packed, has_initial = call.num_sequences, call.varlen, call.has_initial_state
    chunk, prefix = _CHUNK, (slots, heads)
    shape = (batch, seq_len, heads, dim)
    state_shape = (num_sequences, heads, dim, dim)
    dense_chunks = (seq_len + chunk - 1) // chunk
    aligned_dense = not packed and seq_len % chunk == 0
    offset_storage = _offset_lanes(num_sequences)
    search_steps = (num_sequences - 1).bit_length()
    rows, vectors = chunk // 2, dim // 64
    qk_options = (vectors, dtype, call.l2norm, call.scale)
    chunk_jobs = slots * heads
    pairs, panel = (chunk_jobs + 1) // 2, _SOLVE_PANEL
    pack_blocks, solve_blocks = min(_CUBE_BLOCKS, chunk_jobs), min(_CUBE_BLOCKS, pairs)
    value_tiles, half_v = dim // value_tile, value_tile // 2
    state_jobs = num_sequences * heads * value_tiles
    state_blocks = min(_CUBE_BLOCKS, state_jobs)

    @T.prim_func
    def pack_grams(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Beta: T.Tensor(shape[:-1], dtype),
        Offsets: T.Tensor((num_sequences + 1,), "int64"),
        QP: T.Tensor((*prefix, chunk, dim), "float32"),
        KV: T.Tensor((*prefix, chunk, 2 * dim), "float32"),
        BP: T.Tensor((*prefix, chunk), "float32"),
        GramQK: T.Tensor((*prefix, 2 * chunk, chunk), "float32"),
    ):
        with T.MixedKernel(pack_blocks) as (block, sid):
            T.set_hf32_mode(hf32)
            kq_l1 = T.alloc_l1((2 * chunk, dim), "float32")
            left = T.alloc_l0a((2 * chunk, dim), "float32")
            right = T.alloc_l0b((chunk, dim), "float32")
            result = T.alloc_l0c((2 * chunk, chunk), "float32")
            offsets = T.alloc_shared((offset_storage,), "int64")
            q_raw, k_raw, v_raw = tuple(T.alloc_shared((rows, dim), dtype) for _ in range(3))
            b_raw = T.alloc_shared((rows, heads), dtype)
            q_ub, k_ub, v_ub = tuple(T.alloc_shared((rows, dim), "float32") for _ in range(3))
            b_ub = T.alloc_shared((rows,), "float32")
            raw = (offsets, q_raw, k_raw, v_raw, b_raw)
            widened = (q_ub, k_ub, v_ub, b_ub)
            T.annotate_buffer_versions(_versions(kq_l1, left, right, result, *raw, *widened))
            if packed:
                T.copy(Offsets[0 : num_sequences + 1], offsets[0 : num_sequences + 1])
            for job in T.serial(block, chunk_jobs, pack_blocks):
                slot, head = job // heads, job % heads
                if packed:
                    batch_index = T.int32(0)
                    # One extra slot per sequence covers its partial chunk.
                    sequence = T.alloc_var("int32", init=0)
                    if search_steps > 0:
                        for level in T.Unroll(search_steps, explicit=True):
                            step = 1 << (search_steps - level - 1)
                            probe = sequence + T.min(step, num_sequences - 1 - sequence)
                            if slot >= T.cast(offsets[probe], "int32") // chunk + probe:
                                sequence = probe
                    first = T.cast(offsets[sequence], "int32")
                    sequence_end = T.cast(offsets[sequence + 1], "int32")
                    first_slot = first // chunk + sequence
                    remaining = sequence_end - first - (slot - first_slot) * chunk
                    token = sequence_end - T.max(0, remaining)
                    valid = T.min(chunk, remaining)
                else:
                    batch_index = slot // dense_chunks
                    token = (slot % dense_chunks) * chunk
                    valid = T.min(chunk, seq_len - token)
                row = sid * rows
                live = T.max(0, T.min(rows, valid - row))
                if live > 0:
                    begin = token + row
                    end = begin + live
                    for src, dst in zip((Q, K, V), (q_raw, k_raw, v_raw), strict=True):
                        T.copy(src[batch_index, begin:end, head, 0:dim], dst[0:live, 0:dim])
                    T.copy(Beta[batch_index, begin:end, 0:heads], b_raw[0:live, 0:heads])
                with T.SimtVF(threads=min(_THREADS, dim)):
                    for t in T.Parallel(rows):
                        b_ub[t] = T.if_then_else(
                            t < live, T.cast(b_raw[t, head], "float32"), T.float32(0)
                        )
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for t in range(live):
                        _prepare_qk((q_raw, k_raw), (q_ub, k_ub), t, qk_options, mask)
                        for c in T.Unroll(vectors, explicit=True):
                            T.simd.vsts(v_ub[t, c * 64], _unpack_token(v_raw[t, c * 64]), mask)
                    zero = T.simd.vdup(T.float32(0), "float32", mask)
                    for t in range(live, rows):
                        for c in T.Unroll(vectors, explicit=True):
                            for dst in (q_ub, k_ub, v_ub):
                                T.simd.vsts(dst[t, c * 64], zero, mask)
                T.dual_copy(k_ub, kq_l1[0:chunk, 0:dim])
                T.dual_copy(q_ub, kq_l1[chunk : 2 * chunk, 0:dim])
                T.copy(kq_l1, left)
                T.copy(kq_l1[0:chunk, 0:dim], right)
                T.gemm(left, right, result, transpose_B=True, clear_accum=True)
                T.copy(result, GramQK[slot, head, 0 : 2 * chunk, 0:chunk])
                T.copy(q_ub, QP[slot, head, row : row + rows, 0:dim])
                T.copy(k_ub, KV[slot, head, row : row + rows, 0:dim])
                T.copy(v_ub, KV[slot, head, row : row + rows, dim : 2 * dim])
                T.copy(b_ub, BP[slot, head, row : row + rows])

    @T.prim_func
    def solve(
        KV: T.Tensor((*prefix, chunk, 2 * dim), "float32"),
        GramQK: T.Tensor((*prefix, 2 * chunk, chunk), "float32"),
        BP: T.Tensor((*prefix, chunk), "float32"),
        Solved: T.Tensor((*prefix, 2 * dim, chunk), "float32"),
    ):
        with T.MixedKernel(solve_blocks) as (block, sid):
            T.set_hf32_mode(hf32)
            # Pad the final inverse row for a full-width SIMD load.
            rhs = T.alloc_shared((chunk + 1, chunk), "float32")
            gram = T.alloc_shared((chunk + 1, chunk), "float32")
            beta = T.alloc_shared((chunk,), "float32")
            inv_l1 = T.alloc_l1((2 * chunk, chunk), "float32")
            rhs_l1 = T.alloc_l1((2 * dim, chunk), "float32")
            inv_b = T.alloc_l0b((chunk, chunk), "float32")
            rhs_a = T.alloc_l0a((2 * dim, chunk), "float32")
            result = T.alloc_l0c((2 * dim, chunk), "float32")
            cube_tiles = (inv_l1, rhs_l1, rhs_a, inv_b, result)
            T.annotate_buffer_versions(_versions(rhs, gram, beta, *cube_tiles))
            for pair in T.serial(block, pairs, solve_blocks):
                # Both AIVs participate even when the final pair has just one chunk.
                job = T.min(pair * 2 + sid, chunk_jobs - 1)
                slot, head = job // heads, job % heads
                T.copy(GramQK[slot, head, 0:chunk, 0:chunk], gram[0:chunk, 0:chunk])
                T.copy(BP[slot, head, 0:chunk], beta)
                with T.SimdVF():
                    mask = T.simd.pset(32, "PAT_VL32")
                    lanes = T.simd.vci(T.int32(0), "int32")
                    zero = T.simd.vdup(T.float32(0), "float32", mask)
                    for i in range(chunk):
                        weight = T.simd.vld(beta[i], dist="BRC_B32")
                        weighted = T.simd.vmul(T.simd.vld(gram[i, 0]), weight, mask)
                        T.simd.vsts(gram[i, 0], T.simd.vneg(weighted, mask), mask, extent=chunk)
                    T.simd.mem_bar("VST_VLD")
                    rows = T.simd.alloc_local(panel, "float32")
                    for p in range(_CHUNK // panel):
                        base = p * panel
                        for i in T.Unroll(panel, explicit=True):
                            diagonal = T.simd.vcmps(lanes, base + i, mask, op="eq")
                            weight = T.simd.vld(beta[base + i], dist="BRC_B32")
                            rows[i] = T.simd.vsel(weight, zero, diagonal)
                        for j in range(base):
                            pivot = T.simd.vld(rhs[j, 0])
                            for i in T.Unroll(panel, explicit=True):
                                coeff = T.simd.vld(gram[base + i, j], dist="BRC_B32")
                                T.simd.vmula(rows[i], coeff, pivot, mask)
                        for j in T.Unroll(panel, explicit=True):
                            for i in T.Unroll(panel, explicit=True):
                                if i > j:
                                    coeff = T.simd.vld(gram[base + i, base + j], dist="BRC_B32")
                                    T.simd.vmula(rows[i], coeff, rows[j], mask)
                        for i in T.Unroll(panel, explicit=True):
                            T.simd.vsts(rhs[base + i, 0], rows[i], mask, extent=_CHUNK)
                        T.simd.mem_bar("VST_VLD")
                T.dual_copy(rhs[0:chunk, 0:chunk], inv_l1)
                for cube_job in T.serial(pair * 2, T.min(pair * 2 + 2, chunk_jobs)):
                    cube_slot, cube_head = cube_job // heads, cube_job % heads
                    row = (cube_job % 2) * chunk
                    T.copy(inv_l1[row : row + chunk, 0:chunk], inv_b)
                    T.copy(KV[cube_slot, cube_head, 0:chunk, 0 : 2 * dim], rhs_l1, transpose=True)
                    T.copy(rhs_l1, rhs_a)
                    T.gemm(rhs_a, inv_b, result, transpose_B=True, clear_accum=True)
                    T.copy(result, Solved[cube_slot, cube_head, 0 : 2 * dim, 0:chunk])

    @T.prim_func
    def states_project(
        Solved: T.Tensor((*prefix, 2 * dim, chunk), "float32"),
        KP: T.Tensor((*prefix, chunk, 2 * dim), "float32"),
        QP: T.Tensor((*prefix, chunk, dim), "float32"),
        GramQK: T.Tensor((*prefix, 2 * chunk, chunk), "float32"),
        Initial: T.Tensor(state_shape, "float32"),
        Offsets: T.Tensor((num_sequences + 1,), "int64"),
        OP: T.Tensor(shape, dtype),
        Final: T.Tensor(state_shape, "float32"),
    ):
        with T.MixedKernel(state_blocks) as (block, sid):
            T.set_hf32_mode(hf32)
            state_l1 = T.alloc_l1((dim, value_tile), "float32")
            w_l1 = T.alloc_l1((chunk, dim), "float32")
            q_l1 = T.alloc_l1((chunk, dim), "float32")
            d_l1 = T.alloc_l1((value_tile, chunk), "float32")
            kt_l1 = T.alloc_l1((dim, chunk), "float32")
            g_l1 = T.alloc_l1((chunk, chunk), "float32")
            identity_l1 = T.alloc_l1((dim, dim), "float32")
            left = T.alloc_l0a((dim, dim), "float32")
            right = T.alloc_l0b((dim, dim), "float32")
            predicted = T.alloc_l0c((value_tile, chunk), "float32")
            output = T.alloc_l0c((chunk, value_tile), "float32")
            state = T.alloc_l0c((dim, value_tile), "float32")
            state_ub = T.alloc_shared((dim, half_v), "float32")
            pred_ub = T.alloc_shared((half_v, chunk), "float32")
            u_ub = T.alloc_shared((half_v, chunk), "float32")
            d_ub = T.alloc_shared((half_v, chunk), "float32")
            g_ub = T.alloc_shared((chunk, chunk), "float32")
            identity_ub = T.alloc_shared((dim, dim // 2), "float32")
            l1 = (state_l1, w_l1, q_l1, d_l1, kt_l1, g_l1, identity_l1)
            l0 = (left, right, predicted, output, state)
            ub = (state_ub, pred_ub, u_ub, d_ub, g_ub)
            T.annotate_buffer_versions(_versions(*l1, *l0, *ub, identity_ub))
            if has_initial:
                with T.SimtVF(threads=_THREADS):
                    for i, j in T.Parallel(dim, dim // 2):
                        identity_ub[i, j] = T.if_then_else(i == dim // 2 * sid + j, 1.0, 0.0)
                T.dual_copy(identity_ub, identity_l1)
            for job in T.serial(block, state_jobs, state_blocks):
                sequence = job // (heads * value_tiles)
                head = job // value_tiles % heads
                value = (job % value_tiles) * value_tile
                v_end = T.meta_var(value + value_tile)
                if packed:
                    batch = 0
                    first = T.cast(Offsets[sequence], "int32")
                    end = T.cast(Offsets[sequence + 1], "int32")
                    first_slot = first // chunk + sequence
                    count = T.cast(T.ceildiv(T.cast(end - first, "int64"), chunk), "int32")
                else:
                    batch = sequence
                    first, end = 0, seq_len
                    first_slot = sequence * dense_chunks
                    count = dense_chunks
                if has_initial:
                    T.dual_copy(Initial[sequence, head, 0:dim, value:v_end], state_ub)
                else:
                    with T.SimdVF():
                        T.clear(state_ub)
                T.dual_copy(state_ub, state_l1)
                if has_initial:
                    _staged_gemm(identity_l1, state_l1, left, right, state, True, "b")
                for slot in T.serial(first_slot, first_slot + count):
                    T.copy(Solved[slot, head, 0:dim, 0:chunk], w_l1, transpose=True)
                    T.copy(QP[slot, head, 0:chunk, 0:dim], q_l1)
                    T.copy(KP[slot, head, 0:chunk, 0:dim], kt_l1, transpose=True)
                    _staged_gemm(state_l1, w_l1, left, right, predicted, True, "a")
                    T.dual_copy(predicted, pred_ub)
                    _staged_gemm(q_l1, state_l1, left, right, output, True, "b")
                    T.dual_copy(Solved[slot, head, dim + value : dim + v_end, 0:chunk], u_ub)
                    T.copy(GramQK[slot, head, chunk : 2 * chunk, 0:chunk], g_ub)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        for x in range(half_v * chunk // 64):
                            row, col = x * 64 // chunk, x * 64 % chunk
                            u = T.simd.vld(u_ub[row, col])
                            p = T.simd.vld(pred_ub[row, col])
                            delta = T.simd.vsub(u, p, mask)
                            T.simd.vsts(d_ub[row, col], delta, mask)
                    T.dual_copy(d_ub, d_l1)
                    clear = False if has_initial else slot == first_slot
                    _staged_gemm(kt_l1, d_l1, left, right, state, clear, "")
                    T.dual_copy(state, state_ub)
                    T.dual_copy(state_ub, state_l1)
                    with T.SimtVF(threads=_THREADS):
                        for i, j in T.Parallel(chunk, chunk):
                            g_ub[i, j] = T.if_then_else(j <= i, g_ub[i, j], T.float32(0))
                    T.copy(g_ub, g_l1)
                    _staged_gemm(g_l1, d_l1, left, right, output, False, "")
                    token = first + (slot - first_slot) * chunk
                    valid = chunk if aligned_dense else T.min(chunk, end - token)
                    T.copy(output, OP[batch, token : token + valid, head, value:v_end])
                T.dual_copy(state_ub, Final[sequence, head, 0:dim, value:v_end])

    return (
        tilelang.compile(pack_grams, target="ascend", out_idx=[5, 6, 7, 8]),
        tilelang.compile(solve, target="ascend", out_idx=[3]),
        tilelang.compile(states_project, target="ascend", out_idx=[6, 7]),
    )


class _AscendDeltaNetKernel(Kernel, DeltaNetFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: DeltaNetCall) -> str | None:
        if call.dtype not in (torch.float16, torch.bfloat16):
            return "requires float16 or bfloat16 token tensors"
        if call.dim_k not in (64, 128) or call.dim_v != call.dim_k:
            return "requires a square 64-wide or 128-wide state"
        if min(call.batch, call.seq_len, call.heads) < 1:
            return "requires positive batch, sequence length and head count"
        if call.seq_len > 2147483647:
            return "requires sequence positions representable by int32"
        if call.num_sequences < 1:
            return "requires at least one sequence"
        if call.varlen and call.batch != 1:
            return "packed sequences require batch=1"
        if not call.varlen and call.num_sequences != call.batch:
            return "dense calls require num_sequences=batch"
        return super().refusal(call)

    def __init__(self, call: DeltaNetCall):
        reason = type(self).refusal(call)
        if reason is not None:
            raise NotImplementedError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.dtype = call.dtype
        self._call = call
        self._state_shape = (call.num_sequences, call.heads, call.dim_k, call.dim_v)

    def _initial_state(self, q, initial_state, cu_seqlens):
        call = self._call
        if (initial_state is not None) != call.has_initial_state:
            raise ValueError("initial_state presence differs from the specialized call")
        if (cu_seqlens is not None) != call.varlen:
            raise ValueError("cu_seqlens presence differs from the specialized call")
        if initial_state is None:
            return q.new_empty(self._state_shape, dtype=torch.float32)
        return initial_state


class AscendDeltaNetDecodeKernel(_AscendDeltaNetKernel):
    @classmethod
    def refusal(cls, call: DeltaNetCall) -> str | None:
        if call.seq_len != 1 or call.varlen:
            return "requires single-token dense decode"
        return super().refusal(call)

    def __init__(self, call: DeltaNetCall):
        super().__init__(call)
        self.config.update(
            max_blocks=_VECTOR_BLOCKS,
            state_layout="KV",
            reduction_stripes=4,
            value_vector_width=64,
        )
        self._compiled_fn = _build_dense_decode(call, self.dtype_str)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._initial_state(q, initial_state, cu_seqlens)
        return tuple(self._compiled_fn(q, k, v, beta, state))


class AscendDeltaNetChunkedKernel(_AscendDeltaNetKernel):
    @classmethod
    def refusal(cls, call: DeltaNetCall) -> str | None:
        if call.seq_len == 1 and not call.varlen:
            return "requires prefill or packed input"
        offsets_bytes = 8 * _offset_lanes(call.num_sequences) if call.varlen else 0
        beta_bytes = _CHUNK * call.heads
        # Keep UB space available for widened Q/K/V and compiler staging.
        if offsets_bytes + beta_bytes > _PACK_METADATA_BYTES:
            return "requires the packed offsets and beta tile to fit in the 64 KiB staging budget"
        return super().refusal(call)

    def __init__(self, call: DeltaNetCall):
        super().__init__(call)
        sequences, heads, dim, _ = self._state_shape
        dense_chunks = (call.seq_len + _CHUNK - 1) // _CHUNK
        # Packed starts reserve one slot per sequence so partial chunks cannot overlap.
        slots = dense_chunks + sequences - 1 if call.varlen else call.batch * dense_chunks
        if slots > 2147483647:
            raise NotImplementedError("requires chunk indices representable by int32")
        hf32 = "nearest_even" if dim == 128 else None
        # A packed batch can be dominated by a single sequence.
        jobs = heads if call.varlen else sequences * heads
        wide = jobs >= _CUBE_BLOCKS
        value_tile = dim if wide else dim // 2
        self.config.update(
            chunk=_CHUNK,
            cube_blocks=_CUBE_BLOCKS,
            threads=_THREADS,
            solve_blocks=_CUBE_BLOCKS,
            solve_threads=_THREADS,
            solve_panel=_SOLVE_PANEL,
            solve_algorithm="paired_inverse_cube",
            state_value_tile=value_tile,
            hf32=hf32,
        )
        self._pack, self._solve, self._states_project = _build_prefill(
            call, self.dtype_str, slots, value_tile, hf32
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        initial_state = self._initial_state(q, initial_state, cu_seqlens)
        if cu_seqlens is None:
            cu_seqlens = q.new_empty((self._call.num_sequences + 1,), dtype=torch.int64)
        # Device offsets are authoritative and all workspaces belong to this invocation.
        qp, kv, bp, gram_qk = self._pack(q, k, v, beta, cu_seqlens)
        solved = self._solve(kv, gram_qk, bp)
        return self._states_project(solved, kv, qp, gram_qk, initial_state, cu_seqlens)


tileops.backend.register_kernel_type("DeltaNetFwdOp", "ascend", AscendDeltaNetChunkedKernel)
tileops.backend.register_kernel_type("DeltaNetFwdOp", "ascend_decode", AscendDeltaNetDecodeKernel)
