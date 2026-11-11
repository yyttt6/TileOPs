"""SSD chunk-state kernels for Ascend 950."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch
import torch.nn.functional as F
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout

from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.mamba.call_spec import SSDChunkStateCall, SSDChunkStateFwdInterface
from tileops.backend._registry import register_kernel_type


@T.macro
def _weights(delta, steps, same, weights, has_seq: bool):
    with T.SimdVF():
        mask = T.simd.pset(32)
        for part in T.serial(2):
            raw = T.simd.vld(delta[part * 64])
            capped = T.simd.vmins(raw, T.float32(0), mask)
            capped = T.simd.vsel(raw, capped, T.simd.vcmp(raw, raw, mask, op="ne"))
            decay = T.simd.vexp(capped, mask, precision="ftz_false")
            value = T.simd.vmul(decay, T.simd.vld(steps[part * 64]), mask)
            if has_seq:
                value = T.simd.vmul(value, T.simd.vld(same[part * 64]), mask)
            T.simd.vsts(weights[part * 64], value, mask, extent=64)
        T.simd.mem_bar("VST_VLD")


@functools.lru_cache(maxsize=128)
def build(b, c, q, h, p, n, g, dtype, dt_dtype, has_seq):
    groups = b * c * h
    qp = (q + 127) // 128 * 128
    pp = (p + 63) // 64 * 64
    np = (n + 63) // 64 * 64
    nt = max((n + 15) // 16, pp // 16)
    qt = qp // 128
    tasks = groups * nt * qt
    blocks = min(64, tasks)
    dma = n % 16 == 0
    xdma = p % 16 == 0

    @T.prim_func
    def weighted(
        X: T.Tensor((b, c * q, h, p), dtype),
        B: T.Tensor((b, c * q, g, n), dtype),
        DT: T.Tensor((b, h, c, q), dt_dtype),
        DA: T.Tensor((b, h, c, q), "float32"),
        Seq: T.Tensor((b, c * q), "int32"),
        W: T.Tensor((groups, np, qp), "float32"),
        Packed: T.Tensor((groups, pp, qp), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            data = T.alloc_shared((128, 16), dtype)
            xdata = T.alloc_shared((128, 16), dtype)
            out = T.alloc_shared((16, 128), "float32")
            weights = T.alloc_shared((128,), "float32")
            deltas = T.alloc_shared((128,), "float32")
            steps = T.alloc_shared((128,), "float32")
            masks = T.alloc_shared((128,), "float32")
            for step in T.serial(T.ceildiv(tasks, blocks)):
                task = step * blocks + bx
                if task < tasks:
                    group = task // (nt * qt)
                    bi = group // (c * h)
                    ci = group // h % c
                    hi = group % h
                    gi = hi // (h // g)
                    ni = task // qt % nt * 16
                    qi = task % qt * 128
                    count = T.min(128, q - qi)
                    if dma:  # noqa: SIM102
                        if ni < n:
                            T.copy(
                                B[bi, ci * q + qi : ci * q + qi + count, gi, ni : ni + 16],
                                data[:count, :],
                            )
                    if xdma:  # noqa: SIM102
                        if ni < p:
                            T.copy(
                                X[bi, ci * q + qi : ci * q + qi + count, hi, ni : ni + 16],
                                xdata[:count, :],
                            )
                    with T.SimtVF(threads=256):
                        for j in T.Parallel(128):
                            delta = T.alloc_var("float32")
                            dt_value = T.alloc_var("float32")
                            same = T.alloc_var("float32")
                            delta = T.float32(0)
                            dt_value = T.float32(0)
                            same = T.float32(1)
                            if j < count:
                                delta = DA[bi, hi, ci, q - 1] - DA[bi, hi, ci, qi + j]
                                dt_value = T.cast(DT[bi, hi, ci, qi + j], "float32")
                                if has_seq:
                                    end = Seq[bi, ci * q + q - 1]
                                    same = T.cast(
                                        (end >= 0) and (Seq[bi, ci * q + qi + j] == end), "float32"
                                    )
                            deltas[j] = delta
                            steps[j] = dt_value
                            masks[j] = same
                    _weights(deltas, steps, masks, weights, has_seq)
                    with T.SimtVF(threads=256):
                        for i, j in T.Parallel(16, 128):
                            value = T.alloc_var("float32")
                            value = T.float32(0)
                            if ni + i < n and j < count:
                                if dma:
                                    value = weights[j] * T.cast(data[j, i], "float32")
                                else:
                                    value = weights[j] * T.cast(
                                        B[bi, ci * q + qi + j, gi, ni + i], "float32"
                                    )
                            out[i, j] = value
                    if ni < np:
                        T.copy(out, W[group, ni : ni + 16, qi : qi + 128])
                    if ni < pp:
                        with T.SimtVF(threads=256):
                            for i, j in T.Parallel(16, 128):
                                value = T.alloc_var("float32")
                                value = T.float32(0)
                                if ni + i < p and j < count:
                                    if xdma:
                                        value = T.cast(xdata[j, i], "float32")
                                    else:
                                        value = T.cast(
                                            X[bi, ci * q + qi + j, hi, ni + i], "float32"
                                        )
                                out[i, j] = value
                        T.copy(out, Packed[group, ni : ni + 16, qi : qi + 128])
            # The remaining N padding is initialized separately by the wrapper.

    tiles_p = pp // 64
    tiles_n = np // 64
    total = groups * tiles_p * tiles_n
    cubes = min(32, total)

    @T.prim_func
    def contract(
        X: T.Tensor((groups, pp, qp), "float32"),
        W: T.Tensor((groups, np, qp), "float32"),
        O: T.Tensor((groups, pp, np), "float32"),
    ):
        with T.Kernel(cubes) as bx:
            a = T.alloc_l1((64, 128), "float32")
            z = T.alloc_l1((64, 128), "float32")
            acc = T.alloc_l0c((64, 64), "float32")
            T.annotate_buffer_versions({a: 2, z: 2})
            for step in T.serial(T.ceildiv(total, cubes)):
                tile = step * cubes + bx
                if tile < total:
                    group = tile // (tiles_p * tiles_n)
                    pi = tile // tiles_n % tiles_p * 64
                    ni = tile % tiles_n * 64
                    for ki in T.Pipelined(qp // 128, num_stages=2):
                        T.copy(X[group, pi : pi + 64, ki * 128 : ki * 128 + 128], a)
                        T.copy(W[group, ni : ni + 64, ki * 128 : ki * 128 + 128], z)
                        T.gemm(a, z, acc, transpose_B=True, clear_accum=ki == 0)
                    T.copy(acc, O[group, pi : pi + 64, ni : ni + 64])

    return tilelang.compile(weighted, target="ascend"), tilelang.compile(
        contract, target="ascend", out_idx=-1
    )


@T.macro
def _weights_unfused(delta, steps, same, weights, has_seq: bool):
    with T.SimdVF():
        mask = T.simd.pset(32)
        for part in T.serial(2):
            raw = T.simd.vld(delta[part * 64])
            capped = T.simd.vmins(raw, T.float32(0), mask)
            capped = T.simd.vsel(raw, capped, T.simd.vcmp(raw, raw, mask, op="ne"))
            decay = T.simd.vexp(capped, mask, precision="ftz_false")
            value = T.simd.vmul(decay, T.simd.vld(steps[part * 64]), mask)
            if has_seq:
                value = T.simd.vmul(value, T.simd.vld(same[part * 64]), mask)
            T.simd.vsts(weights[part * 64], value, mask, extent=64)
        T.simd.mem_bar("VST_VLD")


@functools.lru_cache(maxsize=128)
def build_unfused(b, c, q, h, p, n, g, dtype, dt_dtype, has_seq):
    groups = b * c * h
    qp = (q + 127) // 128 * 128
    pp = (p + 63) // 64 * 64
    np = (n + 63) // 64 * 64
    nt = (n + 15) // 16
    qt = qp // 128
    tasks = groups * nt * qt
    blocks = min(64, tasks)
    dma = n % 16 == 0

    @T.prim_func
    def weighted(
        B: T.Tensor((b, c * q, g, n), dtype),
        DT: T.Tensor((b, h, c, q), dt_dtype),
        DA: T.Tensor((b, h, c, q), "float32"),
        Seq: T.Tensor((b, c * q), "int32"),
        W: T.Tensor((groups, np, qp), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            data = T.alloc_shared((128, 16), dtype)
            out = T.alloc_shared((16, 128), "float32")
            weights = T.alloc_shared((128,), "float32")
            deltas = T.alloc_shared((128,), "float32")
            steps = T.alloc_shared((128,), "float32")
            masks = T.alloc_shared((128,), "float32")
            for step in T.serial(T.ceildiv(tasks, blocks)):
                task = step * blocks + bx
                if task < tasks:
                    group = task // (nt * qt)
                    bi = group // (c * h)
                    ci = group // h % c
                    hi = group % h
                    gi = hi // (h // g)
                    ni = task // qt % nt * 16
                    qi = task % qt * 128
                    count = T.min(128, q - qi)
                    if dma:
                        T.copy(
                            B[bi, ci * q + qi : ci * q + qi + count, gi, ni : ni + 16],
                            data[:count, :],
                        )
                    with T.SimtVF(threads=256):
                        for j in T.Parallel(128):
                            delta = T.alloc_var("float32")
                            dt_value = T.alloc_var("float32")
                            same = T.alloc_var("float32")
                            delta = T.float32(0)
                            dt_value = T.float32(0)
                            same = T.float32(1)
                            if j < count:
                                delta = DA[bi, hi, ci, q - 1] - DA[bi, hi, ci, qi + j]
                                dt_value = T.cast(DT[bi, hi, ci, qi + j], "float32")
                                if has_seq:
                                    end = Seq[bi, ci * q + q - 1]
                                    same = T.cast(
                                        (end >= 0) and (Seq[bi, ci * q + qi + j] == end), "float32"
                                    )
                            deltas[j] = delta
                            steps[j] = dt_value
                            masks[j] = same
                    _weights_unfused(deltas, steps, masks, weights, has_seq)
                    with T.SimtVF(threads=256):
                        for i, j in T.Parallel(16, 128):
                            value = T.alloc_var("float32")
                            value = T.float32(0)
                            if ni + i < n and j < count:
                                if dma:
                                    value = weights[j] * T.cast(data[j, i], "float32")
                                else:
                                    value = weights[j] * T.cast(
                                        B[bi, ci * q + qi + j, gi, ni + i], "float32"
                                    )
                            out[i, j] = value
                    T.copy(out, W[group, ni : ni + 16, qi : qi + 128])
            # The remaining N padding is initialized separately by the wrapper.

    tiles_p = pp // 64
    tiles_n = np // 64
    total = groups * tiles_p * tiles_n
    cubes = min(32, total)

    @T.prim_func
    def contract(
        X: T.Tensor((groups, pp, qp), "float32"),
        W: T.Tensor((groups, np, qp), "float32"),
        O: T.Tensor((groups, pp, np), "float32"),
    ):
        with T.Kernel(cubes) as bx:
            a = T.alloc_l1((64, 128), "float32")
            z = T.alloc_l1((64, 128), "float32")
            acc = T.alloc_l0c((64, 64), "float32")
            T.annotate_buffer_versions({a: 2, z: 2})
            for step in T.serial(T.ceildiv(total, cubes)):
                tile = step * cubes + bx
                if tile < total:
                    group = tile // (tiles_p * tiles_n)
                    pi = tile // tiles_n % tiles_p * 64
                    ni = tile % tiles_n * 64
                    for ki in T.Pipelined(qp // 128, num_stages=2):
                        T.copy(X[group, pi : pi + 64, ki * 128 : ki * 128 + 128], a)
                        T.copy(W[group, ni : ni + 64, ki * 128 : ki * 128 + 128], z)
                        T.gemm(a, z, acc, transpose_B=True, clear_accum=ki == 0)
                    T.copy(acc, O[group, pi : pi + 64, ni : ni + 64])

    return tilelang.compile(weighted, target="ascend"), tilelang.compile(
        contract, target="ascend", out_idx=-1
    )


@functools.lru_cache(maxsize=64)
def _build_m128(B, NC, Q, H, P, G, N, dtype, has_seq_idx):
    K, KH, M, CM = 256, 128, 128, 64
    PAIRS = H // 2
    JOBS = B * NC * 2
    HEAD_TURNS = PAIRS // 2
    GRID = min(32, JOBS)
    ROUNDS = (JOBS + GRID - 1) // GRID

    @T.macro
    def metadata_load(b, c, h, sid, dt, dA, dt_ub, da_ub, end_da_ub):
        for head in T.serial(2):
            T.copy(dt[b, h + head, c, sid * KH : (sid + 1) * KH], dt_ub[head, 0:KH])
            T.copy(dA[b, h + head, c, sid * KH : (sid + 1) * KH], da_ub[head, 0:KH])
            # Each endpoint destination starts at a separate aligned 32-byte row.
            T.copy(dA[b, h + head, c, K - 1 : K], end_da_ub[head, 0:1])

    @T.macro
    def load_chunk_mask(b, c, sid, seq_idx, seq_ub, end_seq_ub):
        # Chunk metadata is invariant across every head pair.
        if has_seq_idx:
            T.copy(seq_idx[b, c * K + sid * KH : c * K + (sid + 1) * KH], seq_ub)
            T.copy(seq_idx[b, (c + 1) * K - 1 : (c + 1) * K], end_seq_ub[0:1])

    @T.macro
    def metadata_weights(dt_ub, da_ub, end_da_ub, seq_ub, end_seq_ub, weights):
        with T.SimdVF():
            full = S.pset(32)
            zero = S.vdup(T.float32(0), "float32", full)
            if has_seq_idx:
                end_id = S.vld(end_seq_ub[0], "BRC_B32")
                nonnegative = S.vcmps(end_id, T.int32(0), full, "ge")
            for head in range(2):
                end_da = S.vld(end_da_ub[head, 0], "BRC_B32")
                for block in range(KH // 64):
                    delta = S.vsub(end_da, S.vld(da_ub[head, block * 64]), full)
                    weight = S.vmul(
                        S.vexp(S.vmin(delta, zero, full), full),
                        S.vld(dt_ub[head, block * 64]),
                        full,
                    )
                    if has_seq_idx:
                        same = S.vcmp(S.vld(seq_ub[block * 64]), end_id, full, "eq")
                        weight = S.vsel(weight, zero, S.pand(same, nonnegative, full))
                    S.vsts(weights[head, block * 64], weight, full)
            # Explicit SIMD store-to-load visibility, including the next VF.
            S.mem_bar("VST_VLD")

    @T.macro
    def load_x(b, c, h, sid, x_flat, raw_x):
        T.copy(x_flat[b, c * K + sid * KH : c * K + (sid + 1) * KH, h * P : (h + 2) * P], raw_x)

    @T.macro
    def pack_weighted_nz(raw_x, weights, scaled_nz):
        # Final ROUND_R values go directly to the compact-NZ transfer layout.
        with T.SimdVF():
            full = S.pset(32)
            packed_mask = S.pset(16)
            nz_ptr = S.make_ubuf_ptr(T.access_ptr(scaled_nz[0, 0], "w", KH + 1, M), dtype)
            for k in range(KH):
                x0 = S.vcvt(S.vld(raw_x[k, 0], "UNPK_B16"), "float32", full)
                x1 = S.vcvt(S.vld(raw_x[k, P], "UNPK_B16"), "float32", full)
                product0 = S.vmul(x0, S.vld(weights[0, k], "BRC_B32"), full)
                product1 = S.vmul(x1, S.vld(weights[1, k], "BRC_B32"), full)
                rounded0 = S.vcvt(product0, dtype, full, round="ROUND_R", sat=False, part=0)
                rounded1 = S.vcvt(product1, dtype, full, round="ROUND_R", sat=False, part=0)
                # PART_EVEN puts 64 values at even B16 lanes. Join both heads.
                packed, unused = S.vdintlv(rounded0, rounded1)
                nz_ptr = S.vsstb(
                    packed, nz_ptr, T.int32(((KH + 1) << 16) | 1), packed_mask, update=True
                )
            S.mem_bar("VST_VLD")

    @T.macro
    def load_resident_b(b, c, Bmat, b_l1, b_l0):
        # One B transfer for every head pair in this chunk stripe.
        T.copy(Bmat[b, c * K : (c + 1) * K, 0, 0:N], b_l1, transpose=True)
        T.copy(b_l1, b_l0)

    @T.macro
    def publish_a(scaled_nz, a_l1):
        T.dual_copy(scaled_nz[0:KH, 0:M], a_l1)

    @T.macro
    def write_result(job, subhead, acc, out_flat):
        T.copy(acc, out_flat[job * M + subhead * CM : job * M + (subhead + 1) * CM, 0:N])

    @T.prim_func
    def ssd_chunk_state_fwd(
        x: T.Tensor((B, NC * Q, H, P), dtype),
        Bmat: T.Tensor((B, NC * Q, G, N), dtype),
        dt: T.Tensor((B, H, NC, Q), "float32"),
        dA: T.Tensor((B, H, NC, Q), "float32"),
        seq_idx: T.Tensor((B, NC * Q), "int32"),
        states: T.Tensor((B, NC, H, P, N), "float32"),
    ):
        x_flat = T.decl_buffer((B, NC * Q, H * P), dtype, data=x.data)
        out_flat = T.decl_buffer((B * NC * H * P, N), "float32", data=states.data)
        with T.MixedKernel(GRID, sids=2) as (bx, sid):
            raw_x = T.alloc_shared((KH, M), dtype)
            scaled_nz = T.alloc_shared((KH + 1, M), dtype)
            T.annotate_layout({scaled_nz: make_ascend_compact_nz_layout(scaled_nz)})
            dt_ub = T.alloc_shared((2, KH), "float32")
            da_ub = T.alloc_shared((2, KH), "float32")
            end_da_ub = T.alloc_shared((2, 8), "float32")
            weights = T.alloc_shared((2, KH), "float32")
            seq_ub = T.alloc_shared((KH,), "int32")
            end_seq_ub = T.alloc_shared((8,), "int32")
            a_l1 = T.alloc_l1((K, M), dtype)
            b_l1 = T.alloc_l1((N, K), dtype)
            a_l0 = T.alloc_l0a((CM, K), dtype)
            b_l0 = T.alloc_l0b((N, K), dtype)
            acc = T.alloc_l0c((CM, N), "float32")
            # B is invariant throughout the head pipeline; L0B fills its 64KiB pool.
            T.annotate_buffer_versions(
                {
                    b_l1: 1,
                    b_l0: 1,
                    # The inner subhead clock owns two 32KiB L0A epochs.
                    a_l0: (2, "iteration"),
                    # Full metadata epochs can be transferred ahead of their SIMD reader.
                    dt_ub: (2, "iteration"),
                    da_ub: (2, "iteration"),
                    end_da_ub: (2, "iteration"),
                    # Both weights accesses are ordered tasks on PIPE_V at stage 1.
                    weights: 1,
                    # Per-chunk resident data, not head-turn scratch.
                    seq_ub: 1,
                    end_seq_ub: 1,
                    raw_x: (2, "iteration"),
                    scaled_nz: (2, "iteration"),
                    a_l1: (2, "iteration"),
                    acc: (2, "iteration"),
                }
            )
            # A second chunk cannot coexist with the resident full-size L0B.
            for turn in T.serial(ROUNDS):
                job = turn * GRID + bx
                if job < JOBS:
                    stripe = job % 2
                    c = job // 2 % NC
                    b = job // (2 * NC)
                    load_resident_b(b, c, Bmat, b_l1, b_l0)
                    load_chunk_mask(b, c, sid, seq_idx, seq_ub, end_seq_ub)
                    for head_turn in T.Pipelined(
                        HEAD_TURNS, num_stages=2, annotations={"enable_offset": True}
                    ):
                        pair = head_turn * 2 + stripe
                        h = pair * 2
                        output_job = (b * NC + c) * PAIRS + pair
                        # One lookahead epoch: DMA may prepare i+1 while SIMD consumes i.
                        with T.Stage(0):
                            load_x(b, c, h, sid, x_flat, raw_x)
                            metadata_load(b, c, h, sid, dt, dA, dt_ub, da_ub, end_da_ub)
                        with T.Stage(1):
                            metadata_weights(dt_ub, da_ub, end_da_ub, seq_ub, end_seq_ub, weights)
                            pack_weighted_nz(raw_x, weights, scaled_nz)
                            publish_a(scaled_nz, a_l1)
                        with T.Stage(2):
                            # A-L1 remains owned by the complete pair, covering both readers.
                            for subhead in T.Pipelined(
                                2, num_stages=2, annotations={"enable_offset": True}
                            ):
                                with T.Stage(0):
                                    T.copy(
                                        a_l1[0:K, subhead * CM : (subhead + 1) * CM],
                                        a_l0,
                                        transpose=True,
                                    )
                                with T.Stage(1):
                                    T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
                                    write_result(output_job, subhead, acc, out_flat)

    return tilelang.compile(ssd_chunk_state_fwd, target="ascend", out_idx=[5])


class AscendSSDChunkStateM128Kernel(Kernel, SSDChunkStateFwdInterface):
    """Compute FP32 chunk-end states for the Ascend 950 SSD tile geometry."""

    devices = frozenset({"npu"})

    def __init__(self, call: SSDChunkStateCall) -> None:
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.dtype = call.dtype
        self.kernel = _build_m128(
            call.batch,
            call.num_chunks,
            call.chunk_len,
            call.n_heads,
            call.d_head,
            call.n_groups,
            call.d_state,
            self.dtype_str,
            call.has_seq_idx,
        )

    def forward(
        self,
        x: torch.Tensor,
        Bmat: torch.Tensor,
        dt: torch.Tensor,
        dA_cumsum: torch.Tensor,
        seq_idx: torch.Tensor,
    ) -> torch.Tensor:
        # DMA operates on 32-byte blocks. Handle contiguous views with an unaligned offset.
        x, Bmat, dt, dA_cumsum, seq_idx = (
            tensor.clone() if tensor.data_ptr() % 32 else tensor
            for tensor in (x, Bmat, dt, dA_cumsum, seq_idx)
        )
        return self.kernel(x, Bmat, dt, dA_cumsum, seq_idx)


class AscendSSDChunkStateKernel(Kernel, SSDChunkStateFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: SSDChunkStateCall) -> Entry:
        if (
            (call.chunk_len, call.d_head, call.d_state, call.n_groups) == (256, 64, 128, 1)
            and call.n_heads > 0
            and call.n_heads % 4 == 0
            and call.dtype in (torch.float16, torch.bfloat16)
            and call.dt_dtype == torch.float32
        ):
            return call, lambda: AscendSSDChunkStateM128Kernel(call)
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        self.fused = call.batch * call.num_chunks * call.n_heads < 8192
        builder = build if self.fused else build_unfused
        self.weighted, self.compiled = builder(
            call.batch,
            call.num_chunks,
            call.chunk_len,
            call.n_heads,
            call.d_head,
            call.d_state,
            call.n_groups,
            str(call.dtype).split(".")[-1],
            str(call.dt_dtype or call.dtype).split(".")[-1],
            call.has_seq_idx,
        )

    def forward(self, x, B, dt, dA_cumsum, seq_idx):
        c = self.call
        b, nc, q, h, p, n = c.batch, c.num_chunks, c.chunk_len, c.n_heads, c.d_head, c.d_state
        qp = (q + 127) // 128 * 128
        pp = (p + 63) // 64 * 64
        np = (n + 63) // 64 * 64
        if self.fused:
            packed = torch.empty((b * nc * h, pp, qp), device=x.device, dtype=torch.float32)
        else:
            packed = x.float().reshape(b, nc, q, h, p).permute(0, 1, 3, 4, 2).contiguous()
            if qp != q or pp != p:
                packed = F.pad(packed, (0, qp - q, 0, pp - p))
            packed = packed.reshape(b * nc * h, pp, qp)
        # For aligned N every W cell is written by weighted(), including Q
        # padding. Unaligned N retains zero initialization for unwritten rows.
        allocate = torch.empty if n % 64 == 0 else torch.zeros
        w = allocate((b * nc * h, np, qp), device=x.device, dtype=torch.float32)
        if self.fused:
            self.weighted(x, B, dt, dA_cumsum, seq_idx, w, packed)
        else:
            self.weighted(B, dt, dA_cumsum, seq_idx, w)
        result = self.compiled(packed, w)
        return result[:, :p, :n].contiguous().reshape(b, nc, h, p, n)


register_kernel_type("SSDChunkStateFwdOp", "ascend", AscendSSDChunkStateKernel)
