"""Pre-permute: short-route fused SIMD or per-core counting and gather.

Kernel 1 (per core): local histogram -> counts[C, E_pad] in GM.
Kernel 2 (per core): read all counts -> global prefix/block_offsets -> local cursors
-> packed row copy.

There is no software grid barrier: the kernel boundary provides the global
synchronization.  E<=256 uses grouped dhistv2 (one 64-expert group per pass);
E>256 is rejected by the host (the sort fallback is a separate path).
"""

import functools

import torch

from tileops._kernels.moe.ascend.optimized.common import (
    AscendMoEKernel,
    check_dtype,
    check_tensor,
)

__all__ = ["AscendMoEPrePermuteKernel"]

# Retain the original single-token fallback when the sym path is ineligible.
SMALL_FUSED_MAX_TOKENS = 1


def _use_sym(num_tokens, hidden_size, top_k, num_experts):
    """Use the tested short-route specialization; fall back outside its domain."""
    return (
        num_tokens <= 128
        and num_tokens * top_k <= 1024
        and num_experts <= 64
        and hidden_size % 16 == 0
        and hidden_size <= 16384
    )


def _make_sym_fused(num_tokens: int, hidden_size: int, top_k: int, num_experts: int, dtype: str):
    """Compute stable route destinations locally, then read each token only once.

    Each block counts all routes, avoiding a global barrier or a second launch.
    Only block zero writes the shared metadata outputs. Other blocks use their
    own UB inverse indices to scatter disjoint tokens' routes.
    """
    import tilelang.ascend.language as T

    num_routes = num_tokens * top_k
    padded_routes = (num_routes + 63) // 64 * 64
    blocks = min(num_tokens, 64)

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, hidden_size), dtype),
        ids: T.Tensor((num_routes,), "int32"),
        y: T.Tensor((num_routes, hidden_size), dtype),
        ends: T.Tensor((num_experts,), "int32"),
        inverse: T.Tensor((num_routes,), "int32"),
    ):
        with T.Kernel(blocks) as bx:
            ids_ub = T.alloc_shared((padded_routes,), "int32")
            inverse_ub = T.alloc_shared((padded_routes,), "int32")
            ends_ub = T.alloc_shared((num_experts,), "int32")
            row_ub = T.alloc_shared((hidden_size,), dtype)
            T.copy(ids, ids_ub[:num_routes])
            T.ascend_set_flag("MTE2_V", 0)
            T.ascend_wait_flag("MTE2_V", 0)
            with T.SimdVF():
                full = T.simd.pset(32)
                offset = T.simd.alloc_var("int32")
                offset = T.simd.vdup(0, "int32", full)
                for expert in T.serial(num_experts):
                    for chunk in T.serial(padded_routes // 64):
                        valid = T.simd.update_mask(T.min(64, num_routes - chunk * 64), 32)
                        values = T.simd.vld(ids_ub[chunk * 64])
                        selected = T.simd.vcmps(values, expert, valid, "eq")
                        prefix = T.simd.vusqz(selected, "int32")
                        rank = T.simd.vadd(prefix, offset, full)
                        T.simd.vsts(inverse_ub[chunk * 64], rank, selected)
                        flags = T.simd.vdup(1, "int32", selected)
                        count = T.simd.vcadd(flags, full)
                        offset = T.simd.vadd(offset, T.simd.vdupv(count, full), full)
                    T.simd.vsts(ends_ub[expert], offset, full, dist="ONEPT_B32")
            # Scalar destination loads must observe the vector prefix results.
            T.ascend_set_flag("V_S", 0)
            T.ascend_wait_flag("V_S", 0)
            if bx == 0:
                T.ascend_set_flag("V_MTE3", 0)
                T.ascend_wait_flag("V_MTE3", 0)
                T.copy(ends_ub, ends)
                T.copy(inverse_ub[:num_routes], inverse)
            for step in T.serial(T.ceildiv(num_tokens, blocks)):
                token = step * blocks + bx
                if token < num_tokens:
                    # Finish all K writes before reusing the row buffer.
                    T.ascend_set_flag("MTE3_MTE2", 0)
                    T.ascend_wait_flag("MTE3_MTE2", 0)
                    T.copy(x[token, :], row_ub)
                    T.ascend_set_flag("MTE2_MTE3", 0)
                    T.ascend_wait_flag("MTE2_MTE3", 0)
                    for route in T.serial(top_k):
                        dest = inverse_ub[token * top_k + route]
                        T.copy(row_ub, y[dest, :])
            T.ascend_set_flag("MTE3_S", 0)
            T.ascend_wait_flag("MTE3_S", 0)

    return main


@functools.lru_cache(maxsize=32)
def _make_tight_scan_count(num_tokens, top_k, num_experts, num_cores):
    import tilelang.ascend.language as T

    cores = int(num_cores)
    rows_per_core = -(-num_tokens // cores)
    max_routes_this_core = rows_per_core * top_k
    e_pad = (num_experts + 63) // 64 * 64
    ids_cap = max(256, ((max_routes_this_core + 255) // 256) * 256)

    @T.prim_func
    def _tight_scan_count_main(
        flat_ids: T.Tensor((num_tokens * top_k,), "int32"),
        block_counts_gm: T.Tensor((cores * e_pad,), "int32"),
    ):
        with T.Kernel(cores) as core:
            token_start = core * rows_per_core
            token_end = T.min(token_start + rows_per_core, num_tokens)
            rows_this_core = T.max(token_end - token_start, 0)
            route_start = token_start * top_k
            routes_this_core = rows_this_core * top_k

            ids32 = T.alloc_shared((ids_cap,), "int32")
            ids8 = T.alloc_shared((ids_cap,), "uint8")
            counts = T.alloc_shared((e_pad,), "int32")

            if rows_this_core > 0:
                T.copy(
                    flat_ids[route_start : route_start + routes_this_core],
                    ids32[0:routes_this_core],
                )

            # The host limits E to 256; each unrolled group represents 64 bins.
            with T.SimdVF():
                zero_vec = T.simd.vdup(T.int32(0), "int32")
                for group in T.unroll(e_pad // 64):
                    T.simd.vsts(counts[group * 64], zero_vec)

            if routes_this_core > 0:
                for block in range((max_routes_this_core + 255) // 256):
                    base = block * 256
                    if base < routes_this_core:
                        rem = T.min(256, routes_this_core - base)
                        with T.SimdVF():
                            m16 = T.simd.pset(16)
                            for unit in range(4):
                                values = T.simd.vld(ids32[base + unit * 64])
                                low8 = T.simd.vcvt(values, "uint8", part=0)
                                T.simd.vsts(ids8[base + unit * 64], low8, dist="PK4_B32")
                            T.simd.mem_bar("VST_VLD")
                            src = T.simd.vld(ids8[base], dist="NORM")
                            m8 = T.simd.update_mask(rem, 8)
                            freq0 = T.simd.alloc_var("uint16")
                            freq0 = T.simd.vdup(T.uint16(0), "uint16", m16)
                            T.simd.dhistv2(freq0, src, m8, bin=0)
                            wide0 = T.reinterpret(T.simd.vunpack(freq0, part="LOWER"), "int32x64")
                            T.simd.vsts(counts[0], T.simd.vadd(T.simd.vld(counts[0]), wide0))
                            for group in T.unroll(1, e_pad // 64):
                                offset = group * 64
                                shifted = T.simd.vsub(
                                    src, T.simd.vdup(T.uint8(offset), "uint8"), m8
                                )
                                lower = T.simd.vcmps(src, T.int32(offset), m8, op="ge")
                                upper = T.simd.vcmps(shifted, T.int32(64), m8, op="lt")
                                keep = T.simd.pand(lower, upper, m8)
                                freq = T.simd.alloc_var("uint16")
                                freq = T.simd.vdup(T.uint16(0), "uint16", m16)
                                T.simd.dhistv2(freq, shifted, keep, bin=0)
                                wide = T.reinterpret(T.simd.vunpack(freq, part="LOWER"), "int32x64")
                                T.simd.vsts(
                                    counts[group * 64],
                                    T.simd.vadd(T.simd.vld(counts[group * 64]), wide),
                                )

            T.copy(
                counts[0:num_experts], block_counts_gm[core * e_pad : core * e_pad + num_experts]
            )

    return _tight_scan_count_main


@functools.lru_cache(maxsize=32)
def _make_tight_scan_gather(num_tokens, hidden_size, top_k, num_experts, dtype, num_cores):
    import tilelang.ascend.language as T

    cores = int(num_cores)
    rows_per_core = -(-num_tokens // cores)
    e_pad = (num_experts + 63) // 64 * 64

    @T.prim_func
    def _tight_scan_gather_main(
        hidden_states: T.Tensor((num_tokens, hidden_size), dtype),
        flat_ids: T.Tensor((num_tokens * top_k,), "int32"),
        block_counts_gm: T.Tensor((cores * e_pad,), "int32"),
        expert_input: T.Tensor((num_tokens * top_k, hidden_size), dtype),
        physical_ends: T.Tensor((num_experts,), "int32"),
        inverse_indices: T.Tensor((num_tokens * top_k,), "int32"),
    ):
        with T.Kernel(cores) as core:
            token_start = core * rows_per_core
            token_end = T.min(token_start + rows_per_core, num_tokens)
            rows_this_core = T.max(token_end - token_start, 0)
            route_start = token_start * top_k
            routes_this_core = rows_this_core * top_k

            ids32 = T.alloc_shared((max(256, rows_per_core * top_k),), "int32")
            block_counts = T.alloc_shared((cores * e_pad,), "int32")
            expert_totals = T.alloc_shared((e_pad,), "int32")
            expert_prefix = T.alloc_shared((e_pad,), "int32")
            expert_ends = T.alloc_shared((e_pad,), "int32")
            block_offsets = T.alloc_shared((e_pad,), "int32")
            cursors = T.alloc_shared((e_pad,), "int32")
            row_buf = T.alloc_shared((hidden_size,), dtype)

            if rows_this_core > 0:
                T.copy(
                    flat_ids[route_start : route_start + routes_this_core],
                    ids32[0:routes_this_core],
                )

            T.copy(block_counts_gm[0 : cores * e_pad], block_counts[0 : cores * e_pad])

            for g in range(e_pad // 64):
                with T.SimdVF():
                    zero_vec = T.simd.vdup(T.int32(0), "int32")
                    T.simd.vsts(expert_totals[g * 64], zero_vec)
                    T.simd.vsts(expert_prefix[g * 64], zero_vec)
                    T.simd.mem_bar("VST_VLD")
                    for cc in range(cores):
                        row_vec = T.simd.vld(block_counts[cc * e_pad + g * 64])
                        T.simd.vsts(
                            expert_totals[g * 64],
                            T.simd.vadd(T.simd.vld(expert_totals[g * 64]), row_vec),
                        )
                        if cc < core:
                            T.simd.vsts(
                                expert_prefix[g * 64],
                                T.simd.vadd(T.simd.vld(expert_prefix[g * 64]), row_vec),
                            )
                        T.simd.mem_bar("VST_VLD")

            running = T.alloc_var("int32", init=0)
            for e in T.serial(num_experts):
                running = running + expert_totals[e]
                expert_ends[e] = running
                block_offsets[e] = expert_prefix[e] - expert_totals[e] + running
                cursors[e] = block_offsets[e]

            if core == 0:
                for e in T.serial(num_experts):
                    physical_ends[e] = expert_ends[e]

            for r in T.Pipelined(routes_this_core, num_stages=2):
                e = ids32[r]
                if e >= 0 and e < num_experts:
                    slot = cursors[e]
                    cursors[e] = slot + 1
                    inverse_indices[route_start + r] = slot
                    token_row = (route_start + r) // top_k
                    T.copy(hidden_states[token_row, 0:hidden_size], row_buf[0:hidden_size])
                    T.copy(row_buf[0:hidden_size], expert_input[slot, 0:hidden_size])
                else:
                    inverse_indices[route_start + r] = -1

    return _tight_scan_gather_main


@functools.lru_cache(maxsize=32)
def _make_vector_scan_gather(num_tokens, hidden_size, top_k, num_experts, dtype, num_cores):
    """Vector prefix scan and one input-row load per token; measured BF16 shapes."""
    import tilelang.ascend.language as T

    cores = int(num_cores)
    rows_per_core = -(-num_tokens // cores)
    e_pad = (num_experts + 63) // 64 * 64

    @T.prim_func
    def _tight_scan_gather_main(
        hidden_states: T.Tensor((num_tokens, hidden_size), dtype),
        flat_ids: T.Tensor((num_tokens * top_k,), "int32"),
        block_counts_gm: T.Tensor((cores * e_pad,), "int32"),
        expert_input: T.Tensor((num_tokens * top_k, hidden_size), dtype),
        physical_ends: T.Tensor((num_experts,), "int32"),
        inverse_indices: T.Tensor((num_tokens * top_k,), "int32"),
    ):
        with T.Kernel(cores) as core:
            token_start = core * rows_per_core
            token_end = T.min(token_start + rows_per_core, num_tokens)
            rows_this_core = T.max(token_end - token_start, 0)
            route_start = token_start * top_k
            routes_this_core = rows_this_core * top_k

            ids32 = T.alloc_shared((max(256, rows_per_core * top_k),), "int32")
            block_counts = T.alloc_shared((cores * e_pad,), "int32")
            expert_totals = T.alloc_shared((e_pad,), "int32")
            expert_prefix = T.alloc_shared((e_pad,), "int32")
            expert_ends = T.alloc_shared((e_pad,), "int32")
            cursors = T.alloc_shared((e_pad,), "int32")
            row_buf = T.alloc_shared((hidden_size,), dtype)

            if rows_this_core > 0:
                T.copy(
                    flat_ids[route_start : route_start + routes_this_core],
                    ids32[0:routes_this_core],
                )

            T.copy(block_counts_gm[0 : cores * e_pad], block_counts[0 : cores * e_pad])

            for g in range(e_pad // 64):
                with T.SimdVF():
                    zero_vec = T.simd.vdup(T.int32(0), "int32")
                    T.simd.vsts(expert_totals[g * 64], zero_vec)
                    T.simd.vsts(expert_prefix[g * 64], zero_vec)
                    T.simd.mem_bar("VST_VLD")
                    for cc in range(cores):
                        row_vec = T.simd.vld(block_counts[cc * e_pad + g * 64])
                        T.simd.vsts(
                            expert_totals[g * 64],
                            T.simd.vadd(T.simd.vld(expert_totals[g * 64]), row_vec),
                        )
                        if cc < core:
                            T.simd.vsts(
                                expert_prefix[g * 64],
                                T.simd.vadd(T.simd.vld(expert_prefix[g * 64]), row_vec),
                            )
                        T.simd.mem_bar("VST_VLD")

            with T.SimdVF():
                full = T.simd.pset(32)
                lanes = T.simd.vci(0, "int32")
                carry = T.simd.alloc_var("int32")
                carry = T.simd.vdup(0, "int32", full)
                for g in T.serial(e_pad // 64):
                    totals = T.simd.vld(expert_totals[g * 64])
                    scan = T.simd.alloc_var("int32")
                    scan = totals
                    for step in T.unroll(6):
                        shift = 1 << step
                        ix = T.simd.vmax(
                            T.simd.vadds(lanes, -shift, full), T.simd.vdup(0, "int32", full), full
                        )
                        prev = T.simd.vselr(scan, T.reinterpret(ix, "uint32x64"))
                        active = T.simd.vcmps(lanes, shift, full, op="ge")
                        prev = T.simd.vsel(prev, T.simd.vdup(0, "int32", full), active)
                        scan = T.simd.vadd(scan, prev, full)
                    scan = T.simd.vadd(scan, carry, full)
                    T.simd.vsts(expert_ends[g * 64], scan)
                    seed = T.simd.vadd(
                        T.simd.vsub(scan, totals, full), T.simd.vld(expert_prefix[g * 64]), full
                    )
                    T.simd.vsts(cursors[g * 64], seed)
                    carry = T.simd.vselr(scan, T.simd.vdup(63, "uint32", full))

            if core == 0:
                T.copy(expert_ends[0:num_experts], physical_ends)

            for token_local in T.Pipelined(rows_this_core, num_stages=2):
                T.copy(
                    hidden_states[token_start + token_local, 0:hidden_size], row_buf[0:hidden_size]
                )
                for route_slot in T.serial(top_k):
                    r = token_local * top_k + route_slot
                    e = ids32[r]
                    if e >= 0 and e < num_experts:
                        slot = cursors[e]
                        cursors[e] = slot + 1
                        inverse_indices[route_start + r] = slot
                        T.copy(row_buf[0:hidden_size], expert_input[slot, 0:hidden_size])
                    else:
                        inverse_indices[route_start + r] = -1

    return _tight_scan_gather_main


def _make_small_fused(num_tokens, hidden_size, top_k, num_experts, dtype):
    import tilelang.ascend.language as T

    numel = num_tokens * top_k
    e_pad = (num_experts + 63) // 64 * 64
    ids_cap = max(256, ((numel + 255) // 256) * 256)

    @T.prim_func
    def _small_fused_main(
        hidden_states: T.Tensor((num_tokens, hidden_size), dtype),
        flat_ids: T.Tensor((numel,), "int32"),
        expert_input: T.Tensor((numel, hidden_size), dtype),
        physical_ends: T.Tensor((num_experts,), "int32"),
        inverse_indices: T.Tensor((numel,), "int32"),
    ):
        with T.Kernel(1):
            ids32 = T.alloc_shared((ids_cap,), "int32")
            ids8 = T.alloc_shared((ids_cap,), "uint8")
            counts = T.alloc_shared((e_pad,), "int32")
            cursors = T.alloc_shared((e_pad,), "int32")
            row_buf = T.alloc_shared((hidden_size,), dtype)

            T.copy(flat_ids[0:numel], ids32[0:numel])

            if num_experts <= 64:
                with T.SimdVF():
                    T.simd.vsts(counts[0], T.simd.vdup(T.int32(0), "int32"))

                for block in range((numel + 255) // 256):
                    base = block * 256
                    if base < numel:
                        rem = T.min(256, numel - base)
                        with T.SimdVF():
                            m16 = T.simd.pset(16)
                            for unit in range(4):
                                values = T.simd.vld(ids32[base + unit * 64])
                                low8 = T.simd.vcvt(values, "uint8", part=0)
                                T.simd.vsts(ids8[base + unit * 64], low8, dist="PK4_B32")
                            T.simd.mem_bar("VST_VLD")
                            src = T.simd.vld(ids8[base], dist="NORM")
                            m8 = T.simd.update_mask(rem, 8)
                            freq = T.simd.alloc_var("uint16")
                            freq = T.simd.vdup(T.uint16(0), "uint16", m16)
                            T.simd.dhistv2(freq, src, m8, bin=0)
                            wide = T.reinterpret(T.simd.vunpack(freq, part="LOWER"), "int32x64")
                            T.simd.vsts(
                                counts[0],
                                T.simd.vadd(T.simd.vld(counts[0]), wide),
                            )
            else:
                with T.SimtVF(threads=128):
                    for e in T.Parallel(e_pad):
                        counts[e] = T.int32(0)
                    for idx in T.Parallel(numel, coalesced_width=1):
                        T.atomic_add(counts[ids32[idx]], T.int32(1))

            # Single-core serial prefix: no cross-core counts workspace, no
            # grid barrier, no second launch.
            running = T.alloc_var("int32", init=0)
            for e in T.serial(num_experts):
                end = running + counts[e]
                counts[e] = running
                cursors[e] = running
                physical_ends[e] = end
                running = end

            for r in T.Pipelined(numel, num_stages=2):
                e = ids32[r]
                if e >= 0 and e < num_experts:
                    slot = cursors[e]
                    cursors[e] = slot + 1
                    inverse_indices[r] = slot
                    token_row = r // top_k
                    T.copy(
                        hidden_states[token_row, 0:hidden_size],
                        row_buf[0:hidden_size],
                    )
                    T.copy(
                        row_buf[0:hidden_size],
                        expert_input[slot, 0:hidden_size],
                    )
                else:
                    inverse_indices[r] = -1

    return _small_fused_main


def _make_short_rank(t, h, k, e, dtype):
    """Stable route ranks by parallel comparisons; single-token measured specialization."""
    import tilelang.ascend.language as T

    p = t * k
    cap = (p + 63) // 64 * 64
    cores = min(max(t, 32), 64)

    @T.prim_func
    def main(
        x: T.Tensor((t, h), dtype),
        ids: T.Tensor((p,), "int32"),
        y: T.Tensor((p, h), dtype),
        ends: T.Tensor((e,), "int32"),
        inverse: T.Tensor((p,), "int32"),
    ):
        with T.Kernel(cores) as core:
            ib = T.alloc_shared((cap,), "int32")
            dest = T.alloc_shared((k * 64,), "int32")
            tmp = T.alloc_shared((64,), "int32")
            row = T.alloc_shared((h,), dtype)
            T.copy(ids, ib[:p])
            for ee in T.serial(T.ceildiv(e, cores)):
                expert = ee * cores + core
                if expert < e:
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        acc = T.simd.alloc_var("int32")
                        acc = T.simd.vdup(0, "int32", full)
                        for chunk in T.serial(cap // 64):
                            valid = T.simd.update_mask(T.min(64, p - chunk * 64), 32)
                            v = T.simd.vld(ib[chunk * 64])
                            keep = T.simd.vcmps(v, expert, valid, op="le")
                            ones = T.simd.vsel(
                                T.simd.vdup(1, "int32", full), T.simd.vdup(0, "int32", full), keep
                            )
                            acc = T.simd.vadd(acc, ones, full)
                        total = T.simd.vcadd(acc, full)
                        T.simd.vsts(tmp[0], total, full, dist="ONEPT_B32")
                    T.copy(tmp[:1], ends[expert : expert + 1])
            for step in T.serial(T.ceildiv(t, cores)):
                token = step * cores + core
                if token < t:
                    for slot in T.serial(k):
                        route = token * k + slot
                        expert = ib[route]
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            acc = T.simd.alloc_var("int32")
                            acc = T.simd.vdup(0, "int32", full)
                            for chunk in T.serial(cap // 64):
                                valid = T.simd.update_mask(T.min(64, p - chunk * 64), 32)
                                v = T.simd.vld(ib[chunk * 64])
                                idx = T.simd.vci(chunk * 64, "int32")
                                lower = T.simd.vcmps(v, expert, valid, op="lt")
                                equal = T.simd.vcmps(v, expert, valid, op="eq")
                                earlier = T.simd.vcmps(idx, route, valid, op="lt")
                                same = T.simd.pand(equal, earlier, valid)
                                keep = T.simd.por(lower, same, valid)
                                ones = T.simd.vsel(
                                    T.simd.vdup(1, "int32", full),
                                    T.simd.vdup(0, "int32", full),
                                    keep,
                                )
                                acc = T.simd.vadd(acc, ones, full)
                            total = T.simd.vcadd(acc, full)
                            T.simd.vsts(dest[slot * 64], total, full, dist="ONEPT_B32")
                        T.copy(dest[slot * 64 : slot * 64 + 1], inverse[route : route + 1])
                    T.copy(x[token, :], row)
                    for slot in T.serial(k):
                        target = dest[slot * 64]
                        T.copy(row, y[target, :])

    return main


class _CompiledTightScanGather:
    def __init__(self, tight_scan_count, gather, counts):
        self.tight_scan_count = tight_scan_count
        self.gather = gather
        self.tight_scan_counts = counts

    def __call__(self, hidden_states, flat_ids, expert_input, ends, inverse):
        self.tight_scan_count(flat_ids, self.tight_scan_counts)
        self.gather(hidden_states, flat_ids, self.tight_scan_counts, expert_input, ends, inverse)


class AscendMoEPrePermuteKernel(AscendMoEKernel):
    """Dispatch to fused short-route or split counting pre-permute."""

    def __init__(
        self, num_tokens, hidden_size, top_k, num_experts, dtype=torch.bfloat16, num_cores=64
    ):
        super().__init__(num_cores)
        check_dtype(dtype)
        if min(num_tokens, hidden_size, top_k, num_experts) <= 0 or num_tokens * top_k >= 2**31:
            raise ValueError("positive dimensions and fewer than 2**31 routes are required")
        if num_experts > 256:
            raise ValueError("this prototype supports at most 256 local experts")
        self.num_tokens, self.hidden_size = num_tokens, hidden_size
        self.top_k, self.num_experts, self.dtype = top_k, num_experts, dtype
        # Split counting uses one unrolled histogram group per 64 experts.
        self.path = (
            "sym"
            if _use_sym(num_tokens, hidden_size, top_k, num_experts)
            else "fused_small"
            if num_tokens <= SMALL_FUSED_MAX_TOKENS
            else (
                "split_small"
                if num_tokens <= 64
                else (
                    "full"
                    if num_experts <= 64
                    else ("cut_origin" if num_experts <= 128 else "grouped_hist")
                )
            )
        )

    def compile(self, device=None):
        if device is None or device.type not in ("npu", "privateuseone"):
            raise ValueError("an explicit NPU device is required")
        if device not in self._compiled:
            import tilelang

            with torch.npu.device(device):
                measured = self.dtype == torch.bfloat16 and self.config["num_cores"] == 64
                shape = (self.num_tokens, self.hidden_size, self.top_k, self.num_experts)
                if measured and shape == (1, 2048, 8, 128):
                    self._compiled[device] = tilelang.compile(
                        _make_short_rank(
                            self.num_tokens,
                            self.hidden_size,
                            self.top_k,
                            self.num_experts,
                            str(self.dtype).removeprefix("torch."),
                        ),
                        target="ascend",
                        out_idx=[],
                    )
                    return self._compiled[device]
                if _use_sym(self.num_tokens, self.hidden_size, self.top_k, self.num_experts):
                    self._compiled[device] = tilelang.compile(
                        _make_sym_fused(
                            self.num_tokens,
                            self.hidden_size,
                            self.top_k,
                            self.num_experts,
                            str(self.dtype).removeprefix("torch."),
                        ),
                        target="ascend",
                        out_idx=[],
                        pass_configs={
                            tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
                            tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
                            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
                        },
                    )
                    return self._compiled[device]
                if self.num_tokens <= SMALL_FUSED_MAX_TOKENS:
                    # Preserve the original single-token fallback for shapes
                    # outside the verified sym specialization.
                    fused = tilelang.compile(
                        _make_small_fused(
                            self.num_tokens,
                            self.hidden_size,
                            self.top_k,
                            self.num_experts,
                            str(self.dtype).removeprefix("torch."),
                        ),
                        target="ascend",
                        out_idx=None,
                    )
                    self._compiled[device] = fused
                    return self._compiled[device]

                # T=2..64 uses C=min(64,T), so the count->route prefix and
                # count workspace shrink with T.  T>=64 is capped at 64.
                cores = min(self.config["num_cores"], self.num_tokens)
                use_vector_scan = measured and shape in (
                    (32, 2048, 8, 128),
                    (512, 7168, 8, 256),
                    (128, 4096, 8, 128),
                    (64, 7168, 8, 256),
                )
                if use_vector_scan and self.num_tokens == 64:
                    cores = 32
                gather_factory = (
                    _make_vector_scan_gather if use_vector_scan else _make_tight_scan_gather
                )
                e_pad = (self.num_experts + 63) // 64 * 64
                tight_scan_count = tilelang.compile(
                    _make_tight_scan_count(self.num_tokens, self.top_k, self.num_experts, cores),
                    target="ascend",
                    out_idx=None,
                )
                gather = tilelang.compile(
                    gather_factory(
                        self.num_tokens,
                        self.hidden_size,
                        self.top_k,
                        self.num_experts,
                        str(self.dtype).removeprefix("torch."),
                        cores,
                    ),
                    target="ascend",
                    out_idx=None,
                )
            counts = torch.empty((cores * e_pad,), dtype=torch.int32, device=device)
            self._compiled[device] = _CompiledTightScanGather(tight_scan_count, gather, counts)
        return self._compiled[device]

    def __call__(self, hidden_states, local_expert_ids):
        check_tensor(
            hidden_states, "hidden_states", (self.num_tokens, self.hidden_size), self.dtype
        )
        check_tensor(
            local_expert_ids,
            "local_expert_ids",
            (self.num_tokens, self.top_k),
            torch.int32,
            hidden_states.device,
        )
        program = self.compile(hidden_states.device)
        rows = self.num_tokens * self.top_k
        flat_ids = local_expert_ids.flatten().contiguous()
        expert_input = hidden_states.new_empty((rows, self.hidden_size))
        physical_ends = torch.empty(
            (self.num_experts,), dtype=torch.int32, device=hidden_states.device
        )
        inverse_indices = torch.empty((rows,), dtype=torch.int32, device=hidden_states.device)
        program(hidden_states, flat_ids, expert_input, physical_ends, inverse_indices)
        return expert_input, physical_ends, inverse_indices
