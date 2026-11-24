"""Static host KV-block plan standalone fused output implementation: B1H1, BF16/D128, full page128 KV.

Includes the host planner, paged Cube local computation and raw SIMD merge.
SIMD macros adapted from tile-ai/tilelang at a7cb753dafc6b2514cbbd7c8431260c9d05e7df0;
See the repository MIT LICENSE.
"""

from dataclasses import dataclass
from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.ascend.language import simd as S
from tilelang.carver.arch.driver import get_num_vector_cores
from tilelang.layout import make_ascend_compact_nz_layout

from tileops._kernels.ascend.attention.mha_paged_vector import paged_programs
from tileops._kernels.attention.call_spec import AttentionCall, MHAPagedDecodeFwdInterface
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type


@dataclass(frozen=True)
class Plan:
    q: int
    kv: int
    programs: tuple
    # Each segment is (Q tile, first KV block, number of KV blocks, partial slot).
    # slot=-1 means complete Q tile, written directly to final output.
    merges: tuple  # (Q tile, first partial slot, number of partial slots)
    partial_count: int
    strategy: str = "balanced"

    @property
    def max_segments(self):
        return max(map(len, self.programs))

    @property
    def max_parts(self):
        return max((n for _, _, n in self.merges), default=0)

    def describe(self):
        loads = [sum(s[2] for s in program) for program in self.programs]
        return dict(
            strategy=self.strategy,
            Q=self.q,
            KV=self.kv,
            programs=len(loads),
            q_tiles=(self.q + 127) // 128,
            min_kv_blocks=min(loads),
            max_kv_blocks=max(loads),
            segments=sum(map(len, self.programs)),
            max_segments=self.max_segments,
            partial_count=self.partial_count,
            merge_q_tiles=len(self.merges),
            direct_q_tiles=(self.q + 127) // 128 - len(self.merges),
            max_parts=self.max_parts,
            partial_bytes=self.partial_count * 128 * 128 * 4,
            lse_bytes=self.partial_count * 128 * 4,
        )


@lru_cache(maxsize=128)
def make_plan(q, kv, cores=28):
    if type(q) is not int or type(kv) is not int or q <= 0 or kv <= 0 or kv % 128:
        raise ValueError("positive integer Q and positive 128-aligned KV required")
    if type(cores) is not int or not 1 <= cores <= 28:
        raise ValueError("cores must be 1..28")
    tiles, blocks = (q + 127) // 128, kv // 128
    total = tiles * blocks
    if total > 2**31 - 1:
        raise ValueError("int32 scheduling range exceeded")
    cores = min(cores, total)
    programs = [[] for _ in range(cores)]
    owners = [[] for _ in range(tiles)]
    for pid in range(cores):
        first = pid * (total // cores) + min(pid, total % cores)
        stop = first + total // cores + int(pid < total % cores)
        while first < stop:
            tile, begin = divmod(first, blocks)
            end = min(stop, (tile + 1) * blocks)
            idx = len(programs[pid])
            programs[pid].append([tile, begin, end - first, -1])
            owners[tile].append((pid, idx))
            first = end
    merges = []
    slot = 0
    for tile, segments in enumerate(owners):
        if len(segments) > 1:
            merges.append((tile, slot, len(segments)))
            for pid, idx in segments:
                programs[pid][idx][3] = slot
                slot += 1
    plan = Plan(q, kv, tuple(tuple(tuple(s) for s in p) for p in programs), tuple(merges), slot)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    blocks = plan.kv // 128
    tiles = (plan.q + 127) // 128
    slots, direct, groups, intervals, loads = [], set(), {}, {}, []
    for program in plan.programs:
        assert program
        loads.append(sum(s[2] for s in program))
        for tile, first, count, slot in program:
            assert 0 <= tile < tiles and 0 <= first < blocks and 0 < count <= blocks - first
            intervals.setdefault(tile, []).append((first, first + count))
            if slot < 0:
                assert first == 0 and count == blocks and tile not in direct
                direct.add(tile)
            else:
                slots.append(slot)
                groups.setdefault(tile, []).append(slot)
    assert set(intervals) == set(range(tiles))
    for spans in intervals.values():
        cursor = 0
        for first, stop in sorted(spans):
            assert first == cursor
            cursor = stop
        assert cursor == blocks
    assert max(loads) - min(loads) <= (1 if plan.strategy == "balanced" else blocks)
    assert sorted(slots) == list(range(plan.partial_count))
    assert set(groups).isdisjoint(direct)
    assert len(groups) + len(direct) == tiles
    assert len(plan.merges) <= len(plan.programs) - 1
    assert len({tile for tile, _, _ in plan.merges}) == len(plan.merges)
    assert {tile for tile, _, _ in plan.merges} == set(groups)
    for tile, first, count in plan.merges:
        assert count >= 2 and sorted(groups[tile]) == list(range(first, first + count))
    return True


@lru_cache(maxsize=128)
def make_tail_plan(q, kv, cores=28, pieces=None):
    """Whole Q tiles in full waves; split only Q tiles in the final wave."""
    make_plan(q, kv, cores)
    if pieces is not None and (type(pieces) is not int or pieces < 2):
        raise ValueError("pieces must be None or an integer >=2")
    tiles, blocks = (q + 127) // 128, kv // 128
    full, tail = divmod(tiles, cores)
    if not full or not tail or blocks < 2:
        return make_q_plan(q, kv, cores)
    programs = [
        [(tile, 0, blocks, -1) for tile in range(pid * full, (pid + 1) * full)]
        for pid in range(cores)
    ]
    merges, slot, receiver = [], 0, 0
    for extra in range(tail):
        tile = full * cores + extra
        count = min(blocks, pieces or (cores // tail + int(extra < cores % tail)))
        # Explicit larger counts may fill multiple waves of tail fragments.
        if count == 1:
            programs[receiver % cores].append((tile, 0, blocks, -1))
            receiver += 1
            continue
        merges.append((tile, slot, count))
        for part in range(count):
            first = part * (blocks // count) + min(part, blocks % count)
            size = blocks // count + int(part < blocks % count)
            programs[receiver % cores].append((tile, first, size, slot))
            receiver += 1
            slot += 1
    if not merges:
        return make_q_plan(q, kv, cores)
    plan = Plan(q, kv, tuple(tuple(p) for p in programs), tuple(merges), slot, "tail")
    validate_plan(plan)
    return plan


@lru_cache(maxsize=128)
def make_q_plan(q, kv, cores=28):
    """Assign whole consecutive Q tiles; never create partial KV states."""
    # Reuse argument/range validation from the balanced planner.
    make_plan(q, kv, cores)
    tiles, blocks = (q + 127) // 128, kv // 128
    cores = min(cores, tiles)
    programs = []
    for pid in range(cores):
        first = pid * (tiles // cores) + min(pid, tiles % cores)
        count = tiles // cores + int(pid < tiles % cores)
        programs.append(tuple((tile, 0, blocks, -1) for tile in range(first, first + count)))
    plan = Plan(q, kv, tuple(programs), (), 0, "q_only")
    validate_plan(plan)
    return plan


# Initial stage-cost estimates in microseconds, not a fitted latency predictor.
# Fixed for the entire grid: no shape lookup or runtime timing/search on device.
COST_MODEL = dict(
    block_us=0.72, segment_us=2.0, merge_launch_us=12.0, partial_us=0.20, merge_tile_us=0.15
)


def estimate_cost(plan):
    """Critical program work plus selective merge; common call costs cancel."""
    c = COST_MODEL
    local = max(
        sum(s[2] for s in program) * c["block_us"] + len(program) * c["segment_us"]
        for program in plan.programs
    )
    merge = (
        (
            c["merge_launch_us"]
            + plan.partial_count * c["partial_us"]
            + len(plan.merges) * c["merge_tile_us"]
        )
        if plan.merges
        else 0.0
    )
    return dict(local_us=local, merge_us=merge, total_us=local + merge)


@lru_cache(maxsize=128)
def make_stream_plan(q, kv, cores=28):
    """Whole-Q-first host assignment with a remaining-work target per core.

    Independent simplified implementation for fixed 128x128 dense tiles;
    follows the assignment order used by FIA split_core_v2, not its ABI.
    """
    make_plan(q, kv, cores)  # Validate scalar/range constraints.
    tiles, blocks = (q + 127) // 128, kv // 128
    cores = min(cores, tiles * blocks)
    programs, cursor = [], 0
    total = tiles * blocks
    for pid in range(cores):
        if cursor == total:
            break
        # Half-block tolerance, as in the source's whole-row/block assignment.
        target = (total - cursor) / (cores - pid)
        assigned, tasks = 0, []
        while cursor < total:
            tile, first = divmod(cursor, blocks)
            remaining = blocks - first
            if assigned + remaining <= target + 0.5 or pid == cores - 1:
                take = remaining  # Prefer finishing the current Q tile.
            else:
                take = min(remaining, max(0, int(target + 0.5 - assigned)))
                if not tasks:
                    take = max(1, take)
                if take == 0:
                    break
            tasks.append([tile, first, take, -1])
            cursor += take
            assigned += take
            if assigned >= target - 0.5:
                break
        programs.append(tasks)
    assert cursor == total
    owners = {}
    for pid, tasks in enumerate(programs):
        for idx, (tile, _, _, _) in enumerate(tasks):
            owners.setdefault(tile, []).append((pid, idx))
    merges, slot = [], 0
    for tile, entries in sorted(owners.items()):
        if len(entries) > 1:
            merges.append((tile, slot, len(entries)))
            for pid, idx in entries:
                programs[pid][idx][3] = slot
                slot += 1
    if not merges:
        # The direct ABI derives ranges from pid; retain its canonical layout.
        return make_q_plan(q, kv, len(programs))
    plan = Plan(
        q,
        kv,
        tuple(tuple(tuple(s) for s in tasks) for tasks in programs),
        tuple(merges),
        slot,
        "stream",
    )
    validate_plan(plan)
    return plan


@lru_cache(maxsize=128)
def choose_stream_plan(q, kv):
    """Compare split/no-split with an explicit gain threshold; no shape table."""
    direct = make_q_plan(q, kv)
    candidates = [make_stream_plan(q, kv, cores) for cores in range(1, 29)]
    # Also allow the independent final-wave assignment with adaptive pieces.
    # This is our extension, not a claim of reproducing AscendC's final plan.
    candidates.extend(make_tail_plan(q, kv, cores) for cores in range(1, 29))
    split = min(candidates, key=lambda p: (estimate_cost(p)["total_us"], p.partial_count))
    # Existing score already charges merge; require one additional block's gain.
    if (
        estimate_cost(direct)["total_us"] - estimate_cost(split)["total_us"]
        > COST_MODEL["block_us"]
    ):
        return split
    return direct


def _make_merge(sq, slots, merge_count, max_parts, rows):
    groups_per_tile = 128 // rows
    grid = max(1, merge_count * groups_per_tile)
    stats_plane = slots * 128
    stats_size = 2 * stats_plane
    stat_pattern = f"PAT_VL{rows}"

    @T.prim_func
    def merge_simd(
        Parts: T.Tensor((slots, 128, 128), "float32"),
        LSE: T.Tensor((stats_size,), "float32"),
        Merges: T.Tensor((max(1, merge_count), 3), "int32"),
        O: T.Tensor((1, sq, 128), "bfloat16"),
    ):
        with T.Kernel(grid) as core:
            group = core // groups_per_tile
            row = core % groups_per_tile * rows
            tile = Merges[group, 0]
            first = Merges[group, 1]
            pieces = Merges[group, 2]
            qi = tile * 128 + row
            if qi < sq:
                part = T.alloc_shared((rows, 128), "float32")
                # Padded stats keep full-width vector loads inside UB buffers.
                score = T.alloc_shared((64,), "float32")
                local_sum = T.alloc_shared((64,), "float32")
                maximum = T.alloc_shared((64,), "float32")
                total = T.alloc_shared((64,), "float32")
                old_weights = T.alloc_shared((64,), "float32")
                new_weights = T.alloc_shared((64,), "float32")
                acc = T.alloc_shared((rows, 128), "float32")
                out = T.alloc_shared((rows, 128), "bfloat16")
                with T.SimdVF(latency=200):
                    full = S.pset(32, "PAT_ALL")
                    zero = S.vdup(T.float32(0), "float32", full)
                    initial = S.vdup(T.float32(-1e30), "float32", full)
                    S.vsts(maximum[0], initial, full, "NORM_B32")
                    S.vsts(total[0], zero, full, "NORM_B32")
                    for r in range(rows):
                        for c in range(0, 128, 64):
                            S.vsts(acc[r, c], zero, full, "NORM_B32")
                for piece in T.serial(max_parts):
                    if piece < pieces:
                        slot = first + piece
                        T.copy(Parts[slot, row : row + rows, :], part)
                        offset = slot * 128 + row
                        T.copy(LSE[offset : offset + rows], score[0:rows])
                        T.copy(
                            LSE[stats_plane + offset : stats_plane + offset + rows],
                            local_sum[0:rows],
                        )
                        with T.SimdVF(latency=400):
                            full = S.pset(32, "PAT_ALL")
                            stats = S.pset(32, stat_pattern)
                            zero = S.vdup(T.float32(0), "float32", full)
                            old_max = S.vld(maximum[0], "NORM")
                            current_score = S.vld(score[0], "NORM")
                            valid_sum = S.vcmps(
                                S.vld(local_sum[0], "NORM"), T.float32(0), stats, "gt"
                            )
                            current_score = S.vsel(
                                current_score, S.vdup(T.float32(-1e30), "float32", full), valid_sum
                            )
                            peak = S.vmax(old_max, current_score, stats)
                            old_weight = S.vexpdif(old_max, peak, stats)
                            valid = S.vcmps(current_score, T.float32(-1e29), stats, "gt")
                            new_weight = S.vsel(S.vexpdif(current_score, peak, stats), zero, valid)
                            contribution = S.vmul(new_weight, S.vld(local_sum[0], "NORM"), stats)
                            denominator = S.vadd(
                                S.vmul(S.vld(total[0], "NORM"), old_weight, stats),
                                contribution,
                                stats,
                            )
                            S.vsts(maximum[0], peak, stats, "NORM_B32")
                            S.vsts(total[0], denominator, stats, "NORM_B32")
                            S.vsts(old_weights[0], old_weight, stats, "NORM_B32")
                            S.vsts(new_weights[0], new_weight, stats, "NORM_B32")
                            S.mem_bar("VST_VLD")
                            for r in range(rows):
                                alpha = S.vld(old_weights[r], "BRC_B32")
                                beta = S.vld(new_weights[r], "BRC_B32")
                                for c in range(0, 128, 64):
                                    previous = S.vld(acc[r, c], "NORM")
                                    current = S.vld(part[r, c], "NORM")
                                    updated = S.vadd(
                                        S.vmul(previous, alpha, full),
                                        S.vmul(current, beta, full),
                                        full,
                                    )
                                    S.vsts(acc[r, c], updated, full, "NORM_B32")
                with T.SimdVF(latency=200):
                    full = S.pset(32, "PAT_ALL")
                    minimum = S.vdup(T.float32(1e-30), "float32", full)
                    for r in range(rows):
                        denominator = S.vmax(S.vld(total[r], "BRC_B32"), minimum, full)
                        for c in range(0, 128, 64):
                            value = S.vdiv(S.vld(acc[r, c], "NORM"), denominator, full)
                            converted = S.vcvt(value, "bfloat16", full)
                            S.vsts(out[r, c], converted, full, "PK_B32")
                T.copy(out, O[0, qi : qi + rows, :])

    return merge_simd


@lru_cache(maxsize=64)
def build_kernel(sq, kv, pool_rows):
    """Build noncausal paged attention for fixed B1H1 BF16/D128 shapes.

    Q has shape (1, sq, 1, 128). K and V have shape (pool_rows, 1, 128).
    The int32 page table has shape (1, kv // 128) and must contain valid
    physical page IDs. The int32 lengths tensor has shape (1,) and must
    contain kv. All tensors must be contiguous and on the same NPU.
    Call validate_inputs outside the timed path before dispatch.
    """
    plan = choose_stream_plan(sq, kv)
    if pool_rows <= 0 or pool_rows % 128 or pool_rows < kv:
        raise ValueError("whole physical pool pages, at least KV capacity required")
    BR, BC, ROWS, VL, D, STATS_ROWS = 128, 128, 64, 64, 128, 64
    dtype, accum_dtype = "bfloat16", "float32"
    scale = D**-0.5
    page_size = 128
    width = kv // 128
    grid = len(plan.programs)
    max_segments = plan.max_segments
    slots = max(1, plan.partial_count)
    merge_count = len(plan.merges)
    max_parts = plan.max_parts
    has_merge = merge_count > 0
    stats_plane = slots * BR
    stats_size = 2 * stats_plane
    merge_rows = min(8, tilelang.next_power_of_2(sq))
    pipeline_annotations = {"enable_offset": True}

    @T.macro
    def softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub):
        # Online softmax: update the row maximum and denominator, pack P into
        # Cube NZ layout, and return the factor that rescales the previous PV.
        with T.SimdVF(latency=735):
            softmax_full = S.pset(32, "PAT_ALL")
            softmax_one = S.pset(32, "PAT_VL1")
            softmax_b16 = S.pset(16, "PAT_VL128")
            dst_nz_ptr = S.make_ubuf_ptr(T.access_ptr(P_nz_ub[0, 0], "w", ROWS + 1, BC), "bfloat16")
            S.vsts(old_m_ub[0], S.vld(m_ub[0], "NORM"), softmax_full, "NORM_B32")
            # Explicit expansion keeps register indices constant across two rows.
            group = 2
            low = S.alloc_local((group,), T.float32)
            high = S.alloc_local((group,), T.float32)
            for r in range(0, ROWS, group):
                for i in T.unroll(group, explicit=True):
                    low[i] = S.vmuls(S.vld(S_ub[r + i, 0], "NORM"), T.float32(scale), softmax_full)
                    high[i] = S.vmuls(
                        S.vld(S_ub[r + i, VL], "NORM"), T.float32(scale), softmax_full
                    )
                for i in T.unroll(group, explicit=True):
                    S.vsts(S_ub[r + i, 0], low[i], softmax_full, "NORM_B32")
                    S.vsts(S_ub[r + i, VL], high[i], softmax_full, "NORM_B32")
                for i in T.unroll(group, explicit=True):
                    row_max = S.vcmax(S.vmax(low[i], high[i], softmax_full), softmax_full)
                    S.vsts(m_ub[r + i], row_max, softmax_one, "ONEPT_B32")
            S.mem_bar("VST_VLD")
            merged_max = S.vmax(S.vld(m_ub[0], "NORM"), S.vld(old_m_ub[0], "NORM"), softmax_full)
            S.vsts(m_ub[0], merged_max, softmax_full, "NORM_B32")
            S.mem_bar("VST_VLD")
            even_exp = S.alloc_local((group,), T.float32)
            odd_exp = S.alloc_local((group,), T.float32)
            for r in range(0, ROWS, group):
                for i in T.unroll(group, explicit=True):
                    row_max = S.vld(m_ub[r + i], "BRC_B32")
                    even, odd = S.vld2(S_ub[r + i, 0], "DINTLV_B32")
                    even_exp[i] = S.vexpdif(even, row_max, softmax_full)
                    odd_exp[i] = S.vexpdif(odd, row_max, softmax_full)
                for i in T.unroll(group, explicit=True):
                    row_sum = S.vcadd(S.vadd(even_exp[i], odd_exp[i], softmax_full), softmax_full)
                    S.vsts(row_sum_ub[r + i], row_sum, softmax_one, "ONEPT_B32")
                # Store in row order because each store advances the NZ pointer.
                for i in T.unroll(group, explicit=True):
                    even_bf16 = S.vcvt(even_exp[i], "bfloat16", softmax_full, sat=False, part=0)
                    odd_bf16 = S.vcvt(odd_exp[i], "bfloat16", softmax_full, sat=False, part=1)
                    packed = S.vor(
                        T.reinterpret(even_bf16, "uint16x128"),
                        T.reinterpret(odd_bf16, "uint16x128"),
                        softmax_b16,
                    )
                    dst_nz_ptr = S.vsstb(
                        T.reinterpret(packed, "bfloat16x128"),
                        dst_nz_ptr,
                        T.int32((ROWS + 1) << 16 | 1),
                        softmax_b16,
                        update=True,
                    )
            S.mem_bar("VST_VLD")
            alpha_reg = S.vexpdif(S.vld(old_m_ub[0], "NORM"), S.vld(m_ub[0], "NORM"), softmax_full)
            S.vsts(alpha_ub[0], alpha_reg, softmax_full, "NORM_B32")
            l_scaled = S.vmul(S.vld(l_ub[0], "NORM"), alpha_reg, softmax_full)
            l_updated = S.vadd(l_scaled, S.vld(row_sum_ub[0], "NORM"), softmax_full)
            S.vsts(l_ub[0], l_updated, softmax_full, "NORM_B32")

    @T.macro
    def accumulate_output(O_tmp_ub, O_ub, alpha_ub):
        # Preserve the original online-softmax accumulation order.
        with T.SimdVF(latency=318):
            rescale_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                alpha = S.vld(alpha_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    prev = S.vld(O_ub[i, c], "NORM")
                    cur = S.vld(O_tmp_ub[i, c], "NORM")
                    updated = S.vadd(S.vmul(alpha, prev, rescale_mask), cur, rescale_mask)
                    S.vsts(O_ub[i, c], updated, rescale_mask, "NORM_B32")

    @T.macro
    def init_m_l(m_ub, l_ub):
        with T.SimdVF(latency=400):
            init_mask = S.pset(32, "PAT_ALL")
            zero = S.vdup(T.float32(0.0), accum_dtype, init_mask)
            initial_m = S.vdup(-T.infinity(accum_dtype), accum_dtype, init_mask)
            S.vsts(l_ub[0], zero, init_mask, "NORM_B32")
            S.vsts(m_ub[0], initial_m, init_mask, "NORM_B32")

    @T.macro
    def clear_O_ub(O_ub):
        with T.SimdVF(latency=142):
            clear_mask = S.pset(32, "PAT_ALL")
            output_zero = S.vdup(T.float32(0.0), accum_dtype, clear_mask)
            for i in range(ROWS):
                for c in range(0, D, VL):
                    S.vsts(O_ub[i, c], output_zero, clear_mask, "NORM_B32")

    @T.macro
    def write_final_output(output, total, destination):
        cast_view = T.view(output, (ROWS * 2, D), dtype=dtype)
        with T.SimdVF(latency=193):
            mask = S.pset(32, "PAT_ALL")
            minimum = S.vdup(T.float32(1e-30), accum_dtype, mask)
            for r in range(ROWS):
                denominator = S.vmax(S.vld(total[r], "BRC_B32"), minimum, mask)
                for c in range(0, D, VL):
                    normalized = S.vdiv(S.vld(output[r, c], "NORM"), denominator, mask)
                    value = S.vcvt(normalized, dtype, mask)
                    S.vsts(cast_view[r, c], value, mask, "PK_B32")
        T.dual_copy(cast_view[0:ROWS, 0:D], destination)

    @T.macro
    def local_body(Q, K, V, Table, Counts, Tasks, Parts, LSE, O):
        with T.Kernel(grid) as bx:
            Q_l1 = T.alloc_l1((BR, D), dtype)
            K_l1 = T.alloc_l1((BC, D), dtype)
            V_l1 = T.alloc_l1((D, BC), dtype)
            P_l1 = T.alloc_l1((BR, BC), dtype)
            qa = T.alloc_l0a((BR, D), dtype)
            kb = T.alloc_l0b((BC, D), dtype)
            pa = T.alloc_l0a((BR, BC), dtype)
            vb = T.alloc_l0b((D, BC), dtype)
            qk = T.alloc_l0c((BR, BC), accum_dtype)
            pv = T.alloc_l0c((BR, D), accum_dtype)
            scores = T.alloc_shared((ROWS, BC), accum_dtype)
            probs = T.alloc_shared((ROWS + 1, BC), dtype)
            T.annotate_layout({probs: make_ascend_compact_nz_layout(probs)})
            current = T.alloc_shared((ROWS, D), accum_dtype)
            output = T.alloc_shared((ROWS, D), accum_dtype)
            maximum = T.alloc_shared((STATS_ROWS,), accum_dtype)
            total = T.alloc_shared((STATS_ROWS,), accum_dtype)
            alpha = T.alloc_shared((STATS_ROWS,), accum_dtype)
            old_max = T.alloc_shared((STATS_ROWS,), accum_dtype)
            row_sum = T.alloc_shared((STATS_ROWS,), accum_dtype)
            T.annotate_buffer_versions(
                {K_l1: 3, V_l1: 3, qk: 2, pv: 2, scores: 3, probs: 2, current: 1, alpha: 3}
            )
            for work in T.serial(max_segments):
                if work < (
                    Counts[bx]
                    if has_merge
                    else (
                        (sq + BR - 1) // BR // grid
                        + T.if_then_else(bx < ((sq + BR - 1) // BR) % grid, 1, 0)
                    )
                ):
                    if has_merge:
                        task = Tasks[bx, work, 0]
                        begin = Tasks[bx, work, 1]
                        count = Tasks[bx, work, 2]
                        slot = Tasks[bx, work, 3]
                    else:
                        task = (
                            bx * (((sq + BR - 1) // BR) // grid)
                            + T.min(bx, ((sq + BR - 1) // BR) % grid)
                            + work
                        )
                        begin = 0
                        count = width
                        slot = -1
                    qi = task * BR
                    b, h = 0, 0
                    T.copy(Q[b, h, qi : qi + BR, :], Q_l1)
                    init_m_l(maximum, total)
                    clear_O_ub(output)
                    T.copy(Q_l1, qa)
                    for tile in T.Pipelined(count, num_stages=2, annotations=pipeline_annotations):
                        logical = (begin + tile) * BC
                        physical = Table[b, logical // page_size] * page_size + logical % page_size
                        with T.Stage(0):
                            T.copy(K[physical : physical + BC, h, :], K_l1)
                        with T.Stage(1):
                            T.copy(K_l1, kb)
                            T.gemm(qa, kb, qk, transpose_B=True, clear_accum=True)
                        with T.Stage(3):
                            T.dual_copy(qk, scores)
                        with T.Stage(5):
                            softmax(scores, probs, maximum, total, alpha, old_max, row_sum)
                        with T.Stage(6):
                            T.dual_copy(probs[0:ROWS, :], P_l1)
                        with T.Stage(3):
                            T.copy(V[physical : physical + BC, h, :], V_l1, transpose=True)
                        with T.Stage(6):
                            T.copy(P_l1, pa)
                            T.copy(V_l1, vb, transpose=False)
                            T.gemm(pa, vb, pv, transpose_B=True, clear_accum=True)
                        with T.Stage(7):
                            T.dual_copy(pv, current)
                            accumulate_output(current, output, alpha)
                    if has_merge:
                        if slot >= 0:
                            T.dual_copy(output, Parts[slot, :, :])
                            T.dual_copy(maximum, LSE[slot * BR : (slot + 1) * BR])
                            T.dual_copy(
                                total, LSE[stats_plane + slot * BR : stats_plane + (slot + 1) * BR]
                            )
                        else:
                            write_final_output(
                                output, total, O[b, qi : qi + BR, h * D : (h + 1) * D]
                            )
                    else:
                        write_final_output(output, total, O[b, qi : qi + BR, h * D : (h + 1) * D])

    # Both entry signatures inline the same Cube/Vector computation macro.
    # With no fragments, unused workspace and metadata never enter the ABI.
    if has_merge:

        @T.prim_func
        def local_split(
            Q: T.Tensor((1, 1, sq, D), dtype),
            K: T.Tensor((pool_rows, 1, D), dtype),
            V: T.Tensor((pool_rows, 1, D), dtype),
            Table: T.Tensor((1, width), "int32"),
            Counts: T.Tensor((grid,), "int32"),
            Tasks: T.Tensor((grid, max_segments, 4), "int32"),
            Parts: T.Tensor((slots, BR, D), accum_dtype),
            LSE: T.Tensor((stats_size,), accum_dtype),
            O: T.Tensor((1, sq, D), dtype),
        ):
            local_body(Q, K, V, Table, Counts, Tasks, Parts, LSE, O)

        local_entry = local_split
    else:

        @T.prim_func
        def local_direct(
            Q: T.Tensor((1, 1, sq, D), dtype),
            K: T.Tensor((pool_rows, 1, D), dtype),
            V: T.Tensor((pool_rows, 1, D), dtype),
            Table: T.Tensor((1, width), "int32"),
            O: T.Tensor((1, sq, D), dtype),
        ):
            # Aliases are placeholders only: compile-time has_merge=False removes
            # every Counts/Tasks/Parts/LSE reference in the shared macro.
            local_body(Q, K, V, Table, Table, Table, O, O, O)

        local_entry = local_direct

    local_kernel = tilelang.compile(
        local_entry,
        out_idx=[6, 7, 8] if has_merge else [4],
        target="ascend",
        execution_backend="auto",
        pass_configs={},
    )
    if has_merge:
        merge_entry = _make_merge(sq, slots, merge_count, max_parts, merge_rows)
    merge_kernel = (
        tilelang.compile(
            merge_entry, out_idx=[], target="ascend", execution_backend="auto", pass_configs={}
        )
        if has_merge
        else None
    )
    uploaded = {}

    def prepare_plan(device):
        if not has_merge:
            return None, None, None
        if device not in uploaded:
            counts = [len(p) for p in plan.programs]
            tasks = [list(p) + [(-1, -1, -1, -1)] * (max_segments - len(p)) for p in plan.programs]
            merges = list(plan.merges)
            uploaded[device] = tuple(
                torch.tensor(x, dtype=torch.int32, device=device) for x in (counts, tasks, merges)
            )
        return uploaded[device]

    def validate_inputs(q, k, v, table, lengths):
        if (
            tuple(q.shape) != (1, sq, 1, D)
            or tuple(k.shape) != (pool_rows, 1, D)
            or v.shape != k.shape
        ):
            raise ValueError("Q or K/V shape does not match the compiled kernel")
        if tuple(table.shape) != (1, width) or tuple(lengths.shape) != (1,):
            raise ValueError("page table or lengths shape does not match the compiled kernel")
        if not (q.dtype == k.dtype == v.dtype == torch.bfloat16):
            raise TypeError("Q, K and V must use BF16")
        if not (table.dtype == lengths.dtype == torch.int32):
            raise TypeError("page table and lengths must use int32")
        if q.device.type != "npu" or any(
            x.device != q.device or not x.is_contiguous() for x in (q, k, v, table, lengths)
        ):
            raise ValueError("inputs must be contiguous tensors on the same NPU")
        if lengths.item() != kv:
            raise ValueError("the live KV length must match the compiled KV length")
        if table.min().item() < 0 or table.max().item() >= pool_rows // page_size:
            raise ValueError("page table contains an out-of-range physical page ID")

    def run_local(prepared, k, v, table, counts=None, tasks=None):
        if has_merge:
            return local_kernel(prepared, k, v, table, counts, tasks)
        return None, None, local_kernel(prepared, k, v, table)

    def compute(q, k, v, table, lengths):
        counts, tasks, merges = prepare_plan(q.device)
        prepared = q.permute(0, 2, 1, 3).contiguous()
        return run_local(prepared, k, v, table, counts, tasks)

    def finish(parts, lse, output):
        if merge_kernel is not None:
            merges = prepare_plan(output.device)[2]
            merge_kernel(parts, lse, merges, output)
        return output.view(1, sq, 1, D)

    def launch(q, k, v, table, lengths):
        return finish(*compute(q, k, v, table, lengths))

    launch.merge_assignment = dict(
        policy="rawsimd",
        tasks=plan.merges,
        programs=merge_count * (BR // merge_rows),
        max_rows=merge_rows,
    )
    launch.plan = plan
    launch.prepare_plan = prepare_plan
    launch.validate_inputs = validate_inputs
    launch.run_local = run_local
    launch.local_kernel = local_kernel
    launch.merge_kernel = merge_kernel
    launch.compute = compute
    launch.finish = finish
    return launch


class AscendMHAPagedDecodeKernel(Kernel, MHAPagedDecodeFwdInterface):
    """Paged attention with a 28-Cube BF16 schedule and vector split-KV decode."""

    devices = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: AttentionCall) -> str | None:
        if call.heads != call.heads_kv or call.dim not in (64, 128):
            return "requires equal query and KV heads and dimension 64 or 128"
        if call.dtype not in (torch.float16, torch.bfloat16) or call.cache_dtype != call.dtype:
            return "requires matching FP16 or BF16 query and cache"
        if call.page_size <= 0 or call.page_size % 16 or call.seqlen_kv % call.page_size:
            return "requires whole physical pages of a multiple of 16 tokens"
        return super().refusal(call)

    def __init__(self, call: AttentionCall) -> None:
        reason = self.refusal(call)
        if reason:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype

    def forward(
        self,
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        real_seqlen_kv: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        self._require_devices(
            q=q,
            k_pool=k_pool,
            v_pool=v_pool,
            real_seqlen_kv=real_seqlen_kv,
            block_table=block_table,
        )
        if q.numel() == 0 or self.call.max_pages_per_req == 0 or k_pool.shape[0] == 0:
            return torch.zeros_like(q)
        original_geometry = (
            q.dtype == torch.bfloat16
            and self.call.batch == self.call.heads == 1
            and self.call.dim == self.call.page_size == 128
            and not self.call.is_causal
            and torch.npu.get_device_properties(q.device.index).cube_core_num == 28
        )
        # Only the specialized Cube plan needs a page-aligned host length.
        kv = int(real_seqlen_kv.item()) if original_geometry else None
        if kv is None or kv % 128 or kv > k_pool.shape[0]:
            part, merge = paged_programs(
                self.call, int(get_num_vector_cores(device_id=q.device.index))
            )
            return merge(
                part(
                    q.contiguous(),
                    k_pool.contiguous(),
                    v_pool.contiguous(),
                    real_seqlen_kv.contiguous(),
                    block_table.contiguous(),
                )
            )
        if real_seqlen_kv.device != q.device:
            raise ValueError("cache length must be on the query device")
        if real_seqlen_kv.shape != (1,) or real_seqlen_kv.dtype != torch.int32:
            raise ValueError("cache length must be one int32 value")
        if kv == 0:
            return torch.zeros_like(q)
        if kv < 0 or kv % 128 or kv // 128 > block_table.shape[1]:
            raise ValueError("live KV length must be page-aligned and fit the page table")
        pages = block_table[:, : kv // 128]
        program = build_kernel(q.shape[1], kv, k_pool.shape[0])
        program.validate_inputs(q, k_pool, v_pool, pages, real_seqlen_kv)
        return program(q, k_pool, v_pool, pages, real_seqlen_kv)


register_kernel_type("MHAPagedFwdOp", "ascend", AscendMHAPagedDecodeKernel)
