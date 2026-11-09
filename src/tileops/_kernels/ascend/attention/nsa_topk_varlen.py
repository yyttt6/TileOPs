"""Ascend Native Sparse Attention (NSA) compressed-block selection."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.ascend.attention.nsa_common import (
    calibrated_cube_blocks,
    check_inputs,
    vector_aligned,
)
from tileops._kernels.attention.call_spec import NSACall, NSATopKFwdInterface
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type

__all__ = ["AscendNSATopKVarlenKernel"]

# The five manifest call configurations this implementation serves, in both dtypes.
# ``refusal`` reads the same table.
_CONFIGURATIONS = (
    (8, 8192, 32, 2, 128, 256, 16, 32, 1.0),
    (4, 4096, 32, 2, 128, 128, 16, 32, 1.0),
    (3, 5153, 32, 2, 128, 163, 16, 32, 0.08838834764831845),
    (1, 16384, 32, 2, 128, 256, 32, 64, 0.08838834764831845),
    (4, 8192, 64, 2, 64, 256, 8, 32, 0.125),
)


def _make_nsa_topk_varlen_kernel(
    batch: int,
    tokens: int,
    heads: int,
    heads_kv: int,
    dim: int,
    chunk_num: int,
    selected_blocks: int,
    block_size: int,
    scale: float,
    dtype: str,
    num_blocks: int,
):
    """Pair queries within a request; each AIV ranks one complete importance row."""
    assert (
        batch,
        tokens,
        heads,
        heads_kv,
        dim,
        chunk_num,
        selected_blocks,
        block_size,
        scale,
    ) in _CONFIGURATIONS, "Ascend NSA TopK supports the five declared workload configurations"
    assert dtype in ("float16", "bfloat16")
    assert num_blocks > 0
    group = heads // heads_kv
    chunk_pad = max(64, 1 << (chunk_num - 1).bit_length())
    # Compact request-local pairs keep both vectors busy without assigning one
    # permanently inactive parity of token tasks to half of the physical grid.
    # The bound adds at most batch // 2 empty tasks when all lengths are even.
    pair_bound = (tokens + batch) // 2
    tasks = pair_bound * heads_kv
    grid = min(num_blocks, tasks)
    steps = (tasks + grid - 1) // grid

    def make_compute(width):
        @T.macro
        def compute(
            q,
            k_cmp,
            block_indices,
            sid,
            token,
            pair,
            end,
            position,
            kv_head,
            chunk_begin,
            request_chunks,
            reload_keys,
            q_l1,
            k_l1,
            q_l0,
            k_l0,
            scores_l0c,
            queries,
            scores,
            row_maximum,
            importance,
            selected,
        ):
            # Both rows have the same candidate window: block_size is even,
            # and each pair starts at an even request-local position.
            current = position // block_size
            candidates = T.min(current + 1, request_chunks)
            closed = T.min((position + sid + 1) // block_size, request_chunks)
            # Read exactly one request-local query through DMA. Each AIV owns
            # one ND query tile, including the zeroed odd-request tail.
            if token + sid < end:
                T.copy(
                    q[token + sid, kv_head * group : (kv_head + 1) * group, :],
                    queries,
                )
                if scale != 1.0:
                    with T.SimdVF():
                        full = T.simd.pset(32, "PAT_ALL")
                        for g in T.serial(group):
                            for part in T.serial(dim // 64):
                                raw = T.simd.vld(queries[g, part * 64], dist="UNPK_B16")
                                wide = T.simd.vcvt(raw, "float32", full, part=0)
                                scaled = T.simd.vmuls(wide, T.cast(scale, "float32"), full)
                                converted = T.simd.vcvt(
                                    scaled, dtype, full, round="ROUND_R", sat=False, part=0
                                )
                                T.simd.vsts(
                                    queries[g, part * 64],
                                    converted,
                                    full,
                                    dist="PK_B32",
                                    extent=64,
                                )
            else:
                with T.SimdVF():
                    zero = T.reinterpret(T.simd.vdup(T.uint32(0), "uint32"), dtype + "x128")
                    mask = T.simd.pset(16, "PAT_ALL" if dim == 128 else "PAT_VL64")
                    for g in T.serial(group):
                        T.simd.vsts(queries[g, 0], zero, mask, dist="NORM_B16", extent=dim)
            with T.SimdVF():
                full = T.simd.pset(32, "PAT_ALL")
                lanes = T.simd.vci(0, "int32")
                selected_mask = T.simd.vcmps(lanes, selected_blocks, full, op="lt")
                missing = T.simd.vdup(T.int32(-1), "int32", full)
                T.simd.vsts(selected[0], missing, selected_mask, extent=selected_blocks)
            T.dual_copy(queries[:group, :], q_l1)
            key_tile = T.alloc_shared((width // 2, dim), dtype)
            T.annotate_buffer_versions({key_tile: 1})
            if reload_keys:
                load_start = T.cast(chunk_begin + sid * (width // 2), "int32")
                local_valid = T.max(
                    0,
                    T.min(
                        T.min(width // 2, request_chunks - sid * (width // 2)),
                        chunk_num - load_start,
                    ),
                )
                if local_valid < width // 2:
                    with T.SimdVF():
                        zero = T.reinterpret(T.simd.vdup(T.uint32(0), "uint32"), dtype + "x128")
                        mask = T.simd.pset(16, "PAT_ALL" if dim == 128 else "PAT_VL64")
                        for row in T.serial(local_valid, width // 2):
                            T.simd.vsts(key_tile[row, 0], zero, mask, dist="NORM_B16", extent=dim)
                if local_valid > 0:
                    T.copy(
                        k_cmp[load_start : load_start + local_valid, kv_head, :],
                        key_tile[:local_valid, :],
                    )
                T.dual_copy(key_tile[: width // 2, :], k_l1[:width, :])
            T.copy(q_l1, q_l0)
            T.copy(k_l1[:width, :], k_l0[:width, :])
            T.gemm(
                q_l0,
                k_l0[:width, :],
                scores_l0c[:, :width],
                transpose_B=True,
                clear_accum=True,
            )
            T.dual_copy(scores_l0c[:, :width], scores[:, :width])
            # Keep the masked scores and softmax reductions in vector registers.
            # Only the per-head LSE crosses UB between the two SIMD phases.
            with T.SimdVF():
                full = T.simd.pset(32, "PAT_ALL")
                one = T.simd.pset(32, "PAT_VL1")
                negative_infinity = T.simd.vdup(-T.infinity("float32"), "float32", full)
                ids0 = T.simd.vci(0, "int32")
                closed0 = T.simd.vcmps(ids0, T.cast(closed, "int32"), full, op="lt")
                if width >= 128:
                    ids1 = T.simd.vci(64, "int32")
                    closed1 = T.simd.vcmps(ids1, T.cast(closed, "int32"), full, op="lt")
                if width >= 256:
                    ids2 = T.simd.vci(128, "int32")
                    ids3 = T.simd.vci(192, "int32")
                    closed2 = T.simd.vcmps(ids2, T.cast(closed, "int32"), full, op="lt")
                    closed3 = T.simd.vcmps(ids3, T.cast(closed, "int32"), full, op="lt")
                for g in T.serial(group):
                    score0 = T.simd.vsel(T.simd.vld(scores[g, 0]), negative_infinity, closed0)
                    maximum = T.simd.alloc_var("float32")
                    maximum = T.simd.vcmax(score0, full)
                    if width >= 128:
                        score1 = T.simd.vsel(T.simd.vld(scores[g, 64]), negative_infinity, closed1)
                        maximum = T.simd.vmax(maximum, T.simd.vcmax(score1, full), full)
                    if width >= 256:
                        score2 = T.simd.vsel(T.simd.vld(scores[g, 128]), negative_infinity, closed2)
                        score3 = T.simd.vsel(T.simd.vld(scores[g, 192]), negative_infinity, closed3)
                        maximum = T.simd.vmax(maximum, T.simd.vcmax(score2, full), full)
                        maximum = T.simd.vmax(maximum, T.simd.vcmax(score3, full), full)
                    row_max = T.simd.vdupv(maximum, full)
                    total = T.simd.alloc_var("float32")
                    total = T.simd.vcadd(T.simd.vexpdif(score0, row_max, full), full)
                    if width >= 128:
                        total = T.simd.vadd(
                            total, T.simd.vcadd(T.simd.vexpdif(score1, row_max, full), full), full
                        )
                    if width >= 256:
                        total = T.simd.vadd(
                            total, T.simd.vcadd(T.simd.vexpdif(score2, row_max, full), full), full
                        )
                        total = T.simd.vadd(
                            total, T.simd.vcadd(T.simd.vexpdif(score3, row_max, full), full), full
                        )
                    lse = T.simd.vadd(row_max, T.simd.vln(T.simd.vdupv(total, full), full), full)
                    T.simd.vsts(row_maximum[g], lse, one, dist="ONEPT_B32", extent=1)
            with T.SimdVF():
                full = T.simd.pset(32, "PAT_ALL")
                negative_infinity = T.simd.vdup(-T.infinity("float32"), "float32", full)
                priority_score = T.simd.vdup(T.cast(group, "float32"), "float32", full)
                zero = T.simd.vdup(T.float32(0.0), "float32", full)
                for part in T.serial(width // 64):
                    ids = T.simd.vci(part * 64, "int32")
                    visible = T.simd.vcmps(ids, T.cast(candidates, "int32"), full, op="lt")
                    previous = T.simd.vcmps(ids, T.cast(current, "int32"), visible, op="lt")
                    value = T.simd.alloc_var("float32")
                    value = T.simd.vdup(T.float32(0.0), "float32", full)
                    for g in T.serial(group):
                        lse = T.simd.vld(row_maximum[g], dist="BRC_B32")
                        score = T.simd.vld(scores[g, part * 64])
                        probability = T.simd.vexpdif(score, lse, full)
                        value = T.simd.vadd(value, T.simd.vsel(probability, zero, previous), full)
                    first = T.simd.vcmps(ids, T.int32(0), full, op="eq")
                    preceding = T.simd.vcmps(ids, T.cast(current - 1, "int32"), full, op="eq")
                    present = T.simd.vcmps(ids, T.cast(current, "int32"), full, op="eq")
                    value = T.simd.vsel(priority_score, value, first)
                    value = T.simd.vsel(priority_score, value, preceding)
                    value = T.simd.vsel(priority_score, value, present)
                    T.simd.vsts(
                        importance[0, part * 64],
                        T.simd.vsel(value, negative_infinity, visible),
                        full,
                    )

            # Rank the whole candidate window in one VF. Its scores stay in
            # registers across all selections, including stable tied-ID removal.
            with T.SimdVF():
                mask = T.simd.pset(32, "PAT_ALL")
                one = T.simd.pset(32, "PAT_VL1")
                importance0 = T.simd.alloc_var("float32")
                importance0 = T.simd.vld(importance[0, 0])
                if width >= 128:
                    importance1 = T.simd.alloc_var("float32")
                    importance1 = T.simd.vld(importance[0, 64])
                if width >= 256:
                    importance2 = T.simd.alloc_var("float32")
                    importance3 = T.simd.alloc_var("float32")
                    importance2 = T.simd.vld(importance[0, 128])
                    importance3 = T.simd.vld(importance[0, 192])
                sentinel = T.simd.vdup(T.int32(width), "int32", mask)
                negative_infinity = T.simd.vdup(-T.infinity("float32"), "float32", mask)
                for rank in T.serial(T.min(selected_blocks, candidates)):
                    top = T.simd.vcmax(importance0, mask)
                    if width >= 128:
                        top = T.simd.vmax(top, T.simd.vcmax(importance1, mask), mask)
                    if width >= 256:
                        top = T.simd.vmax(top, T.simd.vcmax(importance2, mask), mask)
                        top = T.simd.vmax(top, T.simd.vcmax(importance3, mask), mask)
                    top = T.simd.vdupv(top, mask)
                    ids = T.simd.vci(T.int32(0), "int32")
                    tied = T.simd.vcmp(importance0, top, mask, op="eq")
                    lowest = T.simd.vcmin(T.simd.vsel(ids, sentinel, tied), mask)
                    if width >= 128:
                        ids = T.simd.vci(T.int32(64), "int32")
                        tied = T.simd.vcmp(importance1, top, mask, op="eq")
                        lowest = T.simd.vmin(
                            lowest, T.simd.vcmin(T.simd.vsel(ids, sentinel, tied), mask), mask
                        )
                    if width >= 256:
                        ids = T.simd.vci(T.int32(128), "int32")
                        tied = T.simd.vcmp(importance2, top, mask, op="eq")
                        lowest = T.simd.vmin(
                            lowest, T.simd.vcmin(T.simd.vsel(ids, sentinel, tied), mask), mask
                        )
                        ids = T.simd.vci(T.int32(192), "int32")
                        tied = T.simd.vcmp(importance3, top, mask, op="eq")
                        lowest = T.simd.vmin(
                            lowest, T.simd.vcmin(T.simd.vsel(ids, sentinel, tied), mask), mask
                        )
                    T.simd.vsts(selected[rank], lowest, one, dist="ONEPT_B32", extent=1)
                    lowest = T.simd.vdupv(lowest, mask)
                    ids = T.simd.vci(0, "int32")
                    removed = T.simd.vcmp(ids, lowest, mask, op="eq")
                    importance0 = T.simd.vsel(negative_infinity, importance0, removed)
                    if width >= 128:
                        ids = T.simd.vci(64, "int32")
                        removed = T.simd.vcmp(ids, lowest, mask, op="eq")
                        importance1 = T.simd.vsel(negative_infinity, importance1, removed)
                    if width >= 256:
                        ids = T.simd.vci(128, "int32")
                        removed = T.simd.vcmp(ids, lowest, mask, op="eq")
                        importance2 = T.simd.vsel(negative_infinity, importance2, removed)
                        ids = T.simd.vci(192, "int32")
                        removed = T.simd.vcmp(ids, lowest, mask, op="eq")
                        importance3 = T.simd.vsel(negative_infinity, importance3, removed)
            # Pair compaction gives token = 2 * pair - odd_prefix, with
            # 0 <= odd_prefix <= batch - 1. Keep that explicit bound at the
            # GM access so scheduling can prove successive tasks disjoint.
            with T.SimtVF(threads=128):
                for rank in T.Parallel(selected_blocks):
                    if token + sid < end:
                        block_indices[
                            2 * pair - T.min(batch - 1, T.max(2 * pair - token, 0)) + sid,
                            kv_head,
                            rank,
                        ] = selected[rank]

        return compute

    compute64 = make_compute(64)
    compute128 = make_compute(128)
    compute256 = make_compute(256)

    @T.prim_func
    def main(
        q: T.Tensor((tokens, heads, dim), dtype),
        k_cmp: T.Tensor((chunk_num, heads_kv, dim), dtype),
        offsets: T.Tensor((batch + 1,), "int32"),
        chunk_offsets: T.Tensor((batch + 1,), "int32"),
        token_indices: T.Tensor((tokens, 2), "int32"),
        block_indices: T.Tensor((tokens, heads_kv, selected_blocks), "int32"),
    ):
        with T.MixedKernel(grid, sids=2) as (bx, sid):
            q_l1 = T.alloc_l1((2 * group, dim), dtype)
            k_l1 = T.alloc_l1((chunk_pad, dim), dtype)
            q_l0 = T.alloc_l0a((2 * group, dim), dtype)
            k_l0 = T.alloc_l0b((chunk_pad, dim), dtype)
            scores_l0c = T.alloc_l0c((2 * group, chunk_pad), "float32")
            # Each AIV stages one query in ND; dual_copy packs it into NZ.
            # The shared K tile is split by columns across the two AIVs.
            queries = T.alloc_shared((group, dim), dtype)
            scores = T.alloc_shared((group, chunk_pad), "float32")
            row_maximum = T.alloc_shared((group,), "float32")
            importance = T.alloc_shared((1, chunk_pad), "float32")
            selected = T.alloc_shared((selected_blocks,), "int32")
            T.annotate_buffer_versions(
                {
                    q_l1: 1,
                    k_l1: 1,
                    q_l0: 1,
                    k_l0: 1,
                    scores_l0c: 1,
                    queries: 1,
                    scores: 1,
                    row_maximum: 1,
                    importance: 1,
                    selected: 1,
                }
            )
            cached_request = T.alloc_var("int32", init=-1)
            cached_head = T.alloc_var("int32", init=-1)
            cached_width = T.alloc_var("int32", init=0)
            for step in T.serial(steps):
                task = step * grid + bx
                if task < tasks:
                    pair = task // heads_kv
                    kv_head = task % heads_kv
                    request = T.alloc_var("int32", init=0)
                    position = T.alloc_var("int32", init=0)
                    begin = T.alloc_var("int32", init=0)
                    end = T.alloc_var("int32", init=0)
                    prefix = T.alloc_var("int32", init=0)
                    active = T.alloc_var("bool", init=False)
                    # The number of odd request lengths is runtime metadata.
                    # Locate its compact pair prefix on device rather than
                    # assuming global even tokens begin request-local pairs.
                    for r in T.serial(batch):
                        length = offsets[r + 1] - offsets[r]
                        pair_end = prefix + T.ceildiv(length, 2)
                        if pair >= prefix and pair < pair_end:
                            request = r
                            position = 2 * (pair - prefix)
                            begin = offsets[r]
                            end = offsets[r + 1]
                            active = True
                        prefix = pair_end
                    if active:
                        token = begin + position
                        # Retain the public metadata ABI and use its canonical
                        # query identity after locating the compact pair.
                        request = token_indices[token, 0]
                        position = token_indices[token, 1]
                        chunk_begin = chunk_offsets[request]
                        request_chunks = T.min(
                            chunk_offsets[request + 1] - chunk_begin,
                            T.ceildiv(end - begin, block_size),
                        )
                        candidates = T.min(position // block_size + 1, request_chunks)
                        if candidates <= 3:
                            # All visible blocks are mandatory priority blocks.
                            # Their importance is exactly group, independent of QK.
                            with T.SimtVF(threads=128):
                                for rank in T.Parallel(selected_blocks):
                                    if token + sid < end:
                                        block_indices[
                                            2 * pair
                                            - T.min(batch - 1, T.max(2 * pair - token, 0))
                                            + sid,
                                            kv_head,
                                            rank,
                                        ] = T.if_then_else(rank < candidates, rank, -1)
                        else:
                            width = T.if_then_else(
                                candidates <= 64, 64, T.if_then_else(candidates <= 128, 128, 256)
                            )
                            reload_keys = (
                                cached_request != request
                                or cached_head != kv_head
                                or cached_width != width
                            )
                            if candidates <= 64:
                                compute64(
                                    q,
                                    k_cmp,
                                    block_indices,
                                    sid,
                                    token,
                                    pair,
                                    end,
                                    position,
                                    kv_head,
                                    chunk_begin,
                                    request_chunks,
                                    reload_keys,
                                    q_l1,
                                    k_l1,
                                    q_l0,
                                    k_l0,
                                    scores_l0c,
                                    queries,
                                    scores,
                                    row_maximum,
                                    importance,
                                    selected,
                                )
                            elif candidates <= 128:
                                if chunk_pad >= 128:
                                    compute128(
                                        q,
                                        k_cmp,
                                        block_indices,
                                        sid,
                                        token,
                                        pair,
                                        end,
                                        position,
                                        kv_head,
                                        chunk_begin,
                                        request_chunks,
                                        reload_keys,
                                        q_l1,
                                        k_l1,
                                        q_l0,
                                        k_l0,
                                        scores_l0c,
                                        queries,
                                        scores,
                                        row_maximum,
                                        importance,
                                        selected,
                                    )
                            else:
                                if chunk_pad >= 256:
                                    compute256(
                                        q,
                                        k_cmp,
                                        block_indices,
                                        sid,
                                        token,
                                        pair,
                                        end,
                                        position,
                                        kv_head,
                                        chunk_begin,
                                        request_chunks,
                                        reload_keys,
                                        q_l1,
                                        k_l1,
                                        q_l0,
                                        k_l0,
                                        scores_l0c,
                                        queries,
                                        scores,
                                        row_maximum,
                                        importance,
                                        selected,
                                    )
                            cached_request = request
                            cached_head = kv_head
                            cached_width = width

    return main


@functools.lru_cache(maxsize=64)
def _compile_nsa_topk_varlen_kernel(
    batch: int,
    tokens: int,
    heads: int,
    heads_kv: int,
    dim: int,
    chunk_num: int,
    selected_blocks: int,
    block_size: int,
    scale: float,
    dtype: str,
    num_blocks: int,
):
    return tilelang.compile(
        _make_nsa_topk_varlen_kernel(
            batch,
            tokens,
            heads,
            heads_kv,
            dim,
            chunk_num,
            selected_blocks,
            block_size,
            scale,
            dtype,
            num_blocks,
        ),
        target="ascend",
        out_idx=-1,
    )


class AscendNSATopKVarlenKernel(Kernel, NSATopKFwdInterface):
    """NSA TopK for the five manifest configurations in FP16 and BF16."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: NSACall) -> str | None:
        configuration = (
            call.batch,
            call.c_seq_len,
            call.heads,
            call.heads_kv,
            call.dim,
            call.chunk_num,
            call.selected_blocks,
            call.block_size,
            call.scale,
        )
        if configuration not in _CONFIGURATIONS or call.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            return "requires one of the ten declared NSATopKVarlenFwdOp manifest calls"
        return super().refusal(call)

    def __init__(self, call: NSACall) -> None:
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self._compiled = _compile_nsa_topk_varlen_kernel(
            call.batch,
            call.c_seq_len,
            call.heads,
            call.heads_kv,
            call.dim,
            call.chunk_num,
            call.selected_blocks,
            call.block_size,
            call.scale,
            self.dtype_str,
            calibrated_cube_blocks(),
        )

    def forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        expected = (
            ("q", q, (self.call.c_seq_len, self.call.heads, self.call.dim), self.dtype),
            ("k_cmp", k_cmp, (self.call.chunk_num, self.call.heads_kv, self.call.dim), self.dtype),
            ("offsets", offsets, (self.call.batch + 1,), torch.int32),
            ("chunk_offsets", chunk_offsets, (self.call.batch + 1,), torch.int32),
            ("token_indices", token_indices, (self.call.c_seq_len, 2), torch.int32),
        )
        check_inputs(self, expected)
        normalized = tuple(tensor.contiguous() for _, tensor, _shape, _dtype in expected)
        return self._compiled(
            *(vector_aligned(tensor) if i < 2 else tensor for i, tensor in enumerate(normalized))
        )


register_kernel_type("NSATopKVarlenFwdOp", "ascend", AscendNSATopKVarlenKernel)
