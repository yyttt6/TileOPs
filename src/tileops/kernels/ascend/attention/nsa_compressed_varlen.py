"""Ascend attention over the closed compressed chunks of each packed request."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.layout import make_ascend_compact_nz_layout

from tileops.backend import register_kernel_type
from tileops.kernels.ascend.attention.nsa_common import (
    calibrated_cube_blocks,
    check_inputs,
    normalize_cast_rows,
)
from tileops.kernels.attention.call_spec import NSACall, NSACompressedFwdInterface
from tileops.kernels.kernel_base import Kernel, vector_aligned

__all__ = ["AscendNSACompressedFwdVarlenKernel"]

# The three compressed manifest call configurations, in the dtypes they run in.
# ``refusal`` reads the same table.
_CONFIGURATIONS = (
    (8, 8192, 256, 32, 2, 128, 128, 32, 0.08838834764831845, torch.float16),
    (4, 4096, 128, 32, 2, 128, 128, 32, 0.08838834764831845, torch.float16),
    (2, 1024, 33, 32, 2, 128, 128, 32, 0.08838834764831845, torch.bfloat16),
)


def _make_nsa_compressed_varlen_kernel(
    batch: int,
    tokens: int,
    heads: int,
    heads_kv: int,
    dim_k: int,
    dim_v: int,
    chunk_num: int,
    block_size: int,
    scale: float,
    dtype: str,
    num_blocks: int,
):
    """Build one persistent token/head-tile program, retaining full GM addresses.

    The runtime closed-chunk count can exceed one tile even at a manifest shape:
    request boundaries are inputs, not constants of a compiled specialization.
    """
    assert batch > 0 and tokens > 0 and chunk_num > 0
    assert heads_kv > 0 and heads % heads_kv == 0
    assert heads // heads_kv % 16 == 0
    assert dim_k in (64, 128) and dim_v in (64, 128)
    assert block_size > 0 and num_blocks > 0
    assert dtype in ("float16", "bfloat16")
    head_tile = 16
    vector_rows = 8
    key_tile = 32
    softmax_tile = 64
    head_groups = heads // heads_kv // head_tile
    token_tasks = heads_kv * head_groups
    tasks = tokens * token_tasks
    grid = min(num_blocks, tasks)
    steps = (tasks + grid - 1) // grid
    normalize_rows = normalize_cast_rows(vector_rows, dim_v, dtype)

    @T.macro
    def softmax_simd(scores, maximum, alpha, block_sum, probabilities, valid, first):
        with T.SimdVF():
            full = T.simd.pset(32, "PAT_ALL")
            one = T.simd.pset(32, "PAT_VL1")
            lanes = T.simd.vci(0, "int32")
            valid_mask = T.simd.vcmps(lanes, T.cast(valid, "int32"), full, op="lt")
            negative_infinity = T.simd.vdup(-T.infinity("float32"), "float32", full)
            for row in T.serial(vector_rows):
                score = T.simd.vld(scores[row, 0])
                scaled = T.simd.vmuls(score, scale, full)
                masked = T.simd.vsel(scaled, negative_infinity, valid_mask)
                block_max = T.simd.vdupv(T.simd.vcmax(masked, full), full)
                row_max = T.simd.alloc_var("float32")
                row_max = block_max
                if not first:
                    previous_max = T.simd.vld(maximum[row], dist="BRC_B32")
                    row_max = T.simd.vmax(previous_max, block_max, full)
                    factor = T.simd.vexpdif(previous_max, row_max, full)
                    T.simd.vsts(alpha[row], factor, one, dist="ONEPT_B32", extent=1)
                T.simd.vsts(maximum[row], row_max, one, dist="ONEPT_B32", extent=1)
                probability = T.simd.vexpdif(masked, row_max, full)
                total = T.simd.vcadd(probability, full)
                T.simd.vsts(block_sum[row], total, one, dist="ONEPT_B32", extent=1)
                converted = T.simd.vcvt(
                    probability, dtype, full, round="ROUND_R", sat=False, part=0
                )
                store_mask = T.simd.pset(32, "PAT_VL16")
                T.simd.vsts(
                    probabilities[row, 0],
                    converted,
                    store_mask,
                    dist="PK_B32",
                    extent=16,
                )
                second_indices = T.simd.vci(16, "int32")
                second_indices = T.simd.vand(
                    T.reinterpret(second_indices, "uint32x64"),
                    T.simd.vdup(T.uint32(63), "uint32", full),
                    full,
                )
                second = T.simd.vselr(probability, second_indices)
                converted_second = T.simd.vcvt(
                    second, dtype, full, round="ROUND_R", sat=False, part=0
                )
                T.simd.vsts(
                    probabilities[row, 16],
                    converted_second,
                    store_mask,
                    dist="PK_B32",
                    extent=16,
                )

    @T.prim_func
    def main(
        q: T.Tensor((tokens, heads, dim_k), dtype),
        k_cmp: T.Tensor((chunk_num, heads_kv, dim_k), dtype),
        v_cmp: T.Tensor((chunk_num, heads_kv, dim_v), dtype),
        offsets: T.Tensor((batch + 1,), "int32"),
        chunk_offsets: T.Tensor((batch + 1,), "int32"),
        token_indices: T.Tensor((tokens, 2), "int32"),
        output: T.Tensor((tokens, heads, dim_v), dtype),
        lse: T.Tensor((tokens, heads), dtype),
    ):
        with T.MixedKernel(grid, sids=2) as (bx, sid):
            q_l1 = T.alloc_l1((head_tile, dim_k), dtype)
            k_l1 = T.alloc_l1((key_tile, dim_k), dtype)
            v_l1 = T.alloc_l1((key_tile, dim_v), dtype)
            p_l1 = T.alloc_l1((head_tile, key_tile), dtype)
            q_l0 = T.alloc_l0a((head_tile, dim_k), dtype)
            k_l0 = T.alloc_l0b((key_tile, dim_k), dtype)
            p_l0 = T.alloc_l0a((head_tile, key_tile), dtype)
            v_l0 = T.alloc_l0b((dim_v, key_tile), dtype)
            scores_l0c = T.alloc_l0c((head_tile, key_tile), "float32")
            value_l0c = T.alloc_l0c((head_tile, dim_v), "float32")
            keys = T.alloc_shared((key_tile // 2, dim_k), dtype)
            values = T.alloc_shared((key_tile // 2, dim_v), dtype)
            scores = T.alloc_shared((vector_rows, softmax_tile), "float32")
            probabilities = T.alloc_shared((vector_rows + 1, key_tile), dtype)
            value_tile = T.alloc_shared((vector_rows, dim_v), "float32")
            result = T.alloc_shared((vector_rows, dim_v), dtype)
            block_sum = T.alloc_shared((vector_rows,), "float32")
            maximum = T.alloc_shared((vector_rows,), "float32")
            denominator = T.alloc_shared((vector_rows,), "float32")
            numerator = T.alloc_shared((vector_rows, dim_v), "float32")
            alpha = T.alloc_shared((vector_rows,), "float32")
            T.annotate_layout(
                {
                    probabilities: make_ascend_compact_nz_layout(probabilities),
                }
            )
            # A prepared QK tile and the preceding tile's PV operands coexist.
            # Keep the online softmax state outside that two-slot lifetime.
            T.annotate_buffer_versions(
                {
                    q_l1: 1,
                    k_l1: 2,
                    v_l1: 2,
                    p_l1: 2,
                    q_l0: 1,
                    k_l0: 2,
                    p_l0: 2,
                    v_l0: 2,
                    scores_l0c: 2,
                    value_l0c: 2,
                    keys: 2,
                    values: 2,
                    scores: 2,
                    probabilities: 2,
                    value_tile: 2,
                    result: 1,
                    block_sum: 2,
                    maximum: 1,
                    denominator: 1,
                    numerator: 1,
                    alpha: 2,
                }
            )

            for step in T.serial(steps):
                task = step * grid + bx
                if task < tasks:
                    token_row = task // token_tasks
                    kv_head = task % token_tasks // head_groups
                    head_group = task % head_groups
                    head_start = kv_head * (head_groups * head_tile) + head_group * head_tile
                    request = token_indices[token_row, 0]
                    position = token_indices[token_row, 1]
                    token = offsets[request] + position
                    chunk_begin = chunk_offsets[request]
                    closed = (position + 1) // block_size

                    if closed > 0:
                        T.copy(q[token, head_start : head_start + head_tile, :], q_l1)
                        T.copy(q_l1, q_l0)
                        for tile in T.Pipelined(
                            T.ceildiv(closed, key_tile),
                            num_stages=2,
                            annotations={"enable_offset": True},
                        ):
                            start = tile * key_tile
                            valid = T.min(key_tile, closed - start)
                            # Offset the consumers by one iteration: QK for tile
                            # i+1 can run while the AIVs normalize tile i for PV.
                            with T.Stage(0):
                                # Both AIVs read contiguous rows. The tail is zeroed before
                                # DMA so future summaries, including NaNs, are never read.
                                load_start = T.cast(
                                    chunk_begin + start + sid * (key_tile // 2), "int32"
                                )
                                local_valid = T.max(
                                    0, T.min(key_tile // 2, valid - sid * (key_tile // 2))
                                )
                                if local_valid < key_tile // 2:
                                    with T.SimdVF():
                                        zero_bits = T.simd.vdup(T.uint32(0), "uint32")
                                        zero = T.reinterpret(zero_bits, dtype + "x128")
                                        key_mask = T.simd.pset(
                                            16, "PAT_ALL" if dim_k == 128 else "PAT_VL64"
                                        )
                                        value_mask = T.simd.pset(
                                            16, "PAT_ALL" if dim_v == 128 else "PAT_VL64"
                                        )
                                        for row in T.serial(local_valid, key_tile // 2):
                                            T.simd.vsts(
                                                keys[row, 0],
                                                zero,
                                                key_mask,
                                                dist="NORM_B16",
                                                extent=dim_k,
                                            )
                                            T.simd.vsts(
                                                values[row, 0],
                                                zero,
                                                value_mask,
                                                dist="NORM_B16",
                                                extent=dim_v,
                                            )
                                if local_valid > 0:
                                    T.copy(
                                        k_cmp[
                                            load_start : load_start + local_valid,
                                            kv_head,
                                            :,
                                        ],
                                        keys[:local_valid, :],
                                    )
                                    T.copy(
                                        v_cmp[
                                            load_start : load_start + local_valid,
                                            kv_head,
                                            :,
                                        ],
                                        values[:local_valid, :],
                                    )
                                T.dual_copy(keys, k_l1)
                                T.dual_copy(values, v_l1)
                                T.copy(k_l1, k_l0)
                                # Transpose in the L1-to-L0 path, retaining coalesced GM reads.
                                T.copy(v_l1, v_l0, transpose=True)
                                T.gemm(
                                    q_l0,
                                    k_l0,
                                    scores_l0c,
                                    transpose_B=True,
                                    clear_accum=True,
                                )
                                T.dual_copy(scores_l0c, scores[:, :key_tile])
                            with T.Stage(1):
                                softmax_simd(
                                    scores,
                                    maximum,
                                    alpha,
                                    block_sum,
                                    probabilities,
                                    valid,
                                    tile == 0,
                                )
                                T.dual_copy(probabilities[:vector_rows, :], p_l1)
                                T.copy(p_l1, p_l0)
                                T.gemm(
                                    p_l0,
                                    v_l0,
                                    value_l0c,
                                    transpose_B=True,
                                    clear_accum=True,
                                )
                                T.dual_copy(value_l0c, value_tile)
                                if tile == 0:
                                    with T.SimdVF():
                                        mask = T.simd.pset(32, "PAT_ALL")
                                        one = T.simd.pset(32, "PAT_VL1")
                                        for h in T.serial(vector_rows):
                                            total = T.simd.vld(block_sum[h], "BRC_B32")
                                            T.simd.vsts(
                                                denominator[h],
                                                total,
                                                one,
                                                dist="ONEPT_B32",
                                                extent=1,
                                            )
                                            for c in T.serial(dim_v // 64):
                                                value = T.simd.vld(value_tile[h, c * 64])
                                                T.simd.vsts(
                                                    numerator[h, c * 64],
                                                    value,
                                                    mask,
                                                    dist="NORM_B32",
                                                    extent=64,
                                                )
                                else:
                                    with T.SimdVF():
                                        mask = T.simd.pset(32, "PAT_ALL")
                                        one = T.simd.pset(32, "PAT_VL1")
                                        for h in T.serial(vector_rows):
                                            factor = T.simd.vld(alpha[h], "BRC_B32")
                                            total = T.simd.vadd(
                                                T.simd.vmul(
                                                    T.simd.vld(denominator[h], "BRC_B32"),
                                                    factor,
                                                    mask,
                                                ),
                                                T.simd.vld(block_sum[h], "BRC_B32"),
                                                mask,
                                            )
                                            T.simd.vsts(
                                                denominator[h],
                                                total,
                                                one,
                                                dist="ONEPT_B32",
                                                extent=1,
                                            )
                                            for c in T.serial(dim_v // 64):
                                                updated = T.simd.vadd(
                                                    T.simd.vmul(
                                                        T.simd.vld(numerator[h, c * 64]),
                                                        factor,
                                                        mask,
                                                    ),
                                                    T.simd.vld(value_tile[h, c * 64]),
                                                    mask,
                                                )
                                                T.simd.vsts(
                                                    numerator[h, c * 64],
                                                    updated,
                                                    mask,
                                                    dist="NORM_B32",
                                                    extent=64,
                                                )

                        normalize_rows(numerator, denominator, result)
                        # Each SID owns eight half/bfloat16 LSE elements (16 B).
                        # Exact SIMT GM stores avoid a rounded-up 32 B DMA write.
                        with T.SimtVF(threads=128):
                            for h in T.Parallel(vector_rows):
                                lse[token, head_start + sid * vector_rows + h] = T.cast(
                                    T.if_then_else(
                                        denominator[h] > 0,
                                        maximum[h] + T.log(denominator[h]),
                                        0,
                                    ),
                                    dtype,
                                )
                    else:
                        with T.SimtVF(threads=128):
                            for h, d in T.Parallel(vector_rows, dim_v):
                                result[h, d] = T.cast(0, dtype)
                            for h in T.Parallel(vector_rows):
                                lse[token, head_start + sid * vector_rows + h] = T.cast(0, dtype)
                    T.dual_copy(result, output[token, head_start : head_start + head_tile, :])

    return main


@functools.lru_cache(maxsize=64)
def _compile_nsa_compressed_varlen_kernel(
    batch: int,
    tokens: int,
    heads: int,
    heads_kv: int,
    dim_k: int,
    dim_v: int,
    chunk_num: int,
    block_size: int,
    scale: float,
    dtype: str,
    num_blocks: int,
):
    return tilelang.compile(
        _make_nsa_compressed_varlen_kernel(
            batch,
            tokens,
            heads,
            heads_kv,
            dim_k,
            dim_v,
            chunk_num,
            block_size,
            scale,
            dtype,
            num_blocks,
        ),
        target="ascend",
        out_idx=[-2, -1],
    )


class AscendNSACompressedFwdVarlenKernel(Kernel, NSACompressedFwdInterface):
    """Closed-chunk attention for the three compressed manifest workloads.

    The underlying builder keeps DK and DV independent for diagnostics. Public
    dispatch is restricted to the declared metadata shapes and parameters; the
    contents of offsets and chunk_offsets remain runtime inputs.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: NSACall) -> str | None:
        configuration = (
            call.batch,
            call.c_seq_len,
            call.chunk_num,
            call.heads,
            call.heads_kv,
            call.dim,
            call.dim_v,
            call.block_size,
            call.scale,
            call.dtype,
        )
        if configuration not in _CONFIGURATIONS:
            return "requires one of the three NSACompressedVarlenFwdOp manifest configurations"
        return super().refusal(call)

    def __init__(self, call: NSACall) -> None:
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self._compiled = _compile_nsa_compressed_varlen_kernel(
            call.batch,
            call.c_seq_len,
            call.heads,
            call.heads_kv,
            call.dim,
            call.dim_v,
            call.chunk_num,
            call.block_size,
            call.scale,
            self.dtype_str,
            calibrated_cube_blocks(),
        )

    def forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        v_cmp: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (
            ("q", q, (self.call.c_seq_len, self.call.heads, self.call.dim), self.dtype),
            ("k_cmp", k_cmp, (self.call.chunk_num, self.call.heads_kv, self.call.dim), self.dtype),
            (
                "v_cmp",
                v_cmp,
                (self.call.chunk_num, self.call.heads_kv, self.call.dim_v),
                self.dtype,
            ),
            ("offsets", offsets, (self.call.batch + 1,), torch.int32),
            ("chunk_offsets", chunk_offsets, (self.call.batch + 1,), torch.int32),
            ("token_indices", token_indices, (self.call.c_seq_len, 2), torch.int32),
        )
        check_inputs(self, expected)
        # The manifest accepts strided inputs. Normalize reads without changing
        # the caller's tensors; vectorized payload loads also need aligned bases.
        normalized = tuple(tensor.contiguous() for _, tensor, _shape, _dtype in expected)
        normalized = tuple(
            vector_aligned(tensor) if i < 3 else tensor for i, tensor in enumerate(normalized)
        )
        return self._compiled(*normalized)


register_kernel_type("NSACompressedVarlenFwdOp", "ascend", AscendNSACompressedFwdVarlenKernel)
