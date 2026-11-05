"""Masked MoE grouped GEMM with FP8 block scales on Ascend 950.

Every expert owns one ``max_m``-row slab and ``layout_metadata`` gives its valid
row count; the count is clamped on the device, so no host sync is needed. The
Cube contracts one 128-wide K scale block per MMA, the fixpipe splits each L0C
partial across the two vector sub-cores, and each sub-core folds its half into a
UB accumulator under ``a_scale * b_scale``. The L1 tile spans ``_K_GROUP`` scale
blocks so each GM->L1 descriptor reads 512 contiguous bytes per row instead of
128, which keeps the MTE2 pipe near its width.
"""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.ascend.language import simd as S

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.call_spec import MGroupedGemmFP8Call, MGroupedGemmFP8FwdInterface

__all__ = ["AscendMoEGroupedGemmFP8Kernel"]

# One fp32 scale per 128 K columns of A and per 128x128 block of B.
_SCALE_BLOCK = 128
# fp32 SIMD lanes in one vector register.
_VL = 64
# Scale blocks per GM->L1 tile.
_K_GROUP = 4
# Persistent grid = the device's AI core count; a larger grid adds a tail wave.
_NUM_BLOCKS = 28
# The tallest M tile that divides the slab is used.
_TILE_M_CHOICES = (128, 64)
_NUM_STAGES = 2


def tile_m_for(max_m: int) -> Optional[int]:
    """The M tile this kernel uses for a slab of *max_m* rows, or ``None`` if none fits."""
    for tile_m in _TILE_M_CHOICES:
        if max_m >= tile_m and max_m % tile_m == 0:
            return tile_m
    return None


def make_grouped_gemm_fp8(num_groups: int, max_m: int, n: int, k: int, num_blocks: int):
    """Build the masked FP8 grouped GEMM over ``[E, max_m, ...]`` slabs."""
    tile_m = tile_m_for(max_m)
    if tile_m is None:
        raise ValueError(f"needs max_m divisible by one of {_TILE_M_CHOICES}, got {max_m}")
    tile_n = _SCALE_BLOCK
    if k % _SCALE_BLOCK or n % tile_n:
        raise ValueError(f"needs K and N multiples of {_SCALE_BLOCK}, got K={k}, N={n}")
    scale_k = k // _SCALE_BLOCK
    scale_n = n // _SCALE_BLOCK
    if scale_k % _K_GROUP:
        raise ValueError(f"needs (K / {_SCALE_BLOCK}) divisible by {_K_GROUP}, got K={k}")
    scale_k_pad = (scale_k + 7) // 8 * 8
    n_tiles = n // tile_n
    rows_per_aiv = tile_m // 2
    assert tile_n % _VL == 0

    @T.prim_func
    def main(
        A: T.Tensor((num_groups, max_m, k), "float8_e4m3fn"),
        A_scale: T.Tensor((num_groups, max_m, scale_k), "float32"),
        B: T.Tensor((num_groups, n, k), "float8_e4m3fn"),
        B_scale: T.Tensor((num_groups, scale_n, scale_k), "float32"),
        sizes: T.Tensor((num_groups,), "int32"),
        C: T.Tensor((num_groups, max_m, n), "bfloat16"),
    ):
        with T.MixedKernel(num_blocks, sids=2) as (block_idx, _sub_id):
            a_l1 = T.alloc_l1((tile_m, _K_GROUP * _SCALE_BLOCK), "float8_e4m3fn")
            b_l1 = T.alloc_l1((tile_n, _K_GROUP * _SCALE_BLOCK), "float8_e4m3fn")
            a_l0 = T.alloc_l0a((tile_m, _SCALE_BLOCK), "float8_e4m3fn")
            b_l0 = T.alloc_l0b((tile_n, _SCALE_BLOCK), "float8_e4m3fn")
            l0c = [T.alloc_l0c((tile_m, tile_n), "float32") for _ in range(2)]
            part_ub = [T.alloc_shared((rows_per_aiv, tile_n), "float32") for _ in range(_K_GROUP)]
            acc_ub = T.alloc_shared((rows_per_aiv, tile_n), "float32")
            sfa_ub = T.alloc_shared((rows_per_aiv, scale_k_pad), "float32")
            sfb_ub = T.alloc_shared((scale_k_pad,), "float32")
            out_ub = T.alloc_shared((rows_per_aiv, tile_n), "bfloat16")
            versions = {
                a_l1: _NUM_STAGES,
                b_l1: _NUM_STAGES,
                acc_ub: 1,
                sfa_ub: 1,
                sfb_ub: 1,
                out_ub: 1,
            }
            for buf in part_ub:
                versions[buf] = 1
            for buf in l0c:
                versions[buf] = 1
            T.annotate_buffer_versions(versions)

            # One tile per (expert, M tile in its slab, N tile); a count is clamped to
            # [0, max_m], so an expert's tiles stay inside its slab and empty ones are skipped.
            total = T.alloc_var("int32", init=0)
            tile = T.alloc_var("int32", init=block_idx)
            group = T.alloc_var("int32")
            rem = T.alloc_var("int32")
            found = T.alloc_var("int32")
            m_tile = T.alloc_var("int32")
            n_tile = T.alloc_var("int32")
            m_start = T.alloc_var("int32")
            n_start = T.alloc_var("int32")
            for g in range(num_groups):
                total = total + T.ceildiv(T.max(T.min(sizes[g], max_m), 0), tile_m)
            total = total * n_tiles
            while tile < total:
                group = 0
                rem = tile
                found = 0
                for g in range(num_groups):
                    if found == 0:
                        if rem >= T.ceildiv(T.max(T.min(sizes[g], max_m), 0), tile_m) * n_tiles:
                            rem = (
                                rem - T.ceildiv(T.max(T.min(sizes[g], max_m), 0), tile_m) * n_tiles
                            )
                            group = group + 1
                        else:
                            found = 1
                m_tile = rem // n_tiles
                n_tile = rem % n_tiles
                m_start = m_tile * tile_m
                n_start = n_tile * tile_n
                T.assume(group >= 0 and group < num_groups)
                T.assume(m_start >= 0 and m_start + tile_m <= max_m)
                T.assume(n_start >= 0 and n_start + tile_n <= n)
                n_block = n_start // _SCALE_BLOCK
                # M-split dual copies: each sub-core loads and stores its own half of the rows.
                T.dual_copy(
                    A_scale[group, m_start : m_start + tile_m, 0:scale_k],
                    sfa_ub[0:rows_per_aiv, 0:scale_k],
                )
                T.copy(B_scale[group, n_block, 0:scale_k], sfb_ub[0:scale_k])
                with T.SimdVF():
                    T.fill(acc_ub, T.float32(0.0))
                for kg in T.Pipelined(scale_k // _K_GROUP, num_stages=_NUM_STAGES):
                    T.copy(
                        A[
                            group,
                            m_start : m_start + tile_m,
                            kg * _K_GROUP * _SCALE_BLOCK : (kg + 1) * _K_GROUP * _SCALE_BLOCK,
                        ],
                        a_l1,
                    )
                    T.copy(
                        B[
                            group,
                            n_start : n_start + tile_n,
                            kg * _K_GROUP * _SCALE_BLOCK : (kg + 1) * _K_GROUP * _SCALE_BLOCK,
                        ],
                        b_l1,
                    )
                    # Python-level unroll: each scale-block slot is a separate buffer.
                    for j in tuple(range(_K_GROUP)):
                        kk = kg * _K_GROUP + j
                        T.copy(a_l1[0:tile_m, j * _SCALE_BLOCK : (j + 1) * _SCALE_BLOCK], a_l0)
                        T.copy(b_l1[0:tile_n, j * _SCALE_BLOCK : (j + 1) * _SCALE_BLOCK], b_l0)
                        T.gemm(a_l0, b_l0, l0c[j % 2], transpose_B=True, clear_accum=True)
                        T.dual_copy(l0c[j % 2], part_ub[j])
                        with T.SimdVF():
                            fold_mask = S.pset(32, "PAT_ALL")
                            b_scale = S.vld(sfb_ub[kk], "BRC_B32")
                            for r in range(rows_per_aiv):
                                scale = S.vmul(S.vld(sfa_ub[r, kk], "BRC_B32"), b_scale, fold_mask)
                                for c in range(0, tile_n, _VL):
                                    part = S.vld(part_ub[j][r, c], "NORM")
                                    prev = S.vld(acc_ub[r, c], "NORM")
                                    S.vsts(
                                        acc_ub[r, c],
                                        S.vadd(prev, S.vmul(part, scale, fold_mask), fold_mask),
                                        fold_mask,
                                        "NORM_B32",
                                    )
                with T.SimdVF():
                    cast_mask = S.pset(32, "PAT_ALL")
                    for r in range(rows_per_aiv):
                        for c in range(0, tile_n, _VL):
                            S.vsts(
                                out_ub[r, c],
                                S.vcvt(S.vld(acc_ub[r, c], "NORM"), "bfloat16", cast_mask),
                                cast_mask,
                                "PK_B32",
                            )
                T.dual_copy(
                    out_ub, C[group, m_start : m_start + tile_m, n_start : n_start + tile_n]
                )
                tile = tile + num_blocks

    return main


@functools.lru_cache(maxsize=None)
def _compile(num_groups: int, max_m: int, n: int, k: int, num_blocks: int):
    return tilelang.compile(
        make_grouped_gemm_fp8(num_groups, max_m, n, k, num_blocks),
        target="ascend",
        out_idx=[],
    )


class AscendMoEGroupedGemmFP8Kernel(Kernel, MGroupedGemmFP8FwdInterface):
    """Block-scaled FP8 grouped GEMM over masked ``[E, max_m, ...]`` slabs."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: MGroupedGemmFP8Call) -> str | None:
        if call.kind != "masked":
            return f"serves only the masked layout, got {call.kind}"
        if call.ab_dtype is not torch.float8_e4m3fn:
            return f"serves float8_e4m3fn operands, got {call.ab_dtype}"
        if call.max_m is None or tile_m_for(call.max_m) is None:
            return f"needs max_m divisible by one of {_TILE_M_CHOICES}, got {call.max_m}"
        if call.k % _SCALE_BLOCK or call.n % _SCALE_BLOCK:
            return f"needs K and N multiples of {_SCALE_BLOCK}, got K={call.k}, N={call.n}"
        if (call.k // _SCALE_BLOCK) % _K_GROUP:
            return f"needs (K / {_SCALE_BLOCK}) divisible by {_K_GROUP}, got K={call.k}"
        scale_k = call.k // _SCALE_BLOCK
        a_scale = (call.num_groups, call.max_m, scale_k)
        b_scale = (call.num_groups, call.n // _SCALE_BLOCK, scale_k)
        if tuple(call.a_scale_shape) != a_scale or tuple(call.b_scale_shape) != b_scale:
            return (
                f"reads a_scale {a_scale} and b_scale {b_scale}, got "
                f"{tuple(call.a_scale_shape)} and {tuple(call.b_scale_shape)}"
            )
        return super().refusal(call)

    def __init__(self, call: MGroupedGemmFP8Call) -> None:
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self._kernel = _compile(call.num_groups, call.max_m, call.n, call.k, _NUM_BLOCKS)

    def forward(
        self,
        a: torch.Tensor,
        a_scale: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c = self.call
        self._require_devices(
            a=a,
            a_scale=a_scale,
            b=b,
            b_scale=b_scale,
            layout_metadata=layout_metadata,
            out=out,
        )
        rows = (c.num_groups, c.max_m)
        if a.shape != (*rows, c.k) or a.dtype != c.ab_dtype:
            raise ValueError("a does not match the grouped GEMM specialization")
        if a_scale.shape != (*rows, c.k // _SCALE_BLOCK) or a_scale.dtype is not torch.float32:
            raise ValueError("a_scale must be one float32 scale per row and 128 columns of a")
        if b.shape != (c.num_groups, c.n, c.k) or b.dtype != c.ab_dtype:
            raise ValueError("b does not match the grouped GEMM specialization")
        if b_scale.shape != (c.num_groups, c.n // _SCALE_BLOCK, c.k // _SCALE_BLOCK):
            raise ValueError("b_scale must be one float32 scale per 128x128 block of b")
        if layout_metadata.shape != (c.num_groups,) or layout_metadata.dtype is not torch.int32:
            raise ValueError("layout_metadata must be one int32 valid-row count per slab")
        if out is None:
            out = torch.empty((*rows, c.n), dtype=torch.bfloat16, device=a.device)
        elif out.shape != (*rows, c.n) or out.dtype is not torch.bfloat16:
            raise ValueError("out must be bfloat16 and shaped like the return value")
        self._kernel(a, a_scale, b, b_scale, layout_metadata, out)
        return out
