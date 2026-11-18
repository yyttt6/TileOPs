"""Ascend 950 W4A16 GEMM over the public Marlin-packed unsigned INT4 weight.

Disjoint AIV producers unpack directly into compact NZ tiles while the Cube contracts
the preceding tile. Group metadata stays resident when its complete grid fits UB.
Contractions whose whole K tile fits on chip restore weight order in that launch;
otherwise the cheaper activation operand is permuted once before the fused product.
"""

import functools
from typing import Optional

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_cube_cores
from tilelang.layout import make_ascend_compact_nz_layout

import tileops.backend._registry
from tileops._kernels.ascend.gemm import w4a16_resident
from tileops._kernels.gemm import (
    GEMMW4A16Call,
    GEMMW4A16FwdInterface,
    W4A16RepackCall,
    W4A16RepackFwdInterface,
)
from tileops._kernels.kernel_base import Kernel

__all__ = ["AscendGEMMW4A16Kernel", "AscendW4A16RepackKernel"]

_FP16 = "float16"
_NUM_AIV_CORES = 64
_CAST_STAGES = 2
_UB_BUDGET = 176 * 1024


def _permute_index() -> list[int]:
    """The original K position of each unpacked slot in one 128-value group."""
    order = []
    for slot in range(128):
        word, remainder = divmod(slot, 8)
        column, lane = word % 4, word // 4
        pair, half = remainder % 4, remainder // 4
        order.append(32 * column + 8 * pair + 2 * lane + half)
    return order


def _tiles(m: int, n: int, k: int) -> tuple[int, int, int, int, int]:
    """Widest complete column atom that fills the fixed grid within L1/L0C."""
    if min(m, n, k) <= 0 or n % 16 or k % 128:
        raise ValueError("requires positive dimensions, N divisible by 16, and K divisible by 128")
    blocks = 32
    tile_m = 16 if m <= 16 else 64 if m <= 64 else 128 if m <= 256 else 256
    while tile_m > 16 and -(-m // tile_m) * (n // 16) < blocks:
        tile_m //= 2
    row_tiles = -(-m // tile_m)
    tile_n = 16
    for candidate in range(min(256, n), 15, -16):
        if n % candidate == 0 and row_tiles * (n // candidate) >= blocks:
            tile_n = candidate
            break
    tile_k = (
        256
        if k % 256 == 0 and _tile_k_fits(m, tile_m, tile_n, k, _CAST_STAGES, 256)
        else 128
    )
    if k >= 512 and k % 256 == 0 and tile_k == 128:
        # The paired nibble layout needs whole 256-value K tiles; narrow the column
        # atom over the same legal candidates the search above used.
        for candidate in range(min(256, n), 15, -16):
            if n % candidate:
                continue
            if _tile_k_fits(m, tile_m, candidate, k, _CAST_STAGES, 256):
                tile_n, tile_k = candidate, 256
                break
    stages = _CAST_STAGES
    if tile_k == 256 and _tile_k_fits(m, tile_m, tile_n, k, stages + 1, tile_k):
        stages += 1
    return tile_m, tile_n, tile_k, stages, blocks


def _producer_rows(tile_n: int) -> tuple[int, int]:
    count = 2 if tile_n % 32 == 0 else 1
    return count, tile_n // count


def _metadata_resident(tile_n: int, k: int, stages: int, tile_k: int = 128) -> bool:
    """Whether the full metadata grid fits beside both staged packed/NZ tiles."""
    _, rows = _producer_rows(tile_n)
    packed_columns = tile_k // 2
    packed_rows = rows + (256 - packed_columns) // packed_columns
    staged = stages * (packed_rows * packed_columns + 2 * (rows + 1) * tile_k)
    return staged + 3 * rows * (k // 128) + 512 <= _UB_BUDGET


def _tile_k_fits(
    m: int, tile_m: int, tile_n: int, k: int, stages: int, tile_k: int
) -> bool:
    """Mirror the fused program's capacity checks for one candidate K tile."""
    _, producer_n = _producer_rows(tile_n)
    resident = _metadata_resident(tile_n, k, stages, tile_k)
    packed_columns = tile_k // 2
    packed_rows = producer_n + (256 - packed_columns) // packed_columns
    scale_columns = k // 128 if resident else (2 if tile_k == 256 else 1)
    metadata_versions = 1 if resident else stages
    ub_bytes = stages * (packed_rows * packed_columns + 2 * (producer_n + 1) * tile_k)
    ub_bytes += metadata_versions * 3 * producer_n * scale_columns + 512
    if ub_bytes > _UB_BUDGET:
        return False
    if 2 * (tile_m + tile_n) * tile_k * stages > 512 * 1024:
        return False
    if 4 * tile_m * tile_n > 256 * 1024:
        return False
    return True


def _permute_program(m: int, k: int):
    """Permute the activation by the packed stream's group-local K order."""
    chunk = 4096
    while chunk > 128 and k % chunk:
        chunk //= 2
    chunks = k // chunk

    @T.prim_func
    def main(
        A: T.Tensor((m, k), _FP16), Order: T.Tensor((128,), "uint16"), A2: T.Tensor((m, k), _FP16)
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(_NUM_AIV_CORES) as core:
            order = T.alloc_shared((128,), "uint16")
            T.copy(Order, order)
            for item in T.Persistent([m * chunks], _NUM_AIV_CORES, core, num_stages=_CAST_STAGES):
                row = item // chunks
                col = item % chunks * chunk
                x = T.alloc_shared((chunk,), _FP16)
                out = T.alloc_shared((chunk,), _FP16)
                T.annotate_buffer_versions({x: _CAST_STAGES, out: _CAST_STAGES})
                T.copy(A[row, col : col + chunk], x)
                with T.SimdVF():
                    index = T.simd.vld(order[0], "NORM_B16")
                    for group in T.serial(chunk // 128):
                        value = T.simd.vgather2(x[group * 128], index)
                        T.simd.vsts(out[group * 128], value, dist="NORM_B16")
                T.copy(out, A2[row, col : col + chunk])

    return main


def _fused_program(m, n, k, tile_m, tile_n, tile_k, stages, blocks):
    """Two-stage weight producer/Cube consumer, including persistent M/N tiles."""
    if min(m, n, k, tile_m, tile_n, stages, blocks) <= 0:
        raise ValueError("dimensions, tiles, stages and blocks must be positive")
    if tile_k not in (128, 256) or k % tile_k or n % tile_n or tile_n % 16:
        raise ValueError("requires whole 128-value quantization groups and 16-row NZ atoms")
    producer_count, producer_n = _producer_rows(tile_n)
    resident_metadata = _metadata_resident(tile_n, k, stages, tile_k)
    packed_columns = tile_k // 2
    packed_rows = producer_n + (256 - packed_columns) // packed_columns
    scale_columns = k // 128 if resident_metadata else (2 if tile_k == 256 else 1)
    metadata_versions = 1 if resident_metadata else stages
    ub_bytes = stages * (packed_rows * packed_columns + 2 * (producer_n + 1) * tile_k)
    ub_bytes += metadata_versions * 3 * producer_n * scale_columns + 512
    if ub_bytes > _UB_BUDGET:
        raise ValueError("versioned packed/NZ operands and metadata exceed UB capacity")
    if 2 * (tile_m + tile_n) * tile_k * stages > 512 * 1024:
        raise ValueError("L1 operand versions exceed the 512 KiB capacity")
    if 4 * tile_m * tile_n > 256 * 1024:
        raise ValueError("FP32 accumulator exceeds the 256 KiB L0C capacity")
    mad_k = 64
    if 4 * max(tile_m, tile_n) * mad_k > 64 * 1024:
        raise ValueError("two L0 operand versions exceed the 64 KiB capacity")

    paired = k >= 512 and tile_k == 256

    @T.macro
    def dequantize_paired(stream, scale, zero, weight, row, col, part: int):
        mask_all = T.simd.pset(8, "PAT_ALL")
        mask16 = T.simd.pset(16, "PAT_VL128")
        value = T.simd.vcvt(
            T.reinterpret(stream, "int8x256"), _FP16, mask_all, part=part
        )
        offset = T.simd.vcvt(T.simd.vld(zero[row, col], "BRC_B8"), _FP16, mask_all, part=0)
        value = T.simd.vsub(value, offset, mask16)
        factor = T.simd.vld(scale[row, col], "BRC_B16")
        T.simd.vsstb(
            T.simd.vmul(value, factor, mask16),
            weight[row, part * 128],
            (producer_n + 1) << 16,
            mask16,
        )

    @T.macro
    def dequantize(stream, scale, zero, weight, row, col, part: int):
        mask16 = T.simd.pset(16, "PAT_VL128")
        q = T.simd.vunpack(stream, part=part)
        z = T.simd.vunpack(T.simd.vld(zero[row, col], "BRC_B8"), part=0)
        centered = T.simd.vsub(q, z, mask16)
        signed = T.reinterpret(centered, "int16x128")
        factor = T.simd.vld(scale[row, col], "BRC_B16")
        value = T.simd.vmul(T.simd.vcvt(signed, _FP16, mask16), factor, mask16)
        T.simd.vsstb(value, weight[row, part * 128], (producer_n + 1) << 16, mask16)

    @T.prim_func
    def main(
        A: T.Tensor((m, k), _FP16),
        P: T.Tensor((n, k // 2), "uint8"),
        S: T.Tensor((n, k // 128), _FP16),
        Z: T.Tensor((n, k // 128), "uint8"),
        D: T.Tensor((m, n), _FP16),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.MixedKernel(blocks) as (bx, sid):
            a_l1 = T.alloc_l1((tile_m, tile_k), _FP16)
            b_l1 = T.alloc_l1((tile_n, tile_k), _FP16)
            a_l0 = T.alloc_l0a((tile_m, mad_k), _FP16)
            b_l0 = T.alloc_l0b((tile_n, mad_k), _FP16)
            acc = T.alloc_l0c((tile_m, tile_n), "float32")
            # NORM_B8 reads 256 bytes even when only the first group lanes are used.
            packed = T.alloc_shared((packed_rows, packed_columns), "uint8")
            scale = T.alloc_shared((producer_n, scale_columns), _FP16)
            zero = T.alloc_shared((producer_n, scale_columns), "uint8")
            # The pinned release's compact-NZ copy requires one trailing gap row.
            weight = T.alloc_shared((producer_n + 1, tile_k), _FP16)
            T.annotate_layout({weight: make_ascend_compact_nz_layout(weight)})
            versions = {
                a_l1: stages,
                b_l1: stages,
                a_l0: 2,
                b_l0: 2,
                packed: stages,
                weight: stages,
                scale: metadata_versions,
                zero: metadata_versions,
            }
            T.annotate_buffer_versions(versions)
            scheduler = T.AscendTileScheduler(
                block_m=tile_m, block_n=tile_n, num_cores=blocks, shape_m=m, shape_n=n
            )
            scheduler.init(bx)
            while scheduler.valid():
                first_m, first_n, rows, cols = scheduler.tile()
                producer = first_n + sid * producer_n
                if resident_metadata and sid < producer_count:
                    T.copy(S[producer : producer + producer_n, :], scale)
                    T.copy(Z[producer : producer + producer_n, :], zero)
                for ki in T.Pipelined(k // tile_k, num_stages=stages):
                    start = ki * tile_k
                    T.copy(A[first_m : first_m + rows, start : start + tile_k], a_l1[:rows, :])
                    if sid < producer_count:
                        T.copy(
                            P[producer : producer + producer_n, start // 2 : (start + tile_k) // 2],
                            packed[:producer_n, :],
                        )
                        if not resident_metadata and tile_k == 256:
                            T.copy(
                                S[producer : producer + producer_n, 2 * ki : 2 * ki + 2], scale
                            )
                            T.copy(Z[producer : producer + producer_n, 2 * ki : 2 * ki + 2], zero)
                        elif not resident_metadata:
                            T.copy(S[producer : producer + producer_n, ki : ki + 1], scale)
                            T.copy(Z[producer : producer + producer_n, ki : ki + 1], zero)
                        with T.SimdVF():
                            mask8 = T.simd.pset(8, "PAT_VL128" if tile_k == 256 else "PAT_VL64")
                            fifteen = T.simd.vdup(T.uint8(15), "uint8")
                            for row in T.serial(producer_n):
                                col = ki * (tile_k // 128) if resident_metadata else 0
                                raw = T.simd.vld(packed[row, 0], "NORM_B8")
                                low = T.simd.vand(raw, fifteen, mask8)
                                high = T.simd.vshrs(raw, 4, mask8)
                                stream, _ = T.simd.vintlv(
                                    T.reinterpret(low, "uint8x256"),
                                    T.reinterpret(high, "uint8x256"),
                                )
                                if paired:
                                    dequantize_paired(
                                        stream, scale, zero, weight, row, col, 0
                                    )
                                    dequantize_paired(
                                        stream, scale, zero, weight, row, col + 1, 1
                                    )
                                else:
                                    dequantize(stream, scale, zero, weight, row, col, 0)
                                    if tile_k == 256:
                                        dequantize(
                                            stream, scale, zero, weight, row, col + 1, 1
                                        )
                        T.copy(
                            weight[:producer_n, :],
                            b_l1[sid * producer_n : (sid + 1) * producer_n, :],
                        )
                    for part in T.serial(tile_k // mad_k):
                        T.copy(a_l1[:rows, part * mad_k : (part + 1) * mad_k], a_l0[:rows, :])
                        T.copy(b_l1[:cols, part * mad_k : (part + 1) * mad_k], b_l0[:cols, :])
                        T.gemm(
                            a_l0[:rows, :],
                            b_l0[:cols, :],
                            acc[:rows, :cols],
                            transpose_B=True,
                            clear_accum=ki == 0 and part == 0,
                        )
                T.copy(acc[:rows, :cols], D[first_m : first_m + rows, first_n : first_n + cols])
                scheduler.next_block()

    return main


@functools.lru_cache(maxsize=1)
def _make_repack_paired():
    """The paired nibble layout for any whole-row shape, compiled once.

    The pass only moves nibbles, so a dynamic row count and row length let the
    single compiled kernel serve every manifest shape instead of one build per
    (N, K) pair.
    """
    rows = T.dynamic("rows")
    columns = T.dynamic("columns")
    pairs = columns // 128

    @T.prim_func
    def main(W: T.Tensor((rows, columns), "uint8"), P: T.Tensor((rows, columns), "uint8")):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(_NUM_AIV_CORES) as core:
            for item in T.Persistent(
                [rows * pairs], _NUM_AIV_CORES, core, num_stages=_CAST_STAGES
            ):
                row = item // pairs
                first = item % pairs * 128
                source = T.alloc_shared((128,), "uint8")
                output = T.alloc_shared((128,), "uint8")
                T.annotate_buffer_versions({source: _CAST_STAGES, output: _CAST_STAGES})
                T.copy(W[row, first : first + 128], source)
                with T.SimtVF(threads=256):
                    for j in T.Parallel(128):
                        shift = 4 * (j % 2)
                        low = (source[j // 2] >> shift) & 15
                        high = (source[64 + j // 2] >> shift) & 15
                        output[j] = low | (high << 4)
                T.copy(output, P[row, first : first + 128])

    return tilelang.compile(
        main,
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


@functools.lru_cache(maxsize=256)
def _make_repack(n: int, packed_k: int):
    """Public row-major INT4 input to the paired order the GEMM reads."""
    if min(n, packed_k) <= 0 or packed_k % 64:
        raise ValueError("repack requires positive rows and whole 128-value K groups")
    if packed_k >= 256 and packed_k % 128 == 0:
        return _make_repack_paired()
    chunk = max(c for c in range(64, min(4096, packed_k) + 1, 64) if packed_k % c == 0)
    chunks = packed_k // chunk

    @T.prim_func
    def main(W: T.Tensor((n, packed_k), "uint8"), P: T.Tensor((n, packed_k), "uint8")):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(_NUM_AIV_CORES) as core:
            for item in T.Persistent([n * chunks], _NUM_AIV_CORES, core, num_stages=_CAST_STAGES):
                row = item // chunks
                col = item % chunks * chunk
                source = T.alloc_shared((chunk,), "uint8")
                output = T.alloc_shared((chunk,), "uint8")
                T.annotate_buffer_versions({source: _CAST_STAGES, output: _CAST_STAGES})
                T.copy(W[row, col : col + chunk], source)
                with T.SimtVF(threads=256):
                    for j in T.Parallel(chunk):
                        slot = 2 * (j % 64)
                        word = slot // 8
                        remainder = slot % 8
                        origin = (
                            32 * (word % 4) + 8 * (remainder % 4) + 2 * (word // 4) + remainder // 4
                        )
                        group = j // 64 * 64
                        shift = 4 * (origin % 2)
                        lo = (source[group + origin // 2] >> shift) & 15
                        hi = (source[group + (origin + 8) // 2] >> shift) & 15
                        output[j] = lo | (hi << 4)
                T.copy(output, P[row, col : col + chunk])

    return tilelang.compile(
        main, target="ascend", out_idx=-1, pass_configs={"tl.enable_fast_math": False}
    )


@functools.lru_cache(maxsize=256)
def _make_permute(m: int, k: int):
    return tilelang.compile(
        _permute_program(m, k),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


@functools.lru_cache(maxsize=256)
def _make_fused(m: int, n: int, k: int):
    return tilelang.compile(
        _fused_program(m, n, k, *_tiles(m, n, k)),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


@functools.lru_cache(maxsize=256)
def _make_resident(m: int, n: int, k: int, tile_m: int, tile_n: int, blocks: int):
    return w4a16_resident.compile_kernel(m, n, k, tile_m, tile_n, blocks)


class AscendGEMMW4A16Kernel(Kernel, GEMMW4A16FwdInterface):
    """Public prepacked W4A16 contract, with resource-derived on-chip schedules."""

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: GEMMW4A16Call) -> Optional[str]:
        if call.group_size != 128:
            return f"requires a weight group of 128, got {call.group_size}"
        if call.dtype != torch.float16:
            return "requires float16 operands"
        if min(call.m, call.n, call.k) <= 0 or call.k % 128 or call.n % 16:
            return f"requires positive dimensions, K divisible by 128 and N divisible by 16, got m={call.m}, n={call.n}, k={call.k}"
        return super().refusal(call)

    def __init__(self, call: GEMMW4A16Call):
        device_index = call.device.index
        super().__init__(device_index=device_index)
        name = torch.npu.get_device_name(device_index)
        if "Ascend950" not in name:
            raise RuntimeError(f"These W4A16 schedules require Ascend 950, got {name}")
        if get_num_cube_cores(device_id=device_index or 0) < 32:
            raise RuntimeError("These W4A16 schedules require at least 32 Cube cores")
        tile_m, tile_n, tile_k, _, blocks = _tiles(call.m, call.n, call.k)
        self._resident = (
            w4a16_resident.fits(call.m, call.n, call.k, tile_m, tile_n)
            and not (call.k >= 512 and tile_k == 256)
        )
        order = _permute_index()
        if self._resident:
            inverse = [0] * 128
            for slot, index in enumerate(order):
                inverse[index] = slot
            order = inverse
            self._gemm = _make_resident(call.m, call.n, call.k, tile_m, tile_n, blocks)
        else:
            if call.k >= 512 and tile_k == 256:
                self._permute = None
            else:
                self._permute = _make_permute(call.m, call.k)
            self._gemm = _make_fused(call.m, call.n, call.k)
        self._order = torch.tensor(order, dtype=torch.uint16, device=call.device)

    def forward(self, activation, packed_weight, weight_scale, weight_zero) -> torch.Tensor:
        if self._resident:
            return self._gemm(activation, packed_weight, weight_scale, weight_zero, self._order)
        if self._permute is not None:
            activation = self._permute(activation, self._order)
        return self._gemm(activation, packed_weight, weight_scale, weight_zero)


class AscendW4A16RepackKernel(Kernel, W4A16RepackFwdInterface):
    """The public one-time weight conversion paired with this backend's GEMM."""

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: W4A16RepackCall) -> Optional[str]:
        if min(call.n, call.packed_k) <= 0 or call.packed_k % 64:
            return "requires positive rows and whole 128-value K groups"
        return super().refusal(call)

    def __init__(self, call: W4A16RepackCall):
        super().__init__(device_index=call.device.index)
        self._repack = _make_repack(call.n, call.packed_k)

    def forward(self, packed_weight: torch.Tensor) -> torch.Tensor:
        return self._repack(packed_weight)


tileops.backend._registry.register_kernel_type("GEMMW4A16FwdOp", "ascend", AscendGEMMW4A16Kernel)
tileops.backend._registry.register_kernel_type(
    "GEMMW4A16FwdOp", "ascend_repack", AscendW4A16RepackKernel
)
