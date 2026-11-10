"""Ascend 950 W4A16 GEMM: FP16 activations times a group-scaled unsigned INT4 weight.

Three passes. The weight is unpacked from the marlin-layout INT4 packing the manifest's
prepacked input uses, centred by its group zero point and scaled to FP16; a native FP16
GEMM then accumulates in FP32 and stores the operand dtype.

The packed order is a permutation of K inside every 128-value group, so the activations
are permuted the same way once per call. That pass costs ``M*K`` reads and writes, which
is the cheaper side of the contraction for every decode row this op targets and two
orders below the ``N*K`` weight it would otherwise have to permute.
"""

import functools
from typing import Optional

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_cube_cores

import tileops.backend._registry
from tileops._kernels.gemm import GEMMW4A16Call, GEMMW4A16FwdInterface
from tileops._kernels.kernel_base import Kernel

__all__ = ["AscendGEMMW4A16Kernel"]

_FP16 = "float16"
_NUM_AIV_CORES = 64
_CAST_STAGES = 2


def _permute_index() -> list[int]:
    """The K position each packed-stream slot holds, for one 128-value group.

    A marlin word ``t = 4*L + w`` of the packing carries the four even K values
    ``2*(16w + 4p + L)`` in its low half and the four odd ones in its high half, one
    nibble each at bit ``4p``. Unpacking walks the word byte by byte, so slot ``rem`` of
    the stream is pair ``rem % 4`` and half ``rem // 4``.
    """
    order = []
    for slot in range(128):
        t, rem = divmod(slot, 8)
        w, lane = t % 4, t // 4
        p, half = rem % 4, rem // 4
        order.append(32 * w + 8 * p + 2 * lane + half)
    return order


def _tiles(m: int, n: int, k: int) -> tuple[int, int, int, int, int]:
    """The tile ``(tile_m, tile_n, tile_k, stages, blocks)`` one call runs with.

    The row tile grows with the call up to the 256 rows the L0C budget allows.
    The column tile is the widest one that still cuts at least one tile per Cube
    core: a narrower tile costs a little reuse but leaves no core idle, and an
    idle core is the larger loss. Shapes that already reach 32 tiles keep the
    widest tile. Sixteen-bit operands double the L1 footprint, so the K stage is
    128 deep rather than the 256 the one-byte GEMMs use.
    """
    blocks = 32
    tile_m = 16 if m <= 16 else 64 if m <= 64 else 128 if m <= 256 else 256
    row_tiles = -(-m // tile_m)
    tile_n = n
    for candidate in range(min(256, n), 0, -16):
        if n % candidate == 0 and row_tiles * (n // candidate) >= blocks:
            tile_n = candidate
            break
    while tile_m > 16 and 2 * (tile_m + tile_n) * 128 * 2 * 2 > 512 * 1024:
        tile_m -= 64
    return tile_m, tile_n, 128, 2, blocks


def _dequant_program(n: int, k: int, group_size: int):
    """Build the unpack pass: ``wb[n, k] = fp16((q - zero) * scale)`` in packed order.

    One vector step covers a whole group: the 64 packed bytes give 128 nibbles in the
    stream order the packing defines, the zero point is one integer subtract before the
    widening (so the signed conversion carries the sign), and the group scale is one
    FP16 multiply.

    Args:
        n: Weight rows.
        k: Contracted length.
        group_size: Values per scale group; the vector step consumes exactly one.

    Raises:
        ValueError: ``group_size`` is not the 128 values one vector step covers, or
            the shapes do not divide into whole steps.
    """
    scale_cols = k // group_size
    # The FP16 row buffer is four times the packed one, so a long K row is walked in
    # chunks that keep both inside the vector unit's unified buffer.
    chunk = 4096
    while chunk > 128 and k % chunk:
        chunk //= 2
    chunks = k // chunk
    groups = chunk // group_size
    if group_size != 128:
        raise ValueError(f"this unpack step consumes one 128-value group, got {group_size}")
    if k % group_size or n % 2:
        raise ValueError(f"requires whole groups and even N, got n={n}, k={k}")

    @T.prim_func
    def main(
        W: T.Tensor((n, k // 2), "uint8"),
        S: T.Tensor((n, scale_cols), _FP16),
        Z: T.Tensor((n, scale_cols), "uint8"),
        WB: T.Tensor((n, k), _FP16),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(_NUM_AIV_CORES) as core:
            for item in T.Persistent([n * chunks], _NUM_AIV_CORES, core, num_stages=_CAST_STAGES):
                row = item // chunks
                col = item % chunks
                packed_ub = T.alloc_shared((chunk // 2,), "uint8")
                scale_ub = T.alloc_shared((groups,), _FP16)
                zero_ub = T.alloc_shared((groups,), "uint8")
                out_ub = T.alloc_shared((chunk,), _FP16)
                T.annotate_buffer_versions(
                    {
                        packed_ub: _CAST_STAGES,
                        scale_ub: _CAST_STAGES,
                        zero_ub: _CAST_STAGES,
                        out_ub: _CAST_STAGES,
                    }
                )
                T.copy(W[row, col * (chunk // 2) : (col + 1) * (chunk // 2)], packed_ub)
                T.copy(S[row, col * groups : (col + 1) * groups], scale_ub)
                T.copy(Z[row, col * groups : (col + 1) * groups], zero_ub)
                with T.SimdVF():
                    mask_b8 = T.simd.pset(8, "PAT_VL64")
                    mask_128 = T.simd.pset(8, "PAT_VL128")
                    mask_b16 = T.simd.pset(16, "PAT_VL128")
                    fifteen = T.simd.vdup(T.uint8(15), "uint8")
                    src = T.simd.make_ubuf_ptr(
                        T.access_ptr(packed_ub[0], "r", extent=chunk // 2), "uint8"
                    )
                    for g in T.serial(groups):
                        packed, src = T.simd.vld(src, "NORM_B8", post_inc=group_size // 2)
                        low = T.simd.vand(packed, fifteen, mask_b8)
                        high = T.simd.vand(T.simd.vshrs(packed, 4, mask_b8), fifteen, mask_b8)
                        stream, _ = T.simd.vintlv(
                            T.reinterpret(low, "uint8x256"),
                            T.reinterpret(high, "uint8x256"),
                        )
                        zero = T.simd.vld(zero_ub[g], "BRC_B8")
                        centred = T.simd.vsub(stream, zero, mask_128)
                        wide = T.simd.vunpack(T.reinterpret(centred, "int8x256"), part=0)
                        scale = T.simd.vld(scale_ub[g], "BRC_B16")
                        T.simd.vsts(
                            out_ub[g * group_size],
                            T.simd.vmul(T.simd.vcvt(wide, _FP16, mask_b16), scale, mask_b16),
                            mask_b16,
                            "NORM_B16",
                        )
                T.copy(out_ub, WB[row, col * chunk : (col + 1) * chunk])

    return main


def _permute_program(m: int, k: int):
    """Build the activation permute: ``a2[m, j] = a[m, order[j % 128] + (j // 128) * 128]``."""
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
            order_ub = T.alloc_shared((128,), "uint16")
            T.copy(Order[0:128], order_ub)
            for item in T.Persistent([m * chunks], _NUM_AIV_CORES, core, num_stages=_CAST_STAGES):
                row = item // chunks
                c = item % chunks
                x_ub = T.alloc_shared((chunk,), _FP16)
                out_ub = T.alloc_shared((chunk,), _FP16)
                T.annotate_buffer_versions({x_ub: _CAST_STAGES, out_ub: _CAST_STAGES})
                T.copy(A[row, c * chunk : (c + 1) * chunk], x_ub)
                with T.SimdVF():
                    gather_index = T.simd.vld(order_ub[0], "NORM_B16")
                    for gg in T.serial(chunk // 128):
                        gathered = T.simd.vgather2(x_ub[gg * 128], gather_index)
                        T.simd.vsts(out_ub[gg * 128], gathered, dist="NORM_B16")
                T.copy(out_ub, A2[row, c * chunk : (c + 1) * chunk])

    return main


def _gemm_program(
    m: int, n: int, k: int, tile_m: int, tile_n: int, tile_k: int, stages: int, blocks: int
):
    """Build the FP16 NT product: one persistent block per Cube core, FP32 accumulation."""
    mad_k = 64  # one MAD step: two L0 versions of a 128-row FP16 operand take 32 KiB
    k_tiles = -(-k // tile_k)
    if k % tile_k or tile_k % mad_k or n % tile_n:
        raise ValueError(f"this schedule requires whole K and N tiles, got n={n}, k={k}")
    if 2 * (tile_m + tile_n) * tile_k * 2 * stages > 512 * 1024:
        raise ValueError("L1 operand versions exceed the 512 KiB capacity")
    if 4 * tile_m * tile_n > 256 * 1024:
        raise ValueError("FP32 accumulator exceeds the 256 KiB L0C capacity")
    if 2 * 2 * max(tile_m, tile_n) * mad_k > 64 * 1024:
        raise ValueError("two L0 versions exceed the 64 KiB per-operand capacity")

    @T.prim_func
    def main(
        A: T.Tensor((m, k), _FP16),
        B: T.Tensor((n, k), _FP16),
        D: T.Tensor((m, n), _FP16),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(blocks) as bx:
            a_l1 = T.alloc_l1((tile_m, tile_k), _FP16)
            b_l1 = T.alloc_l1((tile_n, tile_k), _FP16)
            a_l0 = T.alloc_l0a((tile_m, mad_k), _FP16)
            b_l0 = T.alloc_l0b((tile_n, mad_k), _FP16)
            acc = T.alloc_l0c((tile_m, tile_n), "float32")
            T.annotate_buffer_versions({a_l1: stages, b_l1: stages, a_l0: 2, b_l0: 2})
            sched = T.AscendTileScheduler(
                block_m=tile_m,
                block_n=tile_n,
                num_cores=blocks,
                shape_m=m,
                shape_n=n,
            )
            sched.init(bx)
            while sched.valid():
                m_start, n_start, rows, cols = sched.tile()
                for ki in T.Pipelined(k_tiles, num_stages=stages):
                    start = ki * tile_k
                    T.copy(
                        A[m_start : m_start + rows, start : start + tile_k], a_l1[0:rows, 0:tile_k]
                    )
                    T.copy(
                        B[n_start : n_start + cols, start : start + tile_k], b_l1[0:cols, 0:tile_k]
                    )
                    for m_slice in range(tile_k // mad_k):
                        k_slice = m_slice * mad_k
                        T.copy(a_l1[0:rows, k_slice : k_slice + mad_k], a_l0[0:rows, 0:mad_k])
                        T.copy(b_l1[0:cols, k_slice : k_slice + mad_k], b_l0[0:cols, 0:mad_k])
                        T.gemm(
                            a_l0[0:rows, 0:mad_k],
                            b_l0[0:cols, 0:mad_k],
                            acc[0:rows, 0:cols],
                            transpose_B=True,
                            clear_accum=ki == 0 and m_slice == 0,
                        )
                T.copy(acc[0:rows, 0:cols], D[m_start : m_start + rows, n_start : n_start + cols])
                sched.next_block()

    return main


def _fused_program(
    m: int, n: int, k: int, tile_m: int, tile_n: int, tile_k: int, stages: int, blocks: int
):
    """Build the FP16 product with each weight tile unpacked into L1 by the vector unit.

    The cube never reads the packed weight: for every ``(n_tile, k_tile)`` the vector
    unit reads the packed bytes, unpacks them to FP16 in unified memory and copies that
    tile into L1. One call therefore halves the weight traffic of the two-pass path —
    it reads ``N*K/2`` bytes and writes nothing to GM — at the cost of unpacking the
    tile once per row tile, so it pays only while the call has a single row tile.
    """
    mad_k = 64  # one MAD step: two L0 versions of a 128-row FP16 operand take 32 KiB
    k_tiles = -(-k // tile_k)
    if k % tile_k or tile_k % mad_k or n % tile_n or k % 128:
        raise ValueError(f"this schedule requires whole K and N tiles, got n={n}, k={k}")
    if m > tile_m:
        raise ValueError(f"the fused path serves one row tile, got m={m} over tile_m={tile_m}")
    if 2 * (tile_m + tile_n) * tile_k * 2 * stages > 512 * 1024:
        raise ValueError("L1 operand versions exceed the 512 KiB capacity")
    if 4 * tile_m * tile_n > 256 * 1024:
        raise ValueError("FP32 accumulator exceeds the 256 KiB L0C capacity")

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
            packed_ub = T.alloc_shared((tile_n, tile_k // 2), "uint8")
            weight_ub = T.alloc_shared((tile_n, tile_k), _FP16)
            scale_ub = T.alloc_shared((tile_n, tile_k // 128), _FP16)
            zero_ub = T.alloc_shared((tile_n, tile_k // 128), "uint8")
            T.annotate_buffer_versions(
                {
                    a_l1: stages,
                    b_l1: stages,
                    a_l0: 2,
                    b_l0: 2,
                    packed_ub: stages,
                    weight_ub: stages,
                    scale_ub: stages,
                    zero_ub: stages,
                }
            )
            sched = T.AscendTileScheduler(
                block_m=tile_m,
                block_n=tile_n,
                num_cores=blocks,
                shape_m=m,
                shape_n=n,
            )
            sched.init(bx)
            while sched.valid():
                m_start, n_start, rows, cols = sched.tile()
                for ki in T.Pipelined(k_tiles, num_stages=stages):
                    start = ki * tile_k
                    T.copy(
                        A[m_start : m_start + rows, start : start + tile_k], a_l1[0:rows, 0:tile_k]
                    )
                    # N is a whole multiple of the column tile on this path, so the tile
                    # extent is static even though the scheduler reports it symbolically.
                    T.copy(
                        P[n_start : n_start + tile_n, start // 2 : (start + tile_k) // 2],
                        packed_ub,
                    )
                    T.copy(S[n_start : n_start + tile_n, ki : ki + 1], scale_ub)
                    T.copy(Z[n_start : n_start + tile_n, ki : ki + 1], zero_ub)
                    with T.SimdVF():
                        mask_b8 = T.simd.pset(8, "PAT_VL64")
                        mask_128 = T.simd.pset(8, "PAT_VL128")
                        mask_b16 = T.simd.pset(16, "PAT_VL128")
                        fifteen = T.simd.vdup(T.uint8(15), "uint8")
                        for r in T.serial(tile_n):
                            src = T.simd.make_ubuf_ptr(
                                T.access_ptr(packed_ub[r, 0], "r", extent=tile_k // 2), "uint8"
                            )
                            packed, src = T.simd.vld(src, "NORM_B8", post_inc=64)
                            low = T.simd.vand(packed, fifteen, mask_b8)
                            high = T.simd.vand(T.simd.vshrs(packed, 4, mask_b8), fifteen, mask_b8)
                            stream, _ = T.simd.vintlv(
                                T.reinterpret(low, "uint8x256"), T.reinterpret(high, "uint8x256")
                            )
                            zero = T.simd.vld(zero_ub[r, 0], "BRC_B8")
                            centred = T.simd.vsub(stream, zero, mask_128)
                            wide = T.simd.vunpack(T.reinterpret(centred, "int8x256"), part=0)
                            scale = T.simd.vld(scale_ub[r, 0], "BRC_B16")
                            T.simd.vsts(
                                weight_ub[r, 0],
                                T.simd.vmul(T.simd.vcvt(wide, _FP16, mask_b16), scale, mask_b16),
                                mask_b16,
                                "NORM_B16",
                            )
                    T.copy(weight_ub, b_l1[0:tile_n, 0:tile_k])
                    for m_slice in range(tile_k // mad_k):
                        k_slice = m_slice * mad_k
                        T.copy(a_l1[0:rows, k_slice : k_slice + mad_k], a_l0[0:rows, 0:mad_k])
                        T.copy(b_l1[0:cols, k_slice : k_slice + mad_k], b_l0[0:cols, 0:mad_k])
                        T.gemm(
                            a_l0[0:rows, 0:mad_k],
                            b_l0[0:cols, 0:mad_k],
                            acc[0:rows, 0:cols],
                            transpose_B=True,
                            clear_accum=ki == 0 and m_slice == 0,
                        )
                T.copy(acc[0:rows, 0:cols], D[m_start : m_start + rows, n_start : n_start + cols])
                sched.next_block()

    return main


@functools.lru_cache(maxsize=256)
def _make_dequant(n: int, k: int, group_size: int):
    """Compile the unpack pass for ``(n, k)`` and its group size."""
    return tilelang.compile(
        _dequant_program(n, k, group_size),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


@functools.lru_cache(maxsize=256)
def _make_permute(m: int, k: int):
    """Compile the activation permute for ``(m, k)``."""
    return tilelang.compile(
        _permute_program(m, k),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


@functools.lru_cache(maxsize=256)
def _make_fused(m: int, n: int, k: int):
    """Compile the fused product serving ``(m, n, k)``."""
    tile_m, tile_n, tile_k, stages, blocks = _tiles(m, n, k)
    return tilelang.compile(
        _fused_program(m, n, k, tile_m, tile_n, tile_k, stages, blocks),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


@functools.lru_cache(maxsize=256)
def _make_gemm(m: int, n: int, k: int):
    """Compile the FP16 product serving ``(m, n, k)``."""
    tile_m, tile_n, tile_k, stages, blocks = _tiles(m, n, k)
    return tilelang.compile(
        _gemm_program(m, n, k, tile_m, tile_n, tile_k, stages, blocks),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


class AscendGEMMW4A16Kernel(Kernel, GEMMW4A16FwdInterface):
    """W4A16 NT matmul on the Ascend 950 matrix unit.

    The prepacked weight carries the manifest's marlin-layout INT4 order. Unpacking
    reproduces ``(q - zero) * scale`` in FP16 exactly as the reference dequantizes it,
    and the activation is permuted by the same group-local index so the contraction sees
    both operands in one K order.
    """

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: GEMMW4A16Call) -> Optional[str]:
        if call.group_size != 128:
            return f"this schedule serves a weight group of 128, got {call.group_size}"
        if call.dtype != torch.float16:
            return "requires float16 operands"
        if call.k % 128 or call.n % 16:
            return (
                f"requires K a multiple of 128 and N a multiple of 16, got n={call.n}, k={call.k}"
            )
        return super().refusal(call)

    def __init__(self, call: GEMMW4A16Call):
        device_index = call.device.index
        super().__init__(device_index=device_index)
        name = torch.npu.get_device_name(device_index)
        if "Ascend950" not in name:
            raise RuntimeError(f"These W4A16 schedules require Ascend 950, got {name}")
        # The schedules launch a fixed grid of 32 blocks. Larger devices can run
        # that grid unchanged; do not enlarge it without retuning the schedule.
        if get_num_cube_cores(device_id=device_index or 0) < 32:
            raise RuntimeError("These W4A16 schedules require at least 32 Cube cores")
        self._permute = _make_permute(call.m, call.k)
        # The fused path unpacks each weight tile once per row tile, so it pays only
        # while the call has one row tile; above that the two-pass path wins.
        tile_m, tile_n = _tiles(call.m, call.n, call.k)[0], _tiles(call.m, call.n, call.k)[1]
        # The fused path's unified-memory-to-L1 scatter needs a static column tile, which
        # only the 128-wide-and-up tiles give here.
        self._fused = tile_m >= call.m and tile_n >= 128 and call.n % tile_n == 0
        if self._fused:
            self._gemm = _make_fused(call.m, call.n, call.k)
        else:
            self._dequant = _make_dequant(call.n, call.k, call.group_size)
            self._gemm = _make_gemm(call.m, call.n, call.k)
        self._order = torch.tensor(_permute_index(), dtype=torch.uint16, device=call.device)

    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        permuted = self._permute(activation, self._order)
        if self._fused:
            return self._gemm(permuted, packed_weight, weight_scale, weight_zero)
        weight = self._dequant(packed_weight, weight_scale, weight_zero)
        return self._gemm(permuted, weight)


tileops.backend._registry.register_kernel_type("GEMMW4A16FwdOp", "ascend", AscendGEMMW4A16Kernel)
