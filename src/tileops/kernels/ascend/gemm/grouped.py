"""Grouped GEMM on Ascend 950: DeepGEMM's m-grouped and k-grouped tile walks.

Every group's rows are packed back to back, so a group starts on any row and holds
any number of them. One device-side walk decodes the group of each tile from the
group sizes, which keeps the schedule free of a host-side sync for a group count a
call only knows on the device.
"""

import functools
from typing import Optional

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_cube_cores

import tileops.backend
from tileops.kernels.gemm.grouped.call_spec import (
    GroupedGemmCall,
    GroupedGemmFwdInterface,
)
from tileops.kernels.kernel_base import Kernel

__all__ = ["AscendGroupedGemmKernel"]

_DTYPE_NAMES = {torch.float16: "float16", torch.bfloat16: "bfloat16"}

# One row per served call, ``(numel, experts, n, k, transpose_a, transpose_b)``,
# with the tiles measured for it: ``(tile_m, tile_n, tile_k, stages, blocks)``.
# Under NT/NN the tile is the output tile and the K loop runs over ``k``; under TN
# the tile is the output tile over ``n`` and ``k`` and the K loop runs over the group.
_CONFIGS: dict[tuple[int, int, int, int, bool, bool], tuple[int, int, int, int, int]] = {
    # moe-fwd: 256 rows a group, so one M tile covers the whole group.
    (4096, 16, 4096, 4096, False, True): (256, 256, 256, 2, 32),
    # moe-dgrad: the same tile with the weights stored (k, n).
    (4096, 16, 4096, 4096, False, False): (256, 256, 256, 2, 32),
    # moe-wgrad: the group is the contraction, 256 rows deep; 64 keeps two L0
    # versions of both operands inside the 64 KiB they each own.
    (4096, 16, 4096, 4096, True, False): (256, 256, 64, 1, 32),
    # qwen3-30b: 64 rows a group at most, so a 64-row tile holds any group whole.
    (8, 16, 1536, 2048, False, True): (64, 256, 256, 2, 32),
    (1024, 16, 1536, 2048, False, True): (64, 256, 256, 2, 32),
    (257, 16, 1536, 2048, False, True): (64, 256, 256, 2, 32),
}


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _m_grouped_program(
    numel: int,
    experts: int,
    n: int,
    k: int,
    trans_b: bool,
    dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    stages: int,
    blocks: int,
):
    """Build the m-grouped walk: groups split the rows of ``a`` and ``d``.

    Tiles are taken group-major, so the cores in flight are on one group's weight
    tile and read it once from HBM. The M and N tails narrow the L1 copies, the L0
    stage and the MAD together: the accumulator only ever holds rows and columns the
    store writes, which the fixpipe takes the output pitch from.
    """
    mad_k = 64  # one MAD step: two L0 versions of a 256-row operand take 32 KiB each
    n_tiles = -(-n // tile_n)
    k_tiles = -(-k // tile_k)
    b_shape = (experts, n, k) if trans_b else (experts, k, n)
    if k % tile_k or tile_k % mad_k:
        raise ValueError(f"this schedule requires whole K tiles and MAD slices, got k={k}")
    if 2 * (tile_m + tile_n) * tile_k * stages > 512 * 1024:
        raise ValueError("L1 operand versions exceed the 512 KiB capacity")
    if 4 * tile_m * tile_n > 256 * 1024:
        raise ValueError("FP32 accumulator exceeds the 256 KiB L0C capacity")

    @T.prim_func
    def main(
        A: T.Tensor((numel, k), dtype),
        B: T.Tensor(b_shape, dtype),
        Sizes: T.Tensor((experts,), "int32"),
        Offsets: T.Tensor((experts,), "int32"),
        D: T.Tensor((numel, n), dtype),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(blocks) as bx:
            a_l1 = T.alloc_l1((tile_m, tile_k), dtype)
            b_l1 = T.alloc_l1((tile_n, tile_k), dtype)
            a_l0 = T.alloc_l0a((tile_m, mad_k), dtype)
            b_l0 = T.alloc_l0b((tile_n, mad_k), dtype)
            acc = T.alloc_l0c((tile_m, tile_n), "float32")
            T.annotate_buffer_versions({a_l1: stages, b_l1: stages, a_l0: 2, b_l0: 2})
            total = T.alloc_var("int32", init=0)
            tile = T.alloc_var("int32", init=bx)
            group = T.alloc_var("int32")
            rem = T.alloc_var("int32")
            found = T.alloc_var("int32")
            for g in range(experts):
                total = total + T.ceildiv(Sizes[g], tile_m)
            total = total * n_tiles
            while tile < total:
                # Walk the groups until this tile lands in one.
                group = 0
                rem = tile
                found = 0
                for g in range(experts):
                    if found == 0:
                        if rem >= T.ceildiv(Sizes[g], tile_m) * n_tiles:
                            rem = rem - T.ceildiv(Sizes[g], tile_m) * n_tiles
                            group = group + 1
                        else:
                            found = 1
                m_tile = rem // n_tiles
                n_tile = rem % n_tiles
                row = Offsets[group] + m_tile * tile_m
                rows = T.min(Sizes[group] - m_tile * tile_m, tile_m)
                cols = T.min(n - n_tile * tile_n, tile_n)
                n_start = n_tile * tile_n
                # State what the packed layout guarantees, so the copies are not padded.
                T.assume(group >= 0 and group < experts)
                T.assume(rows > 0 and rows <= tile_m)
                T.assume(row >= 0 and row + rows <= numel)
                T.assume(n_start >= 0 and n_start + cols <= n)
                for ki in T.Pipelined(k_tiles, num_stages=stages):
                    start = ki * tile_k
                    # Tail lanes hold whatever the previous tile left; the L0 stage below
                    # narrows the MAD to the live region, and the store uses the same one.
                    T.copy(A[row : row + rows, start : start + tile_k], a_l1[0:rows, 0:tile_k])
                    if trans_b:
                        T.copy(
                            B[group, n_start : n_start + cols, start : start + tile_k],
                            b_l1[0:cols, 0:tile_k],
                        )
                    else:
                        T.copy(
                            B[group, start : start + tile_k, n_start : n_start + cols],
                            b_l1[0:cols, 0:tile_k],
                            transpose=True,
                        )
                    # One L1 stage feeds tile_k // mad_k MAD steps, so the MTE1 traffic
                    # and the MAD issue overlap inside a stage.
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
                T.copy(acc[0:rows, 0:cols], D[row : row + rows, n_start : n_start + cols])
                tile = tile + blocks

    return main


def _k_grouped_program(
    numel: int,
    experts: int,
    n: int,
    k: int,
    dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    stages: int,
    blocks: int,
):
    """Build the k-grouped walk: the groups split the contraction, the output keeps a group axis.

    Every group owns the whole ``n x k`` output, so all groups share one tile space
    and only the contraction they read differs; that contraction is a row count, so
    each step shrinks the copies and the L0 tiles to the group's own tail. A group
    the routing leaves empty writes zeros: it is a zero tile times a zero tile.
    """
    zero_k = 16  # one MAD step of the 16-bit cube; the shortest legal contraction
    n_tiles = -(-n // tile_m)
    k_tiles = -(-k // tile_n)
    if 4 * max(tile_m, tile_n) * tile_k > 64 * 1024:
        raise ValueError("two 16-bit L0 versions exceed the 64 KiB per-operand capacity")
    if 2 * (tile_m + tile_n) * tile_k * stages > 512 * 1024:
        raise ValueError("L1 operand versions exceed the 512 KiB capacity")
    if 4 * tile_m * tile_n > 256 * 1024:
        raise ValueError("FP32 accumulator exceeds the 256 KiB L0C capacity")

    @T.prim_func
    def main(
        A: T.Tensor((numel, n), dtype),
        B: T.Tensor((numel, k), dtype),
        Sizes: T.Tensor((experts,), "int32"),
        Offsets: T.Tensor((experts,), "int32"),
        ZeroA: T.Tensor((tile_m, zero_k), dtype),
        ZeroB: T.Tensor((tile_n, zero_k), dtype),
        D: T.Tensor((experts, n, k), dtype),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(blocks) as bx:
            # The group axis is the contraction, so both operands are staged in their
            # GM orientation and transposed on the way into L0.
            a_l1 = T.alloc_l1((tile_k, tile_m), dtype)
            b_l1 = T.alloc_l1((tile_k, tile_n), dtype)
            a_l0 = T.alloc_l0a((tile_m, tile_k), dtype)
            b_l0 = T.alloc_l0b((tile_n, tile_k), dtype)
            z_a = T.alloc_l1((tile_m, zero_k), dtype)
            z_b = T.alloc_l1((tile_n, zero_k), dtype)
            acc = T.alloc_l0c((tile_m, tile_n), "float32")
            T.annotate_buffer_versions({a_l1: stages, b_l1: stages, a_l0: 2, b_l0: 2})
            T.copy(ZeroA, z_a)
            T.copy(ZeroB, z_b)
            total = experts * n_tiles * k_tiles
            # The decoded tile state is named scalars, read as loads inside the copies
            # the walk nests below it.
            tile = T.alloc_var("int32", init=bx)
            group = T.alloc_var("int32")
            m_tile = T.alloc_var("int32")
            n_tile = T.alloc_var("int32")
            rows = T.alloc_var("int32")
            cols = T.alloc_var("int32")
            start = T.alloc_var("int32")
            depth = T.alloc_var("int32")
            while tile < total:
                group = tile // (n_tiles * k_tiles)
                m_tile = tile % (n_tiles * k_tiles) // k_tiles
                n_tile = tile % k_tiles
                rows = T.min(n - m_tile * tile_m, tile_m)
                cols = T.min(k - n_tile * tile_n, tile_n)
                # State what the packed layout guarantees, so the copies are not padded.
                T.assume(group >= 0 and group < experts)
                T.assume(m_tile >= 0 and m_tile < n_tiles)
                T.assume(n_tile >= 0 and n_tile < k_tiles)
                T.assume(rows > 0 and rows <= tile_m)
                T.assume(cols > 0 and cols <= tile_n)
                T.assume(m_tile * tile_m + rows <= n)
                T.assume(n_tile * tile_n + cols <= k)
                # Open every tile with a zero contraction, so a group the routing left
                # empty stores zeros through the same path as any other tile.
                T.copy(z_a, a_l0[0:tile_m, 0:zero_k])
                T.copy(z_b, b_l0[0:tile_n, 0:zero_k])
                T.gemm(
                    a_l0[0:tile_m, 0:zero_k],
                    b_l0[0:tile_n, 0:zero_k],
                    acc,
                    transpose_B=True,
                    clear_accum=True,
                )
                for step in range(T.ceildiv(Sizes[group], tile_k)):
                    start = Offsets[group] + step * tile_k
                    depth = T.min(Sizes[group] - step * tile_k, tile_k)
                    T.assume(depth > 0 and depth <= tile_k)
                    T.assume(start >= 0 and start + depth <= numel)
                    T.copy(
                        A[start : start + depth, m_tile * tile_m : m_tile * tile_m + rows],
                        a_l1[0:depth, 0:rows],
                    )
                    T.copy(
                        B[start : start + depth, n_tile * tile_n : n_tile * tile_n + cols],
                        b_l1[0:depth, 0:cols],
                    )
                    T.copy(a_l1[0:depth, 0:rows], a_l0[0:rows, 0:depth], transpose=True)
                    T.copy(b_l1[0:depth, 0:cols], b_l0[0:cols, 0:depth], transpose=True)
                    T.gemm(
                        a_l0[0:rows, 0:depth],
                        b_l0[0:cols, 0:depth],
                        acc[0:rows, 0:cols],
                        transpose_B=True,
                        clear_accum=False,
                    )
                T.copy(
                    acc[0:rows, 0:cols],
                    D[
                        group,
                        m_tile * tile_m : m_tile * tile_m + rows,
                        n_tile * tile_n : n_tile * tile_n + cols,
                    ],
                )
                tile = tile + blocks

    return main


@functools.lru_cache(maxsize=32)
def _make_grouped_kernel(
    numel: int, experts: int, n: int, k: int, transpose_a: bool, transpose_b: bool, dtype: str
):
    """Compile the schedule serving ``(numel, experts, n, k)``, its layouts and ``dtype``."""
    tile_m, tile_n, tile_k, stages, blocks = _CONFIGS[
        numel, experts, n, k, transpose_a, transpose_b
    ]
    if transpose_a:
        program = _k_grouped_program(
            numel, experts, n, k, dtype, tile_m, tile_n, tile_k, stages, blocks
        )
    else:
        program = _m_grouped_program(
            numel, experts, n, k, transpose_b, dtype, tile_m, tile_n, tile_k, stages, blocks
        )
    return tilelang.compile(
        program,
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


class AscendGroupedGemmKernel(Kernel, GroupedGemmFwdInterface):
    """One GEMM per group on the Ascend 950 matrix unit, rows packed without padding.

    Each group's rows are the contiguous slice ``[batch_offsets[g],
    batch_offsets[g] + batch_sizes[g])`` of ``a``; a group may hold no rows. The
    contraction accumulates in float32 and is written in the operand dtype.
    """

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: GroupedGemmCall) -> Optional[str]:
        if call.dtype not in _DTYPE_NAMES:
            return "requires float16 or bfloat16 operands"
        if call.transpose_a and call.transpose_b:
            return "no schedule serves transpose_a=True with transpose_b=True"
        shape = (call.numel, call.num_experts, call.n, call.k, call.transpose_a, call.transpose_b)
        if shape not in _CONFIGS:
            return (
                f"no schedule for numel={call.numel}, experts={call.num_experts}, "
                f"n={call.n}, k={call.k}, transpose_a={call.transpose_a}, "
                f"transpose_b={call.transpose_b}"
            )
        return super().refusal(call)

    def __init__(self, call: GroupedGemmCall):
        device_index = call.device.index
        super().__init__(device_index=device_index)
        name = torch.npu.get_device_name(device_index)
        if "Ascend950" not in name:
            raise RuntimeError(f"These grouped GEMM schedules require Ascend 950, got {name}")
        # The schedules launch a fixed grid of 32 blocks. Larger devices can run
        # that grid unchanged; do not enlarge it without retuning the schedule.
        if get_num_cube_cores(device_id=device_index or 0) < 32:
            raise RuntimeError("These grouped GEMM schedules require at least 32 Cube cores")
        dtype = _dtype_name(call.dtype)
        self._compiled_fn = _make_grouped_kernel(
            call.numel,
            call.num_experts,
            call.n,
            call.k,
            call.transpose_a,
            call.transpose_b,
            dtype,
        )
        # The k-grouped walk opens every tile by multiplying these two zero tiles.
        self._zero_tiles = ()
        if call.transpose_a:
            tile_m, tile_n = _CONFIGS[
                call.numel, call.num_experts, call.n, call.k, call.transpose_a, call.transpose_b
            ][:2]
            self._zero_tiles = (
                torch.zeros((tile_m, 16), dtype=call.dtype, device=call.device),
                torch.zeros((tile_n, 16), dtype=call.dtype, device=call.device),
            )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
    ) -> torch.Tensor:
        return self._compiled_fn(a, b, batch_sizes, batch_offsets, *self._zero_tiles)


tileops.backend.register_kernel_type("GroupedGemmFwdOp", "ascend", AscendGroupedGemmKernel)
