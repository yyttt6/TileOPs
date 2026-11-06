"""Block-aligned MoE routing for Ascend, specialized to 64-route blocks."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._csrc import csrc_include
from tileops.backend import register_kernel_type
from tileops.kernels.ascend.moe.pre_permute import route_pass_configs
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.call_spec import PermuteAlignFwdInterface


def make_sharded(e, grid):
    """Build a dynamic-route kernel with a fixed-size UB working set.

    ``grid`` is the physical AIV launch count; block b handles logical 256-route
    tiles b + k * grid. The first input pass builds per-block histograms, and
    the second reads the input again to scatter routes. UB does not grow with N.
    """
    n = T.dynamic("n")
    p = n + (e + 1) * 63
    blocks = (p + 63) // 64

    @T.prim_func
    def main(
        ids: T.Tensor((n,), "int32"),
        out: T.Tensor((p,), "int32"),
        owner: T.Tensor((blocks,), "int32"),
        count: T.Tensor((1,), "int32"),
        workspace: T.Tensor((grid, e), "int32"),
    ):
        T.func_attr({"ascend.batch_schedule": True})
        with T.Kernel(grid) as block:
            data = T.alloc_shared((512,), "int32")
            positions = T.alloc_shared((256,), "int32")
            hist = T.alloc_shared((e,), "int32")
            row = T.alloc_shared((min(grid, 8), e), "int32")
            starts = T.alloc_shared((e,), "int32")
            totals = T.alloc_shared((e,), "int32")
            bases = T.alloc_shared((e,), "int32")
            total = T.alloc_var("int32")
            x = T.alloc_var("int32")
            tiles = (n - 1) // 256 + 1
            rounds = (tiles - 1) // grid + 1
            with T.SimtVF(threads=128):
                t = T.get_thread_binding()
                for j in T.serial(T.ceildiv(e, 128)):
                    if j * 128 + t < e:
                        hist[j * 128 + t] = 0
                        totals[j * 128 + t] = 0
                        bases[j * 128 + t] = 0
            T.ascend_set_flag("V_S", 0)
            T.ascend_wait_flag("V_S", 0)
            if block < tiles:
                T.copy(
                    ids[block * 256 : block * 256 + T.min(256, n - block * 256)],
                    data[: T.min(256, n - block * 256)],
                )
                T.ascend_set_flag("MTE2_S", 0)
            # Double-buffer input; S_MTE2 permits reuse only after scalar reads.
            for r in T.serial(rounds):
                tile = block + r * grid
                if tile < tiles:
                    if r + 1 < rounds and block + (r + 1) * grid < tiles:
                        next_begin = (block + (r + 1) * grid) * 256
                        if r % 2 == 0:
                            if r >= 1:
                                T.ascend_wait_flag("S_MTE2", 1)
                            T.copy(
                                ids[next_begin : next_begin + T.min(256, n - next_begin)],
                                data[256 : 256 + T.min(256, n - next_begin)],
                            )
                            T.ascend_set_flag("MTE2_S", 1)
                        else:
                            T.ascend_wait_flag("S_MTE2", 0)
                            T.copy(
                                ids[next_begin : next_begin + T.min(256, n - next_begin)],
                                data[: T.min(256, n - next_begin)],
                            )
                            T.ascend_set_flag("MTE2_S", 0)
                    if r % 2 == 0:
                        T.ascend_wait_flag("MTE2_S", 0)
                    else:
                        T.ascend_wait_flag("MTE2_S", 1)
                    length = T.min(256, n - tile * 256)
                    for j in T.serial(length):
                        x = data[(r % 2) * 256 + j]
                        hist[x] = hist[x] + 1
                    if r + 2 < rounds and block + (r + 2) * grid < tiles:
                        if r % 2 == 0:
                            T.ascend_set_flag("S_MTE2", 0)
                        else:
                            T.ascend_set_flag("S_MTE2", 1)
            # Publish each block's complete histogram before the grid barrier.
            T.ascend_set_flag("S_MTE3", 0)
            T.ascend_wait_flag("S_MTE3", 0)
            T.copy(hist, workspace[block, :])
            T.ascend_set_flag("MTE3_S", 0)
            T.ascend_wait_flag("MTE3_S", 0)
            if grid > 1:
                T.call_extern("handle", "tileops::moe_grid_sync")
            # Stage at most eight workspace rows; the event chain protects reuse.
            for group in T.serial(T.ceildiv(grid, 8)):
                first = group * 8
                batch = T.min(8, grid - first)
                T.copy(workspace[first : first + batch, :], row[:batch, :])
                T.ascend_set_flag("MTE2_S", 0)
                T.ascend_wait_flag("MTE2_S", 0)
                T.ascend_set_flag("S_V", 0)
                T.ascend_wait_flag("S_V", 0)
                with T.SimtVF(threads=128):
                    t = T.get_thread_binding()
                    for j in T.serial(T.ceildiv(e, 128)):
                        expert = j * 128 + t
                        if expert < e:
                            for k in T.serial(batch):
                                totals[expert] = totals[expert] + row[k, expert]
                                if first + k < block:
                                    bases[expert] = bases[expert] + row[k, expert]
                T.ascend_set_flag("V_S", 0)
                T.ascend_wait_flag("V_S", 0)
                T.ascend_set_flag("S_MTE2", 0)
                T.ascend_wait_flag("S_MTE2", 0)
            total = 0
            for expert in T.serial(e):
                starts[expert] = total + bases[expert]
                hist[expert] = starts[expert]
                total = total + T.ceildiv(totals[expert], 64) * 64
            # Re-read input; hist now holds the per-expert scatter cursors.
            if block < tiles:
                T.copy(
                    ids[block * 256 : block * 256 + T.min(256, n - block * 256)],
                    data[: T.min(256, n - block * 256)],
                )
                T.ascend_set_flag("MTE2_S", 0)
            for r in T.serial(rounds):
                tile = block + r * grid
                if tile < tiles:
                    if r + 1 < rounds and block + (r + 1) * grid < tiles:
                        next_begin = (block + (r + 1) * grid) * 256
                        if r % 2 == 0:
                            if r >= 1:
                                T.ascend_wait_flag("S_MTE2", 1)
                            T.copy(
                                ids[next_begin : next_begin + T.min(256, n - next_begin)],
                                data[256 : 256 + T.min(256, n - next_begin)],
                            )
                            T.ascend_set_flag("MTE2_S", 1)
                        else:
                            T.ascend_wait_flag("S_MTE2", 0)
                            T.copy(
                                ids[next_begin : next_begin + T.min(256, n - next_begin)],
                                data[: T.min(256, n - next_begin)],
                            )
                            T.ascend_set_flag("MTE2_S", 0)
                    if r % 2 == 0:
                        T.ascend_wait_flag("MTE2_S", 0)
                    else:
                        T.ascend_wait_flag("MTE2_S", 1)
                    length = T.min(256, n - tile * 256)
                    for j in T.serial(length):
                        x = data[(r % 2) * 256 + j]
                        positions[j] = hist[x]
                        hist[x] = hist[x] + 1
                    T.ascend_set_flag("S_V", 0)
                    T.ascend_wait_flag("S_V", 0)
                    with T.SimtVF(threads=128):
                        t = T.get_thread_binding()
                        for j in T.serial(T.ceildiv(length, 128)):
                            idx = j * 128 + t
                            if idx < length:
                                out[positions[idx]] = tile * 256 + idx
                    T.ascend_set_flag("V_S", 0)
                    T.ascend_wait_flag("V_S", 0)
                    if r + 2 < rounds and block + (r + 2) * grid < tiles:
                        if r % 2 == 0:
                            T.ascend_set_flag("S_MTE2", 0)
                        else:
                            T.ascend_set_flag("S_MTE2", 1)
            T.ascend_set_flag("S_V", 0)
            T.ascend_wait_flag("S_V", 0)
            with T.SimtVF(threads=128):
                t = T.get_thread_binding()
                for expert in T.serial(block * e // grid, (block + 1) * e // grid):
                    expert_base = T.alloc_var("int32")
                    expert_base = starts[expert] - bases[expert]
                    expert_count = totals[expert]
                    padded = T.ceildiv(expert_count, 64) * 64
                    for j in T.serial(T.ceildiv(padded - expert_count, 128)):
                        if expert_count + j * 128 + t < padded:
                            out[expert_base + expert_count + j * 128 + t] = n
                    for j in T.serial(T.ceildiv(padded // 64, 128)):
                        if j * 128 + t < padded // 64:
                            owner[expert_base // 64 + j * 128 + t] = expert
                if block == 0:
                    if t == 0:
                        count[0] = total
                    for j in T.serial(T.ceildiv(p - total, 128)):
                        if total + j * 128 + t < p:
                            out[total + j * 128 + t] = n
                    for j in T.serial(T.ceildiv(blocks - total // 64, 128)):
                        if total // 64 + j * 128 + t < blocks:
                            owner[total // 64 + j * 128 + t] = -1

    return main


@functools.lru_cache(maxsize=32)
def _compile(e, grid):
    """Cache the compiled dynamic-N artifact by expert count and physical grid."""
    configs = route_pass_configs()
    if grid > 1:
        configs[tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = [
            *csrc_include("ascend_moe_grid_sync.h"),
            "-U__cpp_concepts",
        ]
    return tilelang.compile(
        make_sharded(e, grid), target="ascend", out_idx=[], pass_configs=configs
    )


def _vector_capacity(device):
    """Return the physical AIV count capped by the device's configured limit."""
    capacity = getattr(torch.npu.get_device_properties(device), "vector_core_num", None)
    if type(capacity) is not int or capacity <= 0:
        raise RuntimeError("NPU vector_core_num is required to select the pure AIV grid")
    getter = getattr(torch.npu, "get_device_limit", None)
    if getter is None:
        return capacity  # Compatibility with torch_npu versions without this API.
    if not callable(getter):
        raise RuntimeError("NPU get_device_limit is not callable")
    limits = getter(device)  # Do not ignore runtime failures or unknown sentinel values.
    limit = limits.get("vector_core_num") if isinstance(limits, dict) else None
    if type(limit) is not int or limit <= 0:
        raise RuntimeError("NPU get_device_limit returned an invalid vector_core_num")
    return min(capacity, limit)


class AscendMoEPermuteAlignKernel(Kernel, PermuteAlignFwdInterface):
    """Run 64-aligned routing for contiguous int32 expert IDs on Ascend."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call):
        if call.block_size != 64:
            return "requires block_size=64"
        if call.num_routes <= 0 or call.num_routes % 8:
            return "requires a positive route count divisible by 8"
        if not 0 < call.num_experts <= 384 or call.num_experts % 8:
            return "supports 8..384 experts divisible by 8"
        if call.num_routes + (call.num_experts + 1) * 63 + 255 > 2**31 - 1:
            return "route count exceeds int32 indexing capacity"
        return super().refusal(call)

    def __init__(self, call, *, grid: int | None = None):
        reason = self.refusal(call)
        if reason:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        device = call.device.index if call.device is not None else torch.npu.current_device()
        capacity = _vector_capacity(device)
        if grid is None:
            grid = 1 if call.num_routes <= 256 else capacity
        if type(grid) is not int or not 1 <= grid <= capacity:
            raise ValueError(f"grid must be between 1 and effective vector_core_num ({capacity})")
        if grid * call.num_experts > 2**31 - 1:
            raise ValueError("workspace indexing exceeds int32 capacity")
        self.aiv_capacity = capacity
        self.device_index = device
        self.grid = grid
        self.kernels = _compile(call.num_experts, grid)

    def forward(self, topk_ids):
        self._require_devices(topk_ids=topk_ids)
        if self.grid > _vector_capacity(self.device_index):
            raise RuntimeError("cached grid exceeds the current device vector limit")
        c = self.call
        if (
            topk_ids.dtype != torch.int32
            or not topk_ids.is_contiguous()
            or topk_ids.numel() != c.num_routes
        ):
            raise ValueError("requires contiguous int32 topk_ids matching the specialization")
        n, e = c.num_routes, c.num_experts
        p = n + (e + 1) * 63
        out = torch.empty((p,), dtype=torch.int32, device=topk_ids.device)
        owner = torch.empty(((p + 63) // 64,), dtype=torch.int32, device=topk_ids.device)
        count = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
        workspace = torch.empty((self.grid, e), dtype=torch.int32, device=topk_ids.device)
        self.kernels(topk_ids.view(-1), out, owner, count, workspace)
        return out, owner, count


register_kernel_type("MoEPermuteAlignFwdOp", "ascend", AscendMoEPermuteAlignKernel)
