"""Ascend 950 implementations for BmmFwdOp."""

import functools
from typing import Optional

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_cube_cores

import tileops.backend
from tileops.kernels.gemm import BmmCall, BmmFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_K = 128
_NUM_STAGES = 2


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


@functools.lru_cache(maxsize=1)
def _calibrated_num_cube_cores(fallback: int = 36) -> int:
    """This SKU's real Cube core count, queried live (not hardcoded)."""
    try:
        queried = get_num_cube_cores(device_id=0)
    except Exception:
        queried = None
    return int(queried) if queried else fallback


@functools.lru_cache(maxsize=64)
def _make_bmm_kernel(batch: int, m: int, n: int, k: int, dtype: str):
    """Build (and cache) the persistent batched-GEMM kernel for one ``(batch, m, n, k,
    dtype)`` combination."""
    block_m, block_n, block_k = (_BLOCK_M, _BLOCK_N, _BLOCK_K)
    m_tiles = -(-m // block_m)
    n_tiles = -(-n // block_n)
    num_k_blocks = -(-k // block_k)
    total_tiles = batch * m_tiles * n_tiles
    num_blocks = min(_calibrated_num_cube_cores(), total_tiles)

    @T.prim_func
    def main(
        A: T.Tensor((batch, m, k), dtype),
        B: T.Tensor((batch, k, n), dtype),
        D: T.Tensor((batch, m, n), dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            sched = T.AscendBatchedTileScheduler(
                block_m=block_m,
                block_n=block_n,
                num_cores=num_blocks,
                shape_m=m,
                shape_n=n,
                num_groups=batch,
            )
            a_l1 = T.alloc_l1((block_m, block_k), dtype)
            b_l1 = T.alloc_l1((block_n, block_k), dtype)
            a_l0 = T.alloc_l0a((block_m, block_k), dtype)
            b_l0 = T.alloc_l0b((block_n, block_k), dtype)
            res = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions(
                {a_l1: _NUM_STAGES, b_l1: _NUM_STAGES, a_l0: _NUM_STAGES, b_l0: _NUM_STAGES}
            )
            sched.init(bx)
            while sched.valid():
                batch_idx = sched.batch()
                m_idx, n_idx, actual_m, actual_n = sched.tile()
                for k_block_idx in T.Pipelined(num_k_blocks, num_stages=_NUM_STAGES):
                    k_idx = k_block_idx * block_k
                    actual_k = T.min(k - k_idx, block_k)
                    T.copy(
                        A[batch_idx, m_idx : m_idx + actual_m, k_idx : k_idx + actual_k],
                        a_l1[0:actual_m, 0:actual_k],
                    )
                    T.copy(
                        B[batch_idx, k_idx : k_idx + actual_k, n_idx : n_idx + actual_n],
                        b_l1[0:actual_n, 0:actual_k],
                        transpose=True,
                    )
                    T.copy(a_l1[0:actual_m, 0:actual_k], a_l0[0:actual_m, 0:actual_k])
                    T.copy(b_l1[0:actual_n, 0:actual_k], b_l0[0:actual_n, 0:actual_k])
                    T.gemm(
                        a_l0[0:actual_m, 0:actual_k],
                        b_l0[0:actual_n, 0:actual_k],
                        res[0:actual_m, 0:actual_n],
                        transpose_B=True,
                        clear_accum=k_block_idx == 0,
                    )
                T.copy(
                    res[0:actual_m, 0:actual_n],
                    D[batch_idx, m_idx : m_idx + actual_m, n_idx : n_idx + actual_n],
                )
                sched.next_block()

    return tilelang.compile(main, target="ascend", out_idx=-1)


_L1_BLOCK_K = (128, 256)
_L1_NUM_STAGES = (1, 2, 3)
_MAX_CUBE_BLOCKS = 36


@functools.lru_cache(maxsize=64)
def _make_bmm_aligned_l1_kernel(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    num_blocks: int,
):
    """Build (and cache) one aligned L1-direct GEMM candidate."""
    if block_m not in (128, 256) or block_n not in (128, 256):
        raise ValueError(f"block_m/block_n must be 128 or 256, got {block_m}/{block_n}")
    if m % block_m or n % block_n or k % _BLOCK_K:
        raise ValueError(
            f"aligned L1 builder requires M, N and K divisible by {_BLOCK_K}, got M={m}, N={n}, K={k}"
        )
    if block_k not in _L1_BLOCK_K:
        raise ValueError(f"block_k must be one of {_L1_BLOCK_K}, got {block_k}")
    if k % block_k:
        raise ValueError(f"block_k={block_k} must divide K={k}")
    if num_stages not in _L1_NUM_STAGES:
        raise ValueError(f"num_stages must be one of {_L1_NUM_STAGES}, got {num_stages}")
    if not 1 <= num_blocks <= _MAX_CUBE_BLOCKS:
        raise ValueError(f"num_blocks must be in [1, {_MAX_CUBE_BLOCKS}], got {num_blocks}")
    m_tiles = m // block_m
    n_tiles = n // block_n
    num_k_blocks = k // block_k
    total_tiles = batch * m_tiles * n_tiles
    grid = min(num_blocks, total_tiles)
    if grid < 1:
        raise ValueError("aligned L1 builder has an empty output tile grid")

    @T.prim_func
    def main(
        A: T.Tensor((batch, m, k), dtype),
        B: T.Tensor((batch, k, n), dtype),
        D: T.Tensor((batch, m, n), dtype),
    ):
        with T.Kernel(grid) as bx:
            sched = T.AscendBatchedTileScheduler(
                block_m=block_m,
                block_n=block_n,
                num_cores=grid,
                shape_m=m,
                shape_n=n,
                num_groups=batch,
            )
            a_l1 = T.alloc_l1((block_m, block_k), dtype)
            b_l1 = T.alloc_l1((block_n, block_k), dtype)
            res = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions({a_l1: num_stages, b_l1: num_stages})
            sched.init(bx)
            while sched.valid():
                batch_idx = sched.batch()
                m_idx, n_idx, actual_m, actual_n = sched.tile()
                for k_block_idx in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    k_idx = k_block_idx * block_k
                    T.copy(
                        A[batch_idx, m_idx : m_idx + actual_m, k_idx : k_idx + block_k],
                        a_l1[0:actual_m, 0:block_k],
                    )
                    T.copy(
                        B[batch_idx, k_idx : k_idx + block_k, n_idx : n_idx + actual_n],
                        b_l1[0:actual_n, 0:block_k],
                        transpose=True,
                    )
                    T.gemm(a_l1, b_l1, res, transpose_B=True, clear_accum=k_block_idx == 0)
                T.copy(
                    res[0:actual_m, 0:actual_n],
                    D[batch_idx, m_idx : m_idx + actual_m, n_idx : n_idx + actual_n],
                )
                sched.next_block()

    return tilelang.compile(main, target="ascend", out_idx=-1)


_ALIGNED_L1_SELECTIONS = {
    (128, 512, 512, 2048, "bfloat16"): (256, 128, 256, 2, 36),
    (8, 1024, 1024, 1024, "float16"): (256, 128, 256, 2, 36),
    (8, 1024, 1024, 1024, "bfloat16"): (256, 128, 256, 2, 36),
    (8, 2048, 2048, 2048, "float16"): (256, 128, 256, 2, 36),
    (8, 2048, 2048, 2048, "bfloat16"): (256, 128, 256, 2, 36),
    (4, 4096, 4096, 4096, "bfloat16"): (256, 128, 256, 2, 36),
}


class AscendBmmKernel(Kernel, BmmFwdInterface):
    """Batched dense GEMM on Ascend NPUs: ``d[i] = a[i] @ b[i]`` per batch item."""

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: BmmCall) -> Optional[str]:
        return None if call.k % 16 == 0 else f"requires k a multiple of 16, got k={call.k}"

    @classmethod
    def entry_for(cls, call: BmmCall) -> Entry:
        return (call, lambda: cls(call.batch, call.m, call.n, call.k, call.dtype))

    def __init__(self, batch: int, m: int, n: int, k: int, dtype: torch.dtype):
        super().__init__()
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        dtype_name = _dtype_str(dtype)
        selection = _ALIGNED_L1_SELECTIONS.get((batch, m, n, k, dtype_name))
        if selection is None:
            self._compiled_fn = _make_bmm_kernel(batch, m, n, k, dtype_name)
        else:
            block_m, block_n, block_k, num_stages, num_blocks = selection
            self._compiled_fn = _make_bmm_aligned_l1_kernel(
                batch, m, n, k, dtype_name, block_m, block_n, block_k, num_stages, num_blocks
            )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._compiled_fn(a, b)


tileops.backend.register_kernel_type("BmmFwdOp", "ascend", AscendBmmKernel)
