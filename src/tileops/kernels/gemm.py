"""Dense GEMM Cube kernel for Ascend 910B1.

The public operands retain the four TileOPs storage layouts.  Tiles are copied
to L1 in the layout consumed by ``T.gemm_v0`` and accumulated in fp32 L0C.
"""

from functools import lru_cache
from typing import Callable

import tilelang
import tilelang.language as T

from .common import grid_repeat_count, launch_block_count


def _dtype_name(dtype) -> str:
    if dtype == __import__("torch").float16:
        return "float16"
    if dtype == __import__("torch").bfloat16:
        return "bfloat16"
    raise TypeError(f"GemmFwdOp supports torch.float16/torch.bfloat16, got {dtype}")


@lru_cache(maxsize=64)
def _build_gemm(m: int, n: int, k: int, dtype_name: str, trans_a: bool, trans_b: bool) -> Callable:
    # Fill L0C on the large path (128 * 256 * fp32 = 128 KiB) and keep small
    # matrices from paying for mostly empty tiles.  K_L1=256 lets gemm_v0
    # reuse each GM/L1 transfer across four 64-wide L0 steps.
    block_m = 32 if m < 128 else 128
    block_n = 64 if n < 256 else 256
    block_k = 64 if k < 256 else 256
    k_l0_size = 64
    m_tiles = (m + block_m - 1) // block_m
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    logical_blocks = m_tiles * n_tiles
    launch_blocks = launch_block_count(logical_blocks)
    grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)
    a_l1_shape = (block_k, block_m) if trans_a else (block_m, block_k)
    b_l1_shape = (block_n, block_k) if trans_b else (block_k, block_n)
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _factory(dtype: str = dtype_name) -> Callable:
        a_shape = (k, m) if trans_a else (m, k)
        b_shape = (n, k) if trans_b else (k, n)

        @T.prim_func
        def main(
            a: T.Tensor(a_shape, dtype),
            b: T.Tensor(b_shape, dtype),
            c: T.Tensor((m, n), dtype),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, _):
                for grid_repeat in T.serial(grid_repeats):
                    logical_cid = cid + grid_repeat * launch_blocks
                    if logical_cid < logical_blocks:
                        bm = logical_cid // n_tiles
                        bn = logical_cid % n_tiles
                        m0 = bm * block_m
                        n0 = bn * block_n

                        a_l1 = T.alloc_shared(a_l1_shape, dtype)
                        b_l1 = T.alloc_shared(b_l1_shape, dtype)
                        c_l0 = T.alloc_fragment((block_m, block_n), accum_dtype)

                        for kt in T.serial(k_tiles):
                            k0 = kt * block_k
                            if trans_a:
                                T.copy(a[k0:k0 + block_k, m0:m0 + block_m], a_l1)
                            else:
                                T.copy(a[m0:m0 + block_m, k0:k0 + block_k], a_l1)
                            if trans_b:
                                T.copy(b[n0:n0 + block_n, k0:k0 + block_k], b_l1)
                            else:
                                T.copy(b[k0:k0 + block_k, n0:n0 + block_n], b_l1)

                            T.gemm_v0(
                                a_l1,
                                b_l1,
                                c_l0,
                                transpose_A=trans_a,
                                transpose_B=trans_b,
                                init=(kt == 0),
                                kL0Size=k_l0_size,
                            )

                        T.copy(c_l0, c[m0:m0 + block_m, n0:n0 + block_n])

        return main

    return _factory(dtype_name)


def build_gemm_kernel(
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    dtype,
    trans_a: bool,
    trans_b: bool,
) -> Callable:
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError(f"GemmFwdOp expects 2D inputs, got a={a_shape}, b={b_shape}")
    m = a_shape[1] if trans_a else a_shape[0]
    k_a = a_shape[0] if trans_a else a_shape[1]
    n = b_shape[0] if trans_b else b_shape[1]
    k_b = b_shape[1] if trans_b else b_shape[0]
    if k_a != k_b:
        raise ValueError(f"GemmFwdOp contraction mismatch: a={a_shape}, b={b_shape}")
    return _build_gemm(m, n, k_a, _dtype_name(dtype), bool(trans_a), bool(trans_b))


@lru_cache(maxsize=32)
def _build_bmm(batch: int, m: int, n: int, k: int, dtype_name: str) -> Callable:
    block_m = 32 if m < 128 else 128
    block_n = 64 if n < 256 else 256
    block_k = 64 if k < 256 else 256
    m_tiles = (m + block_m - 1) // block_m
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    mn_tiles = m_tiles * n_tiles
    logical_blocks = batch * mn_tiles
    launch_blocks = launch_block_count(logical_blocks)
    grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _factory(dtype: str = dtype_name) -> Callable:
        @T.prim_func
        def main(
            a: T.Tensor((batch, m, k), dtype),
            b: T.Tensor((batch, k, n), dtype),
            c: T.Tensor((batch, m, n), dtype),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, _):
                for grid_repeat in T.serial(grid_repeats):
                    logical_cid = cid + grid_repeat * launch_blocks
                    if logical_cid < logical_blocks:
                        batch_idx = logical_cid // mn_tiles
                        mn_idx = logical_cid % mn_tiles
                        m0 = (mn_idx // n_tiles) * block_m
                        n0 = (mn_idx % n_tiles) * block_n
                        a_l1 = T.alloc_shared((block_m, block_k), dtype)
                        b_l1 = T.alloc_shared((block_k, block_n), dtype)
                        c_l0 = T.alloc_fragment((block_m, block_n), "float")

                        for kt in T.serial(k_tiles):
                            k0 = kt * block_k
                            T.copy(a[batch_idx, m0:m0 + block_m, k0:k0 + block_k], a_l1)
                            T.copy(b[batch_idx, k0:k0 + block_k, n0:n0 + block_n], b_l1)
                            T.gemm_v0(
                                a_l1,
                                b_l1,
                                c_l0,
                                init=(kt == 0),
                                kL0Size=64,
                            )

                        T.copy(c_l0, c[batch_idx, m0:m0 + block_m, n0:n0 + block_n])

        return main

    return _factory(dtype_name)


def build_bmm_kernel(a_shape: tuple[int, ...], b_shape: tuple[int, ...], dtype) -> Callable:
    if len(a_shape) != 3 or len(b_shape) != 3:
        raise ValueError(f"BmmFwdOp expects 3D inputs, got a={a_shape}, b={b_shape}")
    batch, m, k = a_shape
    if b_shape[0] != batch or b_shape[1] != k:
        raise ValueError(f"BmmFwdOp shape mismatch: a={a_shape}, b={b_shape}")
    if k % 16:
        raise ValueError(f"BmmFwdOp requires K divisible by 16, got K={k}")
    return _build_bmm(batch, m, b_shape[2], k, _dtype_name(dtype))


@lru_cache(maxsize=32)
def _build_grouped_gemm(
    batch_sum: int,
    batch_count: int,
    n: int,
    k: int,
    dtype_name: str,
    transpose_a: bool,
    transpose_b: bool,
) -> Callable:
    block_k = 64

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _factory(dtype: str = dtype_name) -> Callable:
        if not transpose_a:
            block_m = 64
            block_n = 128
            max_m_tiles = batch_sum // block_m + batch_count
            n_tiles = (n + block_n - 1) // block_n
            k_tiles = (k + block_k - 1) // block_k
            logical_blocks = batch_count * max_m_tiles * n_tiles
            launch_blocks = launch_block_count(logical_blocks)
            grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)
            b_shape = (batch_count, n, k) if transpose_b else (batch_count, k, n)
            b_l1_shape = (block_n, block_k) if transpose_b else (block_k, block_n)

            @T.prim_func
            def main(
                a: T.Tensor((batch_sum, k), dtype),
                b: T.Tensor(b_shape, dtype),
                batch_sizes: T.Tensor((batch_count,), "int32"),
                batch_offsets: T.Tensor((batch_count,), "int32"),
                batch_padded_offsets: T.Tensor((batch_count,), "int32"),
                output: T.Tensor((batch_sum, n), dtype),
            ):
                with T.Kernel(launch_blocks, is_npu=True) as (cid, _):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            group_tile = logical_cid // n_tiles
                            n_tile = logical_cid % n_tiles
                            prefix = T.alloc_var("int32", init=0)
                            group_idx_var = T.alloc_var("int32", init=0)
                            row_var = T.alloc_var("int32", init=0)
                            for group_scan in T.serial(batch_count):
                                group_tiles = (batch_sizes[group_scan] + block_m - 1) // block_m
                                if (group_tile >= prefix) & (group_tile < prefix + group_tiles):
                                    group_idx_var = group_scan
                                    row_var = (group_tile - prefix) * block_m
                                prefix = prefix + group_tiles
                            group_idx = group_idx_var
                            row_in_group = row_var
                            n0 = n_tile * block_n
                            m0 = batch_offsets[group_idx] + row_in_group
                            valid_m = T.min(block_m, batch_sizes[group_idx] - row_in_group)
                            if row_in_group < batch_sum:
                                a_l1 = T.alloc_shared((block_m, block_k), dtype)
                                b_l1 = T.alloc_shared(b_l1_shape, dtype)
                                c_l0 = T.alloc_fragment((block_m, block_n), "float")

                                for kt in T.serial(k_tiles):
                                    k0 = kt * block_k
                                    for i, j in T.Parallel(block_m, block_k):
                                        a_l1[i, j] = T.if_then_else(
                                            (i < valid_m) & (k0 + j < k),
                                            a[m0 + i, k0 + j],
                                            T.cast(0, dtype),
                                        )
                                    if transpose_b:
                                        for i, j in T.Parallel(block_n, block_k):
                                            b_l1[i, j] = T.if_then_else(
                                                (n0 + i < n) & (k0 + j < k),
                                                b[group_idx, n0 + i, k0 + j],
                                                T.cast(0, dtype),
                                            )
                                    else:
                                        for i, j in T.Parallel(block_k, block_n):
                                            b_l1[i, j] = T.if_then_else(
                                                (k0 + i < k) & (n0 + j < n),
                                                b[group_idx, k0 + i, n0 + j],
                                                T.cast(0, dtype),
                                            )
                                    T.gemm_v0(
                                        a_l1,
                                        b_l1,
                                        c_l0,
                                        transpose_B=transpose_b,
                                        init=(kt == 0),
                                        kL0Size=64,
                                    )

                                for i, j in T.Parallel(block_m, block_n):
                                    if (i < valid_m) & (n0 + j < n):
                                        output[m0 + i, n0 + j] = c_l0[i, j]

        else:
            block_n = 64
            block_out_k = 64
            n_tiles = (n + block_n - 1) // block_n
            out_k_tiles = (k + block_out_k - 1) // block_out_k
            max_m_tiles = (batch_sum + block_k - 1) // block_k
            group_output_tiles = n_tiles * out_k_tiles
            logical_blocks = batch_count * group_output_tiles
            launch_blocks = launch_block_count(logical_blocks)
            grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)
            b_shape = (k, batch_sum) if transpose_b else (batch_sum, k)
            b_l1_shape = (block_out_k, block_k) if transpose_b else (block_k, block_out_k)

            @T.prim_func
            def main(
                a: T.Tensor((batch_sum, n), dtype),
                b: T.Tensor(b_shape, dtype),
                batch_sizes: T.Tensor((batch_count,), "int32"),
                batch_offsets: T.Tensor((batch_count,), "int32"),
                batch_padded_offsets: T.Tensor((batch_count,), "int32"),
                output: T.Tensor((batch_count, n, k), dtype),
            ):
                with T.Kernel(launch_blocks, is_npu=True) as (cid, _):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            group_idx = logical_cid // group_output_tiles
                            output_tile = logical_cid % group_output_tiles
                            n0 = (output_tile // out_k_tiles) * block_n
                            out_k0 = (output_tile % out_k_tiles) * block_out_k
                            group_start = batch_offsets[group_idx]
                            group_size = batch_sizes[group_idx]
                            a_l1 = T.alloc_shared((block_k, block_n), dtype)
                            b_l1 = T.alloc_shared(b_l1_shape, dtype)
                            c_l0 = T.alloc_fragment((block_n, block_out_k), "float")

                            for mt in T.serial(max_m_tiles):
                                m0 = mt * block_k
                                for i, j in T.Parallel(block_k, block_n):
                                    a_l1[i, j] = T.if_then_else(
                                        (m0 + i < group_size) & (n0 + j < n),
                                        a[group_start + m0 + i, n0 + j],
                                        T.cast(0, dtype),
                                    )
                                if transpose_b:
                                    for i, j in T.Parallel(block_out_k, block_k):
                                        b_l1[i, j] = T.if_then_else(
                                            (out_k0 + i < k) & (m0 + j < group_size),
                                            b[out_k0 + i, group_start + m0 + j],
                                            T.cast(0, dtype),
                                        )
                                else:
                                    for i, j in T.Parallel(block_k, block_out_k):
                                        b_l1[i, j] = T.if_then_else(
                                            (m0 + i < group_size) & (out_k0 + j < k),
                                            b[group_start + m0 + i, out_k0 + j],
                                            T.cast(0, dtype),
                                        )
                                T.gemm_v0(
                                    a_l1,
                                    b_l1,
                                    c_l0,
                                    transpose_A=True,
                                    transpose_B=transpose_b,
                                    init=(mt == 0),
                                    kL0Size=64,
                                )

                            for i, j in T.Parallel(block_n, block_out_k):
                                if (n0 + i < n) & (out_k0 + j < k):
                                    output[group_idx, n0 + i, out_k0 + j] = c_l0[i, j]

        return main

    return _factory(dtype_name)


def build_grouped_gemm_kernel(
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    batch_count: int,
    dtype,
    transpose_a: bool,
    transpose_b: bool,
) -> Callable:
    if len(a_shape) != 2:
        raise ValueError(f"GroupedGemmFwdOp expects 2D a, got {a_shape}")
    batch_sum = a_shape[0]
    if transpose_a:
        if len(b_shape) != 2:
            raise ValueError(f"GroupedGemmFwdOp expects 2D b for transpose_a=True, got {b_shape}")
        n = a_shape[1]
        k = b_shape[0] if transpose_b else b_shape[1]
        b_batch_sum = b_shape[1] if transpose_b else b_shape[0]
        if b_batch_sum != batch_sum:
            raise ValueError(f"GroupedGemmFwdOp batch_sum mismatch: a={a_shape}, b={b_shape}")
    else:
        if len(b_shape) != 3 or b_shape[0] != batch_count:
            raise ValueError(f"GroupedGemmFwdOp expects b batch {batch_count}, got {b_shape}")
        k = a_shape[1]
        n = b_shape[1] if transpose_b else b_shape[2]
        b_k = b_shape[2] if transpose_b else b_shape[1]
        if b_k != k:
            raise ValueError(f"GroupedGemmFwdOp K mismatch: a={a_shape}, b={b_shape}")
    # Route 3: runtime group boundaries are scheduled by the Python adapter,
    # while every group's arithmetic is still performed by the proven dense
    # GEMM kernel.  This intentionally launches once per group; it is not a
    # single-kernel grouped GEMM implementation.
    dtype_name = _dtype_name(dtype)
    import torch

    def invoke(a, b, batch_sizes, batch_offsets, batch_padded_offsets=None):
        del batch_padded_offsets  # retained in the public five-input ABI
        if tuple(a.shape) != tuple(a_shape) or tuple(b.shape) != tuple(b_shape):
            raise ValueError("GroupedGemmFwdOp runtime tensor shape mismatch")
        sizes = batch_sizes.detach().cpu().tolist()
        offsets = batch_offsets.detach().cpu().tolist()
        if len(sizes) != batch_count or len(offsets) != batch_count:
            raise ValueError("GroupedGemmFwdOp metadata length mismatch")
        chunks = []
        for group_idx, (offset, size) in enumerate(zip(offsets, sizes)):
            offset, size = int(offset), int(size)
            if offset < 0 or size < 0 or offset + size > batch_sum:
                raise ValueError("GroupedGemmFwdOp metadata has an out-of-range group")
            if size == 0:
                if not transpose_a:
                    chunks.append(torch.empty((0, n), dtype=a.dtype, device=a.device))
                else:
                    chunks.append(torch.zeros((n, k), dtype=a.dtype, device=a.device))
                continue
            if not transpose_a:
                a_group = a[offset : offset + size, :]
                b_group = b[group_idx]
                kernel = build_gemm_kernel(
                    tuple(a_group.shape),
                    tuple(b_group.shape),
                    dtype,
                    False,
                    bool(transpose_b),
                )
                chunk = kernel(a_group, b_group)
            else:
                a_group = a[offset : offset + size, :]
                b_group = b[:, offset : offset + size] if transpose_b else b[offset : offset + size, :]
                kernel = build_gemm_kernel(
                    tuple(a_group.shape),
                    tuple(b_group.shape),
                    dtype,
                    True,
                    bool(transpose_b),
                )
                chunk = kernel(a_group, b_group)
            chunks.append(chunk)

        if not chunks:
            shape = (batch_sum, n) if not transpose_a else (batch_count, n, k)
            return torch.empty(shape, dtype=a.dtype, device=a.device)
        if not transpose_a:
            output = torch.empty((batch_sum, n), dtype=a.dtype, device=a.device)
            for (offset, size), chunk in zip(zip(offsets, sizes), chunks):
                if int(size):
                    output[int(offset) : int(offset) + int(size)].copy_(chunk)
            return output
        return torch.stack(chunks, dim=0)

    return invoke


@lru_cache(maxsize=32)
def _build_gemm_w4a16(m: int, n: int, k: int) -> Callable:
    block_m = 32 if m < 64 else 64
    block_n = 64
    block_k = 128
    m_tiles = (m + block_m - 1) // block_m
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = k // block_k
    logical_blocks = m_tiles * n_tiles
    launch_blocks = launch_block_count(logical_blocks)
    grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        },
        compile_flags=["-O3"],
    )
    def _factory() -> Callable:
        @T.prim_func
        def main(
            activation: T.Tensor((m, k), "float16"),
            packed_weight: T.Tensor((n, k // 2), "uint8"),
            weight_scale: T.Tensor((n, k // 128), "float32"),
            weight_zero: T.Tensor((n, k // 128), "uint8"),
            output: T.Tensor((m, n), "float16"),
        ):
            with T.Kernel(launch_blocks, threads=2, is_npu=True) as (cid):
                for grid_repeat in T.serial(grid_repeats):
                    logical_cid = cid + grid_repeat * launch_blocks
                    if logical_cid < logical_blocks:
                        m0 = (logical_cid // n_tiles) * block_m
                        n0 = (logical_cid % n_tiles) * block_n
                        activation_l1 = T.alloc_shared((block_m, block_k), "float16")
                        packed_ub = T.alloc_shared((block_n, block_k // 2), "uint8")
                        scale_ub = T.alloc_shared((block_n, 1), "float32")
                        zero_ub = T.alloc_shared((block_n, 1), "uint8")
                        weight_l1 = T.alloc_shared((block_n, block_k), "float16")
                        output_l0 = T.alloc_fragment((block_m, block_n), "float")

                        for kt in T.serial(k_tiles):
                            k0 = kt * block_k
                            T.copy(
                                activation[m0:m0 + block_m, k0:k0 + block_k],
                                activation_l1,
                            )
                            T.copy(
                                packed_weight[n0:n0 + block_n, k0 // 2:(k0 + block_k) // 2],
                                packed_ub,
                            )
                            T.copy(weight_scale[n0:n0 + block_n, kt:kt + 1], scale_ub)
                            T.copy(weight_zero[n0:n0 + block_n, kt:kt + 1], zero_ub)
                            for i, j in T.Parallel(block_n, block_k):
                                packed = T.cast(packed_ub[i, j // 2], "int32")
                                quantized = T.if_then_else(j % 2 == 0, packed % 16, packed // 16)
                                weight_l1[i, j] = T.cast(
                                    (
                                        T.cast(quantized, "float")
                                        - T.cast(T.cast(zero_ub[i, 0], "int32"), "float")
                                    )
                                    * scale_ub[i, 0],
                                    "float16",
                                )
                            T.gemm_v0(
                                activation_l1,
                                weight_l1,
                                output_l0,
                                transpose_B=True,
                                init=(kt == 0),
                                kL0Size=64,
                            )

                        T.copy(output_l0, output[m0:m0 + block_m, n0:n0 + block_n])

        return main

    return _factory()


def build_gemm_w4a16_kernel(
    activation_shape: tuple[int, ...],
    packed_weight_shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
    zero_shape: tuple[int, ...],
    activation_dtype,
    packed_dtype,
    scale_dtype,
    zero_dtype,
    group_size: int,
) -> Callable:
    import torch

    if activation_dtype != torch.float16:
        raise TypeError(f"GemmW4A16FwdOp expects float16 activation, got {activation_dtype}")
    if packed_dtype != torch.uint8 or zero_dtype != torch.uint8 or scale_dtype != torch.float32:
        raise TypeError("GemmW4A16FwdOp expects uint8 packed/zero and float32 scale")
    if group_size != 128:
        raise ValueError(f"GemmW4A16FwdOp supports group_size=128, got {group_size}")
    if len(activation_shape) != 2 or len(packed_weight_shape) != 2:
        raise ValueError("GemmW4A16FwdOp expects 2D activation and packed_weight")
    m, k = activation_shape
    n, packed_k = packed_weight_shape
    expected_meta = (n, k // group_size)
    if k % group_size or packed_k != k // 2:
        raise ValueError(f"GemmW4A16FwdOp packed shape mismatch: activation={activation_shape}, packed={packed_weight_shape}")
    if scale_shape != expected_meta or zero_shape != expected_meta:
        raise ValueError(f"GemmW4A16FwdOp metadata must have shape {expected_meta}")
    return _build_gemm_w4a16(m, n, k)
