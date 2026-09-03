"""Three-launch, multi-core Ascend scans.

Each row is split into independent chunks. The first launch writes fp32 local
prefixes and chunk totals, the second scans the short totals array, and the
third applies the preceding-chunk carry and casts at the output boundary.
"""

from functools import lru_cache
import math

import tilelang
import tilelang.language as T
import torch

from .common import MAX_BLOCK_COUNT


_NUM_AI_CORES = 25
_CHUNK = 2048
# Concurrent scalar GM stores need disjoint 32-byte sectors on dav-2201.
_TOTAL_STRIDE = 8
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise TypeError(f"scan supports float16, bfloat16, and float32; got {dtype}") from exc


@lru_cache(maxsize=64)
def _compile_serial_scan(m: int, n: int, dtype_name: str, op_kind: str):
    identity = 0.0 if op_kind == "sum" else 1.0
    cast_n = ((n + 63) // 64) * 64

    @tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((m, n), dtype_name),
            B: T.Tensor((m, cast_n), dtype_name),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                with T.Scope("V"):
                    accum_values = T.alloc_ub((cast_n,), "float32")
                    output_values = T.alloc_ub((cast_n,), dtype_name)
                    for row in T.serial(m):
                        # dav-2201 has no scalar bf16 cast instruction, so the
                        # row is staged in the public dtype with plain scalar
                        # moves and widened as a whole tile.  output_values is
                        # reused as the staging tile; it is free until the
                        # narrowing cast at the end of the row.
                        T.tile.fill(output_values, 0.0)
                        for col in T.serial(n):
                            output_values[col] = A[row, col]
                        if dtype_name == "float32":
                            T.copy(output_values, accum_values)
                        else:
                            T.tile.cast(
                                accum_values, output_values, "CAST_NONE", cast_n
                            )
                        acc = T.alloc_var("float32", init=identity)
                        acc = identity
                        for col in T.serial(n):
                            value = accum_values[col]
                            if op_kind == "sum":
                                acc = acc + value
                            else:
                                acc = acc * value
                            accum_values[col] = acc
                        if dtype_name == "float32":
                            T.copy(accum_values, B[row, 0])
                        else:
                            T.tile.cast(output_values, accum_values, "CAST_RINT", cast_n)
                            T.copy(output_values, B[row, 0])

        return main

    return factory()


@lru_cache(maxsize=128)
def _compile_local_scan(m: int, n: int, dtype_name: str, op_kind: str):
    local_n = ((n + 7) // 8) * 8
    num_chunks = (n + _CHUNK - 1) // _CHUNK
    logical_blocks = m * num_chunks
    launch_blocks = min(logical_blocks, _NUM_AI_CORES)
    grid_repeats = (logical_blocks + launch_blocks - 1) // launch_blocks
    identity = 0.0 if op_kind == "sum" else 1.0

    @tilelang.jit(out_idx=[1], pass_configs=_PASS_CONFIGS)
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((m, n), dtype_name),
            Local: T.Tensor((m, local_n), "float32"),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                # dav-2201 has no scalar bf16 cast instruction, so the chunk
                # crosses the dtype boundary as a whole tile: staged by bulk
                # copy (or scalar moves on the tail) and widened by one
                # T.tile.cast, exactly as the carry stage already stages Local.
                stage_dt = T.alloc_ub((_CHUNK,), dtype_name)
                stage_fp32 = T.alloc_ub((_CHUNK,), "float32")
                with T.Scope("V"):
                    for repeat in T.serial(grid_repeats):
                        logical_cid = cid + repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            row = logical_cid // num_chunks
                            chunk = logical_cid % num_chunks
                            start = chunk * _CHUNK
                            if start + _CHUNK <= n:
                                T.copy(A[row, start], stage_dt)
                            else:
                                T.tile.fill(stage_dt, identity)
                                for lane in T.serial(_CHUNK):
                                    col = start + lane
                                    if col < n:
                                        stage_dt[lane] = A[row, col]
                            T.barrier_all()
                            if dtype_name == "float32":
                                T.copy(stage_dt, stage_fp32)
                            else:
                                T.tile.cast(
                                    stage_fp32, stage_dt, "CAST_NONE", _CHUNK
                                )
                            T.barrier_all()
                            acc = T.alloc_var("float32", init=identity)
                            acc = identity
                            for lane in T.serial(_CHUNK):
                                col = start + lane
                                if col < n:
                                    value = stage_fp32[lane]
                                    if op_kind == "sum":
                                        acc = acc + value
                                    else:
                                        acc = acc * value
                                    Local[row, col] = acc

        return main

    return factory()


@lru_cache(maxsize=128)
def _compile_totals_scan(m: int, n: int, num_chunks: int, op_kind: str):
    local_n = ((n + 7) // 8) * 8
    identity = 0.0 if op_kind == "sum" else 1.0

    @tilelang.jit(out_idx=[1], pass_configs=_PASS_CONFIGS)
    def factory():
        @T.prim_func
        def main(
            Local: T.Tensor((m, local_n), "float32"),
            Totals: T.Tensor((m, num_chunks, _TOTAL_STRIDE), "float32"),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                with T.Scope("V"):
                    if vid == 0:
                        for row in T.serial(m):
                            acc = T.alloc_var("float32", init=identity)
                            acc = identity
                            for chunk in T.serial(num_chunks):
                                if op_kind == "sum":
                                    acc = acc + Local[
                                        row, T.min((chunk + 1) * _CHUNK, n) - 1
                                    ]
                                else:
                                    acc = acc * Local[
                                        row, T.min((chunk + 1) * _CHUNK, n) - 1
                                    ]
                                Totals[row, chunk, 0] = acc

        return main

    return factory()


@lru_cache(maxsize=128)
def _compile_apply_carry(m: int, n: int, dtype_name: str, op_kind: str):
    local_n = ((n + 7) // 8) * 8
    output_align = 8 if dtype_name == "float32" else 16
    output_n = ((n + output_align - 1) // output_align) * output_align
    num_chunks = (n + _CHUNK - 1) // _CHUNK
    logical_blocks = m * num_chunks
    launch_blocks = min(logical_blocks, _NUM_AI_CORES)
    grid_repeats = (logical_blocks + launch_blocks - 1) // launch_blocks
    identity = 0.0 if op_kind == "sum" else 1.0

    @tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
    def factory():
        @T.prim_func
        def main(
            Local: T.Tensor((m, local_n), "float32"),
            Totals: T.Tensor((m, num_chunks, _TOTAL_STRIDE), "float32"),
            Output: T.Tensor((m, output_n), dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                local_ub = T.alloc_ub((_CHUNK,), "float32")
                result_ub = T.alloc_ub((_CHUNK,), "float32")
                carry_ub = T.alloc_ub((1,), "float32")
                carry_vec = T.alloc_ub((_CHUNK,), "float32")
                output_ub = T.alloc_ub((_CHUNK,), dtype_name)

                with T.Scope("V"):
                    for repeat in T.serial(grid_repeats):
                        logical_cid = cid + repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            row = logical_cid // num_chunks
                            chunk = logical_cid % num_chunks
                            start = chunk * _CHUNK
                            carry_ub[0] = identity
                            if chunk > 0:
                                carry_ub[0] = Totals[row, chunk - 1, 0]
                            full = start + _CHUNK <= n
                            if full:
                                T.copy(Local[row, start], local_ub)
                            else:
                                T.tile.fill(local_ub, identity)
                                for lane in T.serial(_CHUNK):
                                    col = start + lane
                                    if col < n:
                                        local_ub[lane] = Local[row, col]
                            T.barrier_all()
                            T.tile.broadcast(carry_vec, carry_ub)
                            if op_kind == "sum":
                                T.tile.add(result_ub, local_ub, carry_vec)
                            else:
                                T.tile.mul(result_ub, local_ub, carry_vec)
                            if dtype_name == "float32":
                                T.copy(result_ub, output_ub)
                            else:
                                T.tile.cast(output_ub, result_ub, "CAST_RINT", _CHUNK)
                            T.barrier_all()
                            if full:
                                T.copy(output_ub, Output[row, start])
                            else:
                                for lane in T.serial(_CHUNK):
                                    col = start + lane
                                    if col < n:
                                        Output[row, col] = output_ub[lane]

        return main

    return factory()


@lru_cache(maxsize=128)
def _compile_scan(m: int, n: int, dtype_name: str, op_kind: str):
    if op_kind not in {"sum", "prod"}:
        raise ValueError(f"unknown scan kind {op_kind!r}")
    num_chunks = (n + _CHUNK - 1) // _CHUNK
    if num_chunks > MAX_BLOCK_COUNT:
        raise ValueError(
            f"scan axis requires {num_chunks} chunks, exceeding the verified "
            f"{MAX_BLOCK_COUNT}-chunk carry scan; recursive totals scan is required"
        )
    return (
        _compile_local_scan(m, n, dtype_name, op_kind),
        _compile_totals_scan(m, n, num_chunks, op_kind),
        _compile_apply_carry(m, n, dtype_name, op_kind),
    )


def build_scan(x, *, dim=-1, op_kind="sum"):
    """Build a three-launch contiguous scan, retaining the host permute fallback."""
    if x is None:
        raise ValueError("scan requires an input tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"scan supports floating dtypes, got {x.dtype}")
    shape = tuple(int(v) for v in x.shape)
    ndim = len(shape)
    axis = int(dim)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"scan dim {dim!r} out of range for rank {ndim}")
    order = tuple(i for i in range(ndim) if i != axis) + (axis,)
    inverse = tuple(order.index(i) for i in range(ndim))
    rows = math.prod(shape[i] for i in range(ndim) if i != axis)
    cols = shape[axis]
    dtype_name = _dtype_name(x.dtype)

    def prepare(inp):
        view = inp if order == tuple(range(ndim)) else inp.permute(order)
        flat = view.reshape(rows, cols)
        return flat if flat.is_contiguous() else flat.contiguous()

    def finish(result):
        result = result[:, :cols]
        result = result.reshape(tuple(shape[i] for i in order))
        return result if inverse == tuple(range(ndim)) else result.permute(inverse)

    if axis != ndim - 1:
        serial_scan = _compile_serial_scan(rows, cols, dtype_name, op_kind)

        def launch(inp):
            return finish(serial_scan(prepare(inp)))

        launch._scan_prepare = prepare
        launch._scan_finish = finish
        launch._scan_kernels = (serial_scan,)
        launch._scan_geometry = {
            "rows": rows,
            "cols": cols,
            "chunk": cols,
            "num_chunks": 1,
            "launch_blocks": 1,
            "mode": "non-last-axis-host-permute-serial",
        }
        return launch

    local_scan, totals_scan, apply_carry = _compile_scan(
        rows, cols, dtype_name, op_kind
    )

    def launch_stages(flat):
        local = local_scan(flat)
        totals = totals_scan(local)
        return local, totals, apply_carry(local, totals)

    def launch(inp):
        _, _, result = launch_stages(prepare(inp))
        return finish(result)

    launch._scan_prepare = prepare
    launch._scan_finish = finish
    launch._scan_kernels = (local_scan, totals_scan, apply_carry)
    launch._scan_geometry = {
        "rows": rows,
        "cols": cols,
        "chunk": _CHUNK,
        "num_chunks": (cols + _CHUNK - 1) // _CHUNK,
        "launch_blocks": min(
            rows * ((cols + _CHUNK - 1) // _CHUNK), _NUM_AI_CORES
        ),
        "mode": "three-launch",
    }
    return launch


__all__ = ["build_scan"]
