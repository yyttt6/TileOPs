"""Three-launch, multi-core Ascend scans.

Each row is split into independent chunks. The first launch writes fp32 local
prefixes and chunk totals, the second scans the short totals array, and the
third applies the preceding-chunk carry and casts at the output boundary.
"""

from functools import lru_cache
import math
import os

import tilelang
import tilelang.language as T
import torch

from .common import MAX_BLOCK_COUNT


_NUM_AI_CORES = 25
#: R347: each AI core carries TWO vector units, addressed by the second element of
#: ``T.Kernel(...) as (cid, vid)``.  The chunked scan stages used to index their work
#: by ``cid`` alone, so ``vid = 0`` and ``vid = 1`` computed the SAME ``logical_cid``
#: and wrote the same bytes -- half the vector units duplicating the other half.
#: ``cid * 2 + vid`` is the in-tree idiom for this (ops/reduction/t301a_kernels.py
#: ``_sort_seed`` / ``_affine``).  ``T.barrier_all()`` is a per-core PIPELINE barrier
#: (tilelang/language/ascend.py:275, ``ascend_pipe_barrier("ALL")``), not a
#: cross-worker one, so a diverging tail guard cannot deadlock on it.
_VEC_PER_CORE = 2
_CHUNK = 2048
#: R357: chunk widths are picked from this grain so that `chunk * itemsize` is always a
#: multiple of the 32-byte vector block for every dtype this file accepts (128 fp16
#: elements = 256 B, 128 fp32 elements = 512 B).  `start = chunk_index * chunk` is then
#: 32-byte aligned too, which is what the full-extent `T.copy` on the GM side needs.
_CHUNK_GRAIN = 128
#: The 50 vector workers a full launch has (25 AI cores x 2 vector units).
_TARGET_WORKERS = _NUM_AI_CORES * _VEC_PER_CORE


def _scan_arm() -> str:
    """T357's control-arm switch.  ``base`` reproduces the pre-T357 tree exactly."""
    return os.environ.get("TILEOPS_T357_ARM", "r357")


def _pick_chunk(rows: int, cols: int) -> int:
    """Chunk width for the three-launch scan.

    R357.  The scan inside one chunk is a SERIAL scalar loop over `chunk` elements, so
    the wall time of stage 1 is (per-worker chunk count) x chunk x (scalar iteration),
    and the number of logical blocks is `rows * ceil(cols / chunk)`.  With `_CHUNK`
    fixed at 2048 the manifest's `[4097]` case produced THREE logical blocks: 2 of the
    25 AI cores ran, and each still paid the full 2048 serial iterations.

    The rule is deliberately one-sided: if the default already produces at least
    `_TARGET_WORKERS` logical blocks the default is returned unchanged, so every shape
    that was already filling the machine compiles a bit-identical kernel.
    """
    if _scan_arm() == "base":
        return _CHUNK
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    if rows * ((cols + _CHUNK - 1) // _CHUNK) >= _TARGET_WORKERS:
        return _CHUNK
    want_chunks = (_TARGET_WORKERS + rows - 1) // rows      # chunks per row wanted
    width = (cols + want_chunks - 1) // want_chunks         # ... and the width that gives
    width = (width // _CHUNK_GRAIN) * _CHUNK_GRAIN
    return max(_CHUNK_GRAIN, min(_CHUNK, width))
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
def _compile_local_scan(m: int, n: int, dtype_name: str, op_kind: str,
                        chunk: int = _CHUNK, tight: bool = False):
    _CHUNK = chunk                       # noqa: F841 -- shadows the module default
    local_n = ((n + 7) // 8) * 8
    num_chunks = (n + _CHUNK - 1) // _CHUNK
    # Only the LAST chunk can be partial, so its extent is a compile-time constant.
    tail_len = n - (num_chunks - 1) * _CHUNK
    logical_blocks = m * num_chunks
    launch_blocks = min((logical_blocks + _VEC_PER_CORE - 1) // _VEC_PER_CORE, _NUM_AI_CORES)
    workers = launch_blocks * _VEC_PER_CORE
    grid_repeats = (logical_blocks + workers - 1) // workers
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
                        logical_cid = cid * _VEC_PER_CORE + vid + repeat * workers
                        if logical_cid < logical_blocks:
                            row = logical_cid // num_chunks
                            chunk = logical_cid % num_chunks
                            start = chunk * _CHUNK
                            if start + _CHUNK <= n:
                                T.copy(A[row, start], stage_dt)
                            else:
                                T.tile.fill(stage_dt, identity)
                                # R357 (tight): `tail_len` is a compile-time constant and
                                # only the last chunk reaches this branch, so the loop no
                                # longer runs `_CHUNK` times with a `col < n` guard that is
                                # false for all but `tail_len` lanes.  Measured motive: the
                                # [4097] case has tail_len == 1 and paid 2048 guarded
                                # scalar iterations here.
                                for lane in T.serial(tail_len if tight else _CHUNK):
                                    col = start + lane
                                    if tight or col < n:
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
                            # R347: the running scan stays in UB.  This loop used
                            # to end with `Local[row, col] = acc`, i.e. ONE SCALAR
                            # GM STORE PER ELEMENT (8.4 M of them on
                            # hidden-state-scan).  The values written are
                            # bit-identical; only the store path changed.
                            #
                            # R357 (tight): the same loop, split so that neither copy
                            # carries the `col < n` guard.  The full-chunk branch is the
                            # hot one (every chunk of every big case); the partial branch
                            # runs `tail_len` times instead of `_CHUNK`.
                            if tight:
                                if start + _CHUNK <= n:
                                    for lane in T.serial(_CHUNK):
                                        value = stage_fp32[lane]
                                        if op_kind == "sum":
                                            acc = acc + value
                                        else:
                                            acc = acc * value
                                        stage_fp32[lane] = acc
                                else:
                                    for lane in T.serial(tail_len):
                                        value = stage_fp32[lane]
                                        if op_kind == "sum":
                                            acc = acc + value
                                        else:
                                            acc = acc * value
                                        stage_fp32[lane] = acc
                            else:
                                for lane in T.serial(_CHUNK):
                                    col = start + lane
                                    if col < n:
                                        value = stage_fp32[lane]
                                        if op_kind == "sum":
                                            acc = acc + value
                                        else:
                                            acc = acc * value
                                        stage_fp32[lane] = acc
                            T.barrier_all()
                            if start + _CHUNK <= n:
                                T.copy(stage_fp32, Local[row, start])
                            else:
                                # Partial-extent UB->GM copies move whole 32-byte
                                # blocks (see the RAGGED-CHUNK note in
                                # normalization_spatial.py), so only the
                                # 8-element-aligned prefix goes out by T.copy;
                                # `start` is a multiple of _CHUNK and therefore of
                                # 8, so [start, start+tail_vec) is block-aligned
                                # and stays inside `local_n = ceil(n/8)*8`.
                                tail = (tail_len if tight else n - start)
                                tail_vec = tail - (tail % 8)
                                if tail_vec:
                                    T.copy(stage_fp32[0:tail_vec],
                                           Local[row, start:start + tail_vec])
                                for lane in T.serial(8):
                                    col = start + tail_vec + lane
                                    if col < n:
                                        Local[row, col] = stage_fp32[tail_vec + lane]
                            # `stage_fp32` is the MTE3 source above and the V
                            # destination of the next repeat's cast: drain the
                            # write-after-read before the loop turns over.
                            T.barrier_all()

        return main

    return factory()


@lru_cache(maxsize=128)
def _compile_totals_scan(m: int, n: int, num_chunks: int, op_kind: str,
                         chunk: int = _CHUNK):
    _CHUNK = chunk                       # noqa: F841 -- shadows the module default
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
def _compile_apply_carry(m: int, n: int, dtype_name: str, op_kind: str,
                         chunk: int = _CHUNK, tight: bool = False):
    _CHUNK = chunk                       # noqa: F841 -- shadows the module default
    local_n = ((n + 7) // 8) * 8
    output_align = 8 if dtype_name == "float32" else 16
    output_n = ((n + output_align - 1) // output_align) * output_align
    num_chunks = (n + _CHUNK - 1) // _CHUNK
    tail_len = n - (num_chunks - 1) * _CHUNK
    logical_blocks = m * num_chunks
    launch_blocks = min((logical_blocks + _VEC_PER_CORE - 1) // _VEC_PER_CORE, _NUM_AI_CORES)
    workers = launch_blocks * _VEC_PER_CORE
    grid_repeats = (logical_blocks + workers - 1) // workers
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
                        logical_cid = cid * _VEC_PER_CORE + vid + repeat * workers
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
                                # R357 (tight): see the twin comment in
                                # `_compile_local_scan`; `tail_len` is compile-time.
                                for lane in T.serial(tail_len if tight else _CHUNK):
                                    col = start + lane
                                    if tight or col < n:
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
                                if tight:
                                    # R357.  This was `_CHUNK` GUARDED SCALAR GM STORES
                                    # (2048 of them for one live element on [4097]).
                                    # `Output` is declared (m, output_n) with
                                    # output_n = ceil(n / output_align) * output_align and
                                    # `start` is a multiple of `_CHUNK` and therefore of
                                    # output_align, so the whole aligned span
                                    # [start, start + out_tail) is inside the buffer.  The
                                    # lanes past `n` receive identity-seeded values and are
                                    # sliced off by `finish` (`result[:, :cols]`), exactly
                                    # as the padding produced by the full-chunk branch on a
                                    # shorter row would be.
                                    out_tail = ((tail_len + output_align - 1)
                                                // output_align) * output_align
                                    T.copy(output_ub[0:out_tail],
                                           Output[row, start:start + out_tail])
                                else:
                                    for lane in T.serial(_CHUNK):
                                        col = start + lane
                                        if col < n:
                                            Output[row, col] = output_ub[lane]

        return main

    return factory()


@lru_cache(maxsize=128)
def _compile_scan(m: int, n: int, dtype_name: str, op_kind: str,
                  chunk: int = _CHUNK, tight: bool = False):
    if op_kind not in {"sum", "prod"}:
        raise ValueError(f"unknown scan kind {op_kind!r}")
    num_chunks = (n + chunk - 1) // chunk
    if num_chunks > MAX_BLOCK_COUNT:
        raise ValueError(
            f"scan axis requires {num_chunks} chunks, exceeding the verified "
            f"{MAX_BLOCK_COUNT}-chunk carry scan; recursive totals scan is required"
        )
    return (
        _compile_local_scan(m, n, dtype_name, op_kind, chunk, tight),
        _compile_totals_scan(m, n, num_chunks, op_kind, chunk),
        _compile_apply_carry(m, n, dtype_name, op_kind, chunk, tight),
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

    chunk_w = _pick_chunk(rows, cols)
    tight = _scan_arm() != "base"
    local_scan, totals_scan, apply_carry = _compile_scan(
        rows, cols, dtype_name, op_kind, chunk_w, tight
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
        "chunk": chunk_w,
        "num_chunks": (cols + chunk_w - 1) // chunk_w,
        "launch_blocks": min(
            (rows * ((cols + chunk_w - 1) // chunk_w) + _VEC_PER_CORE - 1) // _VEC_PER_CORE,
            _NUM_AI_CORES,
        ),
        "vec_per_core": _VEC_PER_CORE,
        "mode": "three-launch",
        "arm": _scan_arm(),
        "tight_tail": tight,
    }
    return launch


__all__ = ["build_scan"]
