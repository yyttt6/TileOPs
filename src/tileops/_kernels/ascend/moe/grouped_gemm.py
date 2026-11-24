"""Build the grouped Cube product with device-resident row boundaries."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.buffer_utils import tensors_overlap
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.moe.call_spec import MGroupedGEMMCall, MGroupedGEMMFwdInterface
from tileops.backend._registry import register_kernel_type


@functools.lru_cache(maxsize=64)
def _product(m, n, k, groups, dtype, output_dtype, kind, max_m, alignment):
    # Small expert groups share weights across all their rows in a tile.
    bm = 16 if m <= groups * 16 else 64 if m <= groups * 64 else 128
    bn = min(128, ((n + 15) // 16) * 16)
    bk = next((size for size in (128, 64, 32, 16) if k % size == 0), 16)
    nt = (n + bn - 1) // bn
    grid = 32
    mad_k = min(64, bk)

    @T.macro
    def multiply_tile(a, b, output, lhs, rhs, al0, bl0, acc, row, col, rows, cols, g):
        for step in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
            first = step * bk
            depth = T.min(k - first, bk)
            T.copy(a[row : row + rows, first : first + depth], lhs[:rows, :depth])
            T.copy(b[g, col : col + cols, first : first + depth], rhs[:cols, :depth])
            for part in T.serial(bk // mad_k):
                off = part * mad_k
                T.copy(lhs[:rows, off : off + mad_k], al0[:rows, :])
                T.copy(rhs[:cols, off : off + mad_k], bl0[:cols, :])
                T.gemm(
                    al0[:rows, :],
                    bl0[:cols, :],
                    acc[:rows, :cols],
                    transpose_B=True,
                    clear_accum=(step == 0 and part == 0),
                )
        T.copy(acc[:rows, :cols], output[row : row + rows, col : col + cols])

    @T.prim_func
    def main(
        a: T.Tensor((m, k), dtype),
        b: T.Tensor((groups, n, k), dtype),
        ends: T.Tensor((m if kind == "per_row" else groups,), "int32"),
        output: T.Tensor((m, n), output_dtype),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(grid) as bx:
            lhs = T.alloc_l1((bm, bk), dtype)
            rhs = T.alloc_l1((bn, bk), dtype)
            al0 = T.alloc_l0a((bm, mad_k), dtype)
            bl0 = T.alloc_l0b((bn, mad_k), dtype)
            acc = T.alloc_l0c((bm, bn), "float32")
            T.annotate_buffer_versions({lhs: 2, rhs: 2, al0: 2, bl0: 2})
            tile = T.alloc_var("int32", init=bx)
            start = T.alloc_var("int32", init=0)
            end = T.alloc_var("int32")
            if kind == "per_row":
                for job in T.serial(T.ceildiv(m * nt, grid)):
                    index = job * grid + bx
                    if index < m * nt:
                        row = index // nt
                        g = ends[row]
                        if (g >= 0) & (g < groups):
                            col = index % nt * bn
                            multiply_tile(
                                a,
                                b,
                                output,
                                lhs,
                                rhs,
                                al0,
                                bl0,
                                acc,
                                row,
                                col,
                                1,
                                T.min(bn, n - col),
                                g,
                            )
            else:
                for g in T.serial(groups):
                    if kind == "masked":
                        start = g * max_m
                        end = start + ends[g]
                    else:
                        start = T.ceildiv(start, alignment) * alignment
                        end = ends[g]
                    jobs = T.ceildiv(T.max(end - start, 0), bm) * nt
                    while tile < jobs:
                        row = start + tile // nt * bm
                        col = (tile % nt) * bn
                        rows = T.min(bm, end - row)
                        cols = T.min(bn, n - col)
                        multiply_tile(
                            a, b, output, lhs, rhs, al0, bl0, acc, row, col, rows, cols, g
                        )
                        tile = tile + grid
                    tile = tile - jobs
                    start = end

    return tilelang.compile(main, target="ascend", out_idx=[])


@functools.lru_cache(maxsize=64)
def _epilogue(m, n, groups, dtype, output_dtype, activation, has_bias, kind, max_m, alignment):
    cols = n // 2 if activation else n
    tile = 256
    tiles = (cols + tile - 1) // tile

    @T.prim_func
    def main(
        product: T.Tensor((m, n), "float32"),
        meta: T.Tensor((m if kind == "per_row" else groups,), "int32"),
        bias: T.Tensor((groups, n), dtype) if has_bias else None,
        output: T.Tensor((m, cols), output_dtype),
    ):
        with T.Kernel(64) as core:
            for job in T.serial(T.ceildiv(m * tiles, 64)):
                idx = job * 64 + core
                if idx < m * tiles:
                    row = idx // tiles
                    start = idx % tiles * tile
                    valid = T.alloc_var("int32", init=1)
                    group = T.alloc_var("int32", init=0)
                    if kind == "masked":
                        group = row // max_m
                        valid = T.cast(row % max_m < meta[group], "int32")
                    elif kind == "per_row":
                        group = meta[row]
                        valid = T.cast((group >= 0) & (group < groups), "int32")
                    else:
                        hi = T.alloc_var("int32", init=groups)
                        while group < hi:
                            middle = (group + hi) // 2
                            if meta[middle] <= row:
                                group = middle + 1
                            else:
                                hi = middle
                        valid = T.cast(group < groups, "int32")
                        if group > 0:  # noqa: SIM102 - guard the group-1 load in device IR.
                            if row < T.ceildiv(meta[group - 1], alignment) * alignment:
                                valid = 0
                    if valid != 0:
                        with T.SimtVF(threads=128):
                            for j in T.Parallel(tile):
                                col = start + j
                                if col < cols:
                                    ucol = 2 * col if activation == "swigluoai" else col
                                    gate = T.alloc_local((1,), "float32")
                                    gate[0] = product[row, ucol]
                                    if has_bias:
                                        gate[0] += T.cast(bias[group, ucol], "float32")
                                    if activation:
                                        vcol = ucol + 1 if activation == "swigluoai" else col + cols
                                        up = T.alloc_local((1,), "float32")
                                        up[0] = product[row, vcol]
                                        if has_bias:
                                            up[0] += T.cast(bias[group, vcol], "float32")
                                        if activation == "swigluoai":
                                            gate[0] = T.min(gate[0], 7.0)
                                            gate[0] *= T.sigmoid(1.702 * gate[0]) * (
                                                1 + T.max(-7.0, T.min(7.0, up[0]))
                                            )
                                        elif activation == "silu_and_mul":
                                            gate[0] *= T.sigmoid(gate[0]) * up[0]
                                        else:
                                            gate[0] *= (
                                                0.5
                                                * (1 + T.erf(gate[0] * 0.7071067811865476))
                                                * up[0]
                                            )
                                    output[row, col] = T.cast(gate[0], output_dtype)

    return tilelang.compile(main, target="ascend", out_idx=[])


class AscendMoEGroupedGEMMKernel(Kernel, MGroupedGEMMFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: MGroupedGEMMCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: MGroupedGEMMCall):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(self, a, b, layout_metadata, out=None, *, bias=None):
        call = self.call
        m = a.numel() // call.k
        activation = call.activation if call.activation != "none" else None
        cols = call.n // 2 if activation else call.n
        shape = (*a.shape[:-1], cols)
        if out is None:
            out = torch.empty(shape, dtype=call.cd_dtype, device=a.device)
        else:
            if (
                tuple(out.shape) != shape
                or out.dtype != call.cd_dtype
                or out.device != a.device
                or not out.is_contiguous()
            ):
                raise ValueError(
                    "out must be contiguous and match the result shape, dtype and device"
                )
            for name, ref in (
                ("a", a),
                ("b", b),
                ("layout_metadata", layout_metadata),
                ("bias", bias),
            ):
                if ref is not None and tensors_overlap(out, ref):
                    raise ValueError(f"out must not overlap {name} in memory")
        if m == 0:
            return out
        dtype = str(call.ab_dtype).removeprefix("torch.")
        dst = str(call.cd_dtype).removeprefix("torch.")
        kind = "masked" if call.kind == "masked" else call.metadata_kind
        epi = bool(activation or bias is not None)
        tmp = (
            torch.empty((m, call.n), dtype=torch.float32, device=a.device)
            if epi
            else out.view(m, cols)
        )
        padded_k = (call.k + 15) // 16 * 16
        if padded_k != call.k:
            a = torch.nn.functional.pad(a, (0, padded_k - call.k))
            b = torch.nn.functional.pad(b, (0, padded_k - call.k))
        gemm = _product(
            m,
            call.n,
            padded_k,
            call.num_groups,
            dtype,
            "float32" if epi else dst,
            kind,
            call.max_m or 1,
            call.alignment,
        )
        gemm(a.view(m, padded_k), b, layout_metadata, tmp)
        if epi:
            program = _epilogue(
                m,
                call.n,
                call.num_groups,
                dtype,
                dst,
                activation,
                bias is not None,
                kind,
                call.max_m or 1,
                call.alignment,
            )
            args = (
                (tmp, layout_metadata, bias, out.view(m, cols))
                if bias is not None
                else (tmp, layout_metadata, out.view(m, cols))
            )
            program(*args)
        return out


register_kernel_type("MoEGroupedGEMMFwdOp", "ascend", AscendMoEGroupedGEMMKernel)
