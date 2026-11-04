"""Ascend argmin: exact ordered keys, first-index ties and split-row reduction."""

import functools
from typing import ClassVar

import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import restore_reduced, rows_for_axes
from tileops.kernels.reduction.call_spec import ArgreduceFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

if T is not None:

    def ordered_key(value):
        value = T.cast(value, "float32")
        bits = T.reinterpret(value, "int32")
        # Reverse numerical order, with NaNs above all numbers and both zeros tied.
        return T.if_then_else(
            T.isnan(value),
            T.int32(2147483647),
            T.if_then_else(bits < 0, bits & T.int32(2147483647), -(bits & T.int32(2147483647))),
        )

    @T.macro
    def _native_argmin(x, keys, indices, rows: int, count, begin, dtype: str, packed_output: bool):
        index_bits = T.view(indices, shape=(rows, 32), dtype="int32")
        with T.SimdVF():
            for r in T.serial(rows):
                mask = T.simd.pset(32)
                lanes = T.simd.vci(T.int32(0), "int32")
                ceiling = T.simd.vdup(T.int32(2147483647), "int32")
                zero = T.simd.vdup(T.int32(0), "int32")
                a = T.simd.alloc_local(1, "int32")
                ai = T.simd.alloc_local(1, "int32")
                a[0] = T.simd.vdup(T.int32(-2147483648), "int32")
                ai[0] = ceiling
                b = T.simd.alloc_local(1, "int32")
                bi = T.simd.alloc_local(1, "int32")
                b[0] = T.simd.vdup(T.int32(-2147483648), "int32")
                bi[0] = ceiling
                c = T.simd.alloc_local(1, "int32")
                ci = T.simd.alloc_local(1, "int32")
                c[0] = T.simd.vdup(T.int32(-2147483648), "int32")
                ci[0] = ceiling
                d = T.simd.alloc_local(1, "int32")
                di = T.simd.alloc_local(1, "int32")
                d[0] = T.simd.vdup(T.int32(-2147483648), "int32")
                di[0] = ceiling
                for i in T.serial(count // 256):
                    if dtype == "float32":
                        av = T.simd.vld(x[r, i * 256])
                        bv = T.simd.vld(x[r, i * 256 + 64])
                        cv = T.simd.vld(x[r, i * 256 + 128])
                        dv = T.simd.vld(x[r, i * 256 + 192])
                    else:
                        low = T.simd.vld(x[r, i * 256])
                        high = T.simd.vld(x[r, i * 256 + 128])
                        av = T.simd.vcvt(low, "float32", part=0)
                        bv = T.simd.vcvt(low, "float32", part=1)
                        cv = T.simd.vcvt(high, "float32", part=0)
                        dv = T.simd.vcvt(high, "float32", part=1)
                    abits = T.reinterpret(av, "int32x64")
                    aabs = T.simd.vand(abits, ceiling, mask)
                    akey = T.simd.vsel(
                        aabs,
                        T.simd.vsub(zero, aabs, mask),
                        T.simd.vcmps(abits, T.int32(0), mask, op="lt"),
                    )
                    akey = T.simd.vsel(
                        ceiling, akey, T.simd.vcmps(aabs, T.int32(2139095040), mask, op="gt")
                    )
                    if dtype == "float32":
                        apos = T.simd.vadds(lanes, T.cast(begin + i * 256 + 0, "int32"), mask)
                    else:
                        apos = T.simd.vadds(
                            T.simd.vmuls(lanes, T.int32(2), mask),
                            T.cast(begin + i * 256 + 0, "int32"),
                            mask,
                        )
                    ai[0] = T.simd.vsel(apos, ai[0], T.simd.vcmp(akey, a[0], mask, op="gt"))
                    a[0] = T.simd.vmax(a[0], akey, mask)
                    bbits = T.reinterpret(bv, "int32x64")
                    babs = T.simd.vand(bbits, ceiling, mask)
                    bkey = T.simd.vsel(
                        babs,
                        T.simd.vsub(zero, babs, mask),
                        T.simd.vcmps(bbits, T.int32(0), mask, op="lt"),
                    )
                    bkey = T.simd.vsel(
                        ceiling, bkey, T.simd.vcmps(babs, T.int32(2139095040), mask, op="gt")
                    )
                    if dtype == "float32":
                        bpos = T.simd.vadds(lanes, T.cast(begin + i * 256 + 64, "int32"), mask)
                    else:
                        bpos = T.simd.vadds(
                            T.simd.vmuls(lanes, T.int32(2), mask),
                            T.cast(begin + i * 256 + 1, "int32"),
                            mask,
                        )
                    bi[0] = T.simd.vsel(bpos, bi[0], T.simd.vcmp(bkey, b[0], mask, op="gt"))
                    b[0] = T.simd.vmax(b[0], bkey, mask)
                    cbits = T.reinterpret(cv, "int32x64")
                    cabs = T.simd.vand(cbits, ceiling, mask)
                    ckey = T.simd.vsel(
                        cabs,
                        T.simd.vsub(zero, cabs, mask),
                        T.simd.vcmps(cbits, T.int32(0), mask, op="lt"),
                    )
                    ckey = T.simd.vsel(
                        ceiling, ckey, T.simd.vcmps(cabs, T.int32(2139095040), mask, op="gt")
                    )
                    if dtype == "float32":
                        cpos = T.simd.vadds(lanes, T.cast(begin + i * 256 + 128, "int32"), mask)
                    else:
                        cpos = T.simd.vadds(
                            T.simd.vmuls(lanes, T.int32(2), mask),
                            T.cast(begin + i * 256 + 128, "int32"),
                            mask,
                        )
                    ci[0] = T.simd.vsel(cpos, ci[0], T.simd.vcmp(ckey, c[0], mask, op="gt"))
                    c[0] = T.simd.vmax(c[0], ckey, mask)
                    dbits = T.reinterpret(dv, "int32x64")
                    dabs = T.simd.vand(dbits, ceiling, mask)
                    dkey = T.simd.vsel(
                        dabs,
                        T.simd.vsub(zero, dabs, mask),
                        T.simd.vcmps(dbits, T.int32(0), mask, op="lt"),
                    )
                    dkey = T.simd.vsel(
                        ceiling, dkey, T.simd.vcmps(dabs, T.int32(2139095040), mask, op="gt")
                    )
                    if dtype == "float32":
                        dpos = T.simd.vadds(lanes, T.cast(begin + i * 256 + 192, "int32"), mask)
                    else:
                        dpos = T.simd.vadds(
                            T.simd.vmuls(lanes, T.int32(2), mask),
                            T.cast(begin + i * 256 + 129, "int32"),
                            mask,
                        )
                    di[0] = T.simd.vsel(dpos, di[0], T.simd.vcmp(dkey, d[0], mask, op="gt"))
                    d[0] = T.simd.vmax(d[0], dkey, mask)
                best = T.simd.vcmax(
                    T.simd.vmax(T.simd.vmax(a[0], b[0], mask), T.simd.vmax(c[0], d[0], mask), mask),
                    mask,
                )
                broadcast = T.simd.vselr(best, zero)
                aindex = T.simd.vsel(ai[0], ceiling, T.simd.vcmp(a[0], broadcast, mask, op="eq"))
                bindex = T.simd.vsel(bi[0], ceiling, T.simd.vcmp(b[0], broadcast, mask, op="eq"))
                cindex = T.simd.vsel(ci[0], ceiling, T.simd.vcmp(c[0], broadcast, mask, op="eq"))
                dindex = T.simd.vsel(di[0], ceiling, T.simd.vcmp(d[0], broadcast, mask, op="eq"))
                position = T.simd.vcmin(
                    T.simd.vmin(
                        T.simd.vmin(aindex, bindex, mask), T.simd.vmin(cindex, dindex, mask), mask
                    ),
                    mask,
                )
                if packed_output:
                    packed, unused = T.simd.vintlv(position, best)
                else:
                    packed, unused = T.simd.vintlv(position, zero)
                two = T.simd.pset(32, "PAT_VL2")
                T.simd.vsts(index_bits[r, 0], packed, two, extent=2)

    @T.macro
    def _native_short_half(x, indices, rows: int, count, dtype: str):
        output_bits = T.view(indices, shape=(rows, 32), dtype="int32")
        with T.SimdVF():
            mask = T.simd.pset(16)
            ceiling = T.simd.vdup(T.int16(32767), "int16")
            zero = T.simd.vdup(T.int16(0), "int16")
            lanes = T.simd.vci(T.int16(0), "int16")
            for r in T.serial(rows):
                a = T.simd.alloc_local(1, "int16")
                b = T.simd.alloc_local(1, "int16")
                ai = T.simd.alloc_local(1, "int16")
                bi = T.simd.alloc_local(1, "int16")
                a[0] = T.simd.vdup(T.int16(-32768), "int16")
                b[0] = a[0]
                ai[0] = ceiling
                bi[0] = ceiling
                for i in T.serial(count // 256):
                    av = T.simd.vld(x[r, i * 256])
                    bv = T.simd.vld(x[r, i * 256 + 128])
                    abits = T.reinterpret(av, "int16x128")
                    aabs = T.simd.vand(abits, ceiling, mask)
                    akey = T.simd.vsel(
                        aabs,
                        T.simd.vsub(zero, aabs, mask),
                        T.simd.vcmps(abits, T.int16(0), mask, op="lt"),
                    )
                    if dtype == "float16":
                        akey = T.simd.vsel(
                            ceiling, akey, T.simd.vcmps(aabs, T.int16(31744), mask, op="gt")
                        )
                    else:
                        akey = T.simd.vsel(
                            ceiling, akey, T.simd.vcmps(aabs, T.int16(32640), mask, op="gt")
                        )
                    apos = T.simd.vadds(lanes, T.cast(i * 256 + 0, "int16"), mask)
                    ai[0] = T.simd.vsel(apos, ai[0], T.simd.vcmp(akey, a[0], mask, op="gt"))
                    a[0] = T.simd.vmax(a[0], akey, mask)
                    bbits = T.reinterpret(bv, "int16x128")
                    babs = T.simd.vand(bbits, ceiling, mask)
                    bkey = T.simd.vsel(
                        babs,
                        T.simd.vsub(zero, babs, mask),
                        T.simd.vcmps(bbits, T.int16(0), mask, op="lt"),
                    )
                    if dtype == "float16":
                        bkey = T.simd.vsel(
                            ceiling, bkey, T.simd.vcmps(babs, T.int16(31744), mask, op="gt")
                        )
                    else:
                        bkey = T.simd.vsel(
                            ceiling, bkey, T.simd.vcmps(babs, T.int16(32640), mask, op="gt")
                        )
                    bpos = T.simd.vadds(lanes, T.cast(i * 256 + 128, "int16"), mask)
                    bi[0] = T.simd.vsel(bpos, bi[0], T.simd.vcmp(bkey, b[0], mask, op="gt"))
                    b[0] = T.simd.vmax(b[0], bkey, mask)
                best = T.simd.vcmax(T.simd.vmax(a[0], b[0], mask), mask)
                broadcast = T.simd.vselr(best, zero)
                aindex = T.simd.vsel(ai[0], ceiling, T.simd.vcmp(a[0], broadcast, mask, op="eq"))
                bindex = T.simd.vsel(bi[0], ceiling, T.simd.vcmp(b[0], broadcast, mask, op="eq"))
                position = T.simd.vcmin(T.simd.vmin(aindex, bindex, mask), mask)
                middle = T.simd.vcvt(position, "int32", part=0)
                wide = T.simd.vcvt(middle, "int64", part=0)
                bits = T.reinterpret(wide, "int32x64")
                two = T.simd.pset(32, "PAT_VL2")
                T.simd.vsts(output_bits[r, 0], bits, two, extent=2)

    @functools.lru_cache(maxsize=256)
    def build(m, n, dtype):
        width = max(32, min(8192, tilelang.next_power_of_2(n)))
        chunks = (n + width - 1) // width
        pitch = min(width, (n + 15) // 16 * 16)
        rows = 1 if chunks > 1 else min(4, m)
        while (
            rows > 1
            and 2 * rows * (pitch * (4 if dtype == "float32" else 2) + 16 * 12) > 192 * 1024
        ):
            rows //= 2
        groups = (m + rows - 1) // rows
        tasks = groups * chunks
        blocks = min(64, tasks)
        native = n % 256 == 0 and n < 2147483647 and m % rows == 0
        index_dtype = "int32" if n < 2147483647 else "int64"

        @T.prim_func
        def main(X: T.Tensor((m, n), dtype), I: T.Tensor((m, chunks), "int64")):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, pitch), dtype)
                keys = T.alloc_shared((rows, 16), "int32")
                indices = T.alloc_shared((rows, 16), "int64")
                index_bits = T.view(indices, shape=(rows, 32), dtype="int32")
                T.annotate_buffer_versions({x: 2, keys: 2, indices: 2})
                for step in T.Pipelined((tasks + blocks - 1) // blocks, num_stages=2):
                    task = step * blocks + bx
                    if task < tasks:
                        row = (task // chunks) * rows
                        chunk = task % chunks
                        begin = chunk * width
                        valid_rows = T.min(rows, m - row)
                        count = T.min(width, n - begin)
                        T.copy(
                            X[row : row + valid_rows, begin : begin + count], x[:valid_rows, :count]
                        )
                        if native and chunks == 1 and dtype != "float32":
                            _native_short_half(x, indices, rows, count, dtype)
                        elif native:
                            _native_argmin(x, keys, indices, rows, count, begin, dtype, chunks > 1)
                        else:
                            with T.SimtVF(threads=min(256, width)):
                                best = T.alloc_reducer((rows,), "int32", op="max")
                                T.reducer_init(best)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < count:
                                        T.reducer_update(best[r], ordered_key(x[r, i]))
                                winners = T.alloc_fragment((rows,), "int32")
                                T.finalize_reducer(best, winners)
                                first = T.alloc_reducer((rows,), index_dtype, op="min")
                                T.reducer_init(first)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < count:  # noqa: SIM102 - preserve validated TileLang DSL structure
                                        if ordered_key(x[r, i]) == winners[r]:
                                            T.reducer_update(
                                                first[r], T.cast(begin + i, index_dtype)
                                            )
                                positions = T.alloc_fragment((rows,), index_dtype)
                                T.finalize_reducer(first, positions)
                                for r in T.Parallel(rows):
                                    keys[r, 0] = winners[r]
                                    if chunks > 1:
                                        index_bits[r, 0] = T.cast(positions[r], "int32")
                                        index_bits[r, 1] = winners[r]
                                    else:
                                        indices[r, 0] = T.cast(positions[r], "int64")
                        T.copy(
                            indices[:valid_rows, :1],
                            I[row : row + valid_rows, chunk : chunk + 1],
                            l2_cache_ctrl=0,
                        )

        return tilelang.compile(main, target="ascend", out_idx=[1]), chunks

    @T.macro
    def _native_finish(keys, indices, y, rows: int, pitch: int, chunks: int):
        input_bits = T.view(indices, shape=(rows, pitch * 2), dtype="int32")
        output_bits = T.view(y, shape=(rows, 32), dtype="int32")
        with T.SimdVF():
            mask = T.simd.pset(32)
            zero = T.simd.vdup(T.int32(0), "int32")
            floor = T.simd.vdup(T.int32(-2147483648), "int32")
            ceiling = T.simd.vdup(T.int32(2147483647), "int32")
            lanes = T.simd.vci(T.int32(0), "int32")
            valid = T.simd.vcmps(lanes, T.int32(chunks), mask, op="lt")
            for r in T.serial(rows):
                raw = T.simd.vld(input_bits[r, 0])
                positions, loaded = T.simd.vdintlv(raw, zero)
                values = T.simd.vsel(loaded, floor, valid)
                best = T.simd.vcmax(values, mask)
                broadcast = T.simd.vselr(best, zero)
                candidates = T.simd.vsel(
                    positions, ceiling, T.simd.vcmp(values, broadcast, mask, op="eq")
                )
                candidates = T.simd.vsel(candidates, ceiling, valid)
                position = T.simd.vcmin(candidates, mask)
                wide = T.simd.vcvt(position, "int64", part=0)
                packed = T.reinterpret(wide, "int32x64")
                two = T.simd.pset(32, "PAT_VL2")
                T.simd.vsts(output_bits[r, 0], packed, two, extent=2)

    @functools.lru_cache(maxsize=256)
    def build_finish(m, chunks, index_dtype):
        width = max(32, tilelang.next_power_of_2(chunks))
        pitch = max(64, (chunks + 15) // 16 * 16)
        rows = min(4, m)
        groups = (m + rows - 1) // rows
        blocks = min(64, groups)
        native = chunks <= 32 and index_dtype == "int32" and m % rows == 0

        @T.prim_func
        def main(I: T.Tensor((m, chunks), "int64"), Y: T.Tensor((m, 1), "int64")):
            with T.Kernel(blocks) as bx:
                keys = T.alloc_shared((rows, pitch), "int32")
                indices = T.alloc_shared((rows, pitch), "int64")
                y = T.alloc_shared((rows, 16), "int64")
                for group in T.serial((groups + blocks - 1) // blocks):
                    row = (group * blocks + bx) * rows
                    if row < m:
                        valid_rows = T.min(rows, m - row)
                        T.copy(I[row : row + valid_rows, :], indices[:valid_rows, :chunks])
                        if native:
                            _native_finish(keys, indices, y, rows, pitch, chunks)
                        else:
                            with T.SimtVF(threads=min(256, width)):
                                best = T.alloc_reducer((rows,), "int32", op="max")
                                T.reducer_init(best)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < chunks:
                                        T.reducer_update(
                                            best[r], T.cast(indices[r, i] >> 32, "int32")
                                        )
                                winners = T.alloc_fragment((rows,), "int32")
                                T.finalize_reducer(best, winners)
                                first = T.alloc_reducer((rows,), index_dtype, op="min")
                                T.reducer_init(first)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < chunks:  # noqa: SIM102 - preserve validated TileLang DSL structure
                                        if T.cast(indices[r, i] >> 32, "int32") == winners[r]:
                                            T.reducer_update(
                                                first[r],
                                                T.cast(
                                                    indices[r, i] & T.int64(4294967295), index_dtype
                                                ),
                                            )
                                positions = T.alloc_fragment((rows,), index_dtype)
                                T.finalize_reducer(first, positions)
                                for r in T.Parallel(rows):
                                    y[r, 0] = T.cast(positions[r], "int64")
                        T.copy(y[:valid_rows, :1], Y[row : row + valid_rows, :], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])

    @functools.lru_cache(maxsize=256)
    def build_wide(m, n, dtype):
        width = max(32, min(8192, tilelang.next_power_of_2(n)))
        chunks = (n + width - 1) // width
        pitch = min(width, (n + 15) // 16 * 16)
        rows = 1 if chunks > 1 else min(4, m)
        while (
            rows > 1
            and 2 * rows * (pitch * (4 if dtype == "float32" else 2) + 16 * 12) > 192 * 1024
        ):
            rows //= 2
        groups = (m + rows - 1) // rows
        tasks = groups * chunks
        blocks = min(64, tasks)
        native = False
        index_dtype = "int32" if n < 2147483647 else "int64"

        @T.prim_func
        def main(
            X: T.Tensor((m, n), dtype),
            K: T.Tensor((m, chunks), "int32"),
            I: T.Tensor((m, chunks), "int64"),
        ):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, pitch), dtype)
                keys = T.alloc_shared((rows, 16), "int32")
                indices = T.alloc_shared((rows, 16), "int64")
                T.annotate_buffer_versions({x: 2, keys: 2, indices: 2})
                for step in T.Pipelined((tasks + blocks - 1) // blocks, num_stages=2):
                    task = step * blocks + bx
                    if task < tasks:
                        row = (task // chunks) * rows
                        chunk = task % chunks
                        begin = chunk * width
                        valid_rows = T.min(rows, m - row)
                        count = T.min(width, n - begin)
                        T.copy(
                            X[row : row + valid_rows, begin : begin + count], x[:valid_rows, :count]
                        )
                        if native:
                            _native_argmin(x, keys, indices, rows, count, begin, dtype, False)
                        else:
                            with T.SimtVF(threads=min(256, width)):
                                best = T.alloc_reducer((rows,), "int32", op="max")
                                T.reducer_init(best)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < count:
                                        T.reducer_update(best[r], ordered_key(x[r, i]))
                                winners = T.alloc_fragment((rows,), "int32")
                                T.finalize_reducer(best, winners)
                                first = T.alloc_reducer((rows,), index_dtype, op="min")
                                T.reducer_init(first)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < count:  # noqa: SIM102 - preserve validated TileLang DSL structure
                                        if ordered_key(x[r, i]) == winners[r]:
                                            T.reducer_update(
                                                first[r], T.cast(begin + i, index_dtype)
                                            )
                                positions = T.alloc_fragment((rows,), index_dtype)
                                T.finalize_reducer(first, positions)
                                for r in T.Parallel(rows):
                                    keys[r, 0] = winners[r]
                                    indices[r, 0] = T.cast(positions[r], "int64")
                        T.copy(
                            keys[:valid_rows, :1],
                            K[row : row + valid_rows, chunk : chunk + 1],
                            l2_cache_ctrl=0,
                        )
                        T.copy(
                            indices[:valid_rows, :1],
                            I[row : row + valid_rows, chunk : chunk + 1],
                            l2_cache_ctrl=0,
                        )

        return tilelang.compile(main, target="ascend", out_idx=[1, 2]), chunks

    @functools.lru_cache(maxsize=256)
    def build_finish_wide(m, chunks, index_dtype):
        width = max(32, tilelang.next_power_of_2(chunks))
        pitch = max(64, (chunks + 15) // 16 * 16)
        rows = min(4, m)
        groups = (m + rows - 1) // rows
        blocks = min(64, groups)
        native = False

        @T.prim_func
        def main(
            K: T.Tensor((m, chunks), "int32"),
            I: T.Tensor((m, chunks), "int64"),
            Y: T.Tensor((m, 1), "int64"),
        ):
            with T.Kernel(blocks) as bx:
                keys = T.alloc_shared((rows, pitch), "int32")
                indices = T.alloc_shared((rows, pitch), "int64")
                y = T.alloc_shared((rows, 16), "int64")
                for group in T.serial((groups + blocks - 1) // blocks):
                    row = (group * blocks + bx) * rows
                    if row < m:
                        valid_rows = T.min(rows, m - row)
                        T.copy(K[row : row + valid_rows, :], keys[:valid_rows, :chunks])
                        T.copy(I[row : row + valid_rows, :], indices[:valid_rows, :chunks])
                        if native:
                            _native_finish(keys, indices, y, rows, pitch, chunks)
                        else:
                            with T.SimtVF(threads=min(256, width)):
                                best = T.alloc_reducer((rows,), "int32", op="max")
                                T.reducer_init(best)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < chunks:
                                        T.reducer_update(best[r], keys[r, i])
                                winners = T.alloc_fragment((rows,), "int32")
                                T.finalize_reducer(best, winners)
                                first = T.alloc_reducer((rows,), index_dtype, op="min")
                                T.reducer_init(first)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < chunks:  # noqa: SIM102 - preserve validated TileLang DSL structure
                                        if keys[r, i] == winners[r]:
                                            T.reducer_update(
                                                first[r], T.cast(indices[r, i], index_dtype)
                                            )
                                positions = T.alloc_fragment((rows,), index_dtype)
                                T.finalize_reducer(first, positions)
                                for r in T.Parallel(rows):
                                    y[r, 0] = T.cast(positions[r], "int64")
                        T.copy(y[:valid_rows, :1], Y[row : row + valid_rows, :], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[2])

    @T.macro
    def _native_leading(x, y, n: int, cols: int, dtype: str):
        output_bits = T.view(y, shape=(cols * 2,), dtype="int32")
        with T.SimdVF():
            mask = T.simd.pset(32)
            zero = T.simd.vdup(T.int32(0), "int32")
            ceiling = T.simd.vdup(T.int32(2147483647), "int32")
            for c in T.serial(cols // 128):
                a = T.simd.alloc_local(1, "int32")
                b = T.simd.alloc_local(1, "int32")
                ai = T.simd.alloc_local(1, "int32")
                bi = T.simd.alloc_local(1, "int32")
                a[0] = T.simd.vdup(T.int32(-2147483648), "int32")
                b[0] = a[0]
                ai[0] = zero
                bi[0] = zero
                for r in T.unroll(n):
                    if dtype == "float32":
                        av = T.simd.vld(x[r, c * 128])
                        bv = T.simd.vld(x[r, c * 128 + 64])
                    else:
                        loaded = T.simd.vld(x[r, c * 128])
                        av = T.simd.vcvt(loaded, "float32", part=0)
                        bv = T.simd.vcvt(loaded, "float32", part=1)
                    abits = T.reinterpret(av, "int32x64")
                    aabs = T.simd.vand(abits, ceiling, mask)
                    akey = T.simd.vsel(
                        aabs,
                        T.simd.vsub(zero, aabs, mask),
                        T.simd.vcmps(abits, T.int32(0), mask, op="lt"),
                    )
                    akey = T.simd.vsel(
                        ceiling, akey, T.simd.vcmps(aabs, T.int32(2139095040), mask, op="gt")
                    )
                    ai[0] = T.simd.vsel(
                        T.simd.vdup(T.cast(r, "int32"), "int32"),
                        ai[0],
                        T.simd.vcmp(akey, a[0], mask, op="gt"),
                    )
                    a[0] = T.simd.vmax(a[0], akey, mask)
                    bbits = T.reinterpret(bv, "int32x64")
                    babs = T.simd.vand(bbits, ceiling, mask)
                    bkey = T.simd.vsel(
                        babs,
                        T.simd.vsub(zero, babs, mask),
                        T.simd.vcmps(bbits, T.int32(0), mask, op="lt"),
                    )
                    bkey = T.simd.vsel(
                        ceiling, bkey, T.simd.vcmps(babs, T.int32(2139095040), mask, op="gt")
                    )
                    bi[0] = T.simd.vsel(
                        T.simd.vdup(T.cast(r, "int32"), "int32"),
                        bi[0],
                        T.simd.vcmp(bkey, b[0], mask, op="gt"),
                    )
                    b[0] = T.simd.vmax(b[0], bkey, mask)
                if dtype == "float32":
                    first = ai[0]
                    second = bi[0]
                else:
                    first, second = T.simd.vintlv(ai[0], bi[0])
                aout, bout = T.simd.vintlv(first, zero)
                cout, dout = T.simd.vintlv(second, zero)
                T.simd.vsts(output_bits[c * 256], aout, mask)
                T.simd.vsts(output_bits[c * 256 + 64], bout, mask)
                T.simd.vsts(output_bits[c * 256 + 128], cout, mask)
                T.simd.vsts(output_bits[c * 256 + 192], dout, mask)

    @functools.lru_cache(maxsize=256)
    def build_short_leading(m, n, dtype):
        cols = min(8192, max(32, tilelang.next_power_of_2(m)))
        while 2 * cols * (n * (4 if dtype == "float32" else 2) + 8) > 192 * 1024:
            cols //= 2
        groups = (m + cols - 1) // cols
        blocks = min(64, groups)

        @T.prim_func
        def main(X: T.Tensor((n, m), dtype), Y: T.Tensor((m,), "int64")):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((n, cols), dtype)
                y = T.alloc_shared((cols,), "int64")
                T.annotate_buffer_versions({x: 2, y: 2})
                for group in T.Pipelined((groups + blocks - 1) // blocks, num_stages=2):
                    col = (group * blocks + bx) * cols
                    if col < m:
                        count = T.min(cols, m - col)
                        T.copy(X[:, col : col + count], x[:, :count])
                        if m % cols == 0 and cols % 128 == 0:
                            _native_leading(x, y, n, cols, dtype)
                        else:
                            with T.SimtVF(threads=256):
                                best = T.alloc_local((1,), "int32")
                                position = T.alloc_local((1,), "int32")
                                for c in T.Parallel(cols):
                                    if c < count:
                                        best[0] = T.int32(-2147483648)
                                        position[0] = T.int32(0)
                                        for r in T.unroll(n):
                                            key = ordered_key(x[r, c])
                                            if key > best[0]:
                                                best[0] = key
                                                position[0] = r
                                        y[c] = T.cast(position[0], "int64")
                        T.copy(y[:count], Y[col : col + count], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])


class AscendArgminKernel(Kernel, ArgreduceFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        if call.op_kind != "argmin":
            raise ValueError("Expected argmin")
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("Unsupported argmin dtype")
        dtype = str(call.dtype).split(".")[-1]
        self.leading = (
            call.axes == tuple(range(len(call.axes)))
            and len(call.axes) < len(call.shape)
            and call.n <= 16
        )
        self.finish = None
        self.wide = call.n >= 2147483647
        if self.leading:
            self.compiled = build_short_leading(call.m, call.n, dtype)
            self.chunks = 1
        elif self.wide:
            self.compiled, self.chunks = build_wide(call.m, call.n, dtype)
            if self.chunks > 1:
                self.finish = build_finish_wide(call.m, self.chunks, "int64")
        else:
            self.compiled, self.chunks = build(call.m, call.n, dtype)
            if self.chunks > 1:
                self.finish = build_finish(
                    call.m, self.chunks, "int32" if call.n < 2147483647 else "int64"
                )

    def forward(self, x):
        if tuple(x.shape) != self.call.shape or x.dtype != self.call.dtype:
            raise ValueError("Input signature changed")
        arg = (
            x.reshape(self.call.n, self.call.m)
            if self.leading
            else rows_for_axes(x, self.call.axes)
        )
        if self.leading:
            result = self.compiled(arg)
        elif self.wide:
            keys, indices = self.compiled(arg)
            result = self.finish(keys, indices) if self.finish is not None else indices
        else:
            indices = self.compiled(arg)
            result = self.finish(indices) if self.finish is not None else indices
        return restore_reduced(result, self.call.shape, self.call.axes, self.call.keepdim)


if T is not None:
    register_kernel_type("ArgminFwdOp", "ascend", AscendArgminKernel)
