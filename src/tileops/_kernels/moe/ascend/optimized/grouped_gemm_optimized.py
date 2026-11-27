"""Grouped GEMM for packed expert rows on Ascend 950.

C[r] = A[r] @ B[g].T + bias[g] over each expert's row range ends[g], computed by
one kernel: every core walks the per-group block list and each tile runs a
K-major L1 pipeline into the Cube. Validated shapes select direct BF16 output,
an incremental expert walk. Vector epilogues always use both AIVs.
"""

import functools

import torch

from tileops._kernels.moe.ascend.optimized.common import (
    AscendMoEKernel,
    check_activation,
    check_dtype,
    check_out,
    check_tensor,
)

__all__ = ["AscendMoEGroupedGEMMOptimizedKernel", "pick_block"]

# On-chip budgets (arch35): L0C 256K, L1 512K, UB ~216K per AIV. Keep headroom.
_L0C_BYTES = 256 * 1024
_L1_BYTES = 448 * 1024
_UB_BYTES = 192 * 1024


def pick_block(m, num_groups, n, k, fused, num_cores, cd_bytes=2):
    """Choose (M, N, K, stages) within L0C, L1 and per-AIV UB budgets.

    Gated N includes both gate/up halves. Round M to 16 and each output N
    tile to 64; prefer larger K when at least two L1 stages fit.
    """
    c_cols = n // 2 if fused else n
    avg = max(1, -(-m // max(num_groups, 1)))
    block_m = min(256, -(-avg // 16) * 16)
    acc_cells = _L0C_BYTES // (4 * (2 if fused else 1))
    epi_cells = _UB_BYTES // (4 * (2 if fused else 1) + cd_bytes)
    # A non-gated tile can afford twice the width: only one B operand is alive in
    # L1 and one accumulator in L0C, so the wider (fewer, longer) tile wins on the
    # short-K down projection. The gated gate/up keeps 256 because it streams two
    # B tiles per k step and a wider tile would force block_k down to 64.
    n_cap = 256 if fused else 512
    tile_n = max(
        64,
        min(
            n_cap,
            (acc_cells // block_m) // 64 * 64,
            -(-c_cols // 64) * 64,
            (epi_cells // block_m) // 64 * 64,
        ),
    )
    block_k, stages = 64, 2
    for candidate in (256, 128, 64):
        if k % candidate:
            continue
        per_stage = (block_m + (2 if fused else 1) * tile_n) * candidate * 2
        fit = min(6, _L1_BYTES // per_stage)
        if fit >= 2:
            block_k, stages = candidate, min(fit, max(-(-k // candidate), 1))
            break
    return block_m, 2 * tile_n if fused else tile_n, block_k, max(stages, 1)


@functools.lru_cache(maxsize=32)
def _grouped_gemm_program(
    m,
    num_groups,
    n,
    k,
    ab_dtype,
    cd_dtype,
    activation,
    has_bias,
    num_cores,
    block_m,
    block_n,
    block_k,
    num_stages,
    direct_output=False,
    group_walk=False,
):
    import tilelang.ascend.language as T

    fused = activation is not None
    interleaved = activation == "swigluoai"  # B as [G, c_cols, 2, k]: row 2j = gate j
    c_cols = n // 2 if fused else n
    tile_n = block_n // 2 if fused else block_n
    # N-split shares short expert rows; narrow tiles keep 64-lane rows via M-split.
    split_n = tile_n >= 128
    epi_m = block_m if split_n else block_m // 2
    epi_n = tile_n // 2 if split_n else tile_n
    k_tiles = (k + block_k - 1) // block_k
    n_tiles = (c_cols + tile_n - 1) // tile_n
    # Upper bound on persistent iterations; the scheduler skips empty experts.
    m_tiles = (m + block_m - 1) // block_m
    iters = ((m_tiles + num_groups) * n_tiles + num_cores - 1) // num_cores

    # FP32 vector math; round only at the output store.
    def v_sigmoid(x, m):
        """1 / (1 + exp(-x))."""
        return T.simd.vdiv(
            T.simd.vdup(1.0, "float32", m),
            T.simd.vadds(T.simd.vexp(T.simd.vneg(x, m), m), 1.0, m),
            m,
        )

    def v_erf(x, m):
        """Abramowitz-Stegun 7.1.26: erf(|x|) = 1 - poly(t) * exp(-x*x), t = 1/(1+0.3275911|x|)."""
        ax = T.simd.vabs(x, m)
        t = T.simd.vdiv(
            T.simd.vdup(1.0, "float32", m), T.simd.vadds(T.simd.vmuls(ax, 0.3275911, m), 1.0, m), m
        )
        # Horner chain over vector constants: vmadd needs a mutable register, and
        # an SSA value plus vdup is just as cheap on 64 lanes.
        poly = T.simd.vdup(1.061405429, "float32", m)
        for coef in (-1.453152027, 1.421413741, -0.284496736, 0.254829592):
            poly = T.simd.vadd(T.simd.vmul(poly, t, m), T.simd.vdup(coef, "float32", m), m)
        poly = T.simd.vmul(poly, t, m)
        erf_abs = T.simd.vsub(
            T.simd.vdup(1.0, "float32", m),
            T.simd.vmul(poly, T.simd.vexp(T.simd.vneg(T.simd.vmul(x, x, m), m), m), m),
            m,
        )
        return T.simd.vsel(T.simd.vneg(erf_abs, m), erf_abs, T.simd.vcmps(x, 0.0, m, op="lt"))

    def apply_gated_activation(gate, up, m):
        """act(gate, up) in float32 on 64 lanes."""
        if activation == "silu_and_mul":
            return T.simd.vmul(T.simd.vmul(gate, v_sigmoid(gate, m), m), up, m)
        if activation == "swigluoai":
            seven = T.simd.vdup(7.0, "float32", m)
            gc = T.simd.vmin(gate, seven, m)
            uc = T.simd.vmax(T.simd.vmin(up, seven, m), T.simd.vneg(seven, m), m)
            return T.simd.vmul(
                T.simd.vmul(gc, v_sigmoid(T.simd.vmuls(gc, 1.702, m), m), m),
                T.simd.vadds(uc, 1.0, m),
                m,
            )
        # gelu_and_mul: 0.5 * g * (1 + erf(g / sqrt(2))) * u
        erf = v_erf(T.simd.vmuls(gate, 0.7071067811865476, m), m)
        return T.simd.vmul(
            T.simd.vmuls(T.simd.vmul(gate, T.simd.vadds(erf, 1.0, m), m), 0.5, m), up, m
        )

    def weight_tile(B, g, n0, part, k0):
        """*tile_n* gate (*part* 0) or up (*part* 1) rows of group *g* at *n0*."""
        if interleaved:
            return B[g, n0 : n0 + tile_n, part, k0 : k0 + block_k]
        row0 = n0 if part == 0 else c_cols + n0
        return B[g, row0 : row0 + tile_n, k0 : k0 + block_k]

    @T.macro
    def compute_tile(
        A,
        B,
        C,
        Bias,
        group,
        r0,
        n0,
        valid_m,
        valid_n,
        sid,
        a_l1,
        b_l1,
        acc,
        up_l1,
        acc_up,
        ub,
        out_ub,
        up_ub,
        bias_ub,
    ):
        """Compute one expert tile, then apply its epilogue and write valid rows."""
        for kt in T.Pipelined(k_tiles, num_stages=num_stages, annotations={"enable_offset": True}):
            k0 = kt * block_k
            with T.Stage(0), T.Task():
                T.copy(A[r0 : r0 + block_m, k0 : k0 + block_k], a_l1)
                T.copy(weight_tile(B, group, n0, 0, k0), b_l1)
                if fused:
                    T.copy(weight_tile(B, group, n0, 1, k0), up_l1)
            with T.Stage(1), T.Task():
                T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=kt == 0)
                if fused:
                    T.gemm(a_l1, up_l1, acc_up, transpose_B=True, clear_accum=kt == 0)
        if direct_output:
            # Keep the full L0C source pitch; only the destination is clipped.
            T.copy(acc, C[r0 : r0 + valid_m, n0 : n0 + valid_n])
        else:
            T.dual_copy(acc, ub)
            if fused:
                T.dual_copy(acc_up, up_ub)
            col0 = n0 + sid * epi_n if split_n else n0
            if has_bias:
                if interleaved:  # bias shares B's [c_cols, 2] pairing
                    T.copy(Bias[group, col0 : col0 + epi_n, 0], bias_ub[0, :])
                    T.copy(Bias[group, col0 : col0 + epi_n, 1], bias_ub[1, :])
                else:
                    T.copy(Bias[group, col0 : col0 + epi_n], bias_ub[0, :])
                    if fused:
                        T.copy(Bias[group, c_cols + col0 : c_cols + col0 + epi_n], bias_ub[1, :])
            lanes = epi_n // 64
            with T.SimdVF():
                m = T.simd.pset(32)  # 64 active float32 lanes
                flat = T.reshape(ub, (epi_m * epi_n,))
                flat_up = T.reshape(up_ub, (epi_m * epi_n,)) if fused else None
                flat_out = T.reshape(out_ub, (epi_m * epi_n,))
                for p in T.serial(epi_m * epi_n // 64):
                    gate = T.simd.vld(flat[p * 64])
                    if has_bias or fused:
                        chunk = p % lanes * 64
                    if has_bias:
                        gate = T.simd.vadd(
                            gate,
                            T.simd.vcvt(T.simd.vld(bias_ub[0, chunk], "UNPK_B16"), "float32"),
                            m,
                        )
                    if fused:
                        up = T.simd.vld(flat_up[p * 64])
                        if has_bias:
                            up = T.simd.vadd(
                                up,
                                T.simd.vcvt(T.simd.vld(bias_ub[1, chunk], "UNPK_B16"), "float32"),
                                m,
                            )
                        gate = apply_gated_activation(gate, up, m)
                    if cd_dtype == "float32":
                        T.simd.vsts(flat_out[p * 64], gate, m)
                    else:
                        T.simd.vsts(flat_out[p * 64], T.simd.vcvt(gate, cd_dtype), m, dist="PK_B32")
            store_row = r0 if split_n else r0 + sid * epi_m
            store_m = valid_m if split_n else T.max(T.min(valid_m - sid * epi_m, epi_m), 0)
            store_n = T.max(T.min(valid_n - sid * epi_n, epi_n), 0) if split_n else valid_n
            T.copy(
                out_ub[0:store_m, 0:store_n],
                C[store_row : store_row + store_m, col0 : col0 + store_n],
            )

    @T.macro
    def tile_loop(A, B, ends, C, Bias, core, sid):
        a_l1 = T.alloc_l1((block_m, block_k), ab_dtype)
        b_l1 = T.alloc_l1((tile_n, block_k), ab_dtype)
        acc = T.alloc_l0c((block_m, tile_n), "float32")
        ub = T.alloc_shared((epi_m, epi_n), "float32")
        out_ub = T.alloc_shared((epi_m, epi_n), cd_dtype)
        bias_ub = T.alloc_shared((2, epi_n), ab_dtype) if has_bias else None
        versions = max(num_stages, 2)
        if fused:
            up_l1 = T.alloc_l1((tile_n, block_k), ab_dtype)
            acc_up = T.alloc_l0c((block_m, tile_n), "float32")
            up_ub = T.alloc_shared((epi_m, epi_n), "float32")
            # Only the L1 operands are multi-versioned: a second UB buffer set
            # would exceed the per-AIV UB budget at block_m=256.
            T.annotate_buffer_versions({a_l1: versions, b_l1: versions, up_l1: versions})
        else:
            up_l1 = acc_up = up_ub = None
            T.annotate_buffer_versions({a_l1: versions, b_l1: versions})
        b_view = T.view(B, (num_groups, c_cols, 2, k)) if interleaved else B
        if group_walk:
            previous_end = T.alloc_var("int32", init=0)
            tile_count = T.alloc_var("int32", init=0)
            for group in T.serial(num_groups):
                group_end = T.cast(ends[group], "int32")
                nb = T.ceildiv(T.max(group_end - previous_end, 0), block_m)
                total = nb * n_tiles
                first = (core - tile_count % num_cores + num_cores) % num_cores
                for local in T.serial(T.ceildiv(T.max(total - first, 0), num_cores)):
                    rel = first + local * num_cores
                    r0 = previous_end + (rel % nb) * block_m
                    n0 = (rel // nb) * tile_n
                    valid_m = T.min(group_end - r0, block_m)
                    valid_n = T.min(c_cols - n0, tile_n)
                    compute_tile(
                        A,
                        b_view,
                        C,
                        Bias,
                        group,
                        r0,
                        n0,
                        valid_m,
                        valid_n,
                        sid,
                        a_l1,
                        b_l1,
                        acc,
                        up_l1,
                        acc_up,
                        ub,
                        out_ub,
                        up_ub,
                        bias_ub,
                    )
                previous_end = group_end
                tile_count = tile_count + total
        else:
            group = T.alloc_var("int32", init=0)
            r0 = T.alloc_var("int32", init=0)
            n0 = T.alloc_var("int32", init=0)
            valid_m = T.alloc_var("int32", init=0)
            valid_n = T.alloc_var("int32", init=0)
            # Retain the validated cursor strategy for large expert counts.
            use_cursor = num_groups >= 512
            seen = T.alloc_var("int32", init=0) if use_cursor else None
            prev = T.alloc_var("int32", init=0) if use_cursor else None
            end = T.alloc_var("int32", init=0) if use_cursor else None
            nb = T.alloc_var("int32", init=0) if use_cursor else None
            if use_cursor:
                end = T.cast(ends[0], "int32")
                nb = T.ceildiv(end, block_m)
            for it in T.serial(iters):
                bidx = it * num_cores + core
                if use_cursor:
                    while (group < num_groups) & (bidx >= (seen + nb) * n_tiles):
                        seen = seen + nb
                        prev = end
                        group = group + 1
                        if group < num_groups:
                            end = T.cast(ends[group], "int32")
                            nb = T.ceildiv(T.max(end - prev, 0), block_m)
                    r0 = 0
                    n0 = 0
                    valid_m = 0
                    valid_n = 0
                    if group < num_groups:
                        rel = bidx - seen * n_tiles
                        r0 = prev + (rel % nb) * block_m
                        n0 = (rel // nb) * tile_n
                        valid_m = T.min(end - r0, block_m)
                        valid_n = T.min(c_cols - n0, tile_n)
                else:
                    group = -1
                    r0 = 0
                    n0 = 0
                    valid_m = 0
                    valid_n = 0
                    scan_g = T.alloc_var("int32", init=0)
                    scan_seen = T.alloc_var("int32", init=0)
                    scan_prev = T.alloc_var("int32", init=0)
                    scan_end = T.alloc_var("int32", init=0)
                    scan_nb = T.alloc_var("int32", init=0)
                    while scan_g < num_groups:
                        scan_end = T.cast(ends[scan_g], "int32")
                        scan_nb = T.ceildiv(T.max(scan_end - scan_prev, 0), block_m)
                        if bidx < (scan_seen + scan_nb) * n_tiles:
                            T.loop_break()
                        scan_seen = scan_seen + scan_nb
                        scan_prev = scan_end
                        scan_g = scan_g + 1
                    if scan_g < num_groups:
                        rel = bidx - scan_seen * n_tiles
                        group = scan_g
                        r0 = scan_prev + (rel % scan_nb) * block_m
                        n0 = (rel // scan_nb) * tile_n
                        valid_m = T.min(scan_end - r0, block_m)
                        valid_n = T.min(c_cols - n0, tile_n)
                if group >= 0:
                    compute_tile(
                        A,
                        b_view,
                        C,
                        Bias,
                        group,
                        r0,
                        n0,
                        valid_m,
                        valid_n,
                        sid,
                        a_l1,
                        b_l1,
                        acc,
                        up_l1,
                        acc_up,
                        ub,
                        out_ub,
                        up_ub,
                        bias_ub,
                    )

    bias_shape = (num_groups, c_cols, 2) if interleaved else (num_groups, n)
    if has_bias:

        @T.prim_func
        def main(
            A: T.Tensor((m, k), ab_dtype),
            B: T.Tensor((num_groups, n, k), ab_dtype),
            ends: T.Tensor((num_groups,), "int32"),
            C: T.Tensor((m, c_cols), cd_dtype),
            Bias: T.Tensor(bias_shape, ab_dtype),
        ):
            with T.MixedKernel(num_cores, sids=2) as (core, sid):
                tile_loop(A, B, ends, C, Bias, core, sid)

    else:

        @T.prim_func
        def main(
            A: T.Tensor((m, k), ab_dtype),
            B: T.Tensor((num_groups, n, k), ab_dtype),
            ends: T.Tensor((num_groups,), "int32"),
            C: T.Tensor((m, c_cols), cd_dtype),
        ):
            if direct_output:
                with T.Kernel(num_cores) as core:
                    tile_loop(A, B, ends, C, None, core, 0)
            else:
                with T.MixedKernel(num_cores, sids=2) as (core, sid):
                    tile_loop(A, B, ends, C, None, core, sid)

    return main


# (packed rows, experts, weight N, K, activation):
# (tile M, tile N including both gate/up halves, tile K, stages,
#  direct BF16 store, incremental expert walk).
_VALIDATED_TILES = {
    (4096, 128, 3072, 4096, "silu_and_mul"): (64, 512, 128, 3, False, False),
    (4096, 128, 4096, 1536, None): (64, 512, 128, 3, False, False),
    (32768, 128, 3072, 4096, "silu_and_mul"): (128, 512, 128, 2, False, False),
    (32768, 128, 4096, 1536, None): (128, 512, 128, 2, True, False),
    (4096, 256, 4096, 7168, "silu_and_mul"): (64, 512, 128, 3, False, True),
    (4096, 256, 7168, 2048, None): (64, 512, 128, 3, True, True),
    (8, 128, 2048, 768, None): (16, 512, 128, 3, True, False),
}


class AscendMoEGroupedGEMMOptimizedKernel(AscendMoEKernel):
    """Tight M-grouped GEMM with an exact per-group block walk.

    ``layout`` is the tight cumulative end array (``ends``, int32) of the packed
    rows: nondecreasing, last entry equal to ``m``, kept on the device. Only
    ``tight_physical_psum`` is served.
    """

    def __init__(
        self,
        m=None,
        num_groups=None,
        n=None,
        k=None,
        ab_dtype=None,
        *,
        activation=None,
        cd_dtype=None,
        has_bias=False,
        num_cores=36,
        block_m=None,
        block_n=None,
        block_k=None,
        num_stages=None,
        rows=None,
        num_experts=None,
        dtype=None,
        out_dtype=None,
    ):
        m = m if m is not None else rows
        num_groups = num_groups if num_groups is not None else num_experts
        ab_dtype = ab_dtype or dtype or torch.bfloat16
        cd_dtype = cd_dtype if cd_dtype is not None else out_dtype
        if m is None or num_groups is None or n is None or k is None:
            raise ValueError("grouped GEMM needs m, num_groups, n and k")
        super().__init__(num_cores)
        check_dtype(ab_dtype)
        check_activation(activation)
        if m < 0 or min(num_groups, n, k) <= 0 or (activation is not None and n % 2):
            raise ValueError("invalid grouped GEMM dimensions")
        if cd_dtype not in (None, ab_dtype, torch.float32):
            raise ValueError("out_dtype must be the operand dtype or float32")
        for value in (block_m, block_n, block_k, num_stages):
            if value is not None and value <= 0:
                raise ValueError("tile blocks and stages must be positive")
        self.m, self.num_groups, self.n, self.k = m, num_groups, n, k
        self.ab_dtype = ab_dtype
        self.cd_dtype = ab_dtype if cd_dtype is None else cd_dtype
        self.activation, self.has_bias = activation, has_bias
        picked = pick_block(
            m,
            num_groups,
            n,
            k,
            activation is not None,
            num_cores,
            4 if self.cd_dtype == torch.float32 else 2,
        )
        # For the tested large-expert BF16 shape, a taller tile reduces
        # repeated weight reads caused by multiple tiles per expert.
        large_expert_shape = (
            m == 4096
            and num_groups == 256
            and ab_dtype == torch.bfloat16
            and not has_bias
            and self.cd_dtype == torch.bfloat16
            and block_n is None
            and block_k is None
            and num_stages is None
            and (
                (activation == "silu_and_mul" and n == 4096 and k == 7168)
                or (activation is None and n == 7168 and k == 2048)
            )
        )
        self.block_m = block_m or (64 if large_expert_shape else picked[0])
        self.block_n = block_n or picked[1]
        self.block_k = block_k or picked[2]
        self.num_stages = num_stages or picked[3]
        self.direct_output = self.group_walk = False
        # Validated BF16 MoE shapes. Keep explicit tile overrides and all other
        # shapes/dtypes/epilogues on the existing implementation.
        if (
            ab_dtype == torch.bfloat16
            and self.cd_dtype == torch.bfloat16
            and not has_bias
            and num_cores == 36
            and all(v is None for v in (block_m, block_n, block_k, num_stages))
        ):
            config = _VALIDATED_TILES.get((m, num_groups, n, k, activation))
            if config is not None:
                (
                    self.block_m,
                    self.block_n,
                    self.block_k,
                    self.num_stages,
                    self.direct_output,
                    self.group_walk,
                ) = config
        self.output_shape = (m, n if activation is None else n // 2)

    def make_program(self):
        return _grouped_gemm_program(
            self.m,
            self.num_groups,
            self.n,
            self.k,
            str(self.ab_dtype).removeprefix("torch."),
            str(self.cd_dtype).removeprefix("torch."),
            self.activation,
            self.has_bias,
            self.config["num_cores"],
            self.block_m,
            self.block_n,
            self.block_k,
            self.num_stages,
            self.direct_output,
            self.group_walk,
        )

    def __call__(self, a, b, layout, bias=None, *, out=None):
        check_tensor(a, "a", (self.m, self.k), self.ab_dtype)
        check_tensor(b, "b", (self.num_groups, self.n, self.k), self.ab_dtype, a.device)
        check_tensor(layout, "layout", (self.num_groups,), torch.int32, a.device)
        if (bias is not None) != self.has_bias:
            raise ValueError("bias presence differs from the built specialization")
        if bias is not None:
            check_tensor(bias, "bias", (self.num_groups, self.n), self.ab_dtype, a.device)
        check_out(out, self.output_shape, self.cd_dtype, a.device, (a, b, layout, bias))
        if a.device.type not in ("npu", "privateuseone"):
            raise ValueError("GroupedGEMM requires tensors on an NPU")
        output = (
            torch.empty(self.output_shape, dtype=self.cd_dtype, device=a.device)
            if out is None
            else out
        )
        if self.m:
            kernel_bias = (
                bias.unflatten(1, (self.n // 2, 2))
                if bias is not None and self.activation == "swigluoai"
                else bias
            )
            self.compile(a.device)(
                a, b, layout, output, *([kernel_bias] if bias is not None else [])
            )
        return output
