"""One fullwidth6 streaming Mean candidate. Strict ordered FP32 carry, no NPU here."""

CASES = {
    "uniform-batched-float16-float32": (2, 2048, 8192, 64, 32, 1, False, "float16", "float32"),
}
VARIANTS = {"mean17-fullwidth6": (6, 8192)}


def supports(variant, *params):
    return variant in VARIANTS and tuple(params) in CASES.values()


def configuration(variant, *params):
    if not supports(variant, *params):
        raise ValueError("Mean17 requires only the exact uniform FP16 batched gate")
    return dict(
        path=variant,
        token_tile=6,
        feature_tile=8192,
        stages=2,
        blocks=64,
        independent_accumulators=8,
        vector_feature_group=512,
        mapping="chunk-owned",
        input_dma="one contiguous strip",
        fp32_state_elements=8192,
        token_order="sequential exact FP32 carry; scale only after final token",
        live_metadata=params[6],
        reciprocal_hoisted=params[6],
        explicit_ub_bytes=246016 if params[6] else 245760,
        output_dma_per_core=1,
        full_row_store=True,
    )


def mean_program(variant, batch, seq, width, chunk, chunks, seq_num, ragged, dtype, accum_dtype):
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    configuration(variant, batch, seq, width, chunk, chunks, seq_num, ragged, dtype, accum_dtype)

    @T.macro
    def reduce_strip(raw, slot, carry, output, live, reciprocal_ub, from_state, final_strip):
        with T.SimdVF():
            full = S.pset(32)
            halves = S.pset(16)
            if from_state or (ragged and final_strip):
                S.mem_bar("VST_VLD")
            if ragged and final_strip:
                reciprocal = S.vld(reciprocal_ub[0])
            even = S.alloc_local((4,), "float32")
            odd = S.alloc_local((4,), "float32")
            for group in T.serial(16):
                for pair in T.Unroll(4, explicit=True):
                    if from_state:
                        even[pair] = S.vld(carry[group * 512 + pair * 128])
                        odd[pair] = S.vld(carry[group * 512 + pair * 128 + 64])
                    else:
                        even[pair] = S.vdup(T.float32(0), "float32", full)
                        odd[pair] = S.vdup(T.float32(0), "float32", full)
                for token in T.serial(live):
                    for pair in T.Unroll(4, explicit=True):
                        packed = S.vld(raw[slot, token * 8192 + group * 512 + pair * 128])
                        value_even = S.vcvt(packed, "float32", halves, part=0)
                        value_odd = S.vcvt(packed, "float32", halves, part=1)
                        even[pair] = S.vadd(even[pair], value_even, full)
                        odd[pair] = S.vadd(odd[pair], value_odd, full)
                for pair in T.Unroll(4, explicit=True):
                    if final_strip:
                        lo, hi = S.vintlv(even[pair], odd[pair])
                        if ragged:
                            average_lo = S.vmul(lo, reciprocal, full)
                            average_hi = S.vmul(hi, reciprocal, full)
                        else:
                            average_lo = S.vmuls(lo, 1.0 / 64, full)
                            average_hi = S.vmuls(hi, 1.0 / 64, full)
                        narrow_lo = S.vcvt(average_lo, dtype, full, sat=False)
                        narrow_hi = S.vcvt(average_hi, dtype, full, sat=False)
                        S.vsts(output[group * 512 + pair * 128], narrow_lo, full, "PK_B32")
                        S.vsts(output[group * 512 + pair * 128 + 64], narrow_hi, full, "PK_B32")
                    else:
                        S.vsts(carry[group * 512 + pair * 128], even[pair], full)
                        S.vsts(carry[group * 512 + pair * 128 + 64], odd[pair], full)

    @T.prim_func
    def main(
        X: T.Tensor((batch, seq, width), dtype),
        Offsets: T.Tensor((seq_num + 1,), "int32"),
        Indices: T.Tensor((chunks, 2), "int32"),
        Y: T.Tensor((batch, chunks, width), dtype),
    ):
        with T.Kernel(64) as core:
            flat = T.reshape(X, (batch * seq * width,))
            raw = T.alloc_shared((2, 49152), dtype)
            carry = T.alloc_shared((8192,), "float32")
            output = T.alloc_shared((8192,), dtype)
            T.annotate_manual_multi_buffer(raw)
            T.annotate_buffer_versions({carry: 1, output: 1})
            if ragged:
                reciprocal_ub = T.alloc_shared((64,), "float32")
            row = core % chunks
            b = core // chunks
            if ragged:
                sequence = Indices[row, 0]
                local_chunk = Indices[row, 1]
                start = Offsets[sequence] + local_chunk * chunk
                end = T.min(start + chunk, Offsets[sequence + 1])
                count = end - start
                with T.SimdVF():
                    full = S.pset(32)
                    ic = S.vdup(T.cast(count, "int32"), "int32", full)
                    denom = S.vcvt(ic, "float32", full)
                    reciprocal = S.vdiv(S.vdup(T.float32(1), "float32", full), denom, full)
                    S.vsts(reciprocal_ub[0], reciprocal, full)
            else:
                start = row * 64
                count = 64
            # Fixed11 stages preserve the two-slot pipeline. Past live count,
            # token loop has zero iterations and only the exact FP32 carry moves.
            # No padded zero tokens are added; final scale occurs only at stage10.
            for stage in T.Pipelined(11, num_stages=2):
                live = T.max(T.min(6, count - stage * 6), 0)
                begin = (b * seq + start + stage * 6) * width
                if live > 0:
                    T.copy(flat[begin : begin + live * width], raw[stage % 2, : live * width])
                if stage == 0:
                    reduce_strip(raw, stage % 2, carry, output, live, carry, False, False)
                elif stage == 10:
                    if ragged:
                        reduce_strip(raw, stage % 2, carry, output, live, reciprocal_ub, True, True)
                    else:
                        reduce_strip(raw, stage % 2, carry, output, live, carry, True, True)
                else:
                    reduce_strip(raw, stage % 2, carry, output, live, carry, True, False)
            with T.Task():
                T.copy(output, Y[b, row, :])
                T.ascend_pipe_barrier("PIPE_MTE3")

    return main.with_attr("tilelang_callee_allocated_outputs", 0)
