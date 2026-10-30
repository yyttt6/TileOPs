"""Exact uniform BF16 pooling using 768-feature DMA tiles and ordered FP32 sums."""

CASES = {"uniform-8k-bfloat16-float32": (1, 8192, 8192, 64, 128, 1, False, "bfloat16", "float32")}
VARIANTS = {"mean03-w768": (768, "flat")}
UB_DEFAULT_BYTES = 248 * 1024


def supports(variant, *params):
    return variant in VARIANTS and tuple(params) in CASES.values()


def configuration(variant, *params):
    if not supports(variant, *params):
        raise ValueError("wide-DMA Mean requires an exact audited static gate")
    bw, mapping = VARIANTS[variant]
    b, seq, width, chunk, chunks, sn, ragged, dtype, accum = params
    assert width % 64 == 0 and bw % 64 == 0
    ft = (width + bw - 1) // bw
    explicit = 2 * (64 + 1) * bw * 2
    assert explicit <= UB_DEFAULT_BYTES
    return dict(
        path=variant,
        feature_tile=bw,
        token_tile=64,
        feature_tiles=ft,
        last_feature_width=width - (ft - 1) * bw,
        stages=2,
        blocks=64,
        mapping="chunk-owned" if ragged else mapping,
        live_metadata=ragged,
        explicit_ub_bytes=explicit,
        default_ub_budget=UB_DEFAULT_BYTES,
        nominal_ub_bytes=256 * 1024,
        simt=False,
        reciprocal_hoisted=False,
        jobs=b * chunks * ft,
    )


def mean_program(variant, batch, seq, width, chunk, chunks, seq_num, ragged, dtype, accum_dtype):
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    cfg = configuration(
        variant, batch, seq, width, chunk, chunks, seq_num, ragged, dtype, accum_dtype
    )
    bw, ft, blocks = cfg["feature_tile"], cfg["feature_tiles"], 64
    tail = cfg["last_feature_width"]
    chunk_jobs = batch * chunks
    jobs = chunk_jobs * ft
    chunk_owned = cfg["mapping"] == "chunk-owned"
    rounds = (
        (chunk_jobs + blocks - 1) // blocks * ft if chunk_owned else (jobs + blocks - 1) // blocks
    )

    @T.macro
    def reduce_tile(staged, output, count, active_width):
        # The active_width argument is compile-time full width or exact tail.
        # Tail uses only initialized whole vectors: no masked-off UB loads,
        # no stale lanes, no SIMT DCache. Full/tail branches are outside tokens.
        with T.SimdVF():
            full = S.pset(32)
            if ragged:
                integer_count = S.vdup(T.cast(count, "int32"), "int32", full)
                denominator = S.vcvt(integer_count, "float32", full)
                reciprocal = S.vdiv(S.vdup(T.float32(1), "float32", full), denominator, full)
            running = S.alloc_local((active_width // 64,), "float32")
            for r in T.Unroll(active_width // 64, explicit=True):
                running[r] = S.vdup(T.float32(0), "float32", full)
            for token in T.serial(count):
                for r in T.Unroll(active_width // 64, explicit=True):
                    value = S.vcvt(S.vld(staged[token, r * 64], dist="UNPK_B16"), "float32", part=0)
                    running[r] = S.vadd(running[r], value, full)
            for r in T.Unroll(active_width // 64, explicit=True):
                if ragged:
                    average = S.vmul(running[r], reciprocal, full)
                else:
                    average = S.vmuls(running[r], 1.0 / 64, full)
                narrowed = S.vcvt(average, dtype, full, sat=False)
                S.vsts(output[r * 64], narrowed, full, "PK_B32")

    if not ragged:

        @T.prim_func
        def uniform(
            X: T.Tensor((batch, seq, width), dtype),
            Offsets: T.Tensor((seq_num + 1,), "int32"),
            Indices: T.Tensor((chunks, 2), "int32"),
            Y: T.Tensor((batch, chunks, width), dtype),
        ):
            with T.Kernel(blocks) as core:
                staged = T.alloc_shared((64, bw), dtype)
                output = T.alloc_shared((bw,), dtype)
                T.annotate_buffer_versions({staged: 2, output: 2})
                for iteration in T.Pipelined(rounds, num_stages=2):
                    # Immutable expression preserves range proofs. A mutable
                    # alloc_var here prevented bounds simplification and added
                    # an unnecessary PIPE_MTE3 drain in first CPU codegen.
                    job = (
                        (iteration // ft * blocks + core) * ft + iteration % ft
                        if chunk_owned
                        else iteration * blocks + core
                    )
                    if job < jobs:
                        feature = job % ft
                        col = feature * bw
                        row = job // ft % chunks
                        b = job // (ft * chunks)
                        valid = T.min(bw, width - col)
                        # UB row pitch is bw*2 even for a narrow final tile.
                        T.copy(
                            X[b, row * 64 : (row + 1) * 64, col : col + valid], staged[:, :valid]
                        )
                        if feature < ft - 1:
                            reduce_tile(staged, output, 64, bw)
                        else:
                            reduce_tile(staged, output, 64, tail)
                        T.copy(output[:valid], Y[b, row, col : col + valid])

        return uniform.with_attr("tilelang_callee_allocated_outputs", 0)

    @T.prim_func
    def ragged_main(
        X: T.Tensor((batch, seq, width), dtype),
        Offsets: T.Tensor((seq_num + 1,), "int32"),
        Indices: T.Tensor((chunks, 2), "int32"),
        Y: T.Tensor((batch, chunks, width), dtype),
    ):
        with T.Kernel(blocks) as core:
            staged = T.alloc_shared((64, bw), dtype)
            output = T.alloc_shared((bw,), dtype)
            T.annotate_buffer_versions({staged: 2, output: 2})
            # Same metadata ownership as production16, including live count.
            for outer in T.serial((chunk_jobs + blocks - 1) // blocks):
                chunk_job = outer * blocks + core
                if chunk_job < chunk_jobs:
                    row = chunk_job % chunks
                    b = chunk_job // chunks
                    sequence = Indices[row, 0]
                    local_chunk = Indices[row, 1]
                    start = Offsets[sequence] + local_chunk * chunk
                    end = T.min(start + chunk, Offsets[sequence + 1])
                    count = end - start
                    for feature in T.Pipelined(ft, num_stages=2):
                        col = feature * bw
                        valid = T.min(bw, width - col)
                        T.copy(X[b, start:end, col : col + valid], staged[:count, :valid])
                        if feature < ft - 1:
                            reduce_tile(staged, output, count, bw)
                        else:
                            reduce_tile(staged, output, count, tail)
                        T.copy(output[:valid], Y[b, row, col : col + valid])

    return ragged_main.with_attr("tilelang_callee_allocated_outputs", 0)
