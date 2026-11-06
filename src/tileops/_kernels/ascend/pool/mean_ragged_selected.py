"""Selected ragged mean programs with live device metadata.

Static shape/dtype gates only; offsets and indices are read by every invocation.
"""

CASES = {
    "ragged-even-float16": (1, 8192, 8192, 64, 128, 4, True, "float16", "float32"),
    "ragged-even-float32": (1, 8192, 8192, 64, 128, 4, True, "float32", "float32"),
    "ragged-batched-float16": (2, 2048, 8192, 64, 32, 2, True, "float16", "float32"),
}


def supports(*params):
    return tuple(params) in CASES.values()


def configuration(*params):
    if not supports(*params):
        raise ValueError("Only the three exact ragged canonical metadata shapes")
    dtype = params[-2]
    width = 256 if dtype == "float32" else 512
    item = 4 if dtype == "float32" else 2
    return dict(
        path="ragged-live-metadata-simd",
        feature_tile=width,
        token_tile=64,
        stages=2,
        explicit_ub_bytes=2 * 65 * width * item,
    )


def mean_program(batch, seq, width, chunk, chunks, seq_num, ragged, dtype, accum_dtype):
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    config = configuration(batch, seq, width, chunk, chunks, seq_num, ragged, dtype, accum_dtype)
    bw = config["feature_tile"]
    feature_tiles = width // bw
    chunk_jobs = batch * chunks
    blocks = 64
    assert chunk_jobs % blocks == 0 and width % bw == 0
    assert config["explicit_ub_bytes"] <= 192 * 1024

    @T.prim_func
    def main(
        X: T.Tensor((batch, seq, width), dtype),
        Offsets: T.Tensor((seq_num + 1,), "int32"),
        Indices: T.Tensor((chunks, 2), "int32"),
        Y: T.Tensor((batch, chunks, width), dtype),
    ):
        with T.Kernel(blocks) as core:
            staged = T.alloc_shared((64, bw), dtype)
            output = T.alloc_shared((bw,), dtype)
            T.annotate_buffer_versions({staged: 2, output: 2})
            # LIFETIME AUDIT:
            # Scalar sequence/start/end/count are loaded once per chunk OUTSIDE
            # the feature pipeline and remain stable through its prologue/drain.
            # staged[f]: MTE2 producer -> one compute VF consumer.
            # output[f]: that VF producer -> MTE3 consumer; both have two versions.
            # running registers live only inside one VF; no UB accumulator or
            # cross-feature recurrence. The next outer chunk reloads metadata.
            for outer in T.serial(chunk_jobs // blocks):
                job = outer * blocks + core
                row = job % chunks
                b = job // chunks
                sequence = Indices[row, 0]
                local_chunk = Indices[row, 1]
                start = Offsets[sequence] + local_chunk * 64
                end = T.min(start + 64, Offsets[sequence + 1])
                count = end - start
                for feature in T.Pipelined(feature_tiles, num_stages=2):
                    col = feature * bw
                    # Runtime extent supports short chunks without reading the
                    # next sequence or padding/stale lanes. No scalar GM staging.
                    T.copy(X[b, start:end, col : col + bw], staged[:count, :])
                    with T.SimdVF():
                        full = S.pset(32)
                        integer_count = S.vdup(T.cast(count, "int32"), "int32", full)
                        denominator = S.vcvt(integer_count, "float32", full)
                        reciprocal = S.vdiv(
                            S.vdup(T.float32(1), "float32", full), denominator, full
                        )
                        for v in T.serial(bw // 64):
                            running = S.alloc_local((1,), "float32")
                            running[0] = S.vdup(T.float32(0), "float32", full)
                            for token in T.serial(count):
                                if dtype == "float32":
                                    value = S.vld(staged[token, v * 64])
                                else:
                                    value = S.vcvt(
                                        S.vld(staged[token, v * 64], dist="UNPK_B16"),
                                        "float32",
                                        part=0,
                                    )
                                running[0] = S.vadd(running[0], value, full)
                            average = S.vmul(running[0], reciprocal, full)
                            if dtype == "float32":
                                S.vsts(output[v * 64], average, full)
                            else:
                                narrowed = S.vcvt(average, dtype, full, sat=False)
                                S.vsts(output[v * 64], narrowed, full, "PK_B32")
                    T.copy(output, Y[b, row, col : col + bw])

    return main.with_attr("tilelang_callee_allocated_outputs", 0)
