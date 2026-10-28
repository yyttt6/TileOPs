# Ascend 950 GEMM

This contribution adds all 20 `GemmFwdOp` manifest cases on Ascend 950
(950DT, 32 Cube cores): 18 BF16 and two float16, `D = A @ B` with contiguous
operands and FP32 accumulation. Seventeen BF16 cases store `b` as `(N, K)`;
`square-1k` stores it as `(K, N)`. The two float16 cases
(`square-1k-float16`, `ds-v3-prefill-mlp-up-float16`) build their BF16 sibling's
schedule with the operand dtype switched; the device runs both 16-bit dtypes at
the same matrix-unit rate. Refs [#8](https://github.com/yyttt6/TileOPs/issues/8).

`AscendGemmKernel` selects a schedule by `(M, N, K)`, dtype and layout and
refuses every other call: a dtype other than float16 or bfloat16, `trans_a`, or
a shape and dtype pair outside the table.
The table is
[`persistent/__init__.py`](../../src/tileops/kernels/ascend/gemm/persistent/__init__.py).

## Schedules

Every case compiles one of eight modules. Three are parameterized factories
shared by several shapes; five are shape-specialized pipelines.

| Module                   | Shapes                                                                                | Origin                                    |
| ------------------------ | ------------------------------------------------------------------------------------- | ----------------------------------------- |
| `dynamic_tail`           | `square-1k`, `qwen3-8b-gate-up-decode`, `qwen3-8b-down-prefill`                       | Public persistent BMM parameterization    |
| `l1_direct_tail`         | `ds-v3-m1-qkv-a`, `ds-v3-m2-qkv-a`, `vit-l16-in21k-head`                              | Public example L1-direct parameterization |
| `swizzled_persistent`    | four prefill projections, `ds-v3-m64-down`, `ds-v3-m96-qkv-a`, `qwen3-8b-gate-up-b32` | Pinned TileLang persistent example        |
| `bf16_nt_128_2112_7168`  | `ds-v3-decode-qkv-a`                                                                  | Existing TileProf schedule, source pinned |
| `bf16_nt_128_7168_2048`  | `ds-v3-decode-down`                                                                   | Existing TileProf schedule, source pinned |
| `bf16_nt_4096_2112_7168` | `ds-v3-prefill-qkv-a`                                                                 | Existing TileProf schedule, source pinned |
| `bf16_nt_16_4096_7168`   | `ds-v3-m16-mlp-up`                                                                    | Existing TileProf schedule, source pinned |
| `bf16_nt_32_4096_7168`   | `ds-v3-m32-mlp-up`                                                                    | Existing TileProf schedule, source pinned |

`square-1k` and `ds-v3-prefill-mlp-up` also serve their float16 dtype case
through the same module; the dtype is a build parameter.

The factories use `T.Pipelined` and `T.Persistent` loops. The recovered
pipelines keep their steady K cycles as `for` loops; their prologue, epilogue
and N tail stay explicit because each step raises a different flag order.

The per-module configuration is the one the campaign measured, not the output of
an unattended search: the five recovered schedules keep their pinned sources,
seven shapes reuse the pinned example's configuration, and only the three cases
below record a comparison against an alternative. The 80% threshold compares
against external baselines, so it is not evidence of a searched optimum.

### Selection notes

- `ds-v3-m1-qkv-a` / `ds-v3-m2-qkv-a`: 16×80 tiles with K=512 instead of
  K=256. Output tiles, L0 buffer versions, MAD sub-K and total MAD work stay
  the same, while the L1 operand footprint grows from 96 KiB to 192 KiB (well
  inside the 512 KiB capacity) and halves the outer copy rounds. The campaign
  moved M=1 from 20.5 µs (16×128) to 15.7 µs (16×80, K=256) and then to
  13.47 µs (K=512).
- `ds-v3-m96-qkv-a`: N=96 instead of N=192. The output goes from 11 tiles to
  22, so 22 rather than 11 of the 32 Cube cores work, and the L0 sub-K grows
  from 64 to 128; the M=96 case moves from 33.7 µs to 25.1 µs.
- `vit-l16-in21k-head`: the L1-direct tail schedule replaces explicit L0
  staging for the non-multiple-of-8 N and moves the case from 51.7 µs to
  42.3 µs.

These are single-change observations on one device, not a cost model.

## Measured performance

The table records the task campaign on one idle 950DT_9582 (32 Cube / 64
Vector, nominal BF16 peak 432 TFLOP/s), under the site's then-current PyTorch
2.7.1 stack; the two float16 rows were measured afterwards with the same
protocol on the same device. The correctness replay below uses the documented
PyTorch 2.13 stack. The protocol adapter follows issue #1's default-clock rules: 3
warm-up calls, 15 independent profiler sessions of one invocation each, a 1 GiB
eviction before every session, and the arithmetic mean of the raw
`ffts_profile` device execution interval unions. No clock lock or platform
setting was changed. **This is a task-protocol result, not a public formal
receipt**: the maintained clock policy and baseline package are still
unpublished, so performance acceptance remains pending.

For the 18 BF16 cases the fastest baseline is the minimum over DeepGEMM,
`torch.matmul` on NPU, ACLNN, CATLASS and the pinned TileLang example, all
measured in the same round on the same device. The two float16 cases are
compared with DeepGEMM's BF16 arm on the same shape, also measured in the same
round: DeepGEMM has no float16 entry point, and the matrix unit runs both
16-bit dtypes at the same rate. Latency is microseconds. Tiles are M×N×K; `s`
is the pipeline stage count, `v` the L1 buffer versions per operand where they
differ.

| Case                               | M×N×K           | Layout | Tiles                           | Candidate µs | TFLOP/s | Peak   | Fastest baseline µs   | Ratio   |
| ---------------------------------- | --------------- | ------ | ------------------------------- | ------------ | ------- | ------ | --------------------- | ------- |
| `square-1k-bfloat16`               | 1024×1024×1024  | NN     | 128×128×128, s3                 | 10.381       | 206.9   | 47.89% | deepgemm 10.062       | 96.93%  |
| `square-1k-float16`                | 1024×1024×1024  | NN     | 128×128×128, s3                 | 10.289       | 208.7   | 48.31% | deepgemm-bf16 8.865   | 86.16%  |
| `ds-v3-decode-qkv-a-bfloat16`      | 128×2112×7168   | NT     | 128×80, L1 K 256, v2, 27 blocks | 25.806       | 150.2   | 34.76% | deepgemm 26.221       | 101.61% |
| `ds-v3-decode-down-bfloat16`       | 128×7168×2048   | NT     | 128×224, L1 K A128/B256, v2     | 15.994       | 235.0   | 54.39% | deepgemm 15.479       | 96.78%  |
| `ds-v3-prefill-qkv-a-bfloat16`     | 4096×2112×7168  | NT     | 256×240, L1 K 128, v4           | 329.267      | 376.6   | 87.19% | deepgemm 292.474      | 88.83%  |
| `ds-v3-prefill-down-bfloat16`      | 4096×7168×2048  | NT     | 256×256×256, s2                 | 291.563      | 412.5   | 95.48% | deepgemm 282.891      | 97.03%  |
| `ds-v3-prefill-mlp-up-bfloat16`    | 4096×4096×7168  | NT     | 256×256×256, s2                 | 566.039      | 424.9   | 98.36% | deepgemm 560.871      | 99.09%  |
| `ds-v3-prefill-mlp-up-float16`     | 4096×4096×7168  | NT     | 256×256×256, s2                 | 566.171      | 424.8   | 98.33% | deepgemm-bf16 560.782 | 99.05%  |
| `ds-v3-prefill-o-proj-bfloat16`    | 4096×7168×16384 | NT     | 256×256×256, s2                 | 2238.755     | 429.7   | 99.48% | deepgemm 2229.831     | 99.60%  |
| `ds-v3-prefill-q-b-bfloat16`       | 4096×24576×1536 | NT     | 256×256×256, s2                 | 747.822      | 413.5   | 95.72% | deepgemm 719.666      | 96.23%  |
| `ds-v3-m16-mlp-up-bfloat16`        | 16×4096×7168    | NT     | 16×128, L1 K 256, vA1/vB2       | 24.097       | 39.0    | 9.03%  | deepgemm 21.910       | 90.92%  |
| `ds-v3-m32-mlp-up-bfloat16`        | 32×4096×7168    | NT     | 32×128, L1 K 256, vA1/vB2       | 26.143       | 71.9    | 16.64% | deepgemm 22.187       | 84.87%  |
| `ds-v3-m64-down-bfloat16`          | 64×7168×2048    | NT     | 64×256×256, s2                  | 14.422       | 130.3   | 30.16% | deepgemm 13.664       | 94.75%  |
| `ds-v3-m96-qkv-a-bfloat16`         | 96×2112×7168    | NT     | 96×96×256, s2                   | 25.083       | 115.9   | 26.82% | deepgemm 23.293       | 92.87%  |
| `ds-v3-m1-qkv-a-bfloat16`          | 1×2112×7168     | NT     | 16×80×512, s2                   | 13.467       | 2.2     | 0.52%  | deepgemm 12.367       | 91.84%  |
| `ds-v3-m2-qkv-a-bfloat16`          | 2×2112×7168     | NT     | 16×80×512, s2                   | 13.474       | 4.5     | 1.04%  | deepgemm 12.320       | 91.43%  |
| `qwen3-8b-gate-up-decode-bfloat16` | 1×24576×4096    | NT     | 16×128×128, s3                  | 68.925       | 2.9     | 0.68%  | deepgemm 58.742       | 85.23%  |
| `qwen3-8b-gate-up-b32-bfloat16`    | 32×24576×4096   | NT     | 32×256×256, s2                  | 67.728       | 95.1    | 22.02% | deepgemm 61.073       | 90.17%  |
| `qwen3-8b-down-prefill-bfloat16`   | 128×4096×12288  | NT     | 128×128×128, s3                 | 54.127       | 238.0   | 55.10% | torch_npu 49.298      | 91.08%  |
| `vit-l16-in21k-head-bfloat16`      | 256×21843×1024  | NT     | 256×256×256, s2                 | 42.277       | 270.9   | 62.70% | deepgemm 38.125       | 90.18%  |

All 20 cases reach at least 80% of their same-round baseline; the floor is
84.87% (`ds-v3-m32-mlp-up`) and the ceiling is 101.61% (`ds-v3-decode-qkv-a`).
DeepGEMM is the fastest reference on 19 cases; `qwen3-8b-down-prefill` is faster
on `torch.matmul`, and the two float16 cases use DeepGEMM's BF16 arm.

## Correctness replay

The 20 cases were replayed through the public benchmark entry point on the same
950DT with the checked-out tree:

```bash
CASES='^(square-1k-bfloat16|square-1k-float16|ds-v3-decode-qkv-a-bfloat16|ds-v3-decode-down-bfloat16|ds-v3-prefill-qkv-a-bfloat16|ds-v3-prefill-down-bfloat16|ds-v3-prefill-mlp-up-bfloat16|ds-v3-prefill-mlp-up-float16|ds-v3-prefill-o-proj-bfloat16|ds-v3-prefill-q-b-bfloat16|ds-v3-m16-mlp-up-bfloat16|ds-v3-m32-mlp-up-bfloat16|ds-v3-m64-down-bfloat16|ds-v3-m96-qkv-a-bfloat16|ds-v3-m1-qkv-a-bfloat16|ds-v3-m2-qkv-a-bfloat16|qwen3-8b-gate-up-decode-bfloat16|qwen3-8b-gate-up-b32-bfloat16|qwen3-8b-down-prefill-bfloat16|vit-l16-in21k-head-bfloat16)$'
python scripts/benchmark_ascend.py \
  --owner <owner>/gemm-bf16-verify \
  --run-dir <new-directory> \
  --guard <site-guard> \
  --op GemmFwdOp \
  --case-regex "$CASES" \
  --verify-only
```

`--verify-only` uses the shared workload reference and numerical policy, checks
every output and the input-preservation rule. The replay passed 20 of 20 cases
with the source fingerprint unchanged, on TileLang 0.1.15, PyTorch 2.13.0+cpu,
torch-npu 2.13.0rc1 and CANN 9.2.

This site maps logical device 0 to `/dev/davinci7`, so the guard is the
repository's `scripts/npu_guard.py` with that one physical node remapped; it
keeps the same host lock path, PID-identity and ancestry checks. The site
exposes no host-wide `/run/lock` reservation visible from this container, so
the run is a site-scoped correctness replay, not a formal guarded benchmark.
Receipts are kept outside the repository; the environment, both source hashes
and the per-case result are recorded in
[`gemm-provenance.json`](gemm-provenance.json).

## Provenance

Each port keeps the measured schedule's kernel body. `gemm-provenance.json`
records, per case, the hash of the measured source, the hash of the ported
module, and the hash of the built TIR. A checker rebuilds each ported
`PrimFunc` and the measured `main` and compares their TVMScript after renaming
the template's operands to the port's names: all 18 BF16 cases are equal. The
two float16 schedules' TIR differs from their BF16 sibling only in the operand
dtype. Seven
`swizzled_persistent` cases also drop an unused 64 KiB shared buffer that the
measured factory allocated for an epilogue branch no case takes; lowering
already eliminates it. Compiling both sides also yields the same generated
kernel source for all 18 BF16 cases after normalizing symbol names and parameter
order.

## Remaining work

- Formal performance acceptance waits for the public clock policy and matching
  baseline package; the numbers above are a task-protocol result.
- A host-wide `/run/lock/ascend-npu` reservation is not visible on this site.
