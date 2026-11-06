# Ascend 950 FP8 GEMM

This contribution adds every `GemmFP8FwdOp` manifest case on Ascend 950 (950DT,
32 Cube cores): FP8 (E4M3) activations against an FP8 weight, with either one
scale per tensor or the reference's per-128-block float32 scale grid, an optional
bias, and a bfloat16 or float16 output. 39 workload cases cover DeepSeek-V3 decode
and prefill projections and their KV-cache shapes.

## Schedule

Two paths, chosen by the scale grid the call carries.

- **Per-tensor scales** — the cube contracts the two FP8 operands into an FP32
  accumulator at the FP8 rate, and one vector pass applies both scales and the
  bias while casting to the output dtype.
- **Per-128-block scales** — `torch._scaled_mm`'s contract is
  `(a * scale_a) @ (b * scale_b).T`, so each operand is folded once
  (`_fold_program`, one FP16 multiply per element) and the product runs as a
  16-bit GEMM. The cube's block-scale slots hold E8M0 factors — a power of two per
  block — so an arbitrary float32 block scale cannot ride the FP8 product itself;
  the alternative, accumulating every 128-deep K block separately and scaling each
  on the vector unit, drains L0C once per block and costs several times the
  contraction. The fold keeps every bit the tolerance can see: the operand has
  four significand bits, and the workload's own policy allows 2% relative.

Both paths use the same tile rule: a 256-deep K stage (128 deep for the 16-bit
path, whose operands are twice as wide), a 16/64/128/256-row tile that grows with
the call, and a 224-wide column tile wherever N allows it.

## Measured performance

One task-protocol round on one idle 950DT_9582 with the documented PyTorch 2.13
stack: 3 warm-up calls, a 1 GiB eviction before every measured call, and the mean
of the raw `ffts_profile` device execution-interval unions over 25 calls in one
profiler window. **This is a task-protocol result, not a public formal receipt.**

| Case            | M×N×K          | Scales     | µs    | TFLOP/s |
| --------------- | -------------- | ---------- | ----- | ------- |
| `decode-qkv-a`  | 128×2112×7168  | per-tensor | 26.0  | 149.2   |
| `prefill-qkv-a` | 4096×2112×7168 | per-tensor | 203.8 | 608.7   |
| `decode-down`   | 128×7168×2048  | per-128    | 65.4  | 57.4    |
| `prefill-down`  | 4096×7168×2048 | per-128    | 455.0 | 264.3   |
| `kv-b`          | 128×32768×512  | per-128    | 174.4 | 24.6    |

The per-tensor path reaches 609 TFLOP/s on prefill (70% of the 864 TFLOP/s peak).
The block path pays the fold and the 16-bit rate: 2.4x the per-tensor cost at the
same shape.

**Baseline.** DeepGEMM's `fp8_gemm_nt` is the like-for-like reference for this op,
and it does not build in this container: its Ascend JIT needs `ascendc_assert`,
which the installed CANN 9.2.0 no longer ships, and pre-deprecation L0C-to-GM
setters. The same package produced the baselines in the grouped-GEMM contribution,
so this is a toolchain pairing, not a kernel problem. `torch_npu` exposes no FP8
matmul entry point.

That is a limit of this container, not of the baseline package. The site snapshot
carries `DeepGEMM-Ascend` rows measured on this same hardware, and they are the
fastest baseline for 6 of the 39 cases; `catlass` is fastest for 15 and
`torch-eager` for 16. Against those rows the per-tensor path is 2.54x
(`ds-v3-decode-qkv-a`, 66.0 µs) and 3.11x (`ds-v3-prefill-qkv-a`, 634.0 µs) faster.
The table above is the absolute measurement; the ratios are in the PR body, which
takes the fastest valid library per case as issue #46 specifies.

## Correctness

All 39 manifest cases were replayed on the same device. The shared runner's
input-preservation check compares inputs with `torch.isclose`, which this
`torch_npu` build does not implement for `float8_e4m3fn`, so the replay uses a
bitwise FP8 comparison through a `uint8` view — the same statement for those
tensors. 39 of 39 cases verified, source fingerprint unchanged, on TileLang
0.1.15, PyTorch 2.13.0+cpu, torch-npu 2.13.0rc1 and CANN 9.2.

The environment, the schedule table and the per-case result are recorded in
[`fp8-gemm-provenance.json`](fp8-gemm-provenance.json).

## Remaining work

- Formal performance acceptance waits for the public clock policy and a matching
  baseline package, and for a DeepGEMM build that pairs with CANN 9.2.
- The block-scale path could return to the FP8 rate if the cube's block-scale
  slots accepted arbitrary float32 factors; today they are E8M0 only.
