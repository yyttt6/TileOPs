# Ascend 950 INT8 W8A8 GEMM

This contribution adds every `GemmINT8W8A8FwdOp` manifest case on Ascend 950
(950DT, 32 Cube cores): INT8 activations against an INT8 weight, contracted in
INT32, then scaled by a per-token (or per-tensor) activation scale and a
per-channel (or per-tensor) weight scale, with an optional bias, and written as
bfloat16 or float16. 29 workload cases cover Llama-3 8B/70B and Qwen2.5-7B/72B
under SmoothQuant, from batch-1 decode to 4096-row prefill, including the
per-tensor and mixed-granularity variants and the one biased QKV projection.

The product is exact: every partial sum is an integer far below the INT32 range,
so the only rounding after the cube is the single cast to the output dtype. The
kernel reproduces the reference's epilogue order — `acc * scale_a * scale_b`,
then `+ bias`, in FP32 — so a result can only differ from the reference by which
side of one rounding boundary it lands on.

## Schedule

Two passes, one kernel each:

- **Product.** A native INT8 NT matmul: a 256-deep K stage with two operand
  versions at every row count, a 16/64/128/256-row tile that grows with the
  call, and a 224-wide column tile wherever N allows it. The accumulator is
  INT32 in L0C, and one 64-deep MAD step feeds each L0 operand version.
- **Epilogue.** `out = cast(x * scale_a * scale_b + bias)` on the vector unit,
  one row per block in column tiles. The INT32-to-FP32 widening runs on the
  vector unit because a scalar `Cast` between those two types is a bit
  reinterpretation in this backend, not a conversion; the bias is widened once
  per block, and a per-tensor scale is one broadcast vector.

## Measured performance

One task-protocol round on one idle 950DT_9582 (32 Cube / 64 Vector) with the
documented PyTorch 2.13 stack: 3 warm-up calls, a 1 GiB eviction before every
measured call, and the mean of the raw `ffts_profile` device execution-interval
unions over 25 calls in one profiler window. No clock lock or platform setting
was changed. **This is a task-protocol result, not a public formal receipt**:
the maintained clock policy and baseline package are still unpublished, so
performance acceptance remains pending.

Latency is microseconds. The two passes are timed in separate windows, so the
`Σ stages` column is their sum, not a single measured call of the complete op.

| Case                      | M×N×K           | Product µs | Epilogue µs | Σ stages µs | TFLOP/s |
| ------------------------- | --------------- | ---------- | ----------- | ----------- | ------- |
| `llama3-8b-qkv-b1`        | 1×6144×4096     | 11.5       | 2.6         | 14.0        | 3.6     |
| `llama3-8b-qkv-b64`       | 64×6144×4096    | 13.6       | 3.2         | 16.7        | 192.9   |
| `llama3-8b-prefill-qkv`   | 4096×6144×4096  | 275.0      | 67.5        | 351.7       | 586.2   |
| `qwen2.5-7b-prefill-down` | 4096×3584×18944 | 665.1      | 50.1        | 717.3       | 775.4   |
| `llama3-70b-prefill-down` | 4096×8192×28672 | 2365.1     | 82.3        | 2448.5      | 785.9   |

Prefill rows reach 586-786 TFLOP/s end to end; the product alone runs at
750-836 TFLOP/s. The epilogue is bandwidth-bound on the INT32 accumulator it
reads back and the output it writes — 6 bytes per element — which is 24% of the
forward time on the shortest prefill case and 3% on the longest.

There is no like-for-like baseline in this container: `torch_npu`'s
`npu_weight_quant_batchmatmul` serves A16W4/A16W8 (BF16 or FP16 activations), and
no ACLNN entry point takes INT8 activations with these scale granularities. The
product pass alone is therefore the same-op reference, and the ratio isolates
the epilogue.

## Correctness

The 29 manifest cases were replayed through the public entry point on the same
950DT:

```bash
python scripts/benchmark_ascend.py \
  --owner <owner>/w8a8-verify \
  --run-dir <new-directory> \
  --guard <site-guard> \
  --op GemmINT8W8A8FwdOp \
  --verify-only
```

The replay passed 29 of 29 cases (`status: passed`, source fingerprint
unchanged) on TileLang 0.1.15, PyTorch 2.13.0+cpu, torch-npu 2.13.0rc1 and
CANN 9.2.

This site maps logical device 0 to `/dev/davinci1`, so the guard is the
repository's `scripts/npu_guard.py` with that one physical node remapped; it
keeps the same host lock path, PID-identity and ancestry checks. Receipts are
kept outside the repository; the environment, the schedule table and the
per-case result are recorded in [`w8a8-gemm-provenance.json`](w8a8-gemm-provenance.json).

## Remaining work

- Formal performance acceptance waits for the public clock policy and matching
  baseline package; the numbers above are a task-protocol result.
- The epilogue reads the INT32 accumulator back from GM. Folding it into the
  Cube launch needs a mixed Cube/AIV kernel, which this TileLang version does
  not schedule; until then a short prefill case pays up to a quarter of its
  forward time there.
