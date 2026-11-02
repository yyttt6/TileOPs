# Ascend 950 W4A16 GEMM

This contribution adds every `GemmW4A16FwdOp` manifest case on Ascend 950 (950DT,
32 Cube cores): FP16 activations against an unsigned INT4 weight with a per-group
zero point and scale, dequantized to FP16 and contracted with FP32 accumulation.
15 workload cases cover the decode shapes the op targets — batch 1 at three hidden
sizes, a long-K streaming case, batched decode and one prefill tile — plus the
single- and multi-tile compile checks.

Weight element `(n, k)` is `(q - zero[n, g]) * scale[n, g]` with `g = k // 128` and
`q` its unsigned INT4 value. The prepacked input is the manifest's marlin-layout
order: inside every 128-value group a word carries the four even K values of one
lane in its low half and the four odd ones in its high half.

## Schedule

Three passes, one kernel each:

- **Unpack.** `wb[n, k] = fp16((q - zero) * scale)` in the packed order, one vector
  step per 128-value group: the 64 packed bytes give 128 nibbles through the same
  byte-nibble interleave the W4A8 kernel uses, the zero point is one integer
  subtract (so the signed widening carries the sign), and the group scale is one
  FP16 multiply. The pass runs at 1.8-1.9 TB/s — the bandwidth bound for reading
  `N*K/2` bytes and writing `2*N*K`.
- **Permute.** The packed order is a fixed permutation of K inside each group, so
  the activation is gathered by the same 128-entry index once per call. It costs
  `M*K` reads and writes: on the batch-1 shapes that is 2-9 µs, two orders below
  the `N*K` weight it would otherwise be permuting.
- **GEMM.** A native FP16 NT matmul: a 128-deep K stage with two operand versions
  (16-bit operands double the L1 footprint, so the stage is half the depth the
  one-byte GEMMs use), a 16/64/128/256-row tile that grows with the call, and a
  224-wide column tile wherever N allows it. FP32 accumulation in L0C, stored
  straight to the FP16 output.

## Measured performance

One task-protocol round on one idle 950DT_9582 (32 Cube / 64 Vector) with the
documented PyTorch 2.13 stack: 3 warm-up calls, a 1 GiB eviction before every
measured call, and the mean of the raw `ffts_profile` device execution-interval
unions over 25 calls in one profiler window. No clock lock or platform setting was
changed. **This is a task-protocol result, not a public formal receipt**: the
maintained clock policy and baseline package are still unpublished, so performance
acceptance remains pending.

Latency is microseconds; the stages are measured separately, so `forward` is their
sum.

| Case                   | M×N×K         | Unpack µs | Permute µs | GEMM µs | Forward µs | TFLOP/s |
| ---------------------- | ------------- | --------- | ---------- | ------- | ---------- | ------- |
| `decode-l2-resident`   | 1×8192×8192   | 93.9      | 2.0        | 41.2    | 137.4      | 1.0     |
| `decode-hbm-streaming` | 1×8192×16384  | 185.1     | 2.3        | 82.3    | 272.4      | 1.0     |
| `decode-non-pow2`      | 1×7168×20480  | 201.9     | 2.2        | 96.3    | 301.1      | 1.0     |
| `decode-long-k`        | 1×8192×81920  | 908.7     | 2.1        | 416.9   | 1325.0     | 1.0     |
| `decode-b32`           | 32×8192×8192  | 94.1      | 2.6        | 41.8    | 139.8      | 30.7    |
| `prefill-mlp-up`       | 512×8192×8192 | 93.8      | 9.4        | 164.9   | 268.4      | 256.0   |

The decode rows are bandwidth-bound on the weight, and the two-pass design is what
costs: the GEMM alone reads the FP16 weight at 3.3 TB/s, but the unpack has already
written those 134 MB once. A kernel that dequantized each weight tile into L1 and
consumed it in the same pass — the shape a fused A16W4 kernel takes — would remove
the round trip; this backend can copy 16-bit unified-memory tiles into L1, so the
fused schedule is a mixed Cube/AIV restructure rather than a hardware limit, and it
is the next step for this op.

**Baseline.** `torch_npu`'s `npu_weight_quant_batchmatmul` serves the same function
(A16W4 with per-group scales) and is the like-for-like baseline on this device; it
was not measured in this round, so the table reports absolute numbers only.

## Correctness

`scripts/benchmark_ascend.py --op GemmW4A16FwdOp --verify-only`: **15/15 cases
verified**, `status: passed`, source fingerprint unchanged, on TileLang 0.1.15,
PyTorch 2.13.0+cpu, torch-npu 2.13.0rc1 and CANN 9.2.

This site maps logical device 0 to `/dev/davinci1`, so the guard is the repository's
`scripts/npu_guard.py` with that one physical node remapped; it keeps the same host
lock path, PID-identity and ancestry checks. Receipts are kept outside the
repository; the environment, the schedule table and the per-case result are recorded
in [`w4a16-gemm-provenance.json`](w4a16-gemm-provenance.json).

## Remaining work

- Formal performance acceptance waits for the public clock policy and a matching
  baseline package; the numbers above are a task-protocol result.
- The unpack writes the FP16 weight to GM and the GEMM reads it back, which the
  batch-1 rows pay for. Fusing the unpack into the GEMM's operand path is the next
  step; 16-bit tile copies into L1 are already supported here.
