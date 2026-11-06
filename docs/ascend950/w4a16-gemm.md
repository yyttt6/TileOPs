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

Two shapes of the same work, chosen by how many row tiles the call has:

- **One row tile** (batch-1 to 128 rows): a *fused* kernel. The vector unit unpacks
  each weight tile into unified memory and copies it into L1 for the cube, so the
  weight never reaches GM — the call reads `N*K/2` bytes and writes only its output.
  The unpack is not duplicated across row tiles here, which is what makes the fusion
  pay; it still serializes against the cube (see below), so it is cheaper than the
  two-pass form rather than free.
- **More row tiles**: the two-pass form below, because the fused kernel would unpack
  the same weight tile once per row tile.

The two-pass form is three passes, one kernel each:

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

| Case                   | M×N×K         | Path     | Forward µs | Vendor A16W4 µs | Ours / vendor |
| ---------------------- | ------------- | -------- | ---------- | --------------- | ------------- |
| `decode-l2-resident`   | 1×8192×8192   | fused    | 113.4      | 43.4            | 2.61x         |
| `decode-hbm-streaming` | 1×8192×16384  | fused    | 190.0      | 74.9            | 2.54x         |
| `decode-non-pow2`      | 1×7168×20480  | fused    | 207.3      | 107.5           | 1.93x         |
| `decode-long-k`        | 1×8192×81920  | fused    | 921.5      | 593.8           | 1.55x         |
| `decode-b32`           | 32×8192×8192  | fused    | 115.9      | 52.7            | 2.20x         |
| `prefill-mlp-up`       | 512×8192×8192 | two-pass | 268.3      | 168.0           | 1.60x         |

The vendor column is `torch_npu`'s `npu_weight_quant_batchmatmul` (A16W4 with
per-group scales) on the same shapes in the same round: the like-for-like baseline
for this function, and the strongest one available here. Fusion took the decode
rows from 137.4 to 113.4 µs (1.2x) and from 301.1 to 207.3 µs (1.45x) against the
two-pass form.

What is left is the unpack itself: it is compute-bound at ten vector operations per
128 values, while the vendor's is roughly twice as cheap per element. That gap is
not reachable from this SIMD surface — the four-bit trick it needs works on 32-bit
lanes, and `vand`/`vor` here are 8- and 16-bit only. Closing it needs a compiler
op (a packed int4-to-FP16 widening) or an overlap the schedule cannot currently
express: the fused kernel serializes its vector stage against the cube, because a
unified-memory-to-L1 copy lowers to a scatter plus a copy on two pipes and `T.Task`
accepts one pipe per task.

**Baseline.** `torch_npu`'s `npu_weight_quant_batchmatmul` serves the same function
(A16W4 with per-group scales) and is the like-for-like baseline on this device; the
vendor column above is that measurement, taken in the same round on the same shapes.

The site snapshot's rows for this op are not usable as they stand. Its `torch-eager`
column records 426 335 µs for `decode-l2-resident` — 0.04 GB/s on a 16.8 MB weight,
which is a fallback path rather than a measurement, and 12 of its 15 rows exceed
100 ms. The `ops-transformer` and `ops-nn` rows are plausible in magnitude but sit on
a flat 12 408-12 420 µs plateau across batch 2/8/32 and `decode-l2-resident`, a 0.1%
spread across four different token counts on one N×K, which is dispatch overhead
rather than a shape-dependent kernel time. Both are raised for re-collection
separately; until then the in-round vendor number is the only baseline used here.

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
