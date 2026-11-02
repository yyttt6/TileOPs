# Ascend 950 W4A8 GEMM

This contribution adds every `GemmW4A8FwdOp` manifest case on Ascend 950 (950DT,
32 Cube cores): an FP8 activation against a signed INT4 weight whose values carry
a 128-element group scale, contracted in FP8, then scaled per token and per
output channel and written as bfloat16. 98 workload cases cover DeepSeek-R1 and
Llama-3 70B up/down projections from batch-1 decode to 4096-row prefill, plus
their eight-way tensor-parallel shards.

The weight is packed two INT4 values per byte, low nibble first, each a signed
two's-complement value in `[-8, 7]`. The reference rounds `q * group_scale` to
`float8_e4m3fn` before contracting, so the kernel materializes exactly that FP8
operand: the value the tensor core multiplies is the value the reference
produces, and the tolerance carries no unpack error.

## Schedule

Three passes, one kernel each. `AscendGemmW4A8Kernel` runs them back to back on
the vector and Cube cores of the same device:

- **Unpack.** `wb[n, k] = fp8(q * group_scale[n, k // 128])`, on the vector
  unit. `UNPK4_B8` puts one packed byte in the low byte of each 32-bit lane, so
  one vector step covers a whole 128-value group: the low nibbles are the even K
  values, the high nibbles the odd ones, `vintlv` re-pairs them into K order and
  `PK4_B32` stores the FP8 lanes back to back. Two rows per block keep the packed
  read and the FP8 write in flight. The pass is bandwidth-bound — it reads
  `N*K/2` bytes and writes `N*K` — and runs at 1.6-1.7 TB/s.
- **GEMM.** A native FP8 NT matmul: a 256-deep K stage with two operand versions
  at every row count, a 16/64/128/256-row tile that grows with the call and does
  not compute rows it has no tokens for, and a 224-wide column tile wherever N
  allows it. The row tile shrinks by 64 rows when the L1 operand budget cannot
  hold the deeper stage at 256 rows. FP32 accumulation in L0C, one 64-deep MAD
  step per L0 operand version.
- **Scale.** `out = bf16(acc * token[m] * channel[n])`, on the vector unit, one
  row per block in column tiles (a wide N is walked in the widest divisor of N
  that keeps the row buffers inside the vector unit's 192 KiB). The pass is
  memory-bound on the FP32 accumulator it reads back.

The unpack has no DeepGEMM counterpart: DeepGEMM's Ascend kernels take MX
E8M0 block scales, whose float32 slots must hold exponent-only factors, so a
per-group FP8 scale grid is not hardware-native there. The GEMM is DeepGEMM's
`fp8_gemm_nt` translated to TileLang — same persistent tile scheduler and L0
staging, with the tiles retuned here.

## Measured performance

One task-protocol round on one idle 950DT_9582 (32 Cube / 64 Vector, nominal FP8
peak 864 TFLOP/s) with the documented PyTorch 2.13 stack: 3 warm-up calls, a
1 GiB eviction before every measured call, and the mean of the raw
`ffts_profile` device execution-interval unions over 25 calls in one profiler
window. No clock lock or platform setting was changed. **This is a task-protocol
result, not a public formal receipt**: the maintained clock policy and baseline
package are still unpublished, so performance acceptance remains pending.

Latency is microseconds. The stages are measured separately, so `forward` is the
sum of the three passes plus their launches.

| Case                | M×N×K           | Unpack µs | GEMM µs | Scale µs | Forward µs | TFLOP/s | FP8 GEMM only µs | Forward / FP8 GEMM |
| ------------------- | --------------- | --------- | ------- | -------- | ---------- | ------- | ---------------- | ------------------ |
| `ds-r1-down-b1`     | 1×7168×18432    | 120.0     | 47.3    | 3.3      | 170.7      | 1.5     | 47.3             | 3.61x              |
| `ds-r1-down-b64`    | 64×7168×18432   | 119.3     | 53.2    | 4.0      | 177.2      | 95.5    | 53.2             | 3.33x              |
| `ds-r1-down-4k`     | 4096×7168×18432 | 119.3     | 1292.8  | 74.3     | 1491.0     | 725.9   | 1292.8           | 1.15x              |
| `ds-r1-up-4k`       | 4096×18432×7168 | 130.6     | 1381.7  | 229.2    | 1749.7     | 618.6   | 1381.7           | 1.27x              |
| `llama-70b-down-4k` | 4096×8192×28672 | 206.2     | 2363.0  | 86.3     | 2659.1     | 723.6   | 2363.0           | 1.13x              |

Prefill rows land at 619-726 TFLOP/s end to end (72-84% of the FP8 peak), and
the GEMM alone at 837 TFLOP/s (97% of peak) on the DeepSeek-R1 down shape. The
unpack's 119 µs on the DeepSeek-R1 weight is the HBM floor for its 198 MB of
traffic.

Two tile decisions carry most of the GEMM's rate. The K stage is 256 deep at
every row count — a 128-deep stage leaves the Cube waiting on its operands even
at batch 1 — and the column tile is 224 wherever N allows it, which splits 7168
into exactly 32 tiles, one per Cube core. Both are why a batch-1 call reads the
FP8 weight at 2.8 TB/s instead of 1.5.

The FP8-GEMM column is the strongest runnable reference on this site, and it is
the right one for this op: the FP8 matmul is the same Cube work, so the ratio is
exactly what the unpack and the scale pass add. DeepGEMM's Ascend FP8 NT kernel
does not build here — its JIT fails inside its own `copy.hpp` against the
installed CANN 9.2.0 (`asc_set_l0c2gm_nz2nd` has no matching overload), and the
same package produced the baselines in the grouped-GEMM contribution five days
ago, so the failure is in the toolchain pairing rather than in this op. No
`torch_npu` or ACLNN entry point takes FP8 activations: `npu_weight_quant_batchmatmul`
serves A16W4/A16W8 (BF16/FP16 activations), which runs at the BF16 tensor-core
rate and is not a like-for-like baseline for this function.

The decode rows are the honest weak point: at batch 1 the unpack costs 2.5x the
GEMM it feeds, because the two-pass design writes the whole FP8 weight to GM and
reads it back, and that pass does not shrink with the token count. Fusing the unpack into the GEMM's operand path would remove
that round trip, but `asc_copy_ub2l1` carries 16- and 32-bit element types only,
so an FP8 tile cannot reach L1 from the vector unit; the fused schedule needs
either that copy to accept FP8 or a Cube-side unpack, and neither exists today.

## Correctness replay

The 98 manifest cases were replayed on the same 950DT with the checked-out tree:

```bash
python scripts/benchmark_ascend.py \
  --owner <owner>/w4a8-verify \
  --run-dir <new-directory> \
  --guard <site-guard> \
  --op GemmW4A8FwdOp \
  --verify-only
```

`--verify-only` uses the shared workload reference and numerical policy, checks
every output and the input-preservation rule. The replay passed 98 of 98 manifest
cases with the source fingerprint unchanged, on TileLang 0.1.15, PyTorch
2.13.0+cpu, torch-npu 2.13.0rc1 and CANN 9.2. One harness gap is worth naming:
the shared preservation check compares every input with `torch.isclose`, which
this `torch_npu` build does not implement for `float8_e4m3fn`; the replay
compares FP8 inputs bitwise through a `uint8` view, which is the same statement
for those tensors.

This site maps logical device 0 to `/dev/davinci1`, so the guard is the
repository's `scripts/npu_guard.py` with that one physical node remapped; it
keeps the same host lock path, PID-identity and ancestry checks. Receipts are
kept outside the repository; the environment, the schedule table and the
per-case result are recorded in [`w4a8-gemm-provenance.json`](w4a8-gemm-provenance.json).

## Remaining work

- Formal performance acceptance waits for the public clock policy and matching
  baseline package; the numbers above are a task-protocol result.
- The unpack writes FP8 to GM and the GEMM reads it back. Folding the unpack into
  the GEMM is blocked on an FP8 UB-to-L1 copy (see above); until then the
  decode rows pay for a full weight round trip per call.
- The scale pass could move into the GEMM epilogue (L0C to UB, scale, store
  bf16), saving its read-back of the FP32 accumulator, which is 4.5% of the
  prefill forward time.
