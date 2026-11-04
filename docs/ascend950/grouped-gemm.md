# Ascend 950 Grouped GEMM

This contribution adds every `GroupedGemmFwdOp` manifest case on Ascend 950
(950DT, 32 Cube cores): one GEMM per group with the groups packed back to back
along one axis, FP32 accumulation and results in the operand dtype. Six calls,
seven dtype cases: `moe-fwd` (NT, bfloat16 and float16), `moe-dgrad` (NN,
float16), `moe-wgrad` (TN, float16, the groups split the contraction), and three
`qwen3-30b-ep8` routes (NT, bfloat16). Refs
[#8](https://github.com/yyttt6/TileOPs/issues/8).

The rows are packed without padding: a group starts on any row and holds any
number of them, including none. `AscendGroupedGemmKernel` selects a schedule by
`(numel, experts, N, K, transpose_a, transpose_b)` and refuses every other call;
the table is in
[`grouped.py`](../../src/tileops/kernels/ascend/gemm/grouped.py).

## Schedule

Both walks are DeepGEMM's grouped kernels translated to TileLang, with the group
boundaries read from the device instead of the host:

- **m-grouped (NT, NN).** One flat tile space ordered group-major; each core takes
  the next free tile and decodes its group from a device-side walk over
  `batch_sizes`, so the cores of a wave read one group's weight tile and share it
  in L2. A 256-deep L1 stage feeds four 64-deep L0 MAD steps, which overlaps the
  MTE1 traffic with the cube inside a stage.
- **k-grouped (TN).** The groups split the contraction and every group owns the
  whole `N x K` output, so all groups share one tile space and only the rows they
  contract differ. Both operands are staged in their GM orientation and
  transposed on the way into L0. Each tile opens with a zero contraction (a zero
  tile times a zero tile), so a group the routing leaves empty stores zeros
  through the same path as any other tile.

DeepGEMM's own grouped kernels require group starts on a 256-row boundary
(m-grouped) or a 64-row one (k-grouped), and the k-grouped entry point takes the
group sizes on the host. This schedule decodes the tightly packed sizes on the
device, so a ragged split — the `qwen3-30b-ep8` rows — is served by the same
kernel and has no DeepGEMM counterpart.

## Measured performance

The table records one task-protocol round on one idle 950DT_9582 (32 Cube / 64
Vector, nominal BF16 peak 432 TFLOP/s) with the documented PyTorch 2.13 stack.
The protocol adapter follows issue #1's default-clock rules: 3 warm-up calls, a
1 GiB eviction before every measured call, and the mean of the raw `ffts_profile`
device execution-interval unions over 25 calls in one profiler window. No clock
lock or platform setting was changed. **This is a task-protocol result, not a
public formal receipt**: the maintained clock policy and baseline package are
still unpublished, so performance acceptance remains pending.

DeepGEMM's grouped kernels need the group starts on their alignment, so they
measure the three `moe-*` cases only, in the same round and on the same device
(its extension loads under PyTorch 2.7, so the two arms run in separate
processes). The `qwen3-30b-ep8` cases have no DeepGEMM entry point and compare
against a per-group `torch.mm` loop. Latency is microseconds; `s` is the L1
stage count and `mad` the L0 MAD depth.

| Case                              | M×G×N×K           | Layout | Tiles                   | Candidate µs | TFLOP/s | Peak   | Fastest baseline µs | Ratio   |
| --------------------------------- | ----------------- | ------ | ----------------------- | ------------ | ------- | ------ | ------------------- | ------- |
| `moe-fwd-bfloat16`                | 4096×16×4096×4096 | NT     | 256×256×256, s2, mad 64 | 335.5        | 409.7   | 94.84% | deepgemm 323.4      | 96.39%  |
| `moe-fwd-float16`                 | 4096×16×4096×4096 | NT     | 256×256×256, s2, mad 64 | 335.6        | 409.6   | 94.82% | deepgemm-bf16 323.4 | 96.36%  |
| `moe-dgrad-float16`               | 4096×16×4096×4096 | NN     | 256×256×256, s2, mad 64 | 337.2        | 407.6   | 94.35% | deepgemm-bf16 323.7 | 96.00%  |
| `moe-wgrad-float16`               | 4096×16×4096×4096 | TN     | 256×256×64 (group), s1  | 616.8        | 222.9   | 51.60% | deepgemm-bf16 750.7 | 121.71% |
| `qwen3-30b-ep8-empty-bfloat16`    | 8×16×1536×2048    | NT     | 64×256×256, s2, mad 64  | 21.0         | 2.4     | 0.56%  | torch.mm loop 42.3  | 201.4%  |
| `qwen3-30b-ep8-routed1k-bfloat16` | 1024×16×1536×2048 | NT     | 64×256×256, s2, mad 64  | 39.9         | 161.5   | 37.38% | torch.mm loop 124.0 | 310.8%  |
| `qwen3-30b-ep8-ragged-bfloat16`   | 257×16×1536×2048  | NT     | 64×256×256, s2, mad 64  | 33.8         | 47.8    | 11.07% | torch.mm loop 90.5  | 267.8%  |

Every `moe-*` dtype case reaches at least 96% of its same-round DeepGEMM
baseline, and `moe-wgrad` is 21.7% faster than DeepGEMM's k-grouped kernel. The
three `qwen3-30b-ep8` rows sit outside DeepGEMM's layout contract and run
2.0-3.1× faster than the per-group `torch.mm` loop, which pays one host launch
per group. For reference, that loop takes 521.2 µs on `moe-fwd`, against 335.5
µs here.

## Correctness replay

The seven dtype cases were replayed through the public benchmark entry point on
the same 950DT with the checked-out tree:

```bash
CASES='^(moe-fwd-bfloat16|moe-fwd-float16|moe-dgrad-float16|moe-wgrad-float16|qwen3-30b-ep8-empty-bfloat16|qwen3-30b-ep8-routed1k-bfloat16|qwen3-30b-ep8-ragged-bfloat16)$'
python scripts/benchmark_ascend.py \
  --owner <owner>/grouped-gemm-verify \
  --run-dir <new-directory> \
  --guard <site-guard> \
  --op GroupedGemmFwdOp \
  --case-regex "$CASES" \
  --verify-only
```

`--verify-only` uses the shared workload reference and numerical policy, checks
every output and the input-preservation rule. The replay passed 7 of 7 dtype
cases with the source fingerprint unchanged, on TileLang 0.1.15, PyTorch
2.13.0+cpu, torch-npu 2.13.0rc1 and CANN 9.2. Beyond the manifest shapes, the
same kernel was checked against the per-group reference on splits that stress
the tail: 1, 16, 17, 32 and 64 rows a group, and a zero-row group, all exact.

This site maps logical device 0 to `/dev/davinci7`, so the guard is the
repository's `scripts/npu_guard.py` with that one physical node remapped; it
keeps the same host lock path, PID-identity and ancestry checks. The site
exposes no host-wide `/run/lock` reservation visible from this container, so the
run is a site-scoped correctness replay, not a formal guarded benchmark.
Receipts are kept outside the repository; the environment, the schedule table
and the per-case result are recorded in
[`grouped-gemm-provenance.json`](grouped-gemm-provenance.json).

## Remaining work

- Formal performance acceptance waits for the public clock policy and matching
  baseline package; the numbers above are a task-protocol result.
- A host-wide `/run/lock/ascend-npu` reservation is not visible on this site.
- The m-grouped walk decodes the group of each tile by walking the group sizes;
  a device-side prefix sum would make that O(log G) at large expert counts, which
  no manifest case needs yet.
