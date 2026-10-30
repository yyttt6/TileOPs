# Ascend 950 development

The `ascend950` branch keeps the upstream GPU implementation and adds Ascend kernels
through TileOPs' `tileops.backends` entry point and `register_kernel_type` interface.
The public family imports, op signatures, workloads and numerical contracts are shared.
For example, `from tileops.gemm import BmmFwdOp` serves NPU tensors through the Ascend
implementation and CUDA tensors through the existing CUDA implementation.

The backend currently implements 41 ops across elementwise, Bmm and Cumsum. Registration
does not compile a kernel or query device properties. On an installation without
`torch_npu`, the Ascend entry point performs no registration. Backend import errors in
an NPU installation remain visible through `tileops.load_failures()`.

## Runtime

Use the canonical [Ascend 950 toolchain contract](toolchain.md) and its
`config/ascend950-toolchain.json` lock. It distinguishes CANN releases from beta
builds and pins native binary/header fingerprints in addition to package versions.
A generic "CANN 9.2" or "TileLang 0.1.15" installation is not an environment match.
Do not install GPU dependency bundles into the isolated NPU environment.

The core/dev dependency set uses 0.1.15 on every target. The upstream GPU benchmark
bundle has a separate constraint: vLLM 0.27.1
[pins TileLang 0.1.12](https://pypi.org/pypi/vllm/0.27.1/json).
Consequently the full GPU `[bench]` extra and image need a baseline dependency
migration before they can use this branch. The unchanged `constraints-runner-lock.txt`
records that upstream image, not a validated 950 environment. Do not override these
dependencies with `--no-deps` to claim a compatible GPU benchmark stack.

## Shared benchmark

Every developer and baseline adapter uses this entry point from the same revision:

```bash
python scripts/benchmark_ascend.py \
  --owner developer-name/experiment-name \
  --run-dir /shared/results/unique-run-name \
  --clock-config /site/ascend950-clocks.json \
  --op BmmFwdOp
```

The launcher acquires the host reservation **before runtime imports** and wraps the
whole foreground process. All users must share `/run/lock/ascend-npu/device-0.lock`.
The guard checks driver processes, device handles, health and sustained idle before
launch, and checks for foreign driver activity while running. It is a cooperative
lock: unrelated programs can ignore it. Never delete a held lock or reset a device to
make a run start. Guard exits 75, 76 and 78 invalidate the run.

Sites with a different physical device mapping can supply their audited Python
guard entry with `--guard /absolute/path/to/npu_guard.py` or `TILEOPS_NPU_GUARD`.
It must implement the same host lock and process-ownership contract; this option
does not disable reservation validation. Containers need the actual host lock and
host PID visibility. A private container lock is insufficient.

The fixed formal protocol is `ascend950-profiler-locked-v1`:

| Item         | Rule                                                                                                                                                    |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Correctness  | Workload reference and existing numerical policy; every output checked; shared inputs preserved                                                         |
| Warmup       | 3 calls, after correctness/JIT setup                                                                                                                    |
| Repeats      | 15 independent profiler sessions, one operator invocation each                                                                                          |
| Cache        | Reset, 1 GiB int32 eviction kernel, synchronize, then profile                                                                                           |
| Clock        | Apply site's verified lock command; independently read both components before/after each sample; restore on exit                                        |
| Timer        | `torch_npu.profiler.profile`, raw CANN `ffts_profile` kernel durations                                                                                  |
| Score        | Arithmetic mean of per-call device execution interval unions; host gaps excluded; overlapping kernels count once; duration sum retained as a diagnostic |
| Evidence     | All 15 samples and raw traces, environment, clock readbacks, code/contract hashes, final guard receipt                                                  |
| Invalid runs | Correctness, profiler, clock, contamination or source-change failure prevents acceptance                                                                |
| Event timing | No event or wall-clock fallback in the Ascend timer                                                                                                     |

The run directory must be new. There is no silent resume, sample trimming or per-developer
warmup/repeat override. Consume a result only after `receipt.json` says `passed` and its
`result.verify_only` is false. A row written before a later guard failure is invalid.
The raw profiler directory can be large; archive it with the result instead of deleting
it before review. Results from other protocol IDs, TileLang versions or clock policies
are not interchangeable, including earlier development measurements.

A clock config supplies absolute argv arrays `lock` and `restore`, plus
`component_mhz: [component0_mhz, component1_mhz]`. The site must use a supported,
verified frequency-control tool; `restore` must restore its prior policy even after a
partially failed lock. The benchmark checks DSMI readback and refuses above-rated
frequencies. **No verified lock command means no formal measurement.** A sampled
frequency readback cannot prove that the frequency never changed between samples.

For correctness alone, add `--verify-only` and omit `--clock-config`. This produces no
latency or performance acceptance. Baselines are not needed for correctness CI.

For additional implementations, pass `--implementation-factory package.module:function`.
The function receives `(op_name, case)` and returns a mapping of names to existing
`benchmarks.api.Implementation` objects. Each uses the same reference, inputs, reset
order and timer as TileOPs. Stateful implementations use private arguments and `reset`.
The shared callable API is `benchmarks.ascend.runner.benchmark_case`. The timer's
short search mode uses a distinct protocol ID and cannot produce formal results.

## CI

Local and GitHub runs share the same commands:

```bash
python scripts/ci/ascend950.py --phase cpu --output /results/unique-cpu-run
python scripts/ci/ascend950.py --phase device --output /results/unique-device-run
```

CPU checks validate the manifest, benchmark contracts, backend dispatch and packaging
coverage. Device checks verify every manifest case for every registered Ascend op.
No matched cases, zero passed tests, import errors and source changes fail the run.
Performance acceptance remains pending until a baseline uses this exact measurement
protocol and toolchain. Correctness CI does not imply a speed target was met.

GitHub's `Ascend 950` workflow runs on pushes and PRs to `ascend950`. NPU execution runs
on trusted pushes or manual dispatch, after preflight. Its runner needs the `ascend-950`
label and `ASCEND_ENV_SCRIPT`, pointing to the prepared runtime environment. The test
entry uses the actual checked-out source, not another editable checkout. Existing GPU
workflows and the 910 `main` branch retain their own hardware scope.

The public website uses a separate `Official Ascend 950 nightly` workflow in
`yyttt6/TileOPs.github.io`, scheduled at 00:00 UTC on the remote production runner.
This branch's local correctness CI does not publish website performance. The
remote workflow freezes the source commit before checks and preserves the shared
protocol, raw samples, source hashes and clock readbacks. No supported host clock
control or reservation means the remote device stage remains blocked.

For multiple Linux users, install the repository, environment and result root in
locations those users can traverse. A checkout under a private `/root` directory is
not a shared installation. Each developer should use a separate worktree and run
directory, with the same benchmark revision and clock policy.
