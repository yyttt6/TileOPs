# Elementwise validation on physical 32-core hardware

The elementwise benchmark runs in the isolated CI environment described in
[toolchain.md](toolchain.md). It requires the `ascend950-arm64-v1` profile and an
Ascend 950DT device with 32 Cube cores and 64 Vector cores. Ordinary pull requests
continue to run CPU checks; this benchmark is an explicit manual workflow.

`scripts/ci/elementwise32/run.py` prepares a separate job workspace and rejects a
missing, incomplete or changed baseline catalog before any device initialization.
The admitted catalog replaces all local 36-core measurements with audited remote
32-core measurements, preserving the original source and verification evidence.

The initial suite contains 70 elementwise operators and 527 exact manifest cases.
Every complete operator command runs under the host `npu-guard`. Hardware and
runtime core limits, complete case coverage, input preservation, source hashes,
guard receipts and all 15 positive finite profiler samples are checked before a
result can pass. The benchmark uses three warmups and 15 independent profiling
sessions, with 1 GiB cache eviction and synchronization outside each measured
window. Version 3 sums distinct kernel task spans, including overlapping kernels,
and averages all 15 samples. The AIC and AIV records of one mixed-core task count
as one kernel. Host launch gaps are excluded. Every raw profiling window is
retained and reparsed on CPU before its measurement can pass.

The fresh eager result is also checked against the same expected outputs and
unchanged workload tolerances. A measured eager callable that fails this check is
diagnostic data and is excluded from the baseline comparison. This is especially
relevant when a target-device reference differs from the established CPU semantic
reference.

The archived local development measurements are preliminary. Physical 32-core CI
results supply the authoritative timings. Numerical comparisons retain the
workload tolerances and the established CPU semantic references where the NPU
reference differs. Public and boundary regressions are run separately through
`payload/tools/ci_elementwise_public.py`; original reference discrepancies must
remain recorded and must not be described as original-reference passes.

Ascend public tests use independent CPU references for Sinusoidal's FP32 phase
calculation, BF16 true division and the float32 floored division modes. The
original tolerances, NaN checks and exact bitwise assertions are retained. The
NPU reference differs on subnormal divisors, subnormal quotients and some integer
boundaries. A preferred-implementation test must subclass an implementation that
serves the selected device; a CUDA-only subclass cannot serve an NPU call.

The initial physical32 public run preserved 770 passes and six failures. Separate
guarded probes established the numerical reference discrepancies without changing
production kernels. Any corrected targeted run is recorded alongside the original
run, with source equivalence for unaffected tests and retained profiler windows.

Private guard, environment and source evidence stays on the CI host. Website data
contains measured values and measurement identifiers, without machine-origin or
timing-mode labels. Missing or unverified TileOPs results retain a NaN ratio.
