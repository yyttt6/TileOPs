# PR performance

PR CPU preflight is followed by one combined measurement/verification pass over changed
operators. Each case completes the normal cold call, three warmups and fifteen profiler
samples, then compares the final timed output against its reference outside the timing
window. There is no separate correctness execution or all-operator correctness pass.
A failed case stops the remaining cases and operators, retains diagnostic/raw evidence,
and invalidates that case's performance. Previously completed evidence remains on disk;
this does not add automatic PR checkpoint/resume support.

Shared inputs are snapshotted before timing and checked afterwards. Existing full-output
coverage, numerical tolerances and negative controls still apply. Custom distribution
probes still run explicitly outside timing: those contracts need multiple executions,
so they cannot be replaced by checking a single saved output. Private arguments/reset
semantics are preserved. The final output is retained without a device copy inside the
profile; snapshots, reference execution and comparisons stay outside the windows.
All device imports and tests use the host npu-guard and canonical server2 toolchain.
Standalone correctness jobs and nightly's default verification path are unchanged.

The selector records changed paths and operator names. Kernel-bearing changes select
explicit operator names. Helper-only changes expand to their kernel family. Shared
backend/manifest changes expand to registered affected operators. Review selection.json
when changing dynamic registrations or dependencies; it is part of the evidence.

The timer is copied from the successfully audited website nightly: three warmups,
fifteen profiler samples, 1 GiB eviction, platform-default clocks, distinct FFTS tasks
plus missing named AICPU work. Raw samples are retained and replayed on CPU.
Guard failures invalidate performance. Native baseline v2 values are preserved and
identified separately from v3 TileOPs measurements. Only identical op/case/params join.
Missing baseline is not performance acceptance. Ratios are informational; this change
does not introduce a numerical merge threshold.

Results and provenance are in the Actions summary and ascend950-pr-performance artifact.
Evidence persists at /opt/tileops-ci/server2-20261030/pr-evidence/ on server2.
For existing PRs, dispatch Ascend 950 with mode=performance and revision=the current PR merge SHA. This also includes current
CI fixes when the PR branch predates them.
Normal PR events run against GitHub's merge revision; manual runs are identified
by their exact commit and are not a substitute for resolving merge conflicts.

### Persistent compilation cache and merge-base selection

PR performance and the official nightly share the server2 cache namespace
`cache/tilelang-v1-<identity>`. `scripts/ci/tilelang_cache_key.py` requires a clean
tracked source tree and a successful canonical toolchain report. It hashes the
Git tree, toolchain lock and verified compiler/header/library fingerprints. Run
IDs, checkout paths and commit messages do not affect the namespace; changed
source trees or toolchains do. TileLang still keys individual programs and their
compile options within that directory. This reuses binaries, not correctness or
performance results. Cache identity design and initial CPU tests are adapted
from PR #97; parallel precompilation is a separate rollout.

For pull-request events, the tested commit must have two parents, with the PR
head as the second parent. Incremental selection compares against the first
parent actually merged, not the possibly stale event payload base. Explicit
manual performance runs retain their specified base behavior. Both the resolved
base and cache identity are preserved with the run artifacts.
