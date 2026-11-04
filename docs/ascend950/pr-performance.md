# PR performance

PR CPU preflight is followed by changed-operator NPU correctness and then performance.
Performance starts only after every selected operator passes correctness. All device
imports and tests use the host npu-guard and canonical server2 toolchain.

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
