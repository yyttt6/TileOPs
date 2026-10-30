# Ascend 950 CI

The maintained branch is `ascend950`. Historical `main` / Ascend 910B workflows are retired.

| Trigger | Checks | Runner |
| --- | --- | --- |
| Pull request or push | CPU manifest, API and benchmark contracts, packaging coverage | `tileops-cpu-server2` |
| Manual `correctness` | CPU checks, then the selected operators on the supplied revision | `tileops-npu-server2` |
| Daily 00:00 UTC or manual `regression` | CPU checks, then every registered Ascend operator's manifest cases | `tileops-npu-server2` |

Use Actions → Ascend 950 → Run workflow. For a PR, supply its exact head SHA and comma-separated manifest operator names. Device checks require a maintainer-triggered run; ordinary PR updates do not request the card. External PR workflow approval remains enabled.

Both runners run in the existing container on server2 (ARM64, Ascend950DT_9582,
32 Cube / 64 Vector). NPU correctness and full regression results from this host
are authoritative. Each device job checks the physical core counts inside the
host shared NPU reservation and rejects other hardware. CPU jobs disable NPU
autoload and use a separate runner workspace; CPU checks do not reserve the card.

Deployment root: `/opt/tileops-ci/server2-20261030`. Runtime snapshots, job Python
environments, runner workspaces and compilation caches are isolated from baseline
and development directories. CPU TorchInductor uses a private GCC 12 extracted
under `compiler/`; the default system compiler is unchanged. Both runners share the existing container's user and
device permissions: this is environment separation, not a security sandbox.
Do not run unreviewed external code with device jobs; selected NPU tests require a
maintainer dispatch. Daily 00:00 UTC runs full registered-operator correctness.

Device jobs use `/usr/local/libexec/ascend-npu/npu_guard.py` and the existing
`/run/lock/ascend-npu` host coordination entry. They wait up to 10 minutes for the
card and never terminate another development or baseline task. Device jobs are
serialized. Regression is correctness-only; performance measurement and baseline
cache refresh remain separate. Receipts, hardware identity and source fingerprints
are retained as Actions artifacts. A green CPU result does not prove NPU correctness.

The supervisor retries a crashed runner. After the platform restarts the enclosing
container, run `/opt/tileops-ci/server2-20261030/start-runners.sh` to restore the
listeners if the platform does not persist foreground services.
