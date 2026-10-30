# Ascend 950 CI

The maintained branch is `ascend950`. Historical `main` / Ascend 910B workflows are retired.

| Trigger | Checks | Runner |
| --- | --- | --- |
| Pull request or push | CPU manifest, API and benchmark contracts, packaging coverage | `tileops-cpu-docker` |
| Manual `correctness` | CPU checks, then the selected operators on the supplied revision | `tileops-npu-docker` |
| Daily 00:00 UTC or manual `regression` | CPU checks, then every registered Ascend operator's manifest cases | `tileops-npu-docker` |

Use Actions → Ascend 950 → Run workflow. For a PR, supply its exact head SHA and comma-separated manifest operator names. Device checks require a maintainer-triggered run; ordinary PR updates do not request the card. External PR workflow approval remains enabled.

Both runners run in Docker on the local server. CPU jobs have no NPU devices or Docker socket. Device jobs use the host's shared NPU lock directory and read-only observation socket, with the host PID namespace so ownership checks use real process identities. They wait up to 10 minutes for the card and fail visibly if it remains unavailable. They never interrupt development jobs. Device jobs are serialized, and have a bounded execution time.

Regression is correctness-only; performance measurement and baseline cache refresh are separate. A green CPU check does not establish device correctness. Results and guard receipts are retained as GitHub Actions artifacts. An operator is included in regression once its implementation is registered in the checked-out commit.

Host deployment configuration: `/opt/tileops-ci/config`; immutable Python environment snapshot: `/opt/tileops-ci/runtime/venv`; containers: `tileops-ci-cpu`, `tileops-ci-npu`; image: `tileops-ci:20261030`. Persistent volumes retain runner configuration and compilation caches. The runners restart automatically with Docker. Deployment evidence and rebuild inputs are in `reports/ci-docker-20261030` in the administration workspace.
