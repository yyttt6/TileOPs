# nightly-bench — Ascend benchmark snapshots

One commit per run. `yyttt6/TileOPs.github.io` (branch `ascend-bench`) renders the
newest one into its Benchmarks page via `scripts/render_bench.sh`.

| file | required | produced by |
|---|---|---|
| `bench_results.xml` | yes | `tileops-ascend-harness/publish/coverage_to_junit.py` |
| `meta.json` | yes — a missing one aborts the render by design | same |
| `test_results.xml` | no | same |

An implementation is a property-name prefix (`<tag>_<metric>`), so the three
backends render as three columns: `hand`, `tilelang`, `mlir`. A backend with no
data on an op is simply absent from that testcase.

Upstream publishes to `tile-ai/TileOPs-nightly`; this branch is the Ascend fork's
equivalent, kept here so no extra repository has to exist.
