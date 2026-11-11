# Golden reference output cache

Nightly and PR performance CI share `/opt/tileops-ci/golden-reference-v1`.
`golden_cache_policy.json` classifies all 206 manifest operators: 2 high, 47 medium,
157 off. Both high and medium attempt cache lookup; off keeps online reference
execution. This policy does not enable missing Ascend implementations or certify
failed references. The catalog contains 2267 manifest cases, including operators
not yet selected by nightly. Classification details are in `golden-cache-catalog.csv`.

For each eligible case the final timed TileOPs output is verified exactly as before.
A reference call hashes its actual tensor bytes (including shape, dtype, device,
stride and scalar inputs). A matching cache entry supplies expected tensors;
otherwise the original reference runs. Different negative-control inputs get
independent keys. Missing, stale, unreadable or corrupt entries fall back online.
Cache files use CPU tensors, SHA256 and `torch.load(weights_only=True)` and are
moved back to their original devices before the original verifier compares them.

New entries remain private in memory until all output checks, negative controls,
coverage checks and custom probes succeed. Exceptions, incomplete verification,
nonfinite outputs, input-mutating or RNG-consuming references do not publish new
entries. The cache never substitutes TileOPs output for an independent reference.
Existing corrupt entries remain preserved and are treated as misses.

Keys include a conservative hash of workloads, benchmark helpers and all in-tree
TileOPs source/manifest files, the pinned external reference overlay, verified
compiler lock, package versions, device, cache protocol and helper source. This
initial implementation invalidates entries on kernel edits as well as reference
edits; it does not promise reuse across every PR. Commit IDs alone are not keys.
External CI toolchain preflight remains mandatory. A new cache protocol changes
nightly checkpoint identity; existing checkpoints are retained but not mixed with
changed harnesses.

Workflow defaults: 256 GiB maximum cache payload and 200 GiB minimum free disk.
Admission is locked and directory publication atomic. Limits stop new writes, not
cache reads or online correctness. No automatic eviction or evidence deletion.
Change `ASCEND950_GOLDEN_CACHE_MAX_BYTES` / `ASCEND950_GOLDEN_CACHE_MIN_FREE_BYTES`
for a deliberate capacity policy. Unset `ASCEND950_GOLDEN_CACHE_DIR` to disable.
The 206-operator interface output estimate is ~295 GiB per normal-output generation;
49 enabled operators account for ~141 GiB, before control outputs/version copies.
The cap intentionally does not promise storage for every historical version.

Per-case results include `golden_cache` events (hit/miss/stored/bypass/not_stored),
keys, fallback reasons and lookup elapsed time including input identity checks.
These events are diagnostic; measured kernel latency and baseline data are unchanged.
Do not interpret cached reference time as improved TileOPs kernel performance.

Validation covers wrong kernel on a cache hit, distinct negative inputs, changed
contracts, corrupt files, incomplete checks, failed probes, input mutation/RNG,
nonfinite results, and disk budgets. Representative guarded device trials additionally
exercise cold population and fresh-process hits with the unmodified profiler and
numerical checks. No baseline remeasurement is required for this infrastructure change.
