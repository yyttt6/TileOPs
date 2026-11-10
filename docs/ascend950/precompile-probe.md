# Guarded precompile cache diagnostic

PR #99 and #100 already supply persistent cache identities, merge-base selection,
compile descriptions and guarded ReLU precompilation. This optional probe reuses
those implementations; it does not add a second automatic precompile step or
replace the trusted performance harness.

Run from a clean checkout in the canonical CI environment. First obtain a passed
receipt from the site's `check-toolchain-v1.py` with NPU autoload disabled. Then:

```bash
npu-guard run --owner ci/relu-cache-probe --wait 1800 --timeout 3600 -- \
  python scripts/ci/precompile_probe.py \
    --source "$PWD" \
    --toolchain /absolute/path/to/passed-toolchain.json \
    --output /absolute/path/to/new-probe-directory \
    --workers 4
```

The output directory must not exist. It owns a new, initially empty cache and
keeps the supplied toolchain receipt, source commit/tree identity, stage logs and
JSON receipts. The caller must supply a fresh canonical toolchain check; hashing
that receipt is provenance, not a replacement for running the check.

Three separate subprocesses inherit the device reservation and each checks it
before candidate imports:

1. Enumerate all ReLU programs without launching the device compiler.
2. Run the merged `precompile.run` driver against the empty cache.
3. Construct and verify all three ReLU cases through the public operator and
   benchmark binder, with complete reference/output checks and no compiler launches.

`result.json` passes only if every process exits successfully, the cold stage
compiles all enumerated programs, and the fresh process verifies all pilot cases
without recompilation. Partial cases, warm caches and subprocess failures cannot
be reported as successful reuse. The default is four workers; the existing 1–16
bound is retained. GEMM is not part of this probe.

The counter observes direct `bisheng` and `ld.lld` launches through Python's
`subprocess.Popen`, matching the supported TileLang Ascend compilation path. It
is not a process-wide audit of native subprocess creation or a device-access
proof. The guard supplies device exclusion. Compiler commands and full logs are
retained, rather than inferring hits from directory counts or short build times.

This is a cache-reuse/correctness diagnostic, not a latency benchmark or an
option-invalidation audit. Formal performance acceptance still uses the unchanged
trusted harness, including its 15 raw profiling windows. A green source-only CI
run does not establish that this opt-in device experiment has been executed.
