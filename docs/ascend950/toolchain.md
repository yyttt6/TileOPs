# Ascend 950 toolchain contract

This file and `config/ascend950-toolchain.json` define one CI profile. A version
string such as "CANN 9.2" or "TileLang 0.1.15" alone is insufficient: prereleases,
source builds and release wheels can expose different compiler interfaces.

## Profile v1

| Component | Exact selection |
| --- | --- |
| Python (canonical CI) | 3.12.15 |
| torch | 2.13.0+cpu |
| torch-npu | 2.13.0rc1 |
| TileLang | official 0.1.15 ARM64 wheel; native binary/template hashes in the lock |
| Device compiler and headers | CANN **9.2.0 release**, from the locked vendor artifact; beta.2 is a different SDK |
| CANN runtime and OPP | **9.1.0**, separately fingerprinted compatibility components |
| Host C++ compiler | GCC 12.3.0 |
| Target | dav-3510, physical Ascend950DT_9582 with 32 Cube / 64 Vector |

This profile upgrades the compiler SDK, while retaining the already deployed runtime
and OPP. It is an explicitly pinned compatibility combination, **not a full CANN
9.2.0 runtime installation**. Do not replace its runtime libraries with the SDK's
libraries implicitly. A runtime upgrade requires a new validated profile.

The vendor artifact and extracted installer hashes are recorded in the lock.
CI's versioned activation entry is
`/opt/tileops-ci/server2-20261030/config/toolchain-v1.env.sh`.

## Release policy

The accepted profile is the one recorded by the protected branch's toolchain lock
and checked by CI. Developers must reproduce failures with that profile before a
compiler mismatch is treated as an operator defect. An author-side passing result
from another profile is useful evidence, but does not replace the canonical run.

The profile pins Python, torch, torch-npu, TileLang, CANN compiler and headers,
host C++ compiler, and the authoritative device class. Critical native binaries and
headers have SHA256 fingerprints. ARM64 CI is the release gate. Local x86_64 work
must use matching release sources; it is not certified by the ARM64 binary hashes.

Use an isolated Python environment, install the exact snapshot in
`requirements/ascend950-ci.lock.txt` with dependencies already accounted for
(`pip install --no-deps -r requirements/ascend950-ci.lock.txt`), and install the
CANN release artifact identified by the lock. Do not install a beta package under
the same name, overwrite a shared environment, or mix headers from another SDK.

The CI deployment's environment entry is
`/opt/tileops-ci/server2-20261030/config/device-env.sh`; per-job Python environments
are prepared by the adjacent `prepare-python.sh`. Those paths are deployment
locations, not portable developer installation paths.

On the canonical ARM64 environment, check the profile without initializing a device:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/ci/check_ascend_toolchain.py
```

The strict ARM64 binary check deliberately rejects an x86_64 machine. Local
development should match the release sources and package versions, then use the
canonical remote CI for certification; do not compare ARM64 native hashes to x86
binaries or conceal a local source build behind the release version string.

A mismatch reports the component and expected profile before operator compilation.
CPU metadata checks do not reserve the card. Any NPU imports that initialize the
runtime, compilation requiring the runtime, or operator execution still belongs
inside the site's shared `npu-guard run` command.

## Portable activation example

On another installation, substitute your actual roots and compiler path; do not
copy the administration server's absolute prefix:

```bash
export ASCEND950_RUNTIME_ROOT=/path/to/cann-9.1.0
export ASCEND950_COMPILER_ROOT=/path/to/cann-9.2.0
source "$ASCEND950_RUNTIME_ROOT/set_env.sh"
export ASCEND_RUNTIME_ROOT="$ASCEND950_RUNTIME_ROOT"
export ASCEND_OPP_PATH="$ASCEND950_RUNTIME_ROOT/opp"
export ASCEND_AICPU_PATH="$ASCEND950_RUNTIME_ROOT"
export LD_LIBRARY_PATH="$ASCEND950_RUNTIME_ROOT/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"
export ASCEND_HOME_PATH="$ASCEND950_COMPILER_ROOT"
export ASCEND_TOOLKIT_HOME="$ASCEND950_COMPILER_ROOT"
export PATH="$ASCEND950_COMPILER_ROOT/bin:$ASCEND950_COMPILER_ROOT/tools/bisheng_compiler/bin:$PATH"
export CXX=/path/to/g++-12
export USE_ASCEND=ON USE_CUDA=OFF USE_ROCM=OFF ASCEND_NPU_ARCH=dav-3510
export TILELANG_CACHE_DIR=/path/to/private-cache/ascend950-arm64-v1/tilelang
```

Use a fresh profile-specific compilation cache. Existing objects compiled with
beta headers must not be treated as release-SDK validation. All these settings
apply only to the calling shell and its children; shared system SDK links and
other developers' runtime installations remain separate.

## Acceptance and upgrades

- Correctness and regression acceptance use server2's physical 32 Cube / 64 Vector
  Ascend950DT_9582. A 36-core development device is not a performance substitute.
- A profile must pass CPU contracts plus representative Cube, Vector, and public
  operator correctness checks before becoming the default. Record the exact cases;
  a smoke test is not full-regression evidence.
- Upgrade through a dedicated toolchain change. Test the candidate in a separate
  prefix and cache, retain the previous profile, then update the lock and CI together.
- Run full registered-operator regression after an upgrade and retain its failures;
  do not relax numerical tolerances to accommodate a toolchain change.
- Unmerged compiler features (such as cross-block batch scheduling) are outside the
  release profile. A PR must supply a correct supported fallback or first land and
  validate a new pinned compiler profile. A hardware core-count check alone does
  not establish that a compiler feature exists.
- Correctness and performance are separate. Remeasure affected TileOPs timings
  when their compilation profile changes. Baseline caches are reusable only when
  their own hardware, source, environment and timing identities still match.
  This contract governs TileOPs development/CI; do not silently rebuild original
  vendor baselines with the TileOPs compiler profile or alter their isolated environments.

The older `development.md` runtime paragraph is superseded by this contract and
its machine-readable lock. It must not be used to equate beta and release builds.
