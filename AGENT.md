# AGENT.md

Instructions for coding agents working in this repository. `CLAUDE.md` points here; this is
the canonical file.

## Project Overview

TileOPs is a spec-driven LLM operator library for **Ascend NPU**, built on TileLang. There is
no GPU path in this repository: every kernel targets AscendC through TileLang, and every
number is taken on an Ascend device.

This project follows **design-first, spec-driven** development: design docs and
`src/tileops/manifest/` are the authoritative spec; code conforms to the spec, not the other
way around.

## Hardware and stack

| | |
|---|---|
| Device | Ascend 910B1 (Atlas A2 training series) |
| Toolkit | CANN 8.5.0 |
| PyTorch | `torch` >= 2.1 with `torch_npu` (validated on 2.7.1) |
| Compiler | TileLang with the Ascend backend, `bishengir-compile` |

`device="npu"`, not `"cuda"`. `torch.npu.synchronize()`, not `torch.cuda.synchronize()`.
`import torch_npu` is required before any NPU tensor is created.

⚠️ `ASCEND_RT_VISIBLE_DEVICES` **renumbers** devices: after setting it, the visible card is
logical id `0`, and `aclrtSetDevice()` takes a logical id in `[0, aclrtGetDeviceCount())`.

## Development Environment

Activate a virtual environment, then `pip install -e '.[dev]' -c constraints.txt && pre-commit install`.
See [docs/development.md](docs/development.md).

## Key References

### Design

- [architecture.md](docs/design/architecture.md) — system modules, data flow, agent production loop, directory structure
- [ops-design.md](docs/design/ops-design.md) — Op interface execution guide (how to add a new op)
- [ops-design-reference.md](docs/design/ops-design-reference.md) — Op interface detail reference (interface tables, codegen, naming, protocol)
- [manifest.md](docs/design/manifest.md) — `src/tileops/manifest/` spec format (signature, workloads, roofline, source)
- [roofline.md](docs/design/roofline.md) — the `roofline` field spec: performance model, authoring, and per-consumer contracts
- [kernel-pattern.md](docs/design/kernel-pattern.md) — the Ascend kernel template, and what a builder may and may not supply

### Process

- [trust-model.md](docs/design/trust-model.md) — trust boundaries (manifest → test → implementation → benchmark), workloads layer contract
- [testing.md](docs/design/testing.md) — test/benchmark framework, core abstractions, tolerances, reporting rules
- [tileops-skills.md](docs/tileops-skills.md) — developer decision guide: which repo-provided skill to use for which task

## Reading the ops manifest

The manifest lives at `src/tileops/manifest/`, one or more YAML files per op family — most
families use a single file; large families may be sharded across multiple files. The
`tileops.manifest` package merges them into a single `ops` dict at runtime.

- **Programmatic reads**: prefer `from tileops.manifest import load_manifest, load_workloads`. Never re-implement the merge.
- **Structural inspection**: parse the relevant family file with `yaml.safe_load` and index `ops` by op name. Pick the file from the op's family field rather than scanning all of them.
- **Edits**: edit the single family file that owns the op. Use a round-trip parser (`ruamel.yaml`) to preserve comments and key order. Op names must remain unique across files — duplicates raise at load time.
- Reserve `Read`/`grep` for targeted line lookups inside one family file, not structural reading.

## Measuring performance

**Device time, not wall clock.** A call's cost is the union of the intervals the device spent
executing its kernels, via `torch_npu.profiler` with `ProfilerActivity.NPU`. A run that cannot
collect device activity **fails** rather than falling back to a host clock. The L2 cache is
evicted between iterations (192 MiB of L2 → a 384 MiB eviction buffer).

**The bar is the fastest available implementation**, whichever library it comes from. Never
pick a baseline by declaration; pick it by measured speed, and record the tier of the winner.

🚨 **A successful call does not prove whose kernel ran.** A custom operator package that is
missing a kernel binary **silently falls back to the CANN built-in** — the call succeeds, the
numbers are correct, and nothing warns. Any claim that a handwritten kernel was measured must
carry loader-level evidence (a trace showing the expected `.o` was opened). Without that
evidence the tier is `vendor`, with no exceptions.

## Domain Rules (load on demand)

Read the relevant context file **before** modifying files in that domain. Do not load them if
your task does not touch that domain.

| When you modify                                                   | Read first                                                                               |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `tests/`                                                          | [.claude/domain-rules/testing-budget.md](.claude/domain-rules/testing-budget.md)         |
| `src/tileops/manifest/`                                           | [.claude/domain-rules/manifest-spec.md](.claude/domain-rules/manifest-spec.md)           |
| `scripts/validate_manifest.py`, `tests/test_validate_manifest.py` | [.claude/domain-rules/manifest-validator.md](.claude/domain-rules/manifest-validator.md) |
| `src/tileops/ops/`                                                | [.claude/domain-rules/ops-design.md](.claude/domain-rules/ops-design.md)                 |
| `src/tileops/kernels/`                                            | [docs/design/kernel-pattern.md](docs/design/kernel-pattern.md)                           |
| `benchmarks/`                                                     | [.claude/domain-rules/benchmark.md](.claude/domain-rules/benchmark.md)                   |
| `workloads/`                                                      | [docs/design/trust-model.md](docs/design/trust-model.md)                                 |
| `docs/design/`                                                    | [.claude/domain-rules/design-docs.md](.claude/domain-rules/design-docs.md)               |
