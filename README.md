<div align="center">

<h1>TileOPs</h1>

<h3>Spec-driven LLM operators for Ascend NPU — built by agents</h3>

<p>The spec is the source; kernels are derived from it and judged against it.</p>

<p>
    <a href="https://yyttt6.github.io/TileOPs.github.io/benchmarks/"><b>Benchmarks</b></a> ·
    <a href="#quick-start"><b>Quick Start</b></a> ·
    <a href="#why-its-different"><b>Why it's different</b></a> ·
    <a href="#how-it-works"><b>How it works</b></a> ·
    <a href="#installation"><b>Installation</b></a>
  </p>
</div>

## Quick Start

```python
import torch
import torch_npu
from tileops.ops import GemmFwdOp

gemm = GemmFwdOp()  # shapes and dtype are inferred at call time

a = torch.randn(1024, 512, device="npu", dtype=torch.float16)
b = torch.randn(1024, 512, device="npu", dtype=torch.float16)

d = gemm(a, b)  # equals a @ b.T
```

Operators are compiled on first use, ACL-Graph compatible, and declare their
`torch.compile(fullgraph=True)` support per op.

## Why it's different

An implementation can be regenerated from its spec; a spec cannot be recovered from an
implementation. The project is organised around the spec rather than around the kernels:

- **The spec is self-contained.** Generation reads it and nothing else, so every constraint on
  the implementation is declared rather than assumed.
- **Acceptance is decidable.** Correctness settles against a declared reference, performance
  against a modelled bound — neither is a judgement call.
- **The operator/kernel split is enforced.** The boundary is checked rather than agreed, because
  an unenforced convention does not survive automated edits.
- **Every number carries its provenance.** A performance figure records which kernel binary
  actually ran, established by tracing the loader — not inferred from the call succeeding.
  A vendor library that silently substitutes its own kernel is detected and reported as such.

## How it works

Each operator is declared in [`src/tileops/manifest/`](src/tileops/manifest/) before it is
implemented. The entry drives code generation, testing, and benchmarking:

```yaml
GemmFwdOp:
  ref_api: "torch.matmul"
  signature: {inputs: {a: {dtype: "float16 | bfloat16"}, b: {dtype: "same_as(a)"}}, ...}
  workloads: [{m: 1024, n: 1024, k: 1024, dtypes: [float16, bfloat16]}]
  roofline: {func: tileops.perf.formulas.gemm_fwd_roofline}
  source: {kernel: ..., op: ..., test: ..., bench: ...}
```

| Field       | Role                                                                            |
| ----------- | ------------------------------------------------------------------------------- |
| `ref_api`   | Reference implementation the tests compare outputs against.                     |
| `signature` | Tensor contract, shape rules, and dtype combinations; enforced at the op layer. |
| `workloads` | Shapes and dtypes the tests and benchmarks cover.                               |
| `roofline`  | Performance model. Efficiency is achieved throughput over the modelled bound.   |
| `source`    | Paths to the kernel, op, test, and benchmark, and the slot-to-kernel map.       |

A validator checks every entry against its implementation in CI, so the declaration and the
code stay in step.

The implementation is split in two layers. **L2**, the Python entry point, owns the
caller-facing contract: validation, dtype casting, and memory layout. **L1**, the TileLang
kernel, owns the NPU implementation. [trust-model.md](docs/design/trust-model.md) defines the
boundary between them.

## How performance is measured

Device time, not wall clock. A call's cost is the union of the intervals the NPU spent
executing its kernels, collected through `torch_npu.profiler`; a run that cannot collect
device activity fails rather than falling back to a different clock. The L2 cache is evicted
between iterations, so a second iteration does not read what the first left behind.

The bar is the **fastest available implementation of the same operator on the same workload**,
whichever library it comes from — an open-source handwritten AscendC kernel, a vendor library,
or an eager composition. Each figure records which one won and which tier it belongs to, so a
comparison cannot be flattered by picking a weak opponent. The
[benchmark pages](https://yyttt6.github.io/TileOPs.github.io/benchmarks/) carry the tables and
the method behind them.

## Installation

TileOPs installs from source.

**Prerequisites**

- Python >= 3.10
- Ascend 910B1 (Atlas A2 training series)
- CANN 8.5.0
- PyTorch >= 2.1 with `torch_npu` (validated on 2.7.1)
- TileLang with the Ascend backend

```bash
git clone https://github.com/yyttt6/TileOPs
cd TileOPs
pip install -e '.[dev]' -c constraints.txt   # constraints.txt pins what CI validates
pre-commit install

python -m pytest -q tests -m smoke           # verify; requires an Ascend NPU
```

## Documentation

|                                                |                                                  |
| ---------------------------------------------- | ------------------------------------------------ |
| [development.md](docs/development.md)          | Build, test, benchmark                           |
| [architecture.md](docs/design/architecture.md) | Module map and the agent production loop         |
| [manifest.md](docs/design/manifest.md)         | The spec format every operator starts from       |
| [ops-design.md](docs/design/ops-design.md)     | Adding an operator, step by step                 |
| [roofline.md](docs/design/roofline.md)         | How performance is scored against Speed-of-Light |
| [trust-model.md](docs/design/trust-model.md)   | What each layer may assume about the others      |
| [kernel-pattern.md](docs/design/kernel-pattern.md) | The Ascend kernel template and its boundary      |

## Contributing

Operators are added through the loop above — start from [ops-design.md](docs/design/ops-design.md),
which walks the path from a manifest entry to a merged kernel.

## License

TileOPs is released under the [MIT License](LICENSE).
