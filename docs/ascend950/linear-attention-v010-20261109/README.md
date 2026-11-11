# Ascend linear attention v0.1.0 acceptance

All ten operators cover 415 manifest cases on server1 physical 32 AIC / 64 AIV, with runtime limits of 32 / 64. The seven newly validated operators cover 340 cases. DeltaNetChunkFwdOp, DeltaNetChunkBwdOp and KDAFwdOp reuse previously accepted server1 measurements after exact compute/dependency source checks and read-only replay of 68 original audits. All ten also pass the final integrated public and compile tests on the delivered source tree.

| Operator               | Cases | Minimum retained ratio | Evidence                                                   |
| ---------------------- | ----: | ---------------------: | ---------------------------------------------------------- |
| GLARecurrentFwdOp      |     8 |               1.122357 | [GLARecurrentFwdOp.json](GLARecurrentFwdOp.json)           |
| GLAFwdOp               |    12 |               0.810305 | [GLAFwdOp.json](GLAFwdOp.json)                             |
| GLAChunkFwdOp          |     8 |              12.820077 | [GLAChunkFwdOp.json](GLAChunkFwdOp.json)                   |
| GLAChunkBwdOp          |     8 |               1.004817 | [GLAChunkBwdOp.json](GLAChunkBwdOp.json)                   |
| DeltaNetRecurrentFwdOp |     8 |               2.452484 | [DeltaNetRecurrentFwdOp.json](DeltaNetRecurrentFwdOp.json) |
| DeltaNetFwdOp          |    18 |               1.031668 | [DeltaNetFwdOp.json](DeltaNetFwdOp.json)                   |
| GDNFwdOp               |   278 |               0.850677 | [GDNFwdOp.json](GDNFwdOp.json)                             |
| DeltaNetChunkFwdOp     |     9 |               0.974579 | [DeltaNetChunkFwdOp.json](DeltaNetChunkFwdOp.json)         |
| DeltaNetChunkBwdOp     |     9 |               0.821163 | [DeltaNetChunkBwdOp.json](DeltaNetChunkBwdOp.json)         |
| KDAFwdOp               |    57 |               0.901816 | [KDAFwdOp.json](KDAFwdOp.json)                             |

The acceptance threshold is baseline/candidate >= 0.8 for every retained constraint. The minimum includes stricter historical constraints; historical local 36-core measurements capped at 32 cores keep that provenance. Device mean15 and independent synchronized host mean15 are never compared across metrics. See [all 415 benchmark rows](BENCHMARK.md), the per-operator CSVs, and [summary/source hashes](SUMMARY.json).

## Validation

- Final public suite: 167 passed, 84 skipped, 45 CUDA-only deselected. Architecture skips and CUDA-only deselections are retained in the raw collection records; they are not counted as passed.
- Original compile contracts: 30 passed across all ten operators (cold fullgraph versus eager, fake metadata, graph ownership), using the installed AscendC Inductor backend.
- New regression tests: 22 passed; input generation and independent reference calculations live in workloads/linear_attention/ascend_cases.py.
- Correctness, guarded input/output buffers, actual generated kernel source and complete device-profile windows are audited in the per-operator reports.
- Toolchain hashes match the initial locked profile. Final public and compile receipts bind the delivered source tree; production snapshot retention and format-only changes have source/AST evidence. KDA reference placement is documented separately below.

## Evidence boundaries

- New performance measurements use the complete FFTS+AICPU device protocol with 15 windows, 3 warmups and a 1 GiB cache flush; host measurements use a separate process. Older baseline protocols remain explicitly identified.
- KDA retains its original FP64 reference mathematics; on NPU inputs the reference runs on CPU and returns results to the input device. The initial integrated run exposed 11 reference-path failures; the failure record and placement-only AST proof are included. Production KDA kernels and numerical tolerances remain unchanged.
- GDN uses the unchanged workload CPU reference. Earlier FLA qualification/exception records remain historical evidence, not a new FLA rerun; native performance constraints are retained.
- DeltaNetFwdOp retains only three input-digest-scoped packed FLA reference exceptions, independently checked against FP64 recurrence. No numerical tolerance was widened.
- One GDN host batch receipt lost process visibility and is excluded from performance acceptance; its valid same-source replacement is used. The invalid receipt remains archived.
- Raw profiles, generated source, input tensors and guard logs remain under reports/linear7-server1-v010-20261109/evidence and the original reports referenced by each JSON. They are retained workspace artifacts, not bundled GitHub downloads. Compact committed reports contain their hashes and paths.

## Test node delta

```text
Base: baseline (merge base 65f2bb847)

File                                     Base    HEAD    Delta
--------------------------------------------------------------
tests/ops/test_deltanet_ascend.py         new       4    (new)
tests/ops/test_gdn_ascend.py              new       4    (new)
tests/ops/test_gla.py                      51      50       -1
tests/ops/test_gla_ascend.py              new       4    (new)
tests/ops/test_gla_training_ascend.py     new      10    (new)
--------------------------------------------------------------
TOTAL                                      51      72      +21

Growth: +41.2%
```

The net +21 is 22 new cases minus the existing CUDA GEMM tiling-rejection test now marked cuda_only; that test remains available on CUDA. dtype0/dtype1 mean FP16/BF16 except the gradient tests, where they mean FP32/BF16. lengths0/lengths1 mean (63,) and (1, 31, 33, 63).

- `test_ascend_gla_k128_scores_match_direct_recurrence[dtype0-lengths0]`: K=128 diagonal-score reduction, FP64 direct recurrence, fixed/packed tails and input preservation.
- `test_ascend_gla_k128_scores_match_direct_recurrence[dtype0-lengths1]`: K=128 diagonal-score reduction, FP64 direct recurrence, fixed/packed tails and input preservation.
- `test_ascend_gla_k128_scores_match_direct_recurrence[dtype1-lengths0]`: K=128 diagonal-score reduction, FP64 direct recurrence, fixed/packed tails and input preservation.
- `test_ascend_gla_k128_scores_match_direct_recurrence[dtype1-lengths1]`: K=128 diagonal-score reduction, FP64 direct recurrence, fixed/packed tails and input preservation.
- `test_ascend_gla_training_preserves_fp32_boundaries[dtype0-3-17-19]`: FP32 chunk-state boundaries, rectangular state, recurrent fallback and input preservation.
- `test_ascend_gla_training_preserves_fp32_boundaries[dtype0-32-128-64]`: FP32 chunk-state boundaries, rectangular state, recurrent fallback and input preservation.
- `test_ascend_gla_training_preserves_fp32_boundaries[dtype0-64-64-128]`: FP32 chunk-state boundaries, rectangular state, recurrent fallback and input preservation.
- `test_ascend_gla_training_preserves_fp32_boundaries[dtype1-3-17-19]`: FP32 chunk-state boundaries, rectangular state, recurrent fallback and input preservation.
- `test_ascend_gla_training_preserves_fp32_boundaries[dtype1-32-128-64]`: FP32 chunk-state boundaries, rectangular state, recurrent fallback and input preservation.
- `test_ascend_gla_training_preserves_fp32_boundaries[dtype1-64-64-128]`: FP32 chunk-state boundaries, rectangular state, recurrent fallback and input preservation.
- `test_ascend_gla_initial_gradient_matches_direct_autograd[dtype0-3-17-19]`: optional initial-state gradient against independent autograd, recurrent/chunked paths.
- `test_ascend_gla_initial_gradient_matches_direct_autograd[dtype0-32-64-64]`: optional initial-state gradient against independent autograd, recurrent/chunked paths.
- `test_ascend_gla_initial_gradient_matches_direct_autograd[dtype1-3-17-19]`: optional initial-state gradient against independent autograd, recurrent/chunked paths.
- `test_ascend_gla_initial_gradient_matches_direct_autograd[dtype1-32-64-64]`: optional initial-state gradient against independent autograd, recurrent/chunked paths.
- `test_normalized_delta_state_and_tail[129-False-dtype0]`: normalized K=128 packing, multi-chunk recurrence, tail, optional initial state and non-aliasing.
- `test_normalized_delta_state_and_tail[129-False-dtype1]`: normalized K=128 packing, multi-chunk recurrence, tail, optional initial state and non-aliasing.
- `test_normalized_delta_state_and_tail[2049-True-dtype0]`: normalized K=128 packing, multi-chunk recurrence, tail, optional initial state and non-aliasing.
- `test_normalized_delta_state_and_tail[2049-True-dtype1]`: normalized K=128 packing, multi-chunk recurrence, tail, optional initial state and non-aliasing.
- `test_ascend_gdn_packed_rectangular_transforms[17-19-dtype0]`: packed empty sequence and tail, rectangular v-first state, raw gate/beta sigmoid/negative eigenvalue/L2 normalization and non-aliasing.
- `test_ascend_gdn_packed_rectangular_transforms[17-19-dtype1]`: packed empty sequence and tail, rectangular v-first state, raw gate/beta sigmoid/negative eigenvalue/L2 normalization and non-aliasing.
- `test_ascend_gdn_packed_rectangular_transforms[128-64-dtype0]`: packed empty sequence and tail, rectangular v-first state, raw gate/beta sigmoid/negative eigenvalue/L2 normalization and non-aliasing.
- `test_ascend_gdn_packed_rectangular_transforms[128-64-dtype1]`: packed empty sequence and tail, rectangular v-first state, raw gate/beta sigmoid/negative eigenvalue/L2 normalization and non-aliasing.

## Static checks

Changed Python files pass Ruff checks and formatting. Repository-wide pre-commit remains blocked by pre-existing target-branch findings: int8 kernel naming/extern rules, unrelated formatting/whitespace, and 19 normalization-report SHA256 cache keys flagged by gitleaks. The cache-key findings were inspected as existing hexadecimal digest fields; unrelated source and reports were not changed.
