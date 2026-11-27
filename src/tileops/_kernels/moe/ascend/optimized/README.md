# Single-device Ascend MoE forward

Import the staged-tight operators from `tileops.ascend_moe`. The entry point accepts router logits, hidden states and local expert weights. It returns one hidden vector per token.

```python
from tileops.ascend_moe import FusedMoEFwdOp

op = FusedMoEFwdOp(top_k=8, renormalize=True)
# hidden: [T, H], logits: [T, E], gate_up: [E, 2*F, H], down: [E, H, F]
output = op(hidden, logits, gate_up, down)
```

| Entry point | Inputs and outputs |
| --- | --- |
| `FusedTopKFwdOp` | Scores router logits and returns FP32 weights and int32 expert IDs, both `[T, K]`. Uses the shared Ascend TopK kernel for softmax, sigmoid and grouped selection. |
| `MoEPrePermuteFwdOp` | Packs hidden rows by expert into `[T*K, H]`. Returns cumulative exclusive expert ends and an inverse map from each route slot to its packed row. |
| `MoEExpertMLPFwdOp` | Computes the grouped gate/up projection with a gated activation, then the grouped down projection. Returns packed expert outputs. |
| `MoEPostPermuteFwdOp` | Gathers packed rows by the inverse map, weights and sums the K routes in FP32, and applies the routing scale. |
| `FusedMoEExpertsFwdOp` | Takes preselected routes and composes pre-permute, expert MLP and post-permute. |
| `FusedMoEFwdOp` | Takes logits, invokes top-k and then the experts pipeline. The experts epilogue applies the routing scale once. |

Use contiguous NPU tensors, with FP16 or BF16 hidden states and expert weights. The combined path supports up to 256 local experts in multiples of 32. Expert weights are output-major. The pre-permute, MLP and post-permute entry points take `ContiguousLayoutSpec.tight_physical_psum()` from `tileops.moe`. Reuse an operator instance for repeated calls on the same stream so it can reuse its compiled kernels and scratch buffers. Use a separate instance per concurrent stream.

The default grouped GEMM configuration uses 36 Cube cores on the measured Ascend 950DT. On a 32-Cube configuration pass `num_cores=32` to the MLP, experts or full forward constructor.
