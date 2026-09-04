# Elementwise Ascend Pattern

## 1. Delivery order

1. Read the manifest and the existing TileOPs workload/ref program. Record the
   broadcast rule, scalar attributes, dtype union, and workload shapes.
2. Add one `@register("<ManifestOpName>")` builder in
   `src/tileops/kernels/families/elementwise.py`.
3. Reuse `kernels/elementwise_binary.py`; do not duplicate its kernel skeleton.
   The builder supplies the scalar operation, alpha policy, and dtype whitelist.
4. Run a focused NPU probe covering same-shape, broadcast, a non-divisible
   output, a non-default scalar, and every supported dtype.
5. Run the harness and save its JSON, complete stdout/stderr, card selection,
   and `npu-smi info` under `docs/reports/Rxxx-data/`.

## 2. Template boundary

The template owns broadcast-shape validation, padded zero-stride metadata,
compile-time path classification, multi-core launch, `(cid, vid)` tile
partitioning, UB copies, tail predicates, BF16 scratch/cast, and output reshape.
An operator builder supplies only its manifest name, scalar expression kind,
alpha parameter, and explicit dtype whitelist. A future `SubFwdOp` is a small
builder calling the same template with `op_kind="sub"`; no new indexing or tail
code is allowed.

## 3. Compile-time broadcast paths

Classification happens in `build_binary_kernel`, so it adds no runtime host
work and never materializes `expand().contiguous()`:

- **same**: both operands have the output shape and contiguous strides. The
  kernel launches `ceildiv(out_numel, 16384)` blocks. Each block has two vector
  contexts; context `vid` owns an 8192-element half and uses `T.copy` for GM/UB
  transfers. The final block falls back to predicated scalar loads/stores. The
  8192-element tile is the measured safe upper bound on dav-2201/910B1;
  16384 reaches a runtime UB out-of-bounds vector exception.
- **tail_broadcast**: exactly one operand is contiguous/direct and the other
  has contiguous active axes followed only by broadcast (zero-stride) suffix
  axes. The suffix repeat must be `2..4096`; this covers
  `[16,256,56,56] + [256,1,1]` with repeat `3136` and 4096 logical units.
  Each unit is split between the two `vid` contexts. The direct side is copied
  in a contiguous tile; the broadcast side loads one source scalar and reuses
  it in UB. FP16/FP32 use `T.tile.broadcast`; BF16 casts the scalar to fp32,
  uses a vector broadcast in fp32 UB, and casts the result back with
  `CAST_RINT`, avoiding dav-2201's unsupported BF16 scalar `Muls`/Broadcast.
- **generic**: all other legal broadcast layouts retain the zero-stride
  `_offset_expr` calculation, but distribute 512-element logical blocks over
  the multi-core `(cid, vid)` launch. This is a correctness fallback, not the
  performance path.

Every path uses `idx < out_numel` (and, for tail broadcast, the unit end) on
both load and store. No shape is rounded up on host and no full tail tile is
written.

## 4. Ascend pitfalls

- `threads` is not CUDA's 256. Ascend exposes two vector contexts through
  `(cid, vid)`. Every tile must offset by `vid * tile`; ignoring `vid` leaves
  half of each block unwritten and produces uninitialized-data NaNs.
- `T.Kernel(block_count, is_npu=True)` is a grid of logical tiles, not a CUDA
  thread grid. Keep UB extents compile-time and bounded; the BF16 path has
  fp32 scratch plus storage-dtype buffers.
- `T.copy` is the vectorized GM/UB transfer for contiguous regions. Use a
  predicated scalar tail only when a full static copy would cross the output
  bound.
- BF16 vector fadd and scalar `Muls`/Broadcast are not interchangeable on
  dav-2201. Use fp32 UB arithmetic, vector alpha multiplication when needed,
  and `CAST_RINT` back to BF16. For BF16 suffix broadcast, cast one loaded
  scalar to fp32 and broadcast it in the vector domain.
- Kernel launches are asynchronous. Harness timing synchronizes before and
  after every call; ad-hoc probes must call `torch.npu.synchronize()` before
  reading results or timing.

## 5. Reproduction

```bash
source /home/dyq/miniconda3/etc/profile.d/conda.sh
conda activate tlx
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /home/dyq/workspace/tilelang-ascend/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=1
cd /home/dyq/workspace/tileops-ascend-harness
export PYTHONPATH=/home/dyq/workspace/tilelang-ascend:/home/dyq/workspace/TileOPs/src:/home/dyq/workspace/TileOPs/src
python bench1.py AddFwdOp --shape-set full --baseline vendor --target ascend --out coverage
```

The command writes `coverage/AddFwdOp.json`. It must contain six correctness
cases and six measured cases, all correctness flags true, and `sol_gate: pass`.
The baseline is labelled `vendor` when the ops-math handwritten add cannot be
built as a Python harness adapter.

## Reduction pattern (MeanFwdOp)

The verified row-reduction template in `kernels/reduction.py` flattens all
non-reduced axes to `m` rows and the reduced axes to a contiguous `n` suffix.
The generic path keeps two vector contexts per 128-row block, accumulates in
fp32, and writes a 64-row vector tile to padded GM storage so the output never
uses scalar GM stores. BF16 input is cast to fp32 before `reduce_sum` and cast
back with `CAST_RINT`; fp32 input bypasses the cast.

For the common full-reduction case (`m == 1`), the kernel uses a separate
single-row path. It processes 16384 elements per UB tile, which is the measured
safe upper bound with the fp32 scratch buffer on dav-2201. The output is padded
to 64 elements and written with one vector copy, then sliced back to the scalar
shape by the host view. Explicit per-tile barriers between copy/reduce were
removed while retaining `AUTO_SYNC`; generated code still emits the required
MTE2/V and V ordering events.

Attempts to split the one-row reduction across cores were not retained:
`T.barrier_all` is pipe-local rather than a cross-core GM barrier, and replacing
it with `T.sync_all` triggered an `AUTO_CV_COMBINE` lowering index failure (the
specialized path therefore leaves AUTO_CV_COMBINE disabled). A correct
cross-core version needs explicit cross flags plus a workspace contract.

## 6. Cube GEMM pattern

`GemmFwdOp` uses one compile-time kernel per `(M, N, K, dtype, trans_a,
trans_b)` signature. The public tensors retain all four storage layouts; the
builder derives logical M/N/K, and `T.gemm_v0` receives the matching transpose
flags. No host transpose, pad, contiguous copy, or dtype conversion is allowed.

For fp16/bf16 with fp32 accumulation, the measured Developer-mode tile is:

- `block_M=128` for M >= 128, otherwise 32;
- `block_N=256` for N >= 256, otherwise 64;
- `K_L1=256` for K >= 256, otherwise 64, with `kL0Size=64`.

The large L0C tile is the capacity-derived safe upper bound:
`128 * 256 * 4 = 128 KiB`, exactly the 910B1 L0C capacity. L1 uses at most
`(128*256 + 256*256)*2 = 192 KiB`, below its 511.75 KiB capacity. Increasing
either L0C axis would exceed the hardware budget. M/N/K grids use ceildiv and
all GM transfers/writeback retain bounded slices, so M/N tails are not rounded
or padded on host.

Two tempting schedules are not the default:

- Fixed Core regressed `ratio_min` because serial per-core tile processing cost
  outweighed launch savings.
- Expert L1/L0 double-buffered `T.mma` cut the large prefill case materially,
  but reduced the harness-wide `ratio_min`; preserve the Developer schedule
  until the custom-kernel host launch floor is addressed.

Generated Ascend C currently declares a mixed AIC/AIV task even though GEMM
compute and `copy_l0c_to_gm` are in the AIC branch. MSProf device time can be an
order of magnitude below synchronized Python end-to-end time. Report both
layers and do not present device-only time as the harness ratio. For writeback
diagnosis, synchronize a NaN fill of the adapter-allocated output before launch,
then require zero NaNs and reference correctness; an asynchronous fill without
that synchronization is not a valid sentinel test.

## 7. Decode attention with a continuous KV cache

Decode has no useful Q-sequence tile: one logical task is one `(batch, query
head)` pair. Launch at most 25 AI Cores and cover all logical tasks with
`task_id = cid + round * launch_blocks`. This grid-stride mapping keeps
`BlockDim <= 25` even when `batch * heads > 65535`; a longer cache increases
only the static per-task KV loop. GQA maps `kv_head = query_head // group_size`.

The cache stays BSHD at its maximum allocation. `cache_seqlens[B]` is read on
device. TileLang cannot use a tensor value as a loop extent, so iterate the
compile-time `ceildiv(max_seqlen_kv, 64)` and guard each block and tail element
with the actual batch length. Do not pad or crop on host and do not include
invalid cache positions in online softmax.

The verified CV chain uses one reusable slot per physical core for fp32 QK
scores, storage-dtype probabilities, and fp32 PV partials. Three ready/free
cross-core semaphore pairs enforce `QK -> softmax -> PV -> accumulate`; local
`AUTO_SYNC` must remain enabled for UB conversion and MTE3 completion. Turning
off local auto-sync produced run-to-run corrupt probabilities even though the
cross-core order looked correct.

MLA stores and computes directly in the compressed latent space. Its score is
`q @ kv.T + q_pe @ k_pe.T`, scaled by `1/sqrt(dim + pe_dim)`, and the value is
the same compressed `kv`; no full K/V decompression appears in the TileOPs
golden or kernel.

This schedule is a correctness pilot, not the final performance pattern. On
the canonical MHA decode set it passed all correctness and SOL gates but only
reached `ratio_min=0.134417` against Torch-NPU SDPA. Filling the cores along
batch/head avoids starvation, but serial KV scanning plus three GM handoffs per
64-token block remains too expensive. A performance round should split KV and
combine online-softmax states, or establish a direct on-chip CV handoff.

## 8. Convolution: tile-local implicit im2col with direct fallback

Keep the public Conv2d layout NCHW with OIHW weights and NCHW output. For
fp16/bf16 calls whose per-group output channels are a multiple of 16, gather
one `Kx64` receptive-field tile inside the kernel and feed it to
`T.gemm_v0`; never materialize the full im2col matrix or transpose tensors on
the host. The K dimension is tiled by 64 and zero-filled under predicates, so
channel/kernel tails, padding, stride, dilation, and spatial tails use the same
kernel. L0C accumulates in fp32 and writes directly to the NCHW output.

Use the direct window path for fp32, bias, depthwise, and narrow grouped
convolution. These cases either cannot use the fp16/bf16 Cube primitive or
leave most of its 16-row fragment idle. The fallback assigns 64 outputs to each
of the two vector contexts, walks `C_in/groups * kH * kW` with boundary
predicates, accumulates in fp32 UB, and performs vector output copies. It is a
complete-domain correctness path, not a throughput schedule.

Both paths cap physical blocks at 65535 and cover excess logical blocks with
`logical_cid = cid + repeat * launch_blocks`. The T075 depthwise workload
`[16,256,56,56]` has 100352 logical direct tiles and verified two grid-stride
rounds with finite, exact output. Generated Cube code is mixed AIC/AIV: the
current UB-to-L1 lowering inserts a per-launch GM workspace to exchange the two
vector halves before Cube GEMM. This is bounded tile staging, not a full-image
im2col buffer, but it is still real GM traffic and the main performance ceiling.

## 9. Scan pilot (Cumsum/Cumprod)

The verified Ascend path is a correctness fallback: one `T.Kernel(1,
is_npu=True)` performs a row-major serial scan, with a fp32 scalar accumulator
and fp32 UB prefix values. FP16 output is cast only at writeback. Non-last axes
are moved to the trailing dimension with a host `permute`/reshape and restored
after the kernel; this is explicitly a boundary fallback, not a native layout
implementation.

The intended two-pass schedule was not made usable on dav-2201. `T.cumsum`
does not lower from Ascend UB, explicit `shared.dyn` staging rejects GM-to-shared
copy, and a two-block `CrossCoreSetFlag`/`CrossCoreWaitFlag` probe compiles but
fails at runtime with 507015 (illegal instruction / unaligned UUB). Therefore a
25-core carry propagation is not claimed. BF16 scalar load/cast is also blocked
by the backend (`not support bf16 type cast`); it needs an aligned vector staging
path before registration is broadened.

## 10. Max-pool indices: explicit dual-output window reduction

Fixed 1D/2D/3D max pooling shares one flattened NCDHW AIV template;
adaptive 2D changes only the per-output window bounds. Use explicit output
parameters with `out_idx=[]` so a test can prefill both value and index buffers
and prove that both are overwritten. Do not combine explicit outputs with
`out_idx=[-1]`.

Each vector context owns 64 outputs and writes both outputs with vector copies.
Values are gathered in storage dtype, cast as a vector to fp32, compared there,
and cast back once. Indices stay `int64` in UB and GM; initialize the int64 UB
with a scalar UB loop because `T.tile.fill` does not support int64 on dav-2201.
The final `T.copy` int64 writeback is supported and avoids scalar GM stores.

Walk window coordinates in row-major spatial order and update a finite maximum
only on strict `>`; equal values therefore retain the first valid coordinate.
Every NaN overwrites both states, so the last NaN visited supplies the value and
index, matching the TileOPs pooling contract. The index is flattened within one
input spatial plane (`L`, `H*W`, or `D*H*W`), without batch/channel offsets.
Cast coordinates to int64 before multiply-add so planes larger than `2**31`
cannot overflow an int32 intermediate.

Cap physical BlockDim with `launch_block_count` and traverse excess logical
tiles with `logical_cid = cid + repeat * launch_blocks`. A correctness gate must
prefill distinct value/index sentinels, check zero remnants in each public
output, include an all-equal window, multiple NaNs, two unrelated non-divisible
artifacts under the AIV audit, and one workload with more than 65535 logical
tiles.
