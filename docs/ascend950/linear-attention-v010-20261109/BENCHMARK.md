# Linear attention benchmark results

Server1: physical 32 AIC / 64 AIV, runtime limit 32 / 64; Ascend 950DT. PyTorch 2.13.0+cpu, torch-npu 2.13.0rc1, TileLang 0.1.15, CANN runtime 9.1 with compiler SDK 9.2. Exact toolchain hashes are included in `linear7-toolchain-final-v2.json`.

Each row selects the lowest retained baseline/candidate ratio for that case. Device and synchronized host measurements are compared only to their corresponding metric. The CSVs retain every baseline constraint, including historical local results; those results are not relabeled as physical-32 remote measurements.

`Logical BW` is the sum of bytes in the declared input/output tensors divided by the displayed candidate time, in decimal TB/s. It is a reproducible tensor-volume throughput indicator, not measured HBM traffic; it excludes intermediates and repeated transfers. Host rows include launch and synchronization overhead. Full tensor shapes, dtypes and feature parameters are in `benchmark-workloads.json`.

## GLARecurrentFwdOp

| Case                    | q shape      | dtype    | Metric | Baseline                           | TileOPs ms | Baseline ms |    Ratio | Logical BW TB/s |
| ----------------------- | ------------ | -------- | ------ | ---------------------------------- | ---------: | ----------: | -------: | --------------: |
| serving-b1-h16-bfloat16 | (1, 16, 128) | bfloat16 | device | qualified_local_eager (historical) | 0.00506520 |  0.02849773 | 5.626181 |        0.211059 |
| serving-b1-h32-bfloat16 | (1, 32, 128) | bfloat16 | device | qualified_local_eager (historical) | 0.00849187 |  0.03182273 | 3.747437 |        0.251784 |
| serving-b1-h48-bfloat16 | (1, 48, 128) | bfloat16 | device | qualified_local_eager (historical) | 0.01180187 |  0.03580260 | 3.033639 |        0.271751 |
| serving-b1-h64-bfloat16 | (1, 64, 128) | bfloat16 | device | qualified_local_eager (historical) | 0.01481367 |  0.03757293 | 2.536370 |        0.288667 |
| serving-b1-h8-bfloat16  | (1, 8, 128)  | bfloat16 | device | qualified_local_eager (historical) | 0.00573693 |  0.02685007 | 4.680212 |        0.093173 |
| serving-b8-h32-bfloat16 | (8, 32, 128) | bfloat16 | device | qualified_local_eager (historical) | 0.05113667 |  0.06929280 | 1.355051 |        0.334494 |
| serving-b8-h48-bfloat16 | (8, 48, 128) | bfloat16 | device | qualified_local_eager (historical) | 0.07550193 |  0.08825380 | 1.168895 |        0.339824 |
| serving-b8-h64-bfloat16 | (8, 64, 128) | bfloat16 | device | qualified_local_eager (historical) | 0.09928907 |  0.11143773 | 1.122357 |        0.344547 |

## GLAFwdOp

| Case                            | q shape            | dtype    | Metric | Baseline                           | TileOPs ms | Baseline ms |     Ratio | Logical BW TB/s |
| ------------------------------- | ------------------ | -------- | ------ | ---------------------------------- | ---------: | ----------: | --------: | --------------: |
| decode-b1-d64-bfloat16          | (1, 1, 32, 64)     | bfloat16 | device | qualified_local_eager (historical) | 0.00340700 |  0.00569607 |  1.671872 |        0.313782 |
| decode-b8-d128-bfloat16         | (8, 1, 32, 128)    | bfloat16 | device | qualified_local_eager (historical) | 0.02164893 |  0.01805573 |  0.834024 |        1.565071 |
| decode-first-token-float16      | (3, 1, 24, 128)    | float16  | device | qualified_local_eager (historical) | 0.00955260 |  0.01054553 |  1.103944 |        0.503607 |
| gla-continue-b4-bfloat16        | (4, 257, 16, 128)  | bfloat16 | device | qualified_local_eager (historical) | 0.23481253 |  0.37941953 |  1.615840 |        0.125385 |
| gla-short-continuation-bfloat16 | (1, 257, 16, 128)  | bfloat16 | device | qualified_local_eager (historical) | 0.07562853 |  0.11839740 |  1.565512 |        0.097325 |
| prefill-16k-bfloat16            | (2, 16384, 4, 64)  | bfloat16 | device | qualified_local_eager (historical) | 1.47850313 |  1.30467040 |  0.882427 |        0.056914 |
| prefill-2k-bfloat16             | (2, 2048, 4, 64)   | bfloat16 | device | qualified_local_eager (historical) | 0.18805360 |  0.17494353 |  0.930285 |        0.056456 |
| prefill-3000-bfloat16           | (2, 3000, 4, 64)   | bfloat16 | device | qualified_local_eager (historical) | 0.28274747 |  0.23522560 |  0.831928 |        0.054788 |
| prefill-4k-bfloat16             | (2, 4096, 4, 64)   | bfloat16 | device | qualified_local_eager (historical) | 0.35861900 |  0.31676787 |  0.883299 |        0.059210 |
| prefill-8k-bfloat16             | (2, 8192, 4, 64)   | bfloat16 | device | qualified_local_eager (historical) | 0.72350820 |  0.58626220 |  0.810305 |        0.058153 |
| varlen-4x1k-bfloat16            | (1, 4096, 4, 64)   | bfloat16 | device | qualified_local_eager (historical) | 0.18032053 |  0.21154120 |  1.173140 |        0.059605 |
| varlen-ragged-float16           | (1, 4096, 16, 128) | float16  | device | qualified_local_eager (historical) | 0.91100300 | 15.55011640 | 17.069226 |        0.101289 |

## GLAChunkFwdOp

| Case               | q shape           | dtype    | Metric | Baseline                           | TileOPs ms | Baseline ms |     Ratio | Logical BW TB/s |
| ------------------ | ----------------- | -------- | ------ | ---------------------------------- | ---------: | ----------: | --------: | --------------: |
| train-16k-bfloat16 | (2, 16384, 4, 64) | bfloat16 | device | qualified_local_eager (historical) | 1.51761907 | 19.45979747 | 12.822584 |        0.055448 |
| train-16k-float16  | (2, 16384, 4, 64) | float16  | device | qualified_local_eager (historical) | 1.51845040 | 19.46665067 | 12.820077 |        0.055417 |
| train-2k-bfloat16  | (2, 2048, 4, 64)  | bfloat16 | device | qualified_local_eager (historical) | 0.19080007 |  2.47023393 | 12.946714 |        0.055644 |
| train-2k-float16   | (2, 2048, 4, 64)  | float16  | device | qualified_local_eager (historical) | 0.19006627 |  2.47341413 | 13.013430 |        0.055859 |
| train-4k-bfloat16  | (2, 4096, 4, 64)  | bfloat16 | device | qualified_local_eager (historical) | 0.36488173 |  4.87376587 | 13.357111 |        0.058193 |
| train-4k-float16   | (2, 4096, 4, 64)  | float16  | device | qualified_local_eager (historical) | 0.36517153 |  4.88259193 | 13.370681 |        0.058147 |
| train-8k-bfloat16  | (2, 8192, 4, 64)  | bfloat16 | device | qualified_local_eager (historical) | 0.73300680 |  9.71552633 | 13.254347 |        0.057399 |
| train-8k-float16   | (2, 8192, 4, 64)  | float16  | device | qualified_local_eager (historical) | 0.72873813 |  9.70330700 | 13.315218 |        0.057736 |

## GLAChunkBwdOp

| Case               | q shape           | dtype    | Metric | Baseline    |  TileOPs ms | Baseline ms |    Ratio | Logical BW TB/s |
| ------------------ | ----------------- | -------- | ------ | ----------- | ----------: | ----------: | -------: | --------------: |
| train-16k-bfloat16 | (2, 16384, 4, 64) | bfloat16 | device | torch-eager | 36.75403367 | 60.17740547 | 1.637301 |        0.006854 |
| train-16k-float16  | (2, 16384, 4, 64) | float16  | device | torch-eager | 36.48143547 | 60.24105100 | 1.651280 |        0.006905 |
| train-2k-bfloat16  | (2, 2048, 4, 64)  | bfloat16 | device | torch-eager |  4.62430947 |  4.64658460 | 1.004817 |        0.006859 |
| train-2k-float16   | (2, 2048, 4, 64)  | float16  | device | torch-eager |  4.61546973 |  4.64852607 | 1.007162 |        0.006872 |
| train-4k-bfloat16  | (2, 4096, 4, 64)  | bfloat16 | device | torch-eager |  9.30423533 | 10.02946387 | 1.077946 |        0.006790 |
| train-4k-float16   | (2, 4096, 4, 64)  | float16  | device | torch-eager |  9.28826753 | 10.04006973 | 1.080941 |        0.006802 |
| train-8k-bfloat16  | (2, 8192, 4, 64)  | bfloat16 | device | torch-eager | 18.52319433 | 22.82057173 | 1.232000 |        0.006807 |
| train-8k-float16   | (2, 8192, 4, 64)  | float16  | device | torch-eager | 18.55941633 | 22.85386213 | 1.231389 |        0.006794 |

## DeltaNetRecurrentFwdOp

| Case                    | q shape      | dtype    | Metric | Baseline                           | TileOPs ms | Baseline ms |    Ratio | Logical BW TB/s |
| ----------------------- | ------------ | -------- | ------ | ---------------------------------- | ---------: | ----------: | -------: | --------------: |
| serving-b1-h16-bfloat16 | (1, 16, 128) | bfloat16 | device | ops-transformer                    | 0.00711980 |  0.03197380 | 4.490828 |        0.149582 |
| serving-b1-h32-bfloat16 | (1, 32, 128) | bfloat16 | host   | sgl-kernel-npu                     | 0.17715727 |  0.99730659 | 5.629499 |        0.012023 |
| serving-b1-h48-bfloat16 | (1, 48, 128) | bfloat16 | device | ops-transformer                    | 0.00978913 |  0.03528140 | 3.604139 |        0.326380 |
| serving-b1-h64-bfloat16 | (1, 64, 128) | bfloat16 | device | ops-transformer                    | 0.00884087 |  0.03845067 | 4.349197 |        0.481850 |
| serving-b1-h8-bfloat16  | (1, 8, 128)  | bfloat16 | device | ops-transformer                    | 0.00780033 |  0.03067013 | 3.931900 |        0.068266 |
| serving-b8-h32-bfloat16 | (8, 32, 128) | bfloat16 | device | flash-linear-attention-npu         | 0.02641653 |  0.08105387 | 3.068301 |        0.645046 |
| serving-b8-h48-bfloat16 | (8, 48, 128) | bfloat16 | device | qualified_local_eager (historical) | 0.03824707 |  0.10136647 | 2.650307 |        0.668282 |
| serving-b8-h64-bfloat16 | (8, 64, 128) | bfloat16 | device | qualified_local_eager (historical) | 0.04964600 |  0.12175600 | 2.452484 |        0.686455 |

## DeltaNetFwdOp

| Case                              | q shape            | dtype    | Metric | Baseline                           | TileOPs ms | Baseline ms |    Ratio | Logical BW TB/s |
| --------------------------------- | ------------------ | -------- | ------ | ---------------------------------- | ---------: | ----------: | -------: | --------------: |
| decode-continued-bfloat16         | (8, 1, 32, 128)    | bfloat16 | device | flash-linear-attention-npu         | 0.03766700 |  0.06699047 | 1.778492 |        0.897791 |
| decode-continued-float16          | (8, 1, 32, 128)    | float16  | device | qualified_local_eager (historical) | 0.03762940 |  0.10686113 | 2.839831 |        0.898688 |
| decode-fresh-bfloat16             | (8, 1, 32, 128)    | bfloat16 | device | flash-linear-attention-npu         | 0.03334253 |  0.05656927 | 1.696610 |        0.511055 |
| decode-l2norm-bfloat16            | (4, 1, 16, 128)    | bfloat16 | device | flash-linear-attention-npu         | 0.01082860 |  0.03250833 | 3.002081 |        0.780735 |
| decode-narrow-b17-bfloat16        | (17, 1, 6, 64)     | bfloat16 | device | qualified_local_eager (historical) | 0.00972113 |  0.01792180 | 1.843592 |        0.349215 |
| deltanet-1.3b-cont-b4-bfloat16    | (4, 257, 16, 128)  | bfloat16 | device | flash-linear-attention-npu         | 0.13108227 |  0.26380940 | 2.012548 |        0.192736 |
| deltanet-1.3b-continue-bfloat16   | (1, 257, 16, 128)  | bfloat16 | host   | flash-linear-attention-npu         | 0.45462263 |  1.56461603 | 3.441571 |        0.013893 |
| deltanet-1.3b-decode-b32-bfloat16 | (32, 1, 16, 128)   | bfloat16 | device | flash-linear-attention-npu         | 0.07633993 |  0.11926020 | 1.562226 |        0.885961 |
| prefill-12k-continued-bfloat16    | (2, 12288, 4, 64)  | bfloat16 | host   | flash-linear-attention-npu         | 1.28016938 |  1.94853063 | 1.522088 |        0.039675 |
| prefill-16k-bfloat16              | (2, 16384, 4, 64)  | bfloat16 | host   | flash-linear-attention-npu         | 1.63459070 |  2.19916602 | 1.345392 |        0.041296 |
| prefill-2k-bfloat16               | (2, 2048, 4, 64)   | bfloat16 | host   | sgl-kernel-npu                     | 0.89494465 |  1.39706197 | 1.561060 |        0.009556 |
| prefill-3000-bfloat16             | (2, 3000, 4, 64)   | bfloat16 | host   | sgl-kernel-npu                     | 1.23090207 |  1.58928934 | 1.291158 |        0.010128 |
| prefill-4k-bfloat16               | (1, 4096, 16, 128) | bfloat16 | device | flash-linear-attention-npu         | 0.38192047 |  0.81935847 | 2.145364 |        0.178803 |
| prefill-8k-bfloat16               | (2, 8192, 4, 64)   | bfloat16 | host   | flash-linear-attention-npu         | 0.94100535 |  1.64526797 | 1.748415 |        0.035937 |
| prefill-l2norm-4k-bfloat16        | (1, 4096, 16, 128) | bfloat16 | device | flash-linear-attention-npu         | 0.38886387 |  0.86996327 | 2.237192 |        0.175610 |
| prefill-l2norm-ragged-bfloat16    | (1, 3000, 6, 64)   | bfloat16 | device | qualified_local_eager (historical) | 0.57425473 |  0.59244040 | 1.031668 |        0.016796 |
| varlen-4x1k-bfloat16              | (1, 4096, 4, 64)   | bfloat16 | host   | sgl-kernel-npu                     | 0.86254325 |  1.46440805 | 1.697779 |        0.010067 |
| varlen-ragged-bfloat16            | (1, 4096, 16, 128) | bfloat16 | host   | sgl-kernel-npu                     | 1.30920196 |  2.15287926 | 1.644421 |        0.057767 |

## GDNFwdOp

| Case                             | q shape              | dtype    | Metric | Baseline                                |  TileOPs ms |  Baseline ms |    Ratio | Logical BW TB/s |
| -------------------------------- | -------------------- | -------- | ------ | --------------------------------------- | ----------: | -----------: | -------: | --------------: |
| decode-b1-bfloat16               | (1, 1, 16, 128)      | bfloat16 | device | qualified_local_eager (historical)      |  0.00759580 |   0.00825273 | 1.086486 |        0.278259 |
| decode-b1-float16                | (1, 1, 16, 128)      | float16  | device | qualified_local_eager (historical)      |  0.00751880 |   0.00836547 | 1.112607 |        0.281109 |
| decode-b8-bfloat16               | (8, 1, 32, 128)      | bfloat16 | device | flash-linear-attention-npu              |  0.03525567 |   0.06805147 | 1.930228 |        0.959210 |
| decode-b8-float16                | (8, 1, 32, 128)      | float16  | device | qualified_local_eager (historical)      |  0.03533707 |   0.07376493 | 2.087466 |        0.957001 |
| decode-beta-l2norm-bfloat16      | (2, 1, 4, 128)       | bfloat16 | device | qualified_local_eager (historical)      |  0.00803473 |   0.00760787 | 0.946872 |        0.131529 |
| decode-fresh-bfloat16            | (8, 1, 32, 128)      | bfloat16 | device | flash-linear-attention-npu              |  0.03540440 |   0.05836380 | 1.648490 |        0.481307 |
| decode-gate-fused-bfloat16       | (32, 1, 8, 64)       | bfloat16 | device | qualified_local_eager (historical)      |  0.03165033 |   0.06037940 | 1.907702 |        0.536361 |
| decode-vfirst-b17-bfloat16       | (17, 1, 6, 128)      | bfloat16 | device | qualified_local_eager (historical)      |  0.03012500 |   0.02813580 | 0.933968 |        0.892819 |
| prefill-12k-bfloat16             | (1, 12288, 16, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  1.10246396 |   2.31358996 | 2.098563 |        0.184280 |
| prefill-12k-float16              | (1, 12288, 16, 128)  | float16  | host   | flash-linear-attention-npu              |  1.08988537 |   3.55681671 | 3.263478 |        0.186406 |
| prefill-3000-bfloat16            | (1, 3000, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  0.51951266 |   1.25967137 | 2.424717 |        0.097000 |
| prefill-32k-bfloat16             | (1, 32768, 16, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  2.63374206 |   4.13949005 | 1.571714 |        0.205038 |
| prefill-32k-float16              | (1, 32768, 16, 128)  | float16  | host   | flash-linear-attention-npu              |  2.62401070 |   7.39522399 | 2.818290 |        0.205798 |
| prefill-4k-bfloat16              | (1, 4096, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  0.49978799 |   1.36342803 | 2.728013 |        0.136897 |
| prefill-4k-float16               | (1, 4096, 16, 128)   | float16  | host   | flash-linear-attention-npu              |  0.50897201 |   1.91980603 | 3.771929 |        0.134427 |
| prefill-continued-bfloat16       | (1, 4096, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  0.53293472 |   1.43595471 | 2.694429 |        0.130350 |
| prefill-narrow-state-bfloat16    | (1, 4096, 16, 64)    | bfloat16 | host   | flash-linear-attention-npu              |  0.50249606 |   1.47604138 | 2.937419 |        0.067819 |
| prefill-raw-4k-bfloat16          | (1, 4096, 16, 128)   | bfloat16 | device | qualified_local_eager (historical)      |  0.33942247 |   1.83269093 | 5.399439 |        0.201577 |
| prefill-raw-ragged-bfloat16      | (1, 3000, 6, 64)     | bfloat16 | device | qualified_local_eager (historical)      |  0.63546620 |   1.89047473 | 2.974941 |        0.039186 |
| prefill-v-first-4k-bfloat16      | (1, 4096, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  0.89664198 |   1.48970736 | 1.661429 |        0.077476 |
| prefill-v-first-narrow-bfloat16  | (1, 3000, 16, 64)    | bfloat16 | host   | flash-linear-attention-npu              |  0.78195467 |   1.41595602 | 1.810790 |        0.032010 |
| q35-27b-1024-7168-bfloat16       | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  2.12368198 |   4.21147862 | 1.983102 |        0.130104 |
| q35-27b-1024-7168-float16        | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  2.07398733 |   4.64961007 | 2.241870 |        0.133222 |
| q35-27b-16384-bfloat16           | (1, 16384, 16, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  3.43641996 |   4.19861798 | 1.221800 |        0.158061 |
| q35-27b-16384-float16            | (1, 16384, 16, 128)  | float16  | host   | flash-linear-attention-npu              |  3.44497670 |   8.60425262 | 2.497623 |        0.157668 |
| q35-27b-16x8192-bfloat16         | (1, 131072, 16, 128) | bfloat16 | host   | flash-linear-attention-npu              | 24.03231996 |  28.66887401 | 1.192930 |        0.181858 |
| q35-27b-16x8192-float16          | (1, 131072, 16, 128) | float16  | host   | sgl-kernel-npu                          | 24.07514462 |  43.96139806 | 1.826008 |        0.181534 |
| q35-27b-2048-6144-bfloat16       | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  2.03540735 |   3.77625801 | 1.855284 |        0.135747 |
| q35-27b-2048-6144-float16        | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  1.93997991 |   4.45734868 | 2.297626 |        0.142424 |
| q35-27b-2048-bfloat16            | (1, 2048, 16, 128)   | bfloat16 | device | flash-linear-attention-npu              |  0.39695593 |   0.81625093 | 2.056276 |        0.177974 |
| q35-27b-2048-float16             | (1, 2048, 16, 128)   | float16  | host   | flash-linear-attention-npu              |  0.61407536 |   2.03511203 | 3.314108 |        0.115047 |
| q35-27b-32768-bfloat16           | (1, 32768, 16, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  6.71985000 |   7.17039263 | 1.067047 |        0.161191 |
| q35-27b-32768-float16            | (1, 32768, 16, 128)  | float16  | host   | flash-linear-attention-npu              |  6.68764866 |  15.72361598 | 2.351143 |        0.161967 |
| q35-27b-32x8192-bfloat16         | (1, 262144, 16, 128) | bfloat16 | host   | flash-linear-attention-npu              | 47.72281737 |  56.77546800 | 1.189692 |        0.183160 |
| q35-27b-32x8192-float16          | (1, 262144, 16, 128) | float16  | host   | flash-linear-attention-npu              | 47.72643807 | 105.54098465 | 2.211374 |        0.183146 |
| q35-27b-4096-4096-bfloat16       | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  1.73189134 |   3.27076539 | 1.888551 |        0.159536 |
| q35-27b-4096-4096-float16        | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  1.71548330 |   4.52483607 | 2.637645 |        0.161062 |
| q35-27b-4096-bfloat16            | (1, 4096, 16, 128)   | bfloat16 | device | flash-linear-attention-npu              |  0.80352953 |   1.57564293 | 1.960902 |        0.171929 |
| q35-27b-4096-float16             | (1, 4096, 16, 128)   | float16  | host   | flash-linear-attention-npu              |  1.01767468 |   2.95803597 | 2.906662 |        0.135751 |
| q35-27b-4x2048-bfloat16          | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  1.77191732 |   3.58855274 | 2.025237 |        0.159483 |
| q35-27b-4x2048-float16           | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  1.74614529 |   4.70665538 | 2.695455 |        0.161837 |
| q35-27b-6144-2048-bfloat16       | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  2.01978336 |   3.76094527 | 1.862054 |        0.136797 |
| q35-27b-6144-2048-float16        | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  1.97145793 |   4.50339202 | 2.284295 |        0.140150 |
| q35-27b-65536-bfloat16           | (1, 65536, 16, 128)  | bfloat16 | host   | flash-linear-attention-npu              | 13.12085272 |  12.95931538 | 0.987689 |        0.164868 |
| q35-27b-65536-float16            | (1, 65536, 16, 128)  | float16  | host   | sgl-kernel-npu                          | 13.16204934 |  30.99825075 | 2.355123 |        0.164352 |
| q35-27b-8192-bfloat16            | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  1.83455469 |   2.62773276 | 1.432355 |        0.148894 |
| q35-27b-8192-float16             | (1, 8192, 16, 128)   | float16  | host   | flash-linear-attention-npu              |  1.80740931 |   4.77326205 | 2.640941 |        0.151130 |
| q35-27b-8x1024-bfloat16          | (1, 8192, 16, 128)   | bfloat16 | device | flash-linear-attention-npu              |  1.48489747 |   3.12667600 | 2.105651 |        0.198784 |
| q35-27b-8x1024-float16           | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  1.72986131 |   4.36333595 | 2.522362 |        0.170635 |
| q35-27b-8x8192-bfloat16          | (1, 65536, 16, 128)  | bfloat16 | host   | flash-linear-attention-npu              | 12.17844005 |  14.71975803 | 1.208674 |        0.179435 |
| q35-27b-8x8192-float16           | (1, 65536, 16, 128)  | float16  | host   | sgl-kernel-npu                          | 12.20271473 |  22.79607734 | 1.868115 |        0.179078 |
| q35-2b-1024-7168-bfloat16        | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  1.19105134 |   2.63392801 | 2.211431 |        0.114889 |
| q35-2b-1024-7168-float16         | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  1.14568338 |   3.33114875 | 2.907565 |        0.119439 |
| q35-2b-16384-bfloat16            | (1, 16384, 16, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  1.42810925 |   2.58930471 | 1.813100 |        0.189434 |
| q35-2b-16384-float16             | (1, 16384, 16, 128)  | float16  | host   | flash-linear-attention-npu              |  1.40255736 |   4.27317259 | 3.046701 |        0.192885 |
| q35-2b-16x8192-bfloat16          | (1, 131072, 16, 128) | bfloat16 | host   | flash-linear-attention-npu              |  9.03401465 |  11.32184528 | 1.253246 |        0.240497 |
| q35-2b-16x8192-float16           | (1, 131072, 16, 128) | float16  | host   | sgl-kernel-npu                          |  9.04284737 |  15.48341534 | 1.712228 |        0.240262 |
| q35-2b-2048-6144-bfloat16        | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  1.13000460 |   2.60556263 | 2.305798 |        0.121096 |
| q35-2b-2048-6144-float16         | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  1.07852131 |   3.33128059 | 3.088748 |        0.126877 |
| q35-2b-2048-bfloat16             | (1, 2048, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  0.55716404 |   1.34436740 | 2.412875 |        0.062341 |
| q35-2b-2048-float16              | (1, 2048, 16, 128)   | float16  | host   | flash-linear-attention-npu              |  0.54290735 |   1.68635733 | 3.106160 |        0.063978 |
| q35-2b-32x8192-bfloat16          | (1, 262144, 16, 128) | bfloat16 | host   | flash-linear-attention-npu              | 17.62036805 |  15.33099463 | 0.870072 |        0.246607 |
| q35-2b-32x8192-float16           | (1, 262144, 16, 128) | float16  | host   | sgl-kernel-npu                          | 17.63308402 |  29.45522806 | 1.670452 |        0.246429 |
| q35-2b-4096-4096-bfloat16        | (1, 8192, 16, 128)   | bfloat16 | device | qualified_local_eager (historical)      |  0.53417980 |   3.54325067 | 6.633067 |        0.256167 |
| q35-2b-4096-4096-float16         | (1, 8192, 16, 128)   | float16  | device | sgl-kernel-npu                          |  0.53483207 |   1.67907673 | 3.139447 |        0.255855 |
| q35-2b-4x2048-bfloat16           | (1, 8192, 16, 128)   | bfloat16 | device | flash-linear-attention-npu              |  0.53354960 |   1.15662093 | 2.167785 |        0.260400 |
| q35-2b-4x2048-float16            | (1, 8192, 16, 128)   | float16  | device | sgl-kernel-npu                          |  0.53204560 |   1.67606513 | 3.150228 |        0.261136 |
| q35-2b-6144-2048-bfloat16        | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  1.16812798 |   2.64066734 | 2.260598 |        0.117144 |
| q35-2b-6144-2048-float16         | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  1.12267130 |   3.38886476 | 3.018573 |        0.121887 |
| q35-2b-65536-bfloat16            | (1, 65536, 16, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  5.00353392 |   7.24867461 | 1.448711 |        0.215645 |
| q35-2b-65536-float16             | (1, 65536, 16, 128)  | float16  | host   | flash-linear-attention-npu              |  5.02773408 |  13.77945275 | 2.740688 |        0.214607 |
| q35-2b-8192-bfloat16             | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  0.84946336 |   1.82541264 | 2.148901 |        0.159855 |
| q35-2b-8192-float16              | (1, 8192, 16, 128)   | float16  | host   | flash-linear-attention-npu              |  0.81966799 |   2.65236143 | 3.235897 |        0.165665 |
| q35-2b-8x1024-bfloat16           | (1, 8192, 16, 128)   | bfloat16 | device | flash-linear-attention-npu              |  0.53556987 |   1.14033160 | 2.129193 |        0.267249 |
| q35-2b-8x1024-float16            | (1, 8192, 16, 128)   | float16  | device | sgl-kernel-npu                          |  0.53910673 |   1.67126180 | 3.100057 |        0.265496 |
| q35-2b-8x8192-bfloat16           | (1, 65536, 16, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  4.65155601 |   6.34132602 | 1.363270 |        0.233540 |
| q35-2b-8x8192-float16            | (1, 65536, 16, 128)  | float16  | host   | sgl-kernel-npu                          |  4.62558667 |   8.80708403 | 1.903993 |        0.234851 |
| q35-35b-1024-7168-bfloat16       | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  1.55480600 |   3.25744261 | 2.095080 |        0.132859 |
| q35-35b-1024-7168-float16        | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  1.60165538 |   3.46595667 | 2.163984 |        0.128973 |
| q35-35b-16384-bfloat16           | (1, 16384, 16, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  2.31034529 |   3.31983728 | 1.436944 |        0.176098 |
| q35-35b-16384-float16            | (1, 16384, 16, 128)  | float16  | host   | flash-linear-attention-npu              |  2.30644662 |   6.03082925 | 2.614771 |        0.176396 |
| q35-35b-16x8192-bfloat16         | (1, 131072, 16, 128) | bfloat16 | host   | flash-linear-attention-npu              | 16.38823667 |  14.24284400 | 0.869089 |        0.199628 |
| q35-35b-16x8192-float16          | (1, 131072, 16, 128) | float16  | host   | sgl-kernel-npu                          | 16.37216931 |  29.83571934 | 1.822344 |        0.199824 |
| q35-35b-2048-6144-bfloat16       | (1, 8192, 16, 128)   | bfloat16 | device | flash-linear-attention-npu              |  1.05858587 |   2.95904327 | 2.795279 |        0.195137 |
| q35-35b-2048-6144-float16        | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  1.48775795 |   3.47796138 | 2.337720 |        0.138846 |
| q35-35b-2048-bfloat16            | (1, 2048, 16, 128)   | bfloat16 | device | flash-linear-attention-npu              |  0.26134273 |   0.59479547 | 2.275921 |        0.201616 |
| q35-35b-2048-float16             | (1, 2048, 16, 128)   | float16  | host   | flash-linear-attention-npu              |  0.51882199 |   1.73045928 | 3.335362 |        0.101559 |
| q35-35b-32768-bfloat16           | (1, 32768, 16, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  4.42245534 |   5.61807401 | 1.270352 |        0.183517 |
| q35-35b-32768-float16            | (1, 32768, 16, 128)  | float16  | host   | flash-linear-attention-npu              |  4.41319604 |  10.80684199 | 2.448756 |        0.183903 |
| q35-35b-32x8192-bfloat16         | (1, 262144, 16, 128) | bfloat16 | host   | flash-linear-attention-npu              | 32.44472472 |  27.59998140 | 0.850677 |        0.201670 |
| q35-35b-32x8192-float16          | (1, 262144, 16, 128) | float16  | host   | flash-linear-attention-npu              | 32.58547264 |  72.66016534 | 2.229833 |        0.200799 |
| q35-35b-4096-4096-bfloat16       | (1, 8192, 16, 128)   | bfloat16 | device | flash-linear-attention-npu              |  1.02205933 |   2.03517387 | 1.991248 |        0.202111 |
| q35-35b-4096-4096-float16        | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  1.29967472 |   3.45289130 | 2.656735 |        0.158939 |
| q35-35b-4096-bfloat16            | (1, 4096, 16, 128)   | bfloat16 | device | flash-linear-attention-npu              |  0.50422273 |   1.12156160 | 2.224338 |        0.204840 |
| q35-35b-4096-float16             | (1, 4096, 16, 128)   | float16  | host   | flash-linear-attention-npu              |  0.73211348 |   2.43181062 | 3.321631 |        0.141077 |
| q35-35b-4x2048-bfloat16          | (1, 8192, 16, 128)   | bfloat16 | device | flash-linear-attention-npu              |  1.02360413 |   1.95821553 | 1.913059 |        0.205904 |
| q35-35b-4x2048-float16           | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  1.31910201 |   3.54893465 | 2.690417 |        0.159778 |
| q35-35b-6144-2048-bfloat16       | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  1.41275336 |   3.11951470 | 2.208110 |        0.146218 |
| q35-35b-6144-2048-float16        | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  1.39752602 |   3.52773195 | 2.524269 |        0.147811 |
| q35-35b-65536-bfloat16           | (1, 65536, 16, 128)  | bfloat16 | host   | sgl-kernel-npu                          |  8.58768541 |  48.49251931 | 5.646751 |        0.188770 |
| q35-35b-65536-float16            | (1, 65536, 16, 128)  | float16  | host   | flash-linear-attention-npu              |  8.58981408 |  20.47898664 | 2.384101 |        0.188723 |
| q35-35b-8192-bfloat16            | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  1.27834864 |   2.20743002 | 1.726782 |        0.159950 |
| q35-35b-8192-float16             | (1, 8192, 16, 128)   | float16  | host   | flash-linear-attention-npu              |  1.26398609 |   3.56849534 | 2.823208 |        0.161768 |
| q35-35b-8x1024-bfloat16          | (1, 8192, 16, 128)   | bfloat16 | device | flash-linear-attention-npu              |  1.02575187 |   1.98988507 | 1.939928 |        0.213651 |
| q35-35b-8x1024-float16           | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  1.31709132 |   3.47061536 | 2.635061 |        0.166391 |
| q35-35b-8x8192-bfloat16          | (1, 65536, 16, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  8.33225132 |   9.67236469 | 1.160834 |        0.196319 |
| q35-35b-8x8192-float16           | (1, 65536, 16, 128)  | float16  | host   | sgl-kernel-npu                          |  8.31180926 |  15.54787202 | 1.870576 |        0.196802 |
| q35-397b-tp1-1024-7168-bfloat16  | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  2.58173666 |   4.61032065 | 1.785744 |        0.134030 |
| q35-397b-tp1-1024-7168-float16   | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  2.53280724 |   5.20367868 | 2.054510 |        0.136619 |
| q35-397b-tp1-16384-bfloat16      | (1, 16384, 16, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  4.15431603 |   4.77212196 | 1.148714 |        0.163559 |
| q35-397b-tp1-16384-float16       | (1, 16384, 16, 128)  | float16  | host   | flash-linear-attention-npu              |  4.12682472 |   9.78733332 | 2.371638 |        0.164649 |
| q35-397b-tp1-16x8192-bfloat16    | (1, 131072, 16, 128) | bfloat16 | host   | sgl-kernel-npu                          | 31.65868201 | 210.46797928 | 6.648034 |        0.172761 |
| q35-397b-tp1-16x8192-float16     | (1, 131072, 16, 128) | float16  | host   | flash-linear-attention-npu              | 31.60337731 |  70.05871534 | 2.216811 |        0.173063 |
| q35-397b-tp1-2048-6144-bfloat16  | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  2.35219800 |   4.13810061 | 1.759248 |        0.147109 |
| q35-397b-tp1-2048-6144-float16   | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  2.36139738 |   5.28275141 | 2.237129 |        0.146536 |
| q35-397b-tp1-2048-bfloat16       | (1, 2048, 16, 128)   | bfloat16 | device | flash-linear-attention-npu              |  0.49277767 |   0.91688020 | 1.860637 |        0.179807 |
| q35-397b-tp1-2048-float16        | (1, 2048, 16, 128)   | float16  | host   | flash-linear-attention-npu              |  0.72555669 |   2.36339861 | 3.257359 |        0.122120 |
| q35-397b-tp1-32768-bfloat16      | (1, 32768, 16, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  8.12390605 |   7.64912603 | 0.941558 |        0.166762 |
| q35-397b-tp1-32768-float16       | (1, 32768, 16, 128)  | float16  | host   | flash-linear-attention-npu              |  8.10976534 |  18.30740937 | 2.257452 |        0.167053 |
| q35-397b-tp1-32x8192-bfloat16    | (1, 262144, 16, 128) | bfloat16 | host   | sgl-kernel-npu                          | 62.88419471 | 167.05682799 | 2.656579 |        0.173951 |
| q35-397b-tp1-32x8192-float16     | (1, 262144, 16, 128) | float16  | host   | flash-linear-attention-npu              | 63.05357069 | 139.34498467 | 2.209946 |        0.173483 |
| q35-397b-tp1-4096-4096-bfloat16  | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  2.21808140 |   3.39630066 | 1.531188 |        0.156004 |
| q35-397b-tp1-4096-4096-float16   | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  2.23135538 |   5.28582794 | 2.368887 |        0.155076 |
| q35-397b-tp1-4096-bfloat16       | (1, 4096, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  1.22495797 |   1.94114599 | 1.584663 |        0.141242 |
| q35-397b-tp1-4096-float16        | (1, 4096, 16, 128)   | float16  | host   | flash-linear-attention-npu              |  1.22451264 |   3.46885935 | 2.832849 |        0.141293 |
| q35-397b-tp1-4x2048-bfloat16     | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  2.25172993 |   3.83105526 | 1.701383 |        0.157398 |
| q35-397b-tp1-4x2048-float16      | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  2.27660670 |   5.70835937 | 2.507398 |        0.155679 |
| q35-397b-tp1-6144-2048-bfloat16  | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  2.37520064 |   4.25520676 | 1.791515 |        0.145685 |
| q35-397b-tp1-6144-2048-float16   | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  2.37284862 |   5.28292861 | 2.226408 |        0.145829 |
| q35-397b-tp1-65536-bfloat16      | (1, 65536, 16, 128)  | bfloat16 | host   | sgl-kernel-npu                          | 15.97425137 |  96.26069131 | 6.025991 |        0.169355 |
| q35-397b-tp1-65536-float16       | (1, 65536, 16, 128)  | float16  | host   | flash-linear-attention-npu              | 15.95826000 |  35.50377395 | 2.224790 |        0.169525 |
| q35-397b-tp1-8192-bfloat16       | (1, 8192, 16, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  2.17724126 |   2.83165133 | 1.300568 |        0.157004 |
| q35-397b-tp1-8192-float16        | (1, 8192, 16, 128)   | float16  | host   | flash-linear-attention-npu              |  2.18592665 |   5.57237938 | 2.549207 |        0.156380 |
| q35-397b-tp1-8x1024-bfloat16     | (1, 8192, 16, 128)   | bfloat16 | device | flash-linear-attention-npu              |  1.97789420 |   3.74698947 | 1.894434 |        0.187672 |
| q35-397b-tp1-8x1024-float16      | (1, 8192, 16, 128)   | float16  | host   | sgl-kernel-npu                          |  2.27283534 |   5.61906140 | 2.472269 |        0.163318 |
| q35-397b-tp1-8x8192-bfloat16     | (1, 65536, 16, 128)  | bfloat16 | host   | flash-linear-attention-npu              | 15.96982054 |  13.95139275 | 0.873610 |        0.171241 |
| q35-397b-tp1-8x8192-float16      | (1, 65536, 16, 128)  | float16  | host   | sgl-kernel-npu                          | 15.96801269 |  29.85675537 | 1.869785 |        0.171260 |
| q35-397b-tp2-1024-7168-bfloat16  | (1, 8192, 8, 128)    | bfloat16 | host   | flash-linear-attention-npu              |  1.49464474 |   3.19752398 | 2.139320 |        0.115757 |
| q35-397b-tp2-1024-7168-float16   | (1, 8192, 8, 128)    | float16  | host   | sgl-kernel-npu                          |  1.48906335 |   3.55381401 | 2.386610 |        0.116191 |
| q35-397b-tp2-16384-bfloat16      | (1, 16384, 8, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  2.23953395 |   3.29445334 | 1.471044 |        0.151701 |
| q35-397b-tp2-16384-float16       | (1, 16384, 8, 128)   | float16  | host   | flash-linear-attention-npu              |  2.22964327 |   6.04657462 | 2.711902 |        0.152374 |
| q35-397b-tp2-16x8192-bfloat16    | (1, 131072, 8, 128)  | bfloat16 | host   | flash-linear-attention-npu              | 15.90524996 |  13.95109395 | 0.877138 |        0.171936 |
| q35-397b-tp2-16x8192-float16     | (1, 131072, 8, 128)  | float16  | host   | sgl-kernel-npu                          | 15.97513333 |  29.69632794 | 1.858910 |        0.171184 |
| q35-397b-tp2-2048-6144-bfloat16  | (1, 8192, 8, 128)    | bfloat16 | host   | flash-linear-attention-npu              |  1.44202939 |   3.08944192 | 2.142426 |        0.119980 |
| q35-397b-tp2-2048-6144-float16   | (1, 8192, 8, 128)    | float16  | host   | sgl-kernel-npu                          |  1.45409405 |   3.56099730 | 2.448946 |        0.118985 |
| q35-397b-tp2-2048-bfloat16       | (1, 2048, 8, 128)    | bfloat16 | device | flash-linear-attention-npu              |  0.26048973 |   0.58946833 | 2.262923 |        0.170073 |
| q35-397b-tp2-2048-float16        | (1, 2048, 8, 128)    | float16  | device | sgl-kernel-npu                          |  0.25474153 |   0.88315327 | 3.466860 |        0.173911 |
| q35-397b-tp2-32768-bfloat16      | (1, 32768, 8, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  4.28342860 |   5.23762734 | 1.222765 |        0.158140 |
| q35-397b-tp2-32768-float16       | (1, 32768, 8, 128)   | float16  | host   | flash-linear-attention-npu              |  4.31048268 |  11.00484058 | 2.553041 |        0.157147 |
| q35-397b-tp2-32x8192-bfloat16    | (1, 262144, 8, 128)  | bfloat16 | host   | flash-linear-attention-npu              | 31.61676666 |  34.47679939 | 1.090459 |        0.172990 |
| q35-397b-tp2-32x8192-float16     | (1, 262144, 8, 128)  | float16  | host   | flash-linear-attention-npu              | 31.58851539 |  69.94447134 | 2.214237 |        0.173144 |
| q35-397b-tp2-4096-4096-bfloat16  | (1, 8192, 8, 128)    | bfloat16 | device | flash-linear-attention-npu              |  1.00181267 |   1.95501147 | 1.951474 |        0.172702 |
| q35-397b-tp2-4096-4096-float16   | (1, 8192, 8, 128)    | float16  | host   | sgl-kernel-npu                          |  1.25336262 |   3.83458994 | 3.059442 |        0.138041 |
| q35-397b-tp2-4096-bfloat16       | (1, 4096, 8, 128)    | bfloat16 | device | flash-linear-attention-npu              |  0.48569747 |   1.11016587 | 2.285715 |        0.178110 |
| q35-397b-tp2-4096-float16        | (1, 4096, 8, 128)    | float16  | host   | flash-linear-attention-npu              |  0.70711200 |   2.42329864 | 3.427036 |        0.122339 |
| q35-397b-tp2-4x2048-bfloat16     | (1, 8192, 8, 128)    | bfloat16 | device | flash-linear-attention-npu              |  0.99458007 |   1.87901420 | 1.889254 |        0.178175 |
| q35-397b-tp2-4x2048-float16      | (1, 8192, 8, 128)    | float16  | host   | sgl-kernel-npu                          |  1.26166133 |   3.95477340 | 3.134576 |        0.140457 |
| q35-397b-tp2-6144-2048-bfloat16  | (1, 8192, 8, 128)    | bfloat16 | host   | flash-linear-attention-npu              |  1.33943728 |   3.04136202 | 2.270627 |        0.129170 |
| q35-397b-tp2-6144-2048-float16   | (1, 8192, 8, 128)    | float16  | host   | sgl-kernel-npu                          |  1.35914137 |   3.73719921 | 2.749677 |        0.127297 |
| q35-397b-tp2-65536-bfloat16      | (1, 65536, 8, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  8.27383660 |   9.86106666 | 1.191837 |        0.163487 |
| q35-397b-tp2-65536-float16       | (1, 65536, 8, 128)   | float16  | host   | sgl-kernel-npu                          |  8.28556524 |  20.76149474 | 2.505743 |        0.163255 |
| q35-397b-tp2-8192-bfloat16       | (1, 8192, 8, 128)    | bfloat16 | host   | flash-linear-attention-npu              |  1.20570809 |   2.16189530 | 1.793050 |        0.141757 |
| q35-397b-tp2-8192-float16        | (1, 8192, 8, 128)    | float16  | host   | flash-linear-attention-npu              |  1.19746995 |   3.58883264 | 2.997013 |        0.142733 |
| q35-397b-tp2-8x1024-bfloat16     | (1, 8192, 8, 128)    | bfloat16 | device | flash-linear-attention-npu              |  0.99568207 |   1.91597347 | 1.924282 |        0.186403 |
| q35-397b-tp2-8x1024-float16      | (1, 8192, 8, 128)    | float16  | host   | sgl-kernel-npu                          |  1.26419465 |   3.54764132 | 2.806246 |        0.146811 |
| q35-397b-tp2-8x8192-bfloat16     | (1, 65536, 8, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  8.15764737 |   9.13476475 | 1.119779 |        0.167615 |
| q35-397b-tp2-8x8192-float16      | (1, 65536, 8, 128)   | float16  | host   | sgl-kernel-npu                          |  8.19253934 |  15.60622997 | 1.904932 |        0.166901 |
| q35-397b-tp4-1024-7168-bfloat16  | (1, 8192, 4, 128)    | bfloat16 | host   | flash-linear-attention-npu              |  1.14375334 |   2.68038263 | 2.343497 |        0.075635 |
| q35-397b-tp4-1024-7168-float16   | (1, 8192, 4, 128)    | float16  | host   | sgl-kernel-npu                          |  1.09048399 |   3.07174468 | 2.816864 |        0.079330 |
| q35-397b-tp4-16384-bfloat16      | (1, 16384, 4, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  1.31738397 |   2.57921269 | 1.957829 |        0.128944 |
| q35-397b-tp4-16384-float16       | (1, 16384, 4, 128)   | float16  | host   | flash-linear-attention-npu              |  1.33381731 |   4.29837328 | 3.222610 |        0.127356 |
| q35-397b-tp4-16x8192-bfloat16    | (1, 131072, 4, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  8.25789872 |  11.08662738 | 1.342548 |        0.165580 |
| q35-397b-tp4-16x8192-float16     | (1, 131072, 4, 128)  | float16  | host   | sgl-kernel-npu                          |  8.22129001 |  15.25560472 | 1.855622 |        0.166317 |
| q35-397b-tp4-2048-6144-bfloat16  | (1, 8192, 4, 128)    | bfloat16 | host   | flash-linear-attention-npu              |  1.08383801 |   2.56772538 | 2.369104 |        0.079816 |
| q35-397b-tp4-2048-6144-float16   | (1, 8192, 4, 128)    | float16  | host   | sgl-kernel-npu                          |  1.07611536 |   3.30735541 | 3.073421 |        0.080389 |
| q35-397b-tp4-2048-bfloat16       | (1, 2048, 4, 128)    | bfloat16 | host   | flash-linear-attention-npu              |  0.47613522 |   1.33426590 | 2.802284 |        0.046523 |
| q35-397b-tp4-2048-float16        | (1, 2048, 4, 128)    | float16  | host   | flash-linear-attention-npu              |  0.50182811 |   1.68235206 | 3.352447 |        0.044141 |
| q35-397b-tp4-32768-bfloat16      | (1, 32768, 4, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  2.48883869 |   4.02521538 | 1.617307 |        0.136084 |
| q35-397b-tp4-32768-float16       | (1, 32768, 4, 128)   | float16  | host   | flash-linear-attention-npu              |  2.48755527 |   7.30879400 | 2.938143 |        0.136154 |
| q35-397b-tp4-32x8192-bfloat16    | (1, 262144, 4, 128)  | bfloat16 | host   | flash-linear-attention-npu              | 16.19025071 |  20.26675536 | 1.251788 |        0.168909 |
| q35-397b-tp4-32x8192-float16     | (1, 262144, 4, 128)  | float16  | host   | sgl-kernel-npu                          | 16.20481132 |  29.25219607 | 1.805155 |        0.168758 |
| q35-397b-tp4-4096-4096-bfloat16  | (1, 8192, 4, 128)    | bfloat16 | device | flash-linear-attention-npu              |  0.48563593 |   1.21590320 | 2.503734 |        0.178133 |
| q35-397b-tp4-4096-4096-float16   | (1, 8192, 4, 128)    | float16  | device | sgl-kernel-npu                          |  0.48721393 |   1.62938233 | 3.344285 |        0.177556 |
| q35-397b-tp4-4096-bfloat16       | (1, 4096, 4, 128)    | bfloat16 | host   | flash-linear-attention-npu              |  0.53634470 |   1.33118465 | 2.481957 |        0.080645 |
| q35-397b-tp4-4096-float16        | (1, 4096, 4, 128)    | float16  | host   | flash-linear-attention-npu              |  0.50762069 |   2.06646534 | 4.070885 |        0.085209 |
| q35-397b-tp4-4x2048-bfloat16     | (1, 8192, 4, 128)    | bfloat16 | device | flash-linear-attention-npu              |  0.49927147 |   1.03109227 | 2.065194 |        0.177468 |
| q35-397b-tp4-4x2048-float16      | (1, 8192, 4, 128)    | float16  | device | sgl-kernel-npu                          |  0.48876660 |   1.64280240 | 3.361118 |        0.181282 |
| q35-397b-tp4-6144-2048-bfloat16  | (1, 8192, 4, 128)    | bfloat16 | host   | flash-linear-attention-npu              |  1.06708935 |   2.49010070 | 2.333545 |        0.081069 |
| q35-397b-tp4-6144-2048-float16   | (1, 8192, 4, 128)    | float16  | host   | sgl-kernel-npu                          |  1.05270526 |   3.41904924 | 3.247869 |        0.082176 |
| q35-397b-tp4-65536-bfloat16      | (1, 65536, 4, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  4.59024791 |   6.92748132 | 1.509174 |        0.147341 |
| q35-397b-tp4-65536-float16       | (1, 65536, 4, 128)   | float16  | host   | flash-linear-attention-npu              |  4.63988138 |  13.24650006 | 2.854922 |        0.145765 |
| q35-397b-tp4-8192-bfloat16       | (1, 8192, 4, 128)    | bfloat16 | host   | flash-linear-attention-npu              |  0.76312662 |   1.80097269 | 2.359992 |        0.111985 |
| q35-397b-tp4-8192-float16        | (1, 8192, 4, 128)    | float16  | host   | flash-linear-attention-npu              |  0.77877203 |   2.67134265 | 3.430199 |        0.109736 |
| q35-397b-tp4-8x1024-bfloat16     | (1, 8192, 4, 128)    | bfloat16 | device | flash-linear-attention-npu              |  0.49868013 |   1.01464660 | 2.034664 |        0.186089 |
| q35-397b-tp4-8x1024-float16      | (1, 8192, 4, 128)    | float16  | device | sgl-kernel-npu                          |  0.48928660 |   1.64398160 | 3.359956 |        0.189662 |
| q35-397b-tp4-8x8192-bfloat16     | (1, 65536, 4, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  4.24041074 |   6.06286470 | 1.429782 |        0.161228 |
| q35-397b-tp4-8x8192-float16      | (1, 65536, 4, 128)   | float16  | host   | sgl-kernel-npu                          |  4.32877997 |   8.56610605 | 1.978873 |        0.157936 |
| q35-397b-tp8-1024-7168-bfloat16  | (1, 8192, 2, 128)    | bfloat16 | host   | flash-linear-attention-npu              |  1.25777333 |   2.20895472 | 1.756242 |        0.034389 |
| q35-397b-tp8-1024-7168-float16   | (1, 8192, 2, 128)    | float16  | host   | flash-linear-attention-npu              |  1.23943938 |   3.28097008 | 2.647140 |        0.034898 |
| q35-397b-tp8-16384-bfloat16      | (1, 16384, 2, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  0.99069066 |   2.16477333 | 2.185115 |        0.085733 |
| q35-397b-tp8-16384-float16       | (1, 16384, 2, 128)   | float16  | host   | flash-linear-attention-npu              |  0.93555531 |   3.60486936 | 3.853187 |        0.090785 |
| q35-397b-tp8-16x8192-bfloat16    | (1, 131072, 2, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  4.18761404 |   7.09813874 | 1.695032 |        0.163260 |
| q35-397b-tp8-16x8192-float16     | (1, 131072, 2, 128)  | float16  | host   | sgl-kernel-npu                          |  4.18929004 |   8.51635799 | 2.032888 |        0.163195 |
| q35-397b-tp8-2048-6144-bfloat16  | (1, 8192, 2, 128)    | bfloat16 | host   | flash-linear-attention-npu              |  1.39388063 |   2.90578073 | 2.084670 |        0.031031 |
| q35-397b-tp8-2048-6144-float16   | (1, 8192, 2, 128)    | float16  | host   | flash-linear-attention-npu              |  1.10045411 |   3.17131672 | 2.881826 |        0.039305 |
| q35-397b-tp8-2048-bfloat16       | (1, 2048, 2, 128)    | bfloat16 | host   | flash-linear-attention-npu              |  0.48737337 |   1.24948537 | 2.563713 |        0.022725 |
| q35-397b-tp8-2048-float16        | (1, 2048, 2, 128)    | float16  | host   | flash-linear-attention-npu              |  0.49520262 |   1.44365070 | 2.915273 |        0.022366 |
| q35-397b-tp8-32768-bfloat16      | (1, 32768, 2, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  1.75273139 |   3.45664667 | 1.972149 |        0.096618 |
| q35-397b-tp8-32768-float16       | (1, 32768, 2, 128)   | float16  | host   | flash-linear-attention-npu              |  1.69620880 |   5.98187130 | 3.526613 |        0.099837 |
| q35-397b-tp8-32x8192-bfloat16    | (1, 262144, 2, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  8.11558605 |  13.91011796 | 1.714000 |        0.168484 |
| q35-397b-tp8-32x8192-float16     | (1, 262144, 2, 128)  | float16  | host   | sgl-kernel-npu                          |  8.09328199 |  15.58323000 | 1.925452 |        0.168948 |
| q35-397b-tp8-4096-4096-bfloat16  | (1, 8192, 2, 128)    | bfloat16 | device | flash-linear-attention-npu              |  0.27161987 |   0.86006713 | 3.166437 |        0.159244 |
| q35-397b-tp8-4096-4096-float16   | (1, 8192, 2, 128)    | float16  | device | sgl-kernel-npu                          |  0.26877573 |   1.10768733 | 4.121233 |        0.160929 |
| q35-397b-tp8-4096-bfloat16       | (1, 4096, 2, 128)    | bfloat16 | host   | flash-linear-attention-npu              |  0.46118538 |   1.37910536 | 2.990349 |        0.046894 |
| q35-397b-tp8-4096-float16        | (1, 4096, 2, 128)    | float16  | host   | flash-linear-attention-npu              |  0.48423931 |   1.96582268 | 4.059610 |        0.044662 |
| q35-397b-tp8-4x2048-bfloat16     | (1, 8192, 2, 128)    | bfloat16 | device | flash-linear-attention-npu              |  0.25302953 |   0.67908253 | 2.683807 |        0.175088 |
| q35-397b-tp8-4x2048-float16      | (1, 8192, 2, 128)    | float16  | device | sgl-kernel-npu                          |  0.25170287 |   0.84950120 | 3.375016 |        0.176011 |
| q35-397b-tp8-6144-2048-bfloat16  | (1, 8192, 2, 128)    | bfloat16 | host   | flash-linear-attention-npu              |  1.00388601 |   2.33497120 | 2.325933 |        0.043086 |
| q35-397b-tp8-6144-2048-float16   | (1, 8192, 2, 128)    | float16  | host   | flash-linear-attention-npu              |  1.01173197 |   3.06823334 | 3.032654 |        0.042752 |
| q35-397b-tp8-65536-bfloat16      | (1, 65536, 2, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  3.02895857 |   5.74782537 | 1.897624 |        0.111644 |
| q35-397b-tp8-65536-float16       | (1, 65536, 2, 128)   | float16  | host   | flash-linear-attention-npu              |  3.04630666 |  11.29261597 | 3.706986 |        0.111008 |
| q35-397b-tp8-8192-bfloat16       | (1, 8192, 2, 128)    | bfloat16 | host   | flash-linear-attention-npu              |  0.64276271 |   1.52062469 | 2.365764 |        0.066478 |
| q35-397b-tp8-8192-float16        | (1, 8192, 2, 128)    | float16  | host   | flash-linear-attention-npu              |  0.60192666 |   2.49472260 | 4.144562 |        0.070988 |
| q35-397b-tp8-8x1024-bfloat16     | (1, 8192, 2, 128)    | bfloat16 | device | flash-linear-attention-npu              |  0.25113140 |   0.61762267 | 2.459361 |        0.184762 |
| q35-397b-tp8-8x1024-float16      | (1, 8192, 2, 128)    | float16  | device | sgl-kernel-npu                          |  0.25372380 |   0.85421540 | 3.366714 |        0.182875 |
| q35-397b-tp8-8x8192-bfloat16     | (1, 65536, 2, 128)   | bfloat16 | device | flash-linear-attention-npu              |  1.93243640 |   3.88252973 | 2.009137 |        0.176894 |
| q35-397b-tp8-8x8192-float16      | (1, 65536, 2, 128)   | float16  | host   | sgl-kernel-npu                          |  2.25489736 |   5.17423330 | 2.294665 |        0.151597 |
| q35-gva-decode-b1-bfloat16       | (1, 1, 16, 128)      | bfloat16 | device | qualified_local_eager (historical)      |  0.00677680 |   0.01308213 | 1.930429 |        0.622566 |
| q35-gva-decode-b1-float16        | (1, 1, 16, 128)      | float16  | device | qualified_local_eager (historical)      |  0.00690527 |   0.01298453 | 1.880381 |        0.610984 |
| q35-gva-decode-b128-bfloat16     | (128, 1, 16, 128)    | bfloat16 | device | flash-linear-attention-npu              |  0.50398253 |   0.98639087 | 1.957193 |        1.071531 |
| q35-gva-decode-b128-float16      | (128, 1, 16, 128)    | float16  | device | qualified_local_eager (historical)      |  0.51016013 |   1.14853840 | 2.251329 |        1.058556 |
| q35-gva-decode-b16-bfloat16      | (16, 1, 16, 128)     | bfloat16 | device | flash-linear-attention-npu              |  0.06726173 |   0.11134753 | 1.655437 |        1.003604 |
| q35-gva-decode-b16-float16       | (16, 1, 16, 128)     | float16  | device | qualified_local_eager (historical)      |  0.06723020 |   0.14349393 | 2.134367 |        1.004074 |
| q35-gva-decode-b256-bfloat16     | (256, 1, 16, 128)    | bfloat16 | device | flash-linear-attention-npu (historical) |  0.99969327 |   1.97372693 | 1.974333 |        1.080397 |
| q35-gva-decode-b256-float16      | (256, 1, 16, 128)    | float16  | device | qualified_local_eager (historical)      |  1.00214140 |   2.29912927 | 2.294216 |        1.077758 |
| q35-gva-decode-b32-bfloat16      | (32, 1, 16, 128)     | bfloat16 | device | flash-linear-attention-npu              |  0.12896847 |   0.21641080 | 1.678013 |        1.046832 |
| q35-gva-decode-b32-float16       | (32, 1, 16, 128)     | float16  | device | qualified_local_eager (historical)      |  0.12893713 |   0.28568173 | 2.215667 |        1.047086 |
| q35-gva-decode-b4-bfloat16       | (4, 1, 16, 128)      | bfloat16 | device | qualified_local_eager (historical)      |  0.01873867 |   0.03873360 | 2.067041 |        0.900599 |
| q35-gva-decode-b4-float16        | (4, 1, 16, 128)      | float16  | device | qualified_local_eager (historical)      |  0.01858307 |   0.03883967 | 2.090057 |        0.908140 |
| q35-gva-decode-b512-bfloat16     | (512, 1, 16, 128)    | bfloat16 | device | flash-linear-attention-npu (historical) |  2.03852773 |   3.89916127 | 1.912734 |        1.059653 |
| q35-gva-decode-b512-float16      | (512, 1, 16, 128)    | float16  | device | qualified_local_eager (historical)      |  2.03771833 |   4.59368767 | 2.254329 |        1.060074 |
| q35-gva-decode-b64-bfloat16      | (64, 1, 16, 128)     | bfloat16 | device | flash-linear-attention-npu              |  0.25859780 |   0.49201447 | 1.902624 |        1.044156 |
| q35-gva-decode-b64-float16       | (64, 1, 16, 128)     | float16  | device | qualified_local_eager (historical)      |  0.25328567 |   0.57286673 | 2.261742 |        1.066055 |
| q35-gva-decode-b8-bfloat16       | (8, 1, 16, 128)      | bfloat16 | device | flash-linear-attention-npu              |  0.03530180 |   0.06775693 | 1.919362 |        0.956100 |
| q35-gva-decode-b8-float16        | (8, 1, 16, 128)      | float16  | device | qualified_local_eager (historical)      |  0.03541327 |   0.07360653 | 2.078502 |        0.953091 |
| q35-symmetric-1024-7168-bfloat16 | (1, 8192, 32, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  1.59010071 |   3.39051460 | 2.132264 |        0.172114 |
| q35-symmetric-1024-7168-float16  | (1, 8192, 32, 128)   | float16  | host   | sgl-kernel-npu                          |  1.62592667 |   3.48789875 | 2.145176 |        0.168321 |
| q35-symmetric-16384-bfloat16     | (1, 16384, 32, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  2.41232794 |   3.36959868 | 1.396824 |        0.224292 |
| q35-symmetric-16384-float16      | (1, 16384, 32, 128)  | float16  | host   | flash-linear-attention-npu              |  2.39594067 |   6.17803733 | 2.578544 |        0.225826 |
| q35-symmetric-16x8192-bfloat16   | (1, 131072, 32, 128) | bfloat16 | host   | flash-linear-attention-npu              | 17.33589076 |  21.88529798 | 1.262427 |        0.250653 |
| q35-symmetric-16x8192-float16    | (1, 131072, 32, 128) | float16  | host   | sgl-kernel-npu                          | 17.60200666 |  29.98105533 | 1.703275 |        0.246864 |
| q35-symmetric-2048-6144-bfloat16 | (1, 8192, 32, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  1.67442005 |   3.59027668 | 2.144191 |        0.163447 |
| q35-symmetric-2048-6144-float16  | (1, 8192, 32, 128)   | float16  | host   | sgl-kernel-npu                          |  1.49829673 |   3.48726865 | 2.327489 |        0.182660 |
| q35-symmetric-2048-bfloat16      | (1, 2048, 32, 128)   | bfloat16 | device | flash-linear-attention-npu              |  0.27107360 |   0.61997120 | 2.287095 |        0.256270 |
| q35-symmetric-2048-float16       | (1, 2048, 32, 128)   | float16  | host   | flash-linear-attention-npu              |  0.54560459 |   1.71300403 | 3.139644 |        0.127323 |
| q35-symmetric-32768-bfloat16     | (1, 32768, 32, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  4.60212197 |   5.62258329 | 1.221737 |        0.234682 |
| q35-symmetric-32768-float16      | (1, 32768, 32, 128)  | float16  | host   | flash-linear-attention-npu              |  4.59304998 |  11.44780259 | 2.492418 |        0.235145 |
| q35-symmetric-32x8192-bfloat16   | (1, 262144, 32, 128) | bfloat16 | host   | flash-linear-attention-npu              | 34.42219527 |  43.43185667 | 1.261740 |        0.252471 |
| q35-symmetric-32x8192-float16    | (1, 262144, 32, 128) | float16  | host   | flash-linear-attention-npu              | 34.48444000 |  78.31264588 | 2.270956 |        0.252015 |
| q35-symmetric-4096-4096-bfloat16 | (1, 8192, 32, 128)   | bfloat16 | device | flash-linear-attention-npu              |  1.07954473 |   2.25798500 | 2.091609 |        0.253513 |
| q35-symmetric-4096-4096-float16  | (1, 8192, 32, 128)   | float16  | host   | sgl-kernel-npu                          |  1.34712467 |   3.48613135 | 2.587831 |        0.203157 |
| q35-symmetric-4096-bfloat16      | (1, 4096, 32, 128)   | bfloat16 | device | flash-linear-attention-npu              |  0.53229067 |   1.18055600 | 2.217878 |        0.257076 |
| q35-symmetric-4096-float16       | (1, 4096, 32, 128)   | float16  | host   | flash-linear-attention-npu              |  0.76379866 |   2.39054269 | 3.129807 |        0.179156 |
| q35-symmetric-4x2048-bfloat16    | (1, 8192, 32, 128)   | bfloat16 | device | flash-linear-attention-npu              |  1.08062527 |   2.16359587 | 2.002170 |        0.257141 |
| q35-symmetric-4x2048-float16     | (1, 8192, 32, 128)   | float16  | host   | sgl-kernel-npu                          |  1.34476072 |   3.48655069 | 2.592692 |        0.206634 |
| q35-symmetric-6144-2048-bfloat16 | (1, 8192, 32, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  1.42672400 |   3.34858870 | 2.347047 |        0.191823 |
| q35-symmetric-6144-2048-float16  | (1, 8192, 32, 128)   | float16  | host   | sgl-kernel-npu                          |  1.43267537 |   3.45000268 | 2.408084 |        0.191026 |
| q35-symmetric-65536-bfloat16     | (1, 65536, 32, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  8.95951934 |  10.69055004 | 1.193206 |        0.240858 |
| q35-symmetric-65536-float16      | (1, 65536, 32, 128)  | float16  | host   | sgl-kernel-npu                          |  8.95373734 |  21.52903474 | 2.404475 |        0.241013 |
| q35-symmetric-8192-bfloat16      | (1, 8192, 32, 128)   | bfloat16 | host   | flash-linear-attention-npu              |  1.31308070 |   2.22234338 | 1.692465 |        0.206827 |
| q35-symmetric-8192-float16       | (1, 8192, 32, 128)   | float16  | host   | flash-linear-attention-npu              |  1.31140398 |   3.63606267 | 2.772649 |        0.207092 |
| q35-symmetric-8x1024-bfloat16    | (1, 8192, 32, 128)   | bfloat16 | device | flash-linear-attention-npu              |  1.08582513 |   2.19760467 | 2.023903 |        0.263635 |
| q35-symmetric-8x1024-float16     | (1, 8192, 32, 128)   | float16  | host   | sgl-kernel-npu                          |  1.37074136 |   3.54462794 | 2.585920 |        0.208837 |
| q35-symmetric-8x8192-bfloat16    | (1, 65536, 32, 128)  | bfloat16 | host   | flash-linear-attention-npu              |  8.85571868 |  11.29954529 | 1.275960 |        0.245339 |
| q35-symmetric-8x8192-float16     | (1, 65536, 32, 128)  | float16  | host   | sgl-kernel-npu                          |  8.85905667 |  15.70710264 | 1.772999 |        0.245246 |
| varlen-4x1k-bfloat16             | (1, 4096, 16, 128)   | bfloat16 | device | flash-linear-attention-npu              |  0.27724187 |   0.62817787 | 2.265812 |        0.258133 |
| varlen-ragged-grouped-float16    | (1, 4096, 4, 128)    | float16  | device | sgl-kernel-npu                          |  0.61155560 |   1.17518213 | 1.921628 |        0.082730 |
| varlen-v-first-ragged-float16    | (1, 4099, 4, 128)    | float16  | host   | sgl-kernel-npu                          |  1.80244800 |   2.95053062 | 1.636957 |        0.028087 |

## DeltaNetChunkFwdOp

| Case                         | q shape           | dtype    | Metric | Baseline                           | TileOPs ms | Baseline ms |    Ratio | Logical BW TB/s |
| ---------------------------- | ----------------- | -------- | ------ | ---------------------------------- | ---------: | ----------: | -------: | --------------: |
| deltanet-1.3b-train-bfloat16 | (1, 16, 256, 128) | bfloat16 | host   | current-qualified-six-output-eager | 1.65237204 |  1.61036737 | 0.974579 |        0.007620 |
| train-16k-bfloat16           | (2, 4, 16384, 64) | bfloat16 | device | current-qualified-six-output-eager | 1.56074107 | 14.54393713 | 9.318610 |        0.107747 |
| train-16k-float16            | (2, 4, 16384, 64) | float16  | device | current-qualified-six-output-eager | 1.56735733 | 14.50428520 | 9.253975 |        0.107292 |
| train-2k-bfloat16            | (2, 4, 2048, 64)  | bfloat16 | host   | current-qualified-six-output-eager | 1.68428998 |  6.79183394 | 4.032461 |        0.012549 |
| train-2k-float16             | (2, 4, 2048, 64)  | float16  | host   | current-qualified-six-output-eager | 1.37648995 |  6.82834933 | 4.960697 |        0.015355 |
| train-4k-bfloat16            | (2, 4, 4096, 64)  | bfloat16 | device | current-qualified-six-output-eager | 0.41259773 |  3.69417580 | 8.953456 |        0.102133 |
| train-4k-float16             | (2, 4, 4096, 64)  | float16  | host   | current-qualified-six-output-eager | 1.82636594 | 13.15061273 | 7.200426 |        0.023073 |
| train-8k-bfloat16            | (2, 4, 8192, 64)  | bfloat16 | device | current-qualified-six-output-eager | 0.78263333 |  7.32156807 | 9.355042 |        0.107519 |
| train-8k-float16             | (2, 4, 8192, 64)  | float16  | device | current-qualified-six-output-eager | 0.78904487 |  7.29559720 | 9.246112 |        0.106646 |

## DeltaNetChunkBwdOp

| Case                         | q shape           | dtype    | Metric | Baseline                                  | TileOPs ms |  Baseline ms |     Ratio | Logical BW TB/s |
| ---------------------------- | ----------------- | -------- | ------ | ----------------------------------------- | ---------: | -----------: | --------: | --------------: |
| deltanet-1.3b-train-bfloat16 | (1, 16, 256, 128) | bfloat16 | device | flash-linear-attention-npu (historical)   | 0.26146147 |   0.21470247 |  0.821163 |        0.060219 |
| train-16k-bfloat16           | (2, 4, 16384, 64) | bfloat16 | device | flash-linear-attention-npu (historical)   | 1.60195353 |   5.59046980 |  3.489783 |        0.136558 |
| train-16k-float16            | (2, 4, 16384, 64) | float16  | device | retained-v010-torch-eager                 | 1.60028667 |  73.64499820 | 46.019879 |        0.136700 |
| train-2k-bfloat16            | (2, 4, 2048, 64)  | bfloat16 | device | flash-linear-attention-npu (historical)   | 0.34908907 |   0.77588353 |  2.222595 |        0.078661 |
| train-2k-float16             | (2, 4, 2048, 64)  | float16  | host   | current-qualified-four-gradient-eager     | 4.00679931 |  28.07063010 |  7.005749 |        0.006853 |
| train-4k-bfloat16            | (2, 4, 4096, 64)  | bfloat16 | device | flash-linear-attention-npu (historical)   | 0.50382267 |   1.46916107 |  2.916028 |        0.108745 |
| train-4k-float16             | (2, 4, 4096, 64)  | float16  | host   | retained-v1-qualified-four-gradient-eager | 3.14379859 |  56.83132798 | 18.077280 |        0.017427 |
| train-8k-bfloat16            | (2, 4, 8192, 64)  | bfloat16 | device | flash-linear-attention-npu (historical)   | 0.82037020 |   2.85779527 |  3.483543 |        0.133409 |
| train-8k-float16             | (2, 4, 8192, 64)  | float16  | host   | retained-v1-qualified-four-gradient-eager | 8.50051002 | 112.98612269 | 13.291687 |        0.012875 |

## KDAFwdOp

| Case                            | q shape              | dtype    | Metric | Baseline                                |   TileOPs ms |    Baseline ms |      Ratio | Logical BW TB/s |
| ------------------------------- | -------------------- | -------- | ------ | --------------------------------------- | -----------: | -------------: | ---------: | --------------: |
| kda-bounded-packed-bfloat16     | (1, 256, 4, 128)     | bfloat16 | device | torch-eager (historical)                |   0.06597800 |    13.57854095 | 205.804070 |        0.035791 |
| kda-raw-gate-decode-b8-bfloat16 | (8, 1, 32, 128)      | bfloat16 | device | torch-eager (historical)                |   0.01837980 |     0.41410703 |  22.530551 |        1.844369 |
| kda-tp1-1024-7168-bfloat16      | (1, 8192, 32, 128)   | bfloat16 | host   | sgl-kernel-npu (historical)             |   6.81776131 |    10.13245728 |   1.486185 |        0.049908 |
| kda-tp1-16384-bfloat16          | (1, 16384, 32, 128)  | bfloat16 | host   | vllm-ascend (historical)                |  13.81471800 |    36.85401405 |   2.667736 |        0.048806 |
| kda-tp1-16x8192-bfloat16        | (1, 131072, 32, 128) | bfloat16 | host   | vllm-ascend (historical)                | 106.25667938 |   140.56336867 |   1.322866 |        0.050921 |
| kda-tp1-4096-bfloat16           | (1, 4096, 32, 128)   | bfloat16 | device | flash-linear-attention-npu (historical) |   3.40933167 |    20.06892133 |   5.886468 |        0.049902 |
| kda-tp1-4x1024-bfloat16         | (1, 4096, 32, 128)   | bfloat16 | host   | sgl-kernel-npu (historical)             |   3.49973794 |     5.68646804 |   1.624827 |        0.050410 |
| kda-tp1-4x1024-float16          | (1, 4096, 32, 128)   | float16  | host   | vllm-ascend (historical)                |   3.47428943 |    29.02252541 |   8.353514 |        0.050780 |
| kda-tp1-4x8192-bfloat16         | (1, 32768, 32, 128)  | bfloat16 | host   | sgl-kernel-npu (historical)             |  26.97126732 |    30.65274997 |   1.136496 |        0.050152 |
| kda-tp1-65536-bfloat16          | (1, 65536, 32, 128)  | bfloat16 | host   | sgl-kernel-npu (historical)             |  54.93931544 |    49.54517062 |   0.901816 |        0.048975 |
| kda-tp1-decode-b1-bfloat16      | (1, 1, 32, 128)      | bfloat16 | device | flash-linear-attention-npu (historical) |   0.00510653 |     0.02569807 |   5.032390 |        0.829394 |
| kda-tp1-decode-b128-bfloat16    | (128, 1, 32, 128)    | bfloat16 | device | flash-linear-attention-npu (historical) |   0.22787347 |     1.03018273 |   4.520854 |        2.379048 |
| kda-tp1-decode-b256-bfloat16    | (256, 1, 32, 128)    | bfloat16 | device | flash-linear-attention-npu (historical) |   0.44860173 |     2.04526953 |   4.559210 |        2.416941 |
| kda-tp1-decode-b32-bfloat16     | (32, 1, 32, 128)     | bfloat16 | device | flash-linear-attention-npu (historical) |   0.06081593 |     0.23026333 |   3.786234 |        2.228536 |
| kda-tp1-decode-b64-bfloat16     | (64, 1, 32, 128)     | bfloat16 | device | flash-linear-attention-npu (historical) |   0.11678993 |     0.51946233 |   4.447835 |        2.320928 |
| kda-tp1-decode-b8-bfloat16      | (8, 1, 32, 128)      | bfloat16 | device | flash-linear-attention-npu (historical) |   0.01788640 |     0.06713493 |   3.753407 |        1.894323 |
| kda-tp1-raw-gate-4096-bfloat16  | (1, 4096, 32, 128)   | bfloat16 | device | flash-linear-attention-npu (historical) |   4.12535153 |    20.26287340 |   4.911793 |        0.041244 |
| kda-tp2-16384-bfloat16          | (1, 16384, 16, 128)  | bfloat16 | host   | vllm-ascend (historical)                |   6.93859737 |    29.88270596 |   4.306736 |        0.048586 |
| kda-tp2-4096-bfloat16           | (1, 4096, 16, 128)   | bfloat16 | host   | sgl-kernel-npu (historical)             |   1.86128672 |     4.39738470 |   2.362551 |        0.045703 |
| kda-tp2-65536-bfloat16          | (1, 65536, 16, 128)  | bfloat16 | host   | sgl-kernel-npu (historical)             |  27.29417197 |    48.24476599 |   1.767585 |        0.049290 |
| kda-tp4-16384-bfloat16          | (1, 16384, 8, 128)   | bfloat16 | host   | vllm-ascend (historical)                |   4.23074532 |    28.78543601 |   6.803869 |        0.039841 |
| kda-tp4-4096-bfloat16           | (1, 4096, 8, 128)    | bfloat16 | device | flash-linear-attention-npu (historical) |   1.01352513 |    19.35750667 |  19.099188 |        0.041965 |
| kda-tp4-65536-bfloat16          | (1, 65536, 8, 128)   | bfloat16 | host   | vllm-ascend (historical)                |  16.57244532 |    38.03822395 |   2.295269 |        0.040589 |
| kda-tp8-1024-7168-bfloat16      | (1, 8192, 4, 128)    | bfloat16 | device | flash-linear-attention-npu (historical) |   1.27865393 |    38.22057920 |  29.891262 |        0.033264 |
| kda-tp8-16384-bfloat16          | (1, 16384, 4, 128)   | bfloat16 | host   | sgl-kernel-npu (historical)             |   2.91480736 |    12.83878349 |   4.404677 |        0.028914 |
| kda-tp8-16x8192-bfloat16        | (1, 131072, 4, 128)  | bfloat16 | device | flash-linear-attention-npu (historical) |  13.49035147 |   614.02897627 |  45.516159 |        0.050134 |
| kda-tp8-4096-bfloat16           | (1, 4096, 4, 128)    | bfloat16 | device | flash-linear-attention-npu (historical) |   0.68516740 |    19.14551447 |  27.942827 |        0.031038 |
| kda-tp8-65536-bfloat16          | (1, 65536, 4, 128)   | bfloat16 | host   | vllm-ascend (historical)                |  11.29343538 |    30.96296672 |   2.741678 |        0.029781 |
| kda-tp8-decode-b256-bfloat16    | (256, 1, 4, 128)     | bfloat16 | device | flash-linear-attention-npu (historical) |   0.06046760 |     0.23047453 |   3.811538 |        2.241374 |
| kda-tp8-decode-b32-bfloat16     | (32, 1, 4, 128)      | bfloat16 | device | flash-linear-attention-npu (historical) |   0.01011693 |     0.04407073 |   4.356136 |        1.674550 |
| kimi-k3-tp1-mixed8k-bfloat16    | (1, 8192, 96, 128)   | bfloat16 | device | torch-eager (historical)                |  25.69134100 |  1569.03126557 |  61.072377 |        0.039735 |
| kimi-k3-tp1-s1024-bfloat16      | (1, 1024, 96, 128)   | bfloat16 | device | torch-eager (historical)                |   3.27684133 |   203.78004174 |  62.187949 |        0.040395 |
| kimi-k3-tp1-s16384-bfloat16     | (1, 16384, 96, 128)  | bfloat16 | device | torch-eager (historical)                |  52.68964820 |  3290.88950786 |  62.457990 |        0.038390 |
| kimi-k3-tp1-s4096-bfloat16      | (1, 4096, 96, 128)   | bfloat16 | device | torch-eager (historical)                |  13.16026440 |   825.23425817 |  62.706511 |        0.038787 |
| kimi-k3-tp2-s32768-bfloat16     | (1, 32768, 48, 128)  | bfloat16 | device | torch-eager (historical)                |  52.28999593 |  4641.89771419 |  88.772195 |        0.038623 |
| kimi-k3-tp2-s4096-bfloat16      | (1, 4096, 48, 128)   | bfloat16 | device | torch-eager (historical)                |   6.53338493 |   575.45819524 |  88.079640 |        0.039064 |
| kimi-k3-tp4-s4096-bfloat16      | (1, 4096, 24, 128)   | bfloat16 | device | torch-eager (historical)                |   3.44014767 |   446.81855010 | 129.883538 |        0.037095 |
| kimi-k3-tp4-s65536-bfloat16     | (1, 65536, 24, 128)  | bfloat16 | device | torch-eager (historical)                |  55.17182260 |  7062.64340926 | 128.011783 |        0.036577 |
| kimi-k3-tp8-4x8192-bfloat16     | (1, 32768, 12, 128)  | bfloat16 | device | torch-eager (historical)                |  12.85387933 |  1165.15207102 |  90.645947 |        0.039463 |
| kimi-k3-tp8-b32-s256-bfloat16   | (32, 256, 12, 128)   | bfloat16 | device | torch-eager (historical)                |   3.37302980 |   132.08554544 |  39.159318 |        0.044825 |
| kimi-k3-tp8-cont256-bfloat16    | (1, 256, 12, 128)    | bfloat16 | device | torch-eager (historical)                |   0.13818513 |    22.04332085 | 159.520205 |        0.039927 |
| kimi-k3-tp8-cont4096-bfloat16   | (1, 4096, 12, 128)   | bfloat16 | device | torch-eager (historical)                |   1.70651967 |   360.13862932 | 211.036905 |        0.037850 |
| kimi-k3-tp8-s131072-bfloat16    | (1, 131072, 12, 128) | bfloat16 | device | torch-eager (historical)                |  54.72887033 | 11366.54349087 | 207.688253 |        0.036858 |
| kimi-k3-tp8-s4096-bfloat16      | (1, 4096, 12, 128)   | bfloat16 | device | torch-eager (historical)                |   1.69760707 |   363.55060437 | 214.154743 |        0.037586 |
| kl48-tp1-mixed8k-bfloat16       | (1, 8192, 32, 128)   | bfloat16 | device | torch-eager (historical)                |   8.03903047 |   913.51516450 | 113.634992 |        0.042328 |
| kl48-tp1-s1024-bfloat16         | (1, 1024, 32, 128)   | bfloat16 | device | torch-eager (historical)                |   1.03146593 |   119.55395315 | 115.906836 |        0.042776 |
| kl48-tp1-s16384-bfloat16        | (1, 16384, 32, 128)  | bfloat16 | device | torch-eager (historical)                |  16.51632767 |  1947.20858072 | 117.895977 |        0.040823 |
| kl48-tp2-s32768-bfloat16        | (1, 32768, 16, 128)  | bfloat16 | device | torch-eager (historical)                |  16.43090333 |  3134.97576837 | 190.797530 |        0.040971 |
| kl48-tp2-s4096-bfloat16         | (1, 4096, 16, 128)   | bfloat16 | device | torch-eager (historical)                |   2.04917640 |   383.94111028 | 187.363621 |        0.041516 |
| kl48-tp4-s4096-bfloat16         | (1, 4096, 8, 128)    | bfloat16 | device | torch-eager (historical)                |   1.19673000 |   318.54454139 | 266.179123 |        0.035544 |
| kl48-tp4-s65536-bfloat16        | (1, 65536, 8, 128)   | bfloat16 | device | torch-eager (historical)                |  19.31050840 |  4980.79408646 | 257.931795 |        0.034834 |
| kl48-tp8-4x8192-bfloat16        | (1, 32768, 4, 128)   | bfloat16 | device | torch-eager (historical)                |   4.09675053 |   766.49321158 | 187.097849 |        0.041273 |
| kl48-tp8-b32-s256-bfloat16      | (32, 256, 4, 128)    | bfloat16 | device | torch-eager (historical)                |   1.02500147 |    61.32613288 |  59.830288 |        0.049170 |
| kl48-tp8-cont256-bfloat16       | (1, 256, 4, 128)     | bfloat16 | device | torch-eager (historical)                |   0.08015347 |    17.15003041 | 213.964924 |        0.022945 |
| kl48-tp8-cont4096-bfloat16      | (1, 4096, 4, 128)    | bfloat16 | device | torch-eager (historical)                |   0.79315773 |   274.75210415 | 346.402856 |        0.027145 |
| kl48-tp8-s131072-bfloat16       | (1, 131072, 4, 128)  | bfloat16 | device | torch-eager (historical)                |  25.21198973 |  8847.45033572 | 350.922336 |        0.026670 |
| kl48-tp8-s4096-bfloat16         | (1, 4096, 4, 128)    | bfloat16 | device | torch-eager (historical)                |   0.78355533 |   283.43640146 | 361.731188 |        0.027144 |
