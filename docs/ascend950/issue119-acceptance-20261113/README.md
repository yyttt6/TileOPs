# Issue 119 acceptance

The 17 alert cases across nine operators meet baseline / worst independent candidate mean >= 1.0 on server2 physical 32 AIC / 64 AIV. The minimum ratio is 1.00606; this result has a small margin.

`FINAL_AUDIT.json` records both candidate means, frozen baseline identities, source hashes, and 1695 audited raw windows. The per-operator reports record complete manifest correctness and affected public tests. The measured kernel tree is `a1aa520be3bd2a8191e44f425f83f38a327f733e`; this documentation addition does not change kernels.

Evidence paths in the reports are relative to the retained development bundle `reports/issue119-server2-20261113` in the development workspace. Raw profiles and guard receipts are retained there and on server2 under `/opt/tileops-dev/issue119-20261113/evidence`; they are not embedded in this summary directory. These are development acceptance results, not a claim that post-merge CI has completed.

The remaining 0.8–1.0 cases and 605 unmatched baselines are outside this PR. This PR must not close issue 119 as a whole. No benchmark contract, reference tolerance, or baseline definition was changed.

## PR review follow-up: BF16 square-512 BMM

The reviewer found an additional BF16 `(16,512,512)` BMM case at 18.838 us (0.7977x). It now shares the existing FP16 explicit-L0 schedule with a dtype-specific cache entry. On server2 physical32 AIC, two independent candidate means are 14.47087 and 14.46653 us, giving **1.03844x** against the unchanged 15.0272 us baseline using the slower round. FP16 remains >=1.0x; its generated C++ is byte-identical.

`REVIEW_FIX_AUDIT.json` records this follow-up separately from the historical 17-case audit above: 16/16 BMM manifest cases pass correctness and >=0.8x performance, 26 public tests pass, and 270 raw windows replay. The other 15 BMM generated kernels are unchanged. No baseline was remeasured. Updated-head CI remains separate.
