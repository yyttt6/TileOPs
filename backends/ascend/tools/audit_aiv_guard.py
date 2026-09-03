#!/usr/bin/env python3
"""AIV-guard audit for generated Ascend C artifacts.

Detects the F013 defect class: a **scalar GM write** (`<GlobalTensor>.SetValue(...)`)
that is NOT inside an `if ASCEND_IS_AIV { ... }` block.

Why this is a bug (see docs/PROJECT_STATE.md §13.6):

The lowered kernel runs on BOTH the cube core (AIC) and the vector cores (AIV) --
the header says `KERNEL_TYPE_MIX_AIC_1_2`. TileLang normalizes the block index with

    auto cid = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV { cid = cid / 2; }

so on AIC `cid` is NOT halved. A tail-block predicate such as
`full = (cid * TILE + vid * SUB) < BODY_END` therefore evaluates differently on AIC,
which then takes the scalar tail branch and writes GM from its own never-populated UB
(reads as 0.0), racing the AIV core that writes the correct value.

Bulk transfers (`copy_ub_to_gm`) are DMA intrinsics and are inert on AIC, so only the
**scalar** `SetValue` path is affected. That is why the corruption lands on exactly the
tail index, is finite (0.0), and is nondeterministic -- it is a core race, not a cache
or numerics problem. Clearing caches cannot help.

Usage:
    python audit_aiv_guard.py <artifact.cu> [more.cu ...]
Exit code 0 = clean, 1 = at least one unguarded scalar GM write.
"""
import re
import sys


def gm_tensors(src: str) -> set:
    """Names declared as AscendC::GlobalTensor<...> -- i.e. real GM buffers."""
    return set(re.findall(r'AscendC::GlobalTensor<[^>]+>\s+([A-Za-z_][A-Za-z0-9_]*)\s*;', src))


def audit(path: str):
    src = open(path).read()
    gm = gm_tensors(src)
    findings = []
    depth = 0
    aiv_stack = []          # brace depths at which an AIV guard opened
    for lineno, line in enumerate(src.split('\n'), 1):
        # An `if ASCEND_IS_AIV` that is not the combined AIC&&AIV form.
        opens_aiv = 'ASCEND_IS_AIV' in line and 'ASCEND_IS_AIC' not in line
        for ch in line:
            if ch == '{':
                depth += 1
                if opens_aiv:
                    aiv_stack.append(depth)
                    opens_aiv = False
            elif ch == '}':
                if aiv_stack and depth == aiv_stack[-1]:
                    aiv_stack.pop()
                depth -= 1
        for m in re.finditer(r'\b([A-Za-z_][A-Za-z0-9_]*)\.SetValue\(', line):
            if m.group(1) in gm and not aiv_stack:
                findings.append((lineno, m.group(1), line.strip()))
    return gm, findings


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    bad = 0
    for path in argv[1:]:
        gm, findings = audit(path)
        if findings:
            bad += 1
            print(f"FAIL  {path}   (GM tensors: {', '.join(sorted(gm))})")
            for lineno, name, text in findings:
                print(f"        line {lineno}: unguarded GM scalar write via {name}.SetValue")
                print(f"                  {text}")
        else:
            print(f"OK    {path}   (GM tensors: {', '.join(sorted(gm))})")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
