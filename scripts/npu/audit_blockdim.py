#!/usr/bin/env python3
"""BlockDim audit for generated Ascend C artifacts.

CANN 8.5 accepts a physical launch count in ``[1, 65535]``. Exceed it and the launch
fails with ``ret 107000`` -- but nothing raises: the output tensor simply keeps
whatever was in it, which reads downstream as a numeric error rather than a launch
error (docs/PROJECT_STATE.md §13.1).

The launch count is a literal in the generated code::

    main_kernel<<<17, nullptr, stream>>>(X_handle, Y_handle, fftsAddr);

so a template that has not been given a grid-stride loop can be caught statically, at
whatever shape you generate it for. That is the whole check: generate an artifact at a
shape whose logical tile count exceeds 65535, and read the number back.

A template WITH grid-stride pins the launch at <= 65535 and loops inside
(``cid + repeat * launch_blocks``); a template WITHOUT it emits the raw tile count and
is broken at that shape. Discovered the expensive way: on 2026-08-26 the unary math
template still emitted a raw count, so ``AbsFwdOp``/``NegFwdOp``/``SigmoidFwdOp`` all
failed at a 16M-element shape with ``BlockDim=65536`` -- long after grid-stride had been
added to the GEMM (T057) and two-pass (T065) templates. Per-template migration had no
gate, so nobody noticed the gap.

Usage:
    python audit_blockdim.py <artifact.cu> [more.cu ...]
Exit code 0 = every launch count within limit, 1 = at least one over.
"""
import re
import sys

MAX_BLOCK_COUNT = 65535   # CANN 8.5 hard limit, inclusive

LAUNCH_RE = re.compile(r'(\w+)\s*<<<\s*([0-9]+)\s*,')


def audit(path):
    """[(kernel_name, launch_count, line_no)] for every launch found."""
    launches = []
    for lineno, line in enumerate(open(path), 1):
        for m in LAUNCH_RE.finditer(line):
            launches.append((m.group(1), int(m.group(2)), lineno))
    return launches


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    over = 0
    for path in argv[1:]:
        launches = audit(path)
        if not launches:
            print(f"?     {path}   no literal <<<N,...>>> launch found "
                  f"(dynamic launch count? report it rather than assuming clean)")
            continue
        bad = [t for t in launches if t[1] > MAX_BLOCK_COUNT]
        if bad:
            over += 1
            print(f"FAIL  {path}")
            for name, count, lineno in bad:
                print(f"        line {lineno}: {name}<<<{count}>>> exceeds "
                      f"{MAX_BLOCK_COUNT} -- launch returns ret 107000 and the output "
                      f"is left untouched, silently")
        else:
            counts = ', '.join(f'{n}<<<{c}>>>' for n, c, _ in launches)
            print(f"OK    {path}   ({counts})")
    return 1 if over else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
