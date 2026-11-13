"""Exact reference-only compatibility for unaffected shared28 Golden outputs.

Only the reviewed source pair and explicit non-MoE oracle set qualify. Runtime,
toolchain, implementation, adapter and input identities remain independently
checked by GoldenCache. A different source tree uses its fresh identity.
"""

import hashlib
import os
from pathlib import Path

from golden_cache import digest_file, reference_source_digest

OLD_SOURCE = "f27a1471dfe7f6633f89f1eb9ebe43f9edf96f63885974f18cab3698d4173dd1"
NEW_SOURCE = "98a9fdb9882b2ecb21ef2064b29f0158cf4cd159084f2a953d878264e2a17636"
UNCHANGED_OPERATORS = frozenset(
    [
        "BatchNormBwdOp",
        "DSADenseFwdOp",
        "DSAPagedFwdOp",
        "DeltaNetChunkBwdOp",
        "DeltaNetChunkFwdOp",
        "DeltaNetFwdOp",
        "EngramGateConvBwdOp",
        "EngramGateConvFwdOp",
        "FP8LightningIndexerFwdOp",
        "GDNFwdOp",
        "GEMMW4A16FwdOp",
        "GEMMW4A8FwdOp",
        "GLAChunkBwdOp",
        "GLAChunkFwdOp",
        "GLAFwdOp",
        "GQADenseBwdOp",
        "GQADenseFwdOp",
        "GQADenseWithLSEFwdOp",
        "GQAPagedFwdOp",
        "GQAVarlenFwdOp",
        "GroupedGEMMFwdOp",
        "HadamardTransformFwdOp",
        "KDAFwdOp",
        "MHAPagedFwdOp",
        "MHCPreFwdOp",
        "MLADenseBwdOp",
        "MLADenseDecodeFwdOp",
        "MLAPagedFwdOp",
        "MLAVarlenBwdOp",
        "MLAVarlenFwdOp",
        "Mamba2FwdOp",
        "NSACompressedVarlenFwdOp",
        "NSATopKVarlenFwdOp",
        "NSAVarlenFwdOp",
        "SSDChunkScanFwdOp",
        "SSDChunkStateFwdOp",
        "SSDStatePassingFwdOp",
        "SelectiveScanFwdOp",
        "TopKMaskFwdOp",
        "TopKTopPMaskFwdOp",
        "TopPMaskFwdOp",
    ]
)


def _with_overlay(source_digest):
    result = hashlib.sha256(bytes.fromhex(source_digest))
    overlay = os.environ.get("ASCEND950_REFERENCE_ROOT")
    if overlay:
        for path in sorted(Path(overlay).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".so"}:
                result.update(str(path.relative_to(overlay)).encode())
                result.update(bytes.fromhex(digest_file(path)))
    return result.hexdigest()


def reuse_unchanged_reference_identity(source, op, identity):
    """Reuse only unchanged independent references; never affected shared math."""
    if op not in UNCHANGED_OPERATORS or identity.get("source_mode") != "independent-reference":
        return identity
    digest, mode = reference_source_digest(source)
    if mode != "independent-reference" or digest != NEW_SOURCE:
        return identity
    if identity.get("source") != _with_overlay(NEW_SOURCE):
        return identity
    return dict(identity, source=_with_overlay(OLD_SOURCE))
