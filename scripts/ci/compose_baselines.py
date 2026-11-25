"""Append a pinned operator release to the website snapshot used by Ascend PR CI."""

import argparse
import hashlib
import json
from pathlib import Path


def compose(site: Path, addition: Path, output: Path) -> str:
    snapshot_bytes = (site / "snapshot.json").read_bytes()
    snapshot = json.loads(snapshot_bytes)
    index = json.loads((site / "latest-release.json").read_bytes())
    assert hashlib.sha256(snapshot_bytes).hexdigest() == index["snapshot_sha256"]
    assert index["count"] == len(index["measurements"]) == len(snapshot["rows"])
    release = json.loads(addition.read_bytes())
    rows = release["rows"]
    assert (
        rows
        and len(rows) == release["row_count"]
        and not release["pending_operators"]
        and not release["rejected_operators"]
    )

    def key(row):
        return row["library"], row["op"], row["case"]

    keys = {key(r) for r in snapshot["rows"] + snapshot.get("historical_rows", [])}
    added = {key(r) for r in rows}
    assert len(added) == len(rows) and not (keys & added), (
        "baseline release overlaps an existing workload"
    )
    ids = {r["measurement_id"] for r in snapshot["rows"] + snapshot.get("historical_rows", [])}
    assert len({r["measurement_id"] for r in rows} - ids) == len(rows)
    snapshot["rows"].extend(rows)
    index["measurements"].extend(
        {k: r[k] for k in ("library", "op", "case", "measurement_id", "protocol")} for r in rows
    )
    index["count"] = len(snapshot["rows"])
    index["id"] = snapshot["baseline_release"] = f"{index['id']}+{addition.parent.name}"
    snapshot["manifest_cases"] += len({(r["op"], r["case"]) for r in rows})
    data = (json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    index["snapshot_sha256"] = hashlib.sha256(data).hexdigest()
    index["additional_release_sha256"] = hashlib.sha256(addition.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output / "snapshot.json").write_bytes(data)
    (output / "latest-release.json").write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    return index["snapshot_sha256"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    parser.add_argument("addition", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(compose(args.site, args.addition, args.output))
