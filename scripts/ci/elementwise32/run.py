"""Prepare a frozen job workspace and run the physical32 elementwise CI suite."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ASSETS = Path(__file__).resolve().parent
REPO = ASSETS.parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--operators', default='all')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / 'workspace'
    shutil.copytree(ASSETS / 'payload', workspace)
    (workspace / 'TileOPs').symlink_to(REPO, target_is_directory=True)
    gate_path = a.baseline_cache / 'BASELINE_ADMISSION.json'
    gate = json.loads(gate_path.read_text())
    assert gate['status'] == 'passed' and gate['server1_complete'] is True
    cache = a.baseline_cache / 'BASELINE32_PRIVATE.json'
    assert digest(cache) == gate['catalog_sha256']
    for name in ('BASELINE_ADMISSION.json', 'BASELINE32_PRIVATE.json'):
        shutil.copy2(a.baseline_cache / name, workspace / name)
    export = json.loads((workspace / 'ELEMENTWISE_EXPORT.json').read_text())
    for name, expected in export['source_files'].items():
        assert digest(REPO / name) == expected, 'Source differs from admitted export: ' + name
    plan = json.loads((workspace / 'ELEMENTWISE_PLAN.json').read_text())
    selected = [r['op'] for r in plan] if a.operators == 'all' else a.operators.split(',')
    assert selected and len(set(selected)) == len(selected)
    assert set(selected) <= {r['op'] for r in plan}
    metadata = dict(phase='prepared', source_revision=subprocess.check_output(
        ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        operators=selected, baseline_catalog_sha256=gate['catalog_sha256'])
    (output / 'job.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if a.prepare_only:
        print(json.dumps(metadata))
        return 0
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        (str(workspace), str(REPO / 'src'), str(REPO))))
    command = [sys.executable, str(workspace / 'tools/ci_elementwise32.py'),
               '--output', str(output / 'measurements')]
    for op in selected:
        command += ['--op', op]
    # The controller acquires the host guard separately for every complete op.
    result = subprocess.run(command, cwd=REPO, env=env)
    metadata.update(phase='finished', measurement_exit_code=result.returncode)
    (output / 'job.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
