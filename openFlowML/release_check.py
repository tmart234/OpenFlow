"""
Pre-release check for a trained model (the Release Model workflow).

Given the artifacts a "Weekly ML Training" run uploaded, this:
- confirms everything inference needs is present, plus at least one mobile model;
- checks each manifest.json sha256 against the file;
- requires the quality gate in quality_report.json to have passed;
- writes the release notes: provenance, training setup, quality report, checksums.

Standard library only: the release job does not install TensorFlow.

  python release_check.py --dir model_release --notes release_notes.md \
      --run-url <url> --commit <sha> --branch <branch>

Exit status 0 means the model may be published; 1 means it may not, and the
reasons are printed.
"""

import argparse
import hashlib
import json
import os
import sys
from typing import List, Optional

REQUIRED = ['lstm_model.h5', 'scalers.json', 'station_index.json', 'basin_index.json',
            'training_config.json', 'upstream_gauges.json', 'basin_zones.json',
            'manifest.json', 'quality_report.json']
MOBILE = ['lstm_model.tflite', 'lstm_model.mlpackage.zip']


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def problems(model_dir: str) -> List[str]:
    """Reasons the artifacts in model_dir must not be released (empty list = OK)."""
    out = [f"missing {name}" for name in REQUIRED if not os.path.exists(os.path.join(model_dir, name))]
    if not any(os.path.exists(os.path.join(model_dir, m)) for m in MOBILE):
        out.append(f"no mobile model ({' or '.join(MOBILE)}): the export step failed")
    manifest_path = os.path.join(model_dir, 'manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        for entry in manifest.get('files', []):
            path = os.path.join(model_dir, entry['name'])
            if not os.path.exists(path):
                out.append(f"manifest lists {entry['name']} but it is missing")
            elif _sha256(path) != entry['sha256']:
                out.append(f"sha256 mismatch for {entry['name']}")
    report_path = os.path.join(model_dir, 'quality_report.json')
    if os.path.exists(report_path):
        with open(report_path) as f:
            gate = json.load(f).get('gate', {})
        if not gate.get('passed'):
            failed = [c['check'] + ': ' + c['detail'] for c in gate.get('checks', []) if not c.get('passed')]
            out.append("quality gate failed" + (" -- " + "; ".join(failed) if failed else ''))
    return out


def release_notes(model_dir: str, run_url: Optional[str] = None, commit: Optional[str] = None,
                  branch: Optional[str] = None) -> str:
    with open(os.path.join(model_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    with open(os.path.join(model_dir, 'training_config.json')) as f:
        config = json.load(f)
    lines = []
    prov = [f"training run {run_url}" if run_url else None,
            f"commit `{commit[:12]}`" if commit else None,
            f"branch `{branch}`" if branch else None]
    lines.append("Trained by " + ", ".join(p for p in prov if p) + "." if any(prov) else "")
    setup = {
        'Model version (manifest)': manifest.get('model_version'),
        'Stations': config.get('num_stations'),
        'Training years': config.get('training_years'),
        'Ensemble members': config.get('ensemble_size'),
        'Refit on all data': config.get('final_refit'),
        'Disabled sources': ', '.join(config.get('disabled_sources') or []) or 'none',
        'TFLite mode': manifest.get('tflite_mode'),
    }
    lines += ['', '| Setting | Value |', '|---|---|']
    lines += [f"| {k} | {v} |" for k, v in setup.items() if v is not None]
    report_md = os.path.join(model_dir, 'quality_report.md')
    if os.path.exists(report_md):
        with open(report_md) as f:
            lines += ['', f.read().strip()]
    lines += ['', '<details><summary>Checksums (sha256)</summary>', '', '| File | Bytes | sha256 |', '|---|---|---|']
    lines += [f"| {e['name']} | {e['bytes']:,} | `{e['sha256']}` |" for e in manifest.get('files', [])]
    lines += ['', '</details>', '',
              'Download every file into one directory; `docs/INFERENCE.md` describes how to use them.']
    return '\n'.join(lines).strip() + '\n'


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--dir', required=True)
    p.add_argument('--notes', help='write release notes (markdown) here')
    p.add_argument('--run-url')
    p.add_argument('--commit')
    p.add_argument('--branch')
    args = p.parse_args(argv)
    issues = problems(args.dir)
    if issues:
        print("Not releasing:\n" + "\n".join(f"- {i}" for i in issues))
        return 1
    if args.notes:
        with open(args.notes, 'w') as f:
            f.write(release_notes(args.dir, args.run_url, args.commit, args.branch))
    print("Release check passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
