"""Pre-release checks for a trained model (release_check.py); no TensorFlow."""
import hashlib
import json
import os

import release_check


def _write(d, name, data):
    path = os.path.join(d, name)
    with open(path, 'w' if isinstance(data, str) else 'wb') as f:
        f.write(data)
    return path


def _model_dir(tmp_path, gate_passed=True):
    d = str(tmp_path)
    files = {name: '{}' for name in release_check.REQUIRED if name not in ('manifest.json', 'quality_report.json')}
    files['lstm_model.h5'] = 'weights'
    files['lstm_model.tflite'] = 'tflite'
    files['training_config.json'] = json.dumps({'num_stations': 7, 'training_years': 15, 'ensemble_size': 3,
                                                 'final_refit': True, 'disabled_sources': ['SMAP']})
    for name, data in files.items():
        _write(d, name, data)
    manifest = {'model_version': 'model-212', 'tflite_mode': 'builtins',
                'files': [{'name': n, 'bytes': len(v), 'sha256': hashlib.sha256(v.encode()).hexdigest()}
                          for n, v in files.items()]}
    _write(d, 'manifest.json', json.dumps(manifest))
    _write(d, 'quality_report.json', json.dumps({'gate': {'passed': gate_passed, 'checks': [
        {'check': 'day 1 close to persistence', 'passed': gate_passed, 'detail': 'model 2.9% vs 3.8%'}]}}))
    _write(d, 'quality_report.md', '# OpenFlow forecast quality\n\n**Quality gate: PASSED**\n')
    return d


def test_complete_passing_model_releases_with_notes(tmp_path):
    d = _model_dir(tmp_path)
    assert release_check.problems(d) == []
    notes = str(tmp_path / 'notes.md')
    assert release_check.main(['--dir', d, '--notes', notes, '--run-url', 'https://x/runs/1',
                               '--commit', 'abcdef1234567890', '--branch', 'dev']) == 0
    text = open(notes).read()
    assert 'https://x/runs/1' in text and '`abcdef123456`' in text and '`dev`' in text
    assert '| Training years | 15 |' in text and 'Quality gate: PASSED' in text
    assert 'lstm_model.tflite' in text and 'SMAP' in text


def test_failed_gate_tampered_file_or_missing_mobile_model_blocks_release(tmp_path, capsys):
    d = _model_dir(tmp_path, gate_passed=False)
    assert any('quality gate failed' in p and 'day 1' in p for p in release_check.problems(d))
    _write(d, 'lstm_model.h5', 'tampered')
    assert 'sha256 mismatch for lstm_model.h5' in release_check.problems(d)
    os.remove(os.path.join(d, 'lstm_model.tflite'))
    issues = release_check.problems(d)
    assert any(i.startswith('no mobile model') for i in issues)
    assert 'manifest lists lstm_model.tflite but it is missing' in issues
    assert release_check.main(['--dir', d]) == 1
    assert 'Not releasing' in capsys.readouterr().out


def test_missing_required_file_blocks_release(tmp_path):
    d = _model_dir(tmp_path)
    os.remove(os.path.join(d, 'basin_zones.json'))
    assert 'missing basin_zones.json' in release_check.problems(d)
