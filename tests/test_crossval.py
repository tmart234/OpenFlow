"""
Leave-years-out cross-validation (crossval.py): an end-to-end run on a
synthetic multi-year dataset (needs tensorflow; CI model-smoke job) and the
report comparison (no tensorflow).
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

import crossval


def _report(mean, folds):
    leads = 14
    return {'n_folds': len(folds), 'windows': 100, 'mean_error_pct': mean,
            'pooled': {'all': {'error_pct': {'model': [mean] * leads, 'persistence': [mean + 1] * leads}}},
            'folds': [{'fold': i, 'test_water_years': [2020 + i], 'windows': 20, 'mean_error_pct': f}
                      for i, f in enumerate(folds)]}


def test_compare_reports_b_minus_a_by_lead_and_fold(tmp_path):
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text(json.dumps(_report(8.0, [8.5, 7.5, 8.0])))
    b.write_text(json.dumps(_report(7.5, [8.0, 7.7, 7.0])))
    md = crossval.compare(str(a), str(b))
    assert '| B − A | -0.50' in md
    assert 'B is better in 2 of 3 folds' in md


def test_cross_validation_end_to_end_on_synthetic_years(tmp_path, monkeypatch):
    pytest.importorskip('tensorflow')
    import normalize_data
    import train
    from test_train_smoke import _station
    monkeypatch.setattr(train, 'EPOCHS', 1)
    frame = pd.concat([_station('USGS:08276500', '13020101', True, n_days=5 * 365, seed=1),
                       _station('USGS:09114500', '14020002', False, n_days=5 * 365, seed=2)],
                      ignore_index=True)
    base = str(tmp_path)
    data = normalize_data.normalize_data(frame, artifacts_dir=base)
    os.makedirs(os.path.join(base, 'openFlowML'))
    data.to_csv(os.path.join(base, 'openFlowML', 'normalized_data.csv'), index=False)

    assert crossval.main(['--base-path', base, '--folds', '3']) == 0

    with open(os.path.join(base, 'cv_report.json')) as f:
        report = json.load(f)
    assert report['n_folds'] == 3 and len(report['folds']) == 3
    years = [y for fold in report['folds'] for y in fold['test_water_years']]
    assert years == sorted(set(years))
    assert report['windows'] == sum(fold['windows'] for fold in report['folds'])
    assert set(report['pooled']['stations']) == {'USGS:08276500', 'USGS:09114500'}
    assert 'band' in report['pooled']['all']            # the band is cross-validated too
    assert 'Cross-validation: 3 leave-water-years-out folds' in open(os.path.join(base, 'cv_report.md')).read()
