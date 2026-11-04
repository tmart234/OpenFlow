"""
Forecast verification (evaluate.py): unscaling to cfs, the persistence and
upstream-change references, event scoring and the quality gate, on synthetic
windows with an identity scaler so scaled values are cfs.
"""

import json

import numpy as np
import pandas as pd
import pytest

import evaluate
import windowing

E, D = windowing.ENCODER_FEATURES, windowing.DECODER_FEATURES
SCALERS = {c: {'mean': 0.0, 'scale': 1.0, 'transform': 'identity'}
           for c in ('Min Flow', 'Max Flow', 'upstream_flow')}


def _sample(site, history, future, up_hist=None, up_future=None, known_days=0, day=0):
    enc = np.zeros((windowing.ENCODER_DAYS, len(E)))
    enc[:, E.index('Max Flow')] = history
    enc[:, E.index('Min Flow')] = history
    dec = np.zeros((windowing.DECODER_DAYS, len(D)))
    if up_hist is not None:
        enc[:, E.index('upstream_flow')] = up_hist
        enc[:, E.index('upstream_observed')] = 1.0
        dec[:, D.index('upstream_flow')] = up_future
        dec[:known_days, D.index('upstream_known')] = 1.0
    target = np.column_stack([future, future])
    return windowing.WindowedSample(
        encoder_X=enc, decoder_X=dec, target_Y=target, persistence_anchor=target[0] * 0 + history[-1],
        station_idx=1, basin_idx=1, site_id=site,
        anchor_date=pd.Timestamp('2026-01-01') + pd.Timedelta(days=day))


def _windows():
    flat_hist = np.full(windowing.ENCODER_DAYS, 100.0)
    rise = np.array([110, 150, 200, 260, 220, 180, 150, 130, 120, 110, 105, 100, 100, 100], float)
    calm = np.full(windowing.DECODER_DAYS, 100.0)
    up_hist = np.full(windowing.ENCODER_DAYS, 50.0)
    up_future = np.array([80, 120] + [120] * 12, float)
    return [_sample('A', flat_hist, rise, day=0),
            _sample('A', flat_hist, calm, day=1),
            _sample('B', flat_hist, rise, up_hist, up_future, known_days=2, day=0)]


def _pred(samples, kind):
    if kind == 'perfect':
        return np.stack([s.target_Y for s in samples])
    return np.stack([np.tile(s.encoder_X[-1, [E.index('Min Flow'), E.index('Max Flow')]],
                             (windowing.DECODER_DAYS, 1)) for s in samples])


def test_forecasts_unscale_and_build_references():
    samples = _windows()
    fc = evaluate.forecasts(samples, _pred(samples, 'persistence'), SCALERS)
    assert fc['obs'].shape == (3, 14) and (fc['q0'] == 100).all()
    np.testing.assert_allclose(fc['model'], fc['persistence'])
    # Upstream reference: +30 then +70 on the 2 known days, then held.
    np.testing.assert_allclose(fc['upstream'][2, :4], [130, 170, 170, 170])
    np.testing.assert_allclose(fc['upstream'][0], fc['persistence'][0])   # no upstream gauges
    assert list(fc['site']) == ['A', 'A', 'B']
    assert fc['issued'][0] == pd.Timestamp('2026-03-01')                  # last encoder day


def test_perfect_model_passes_and_persistence_fails_the_gate():
    samples = _windows()
    good = evaluate.build_report(evaluate.forecasts(samples, _pred(samples, 'perfect'), SCALERS))
    assert good['all']['error_pct']['model'][0] == 0
    assert good['all']['rises']['windows'] == 2
    assert good['all']['rises']['model'] == {'peak_ratio': 1.0, 'timing_days': 0.0}
    assert good['gate']['passed']

    flat = evaluate.build_report(evaluate.forecasts(samples, _pred(samples, 'persistence'), SCALERS))
    failed = {c['check'] for c in flat['gate']['checks'] if not c['passed']}
    assert 'rising-flow peaks not flattened' in failed
    assert 'beats persistence from day 3' in failed
    assert flat['all']['rises']['persistence']['peak_ratio'] == pytest.approx(100 / 260, rel=1e-3)
    assert set(flat['stations']) == {'A', 'B'}


def test_report_files_and_gate_cli(tmp_path):
    samples = _windows()
    rep = evaluate.build_report(evaluate.forecasts(samples, _pred(samples, 'perfect'), SCALERS), 'm-1')
    paths = evaluate.write_report(rep, str(tmp_path), names={'A': 'River A'})
    md = open(paths[1]).read()
    assert 'River A' in md and 'Quality gate: PASSED' in md
    assert json.load(open(paths[0]))['model_version'] == 'm-1'
    assert evaluate.main(['--base-path', str(tmp_path), '--gate']) == 0
    rep['gate']['passed'] = False
    evaluate.write_report(rep, str(tmp_path))
    assert evaluate.main(['--base-path', str(tmp_path), '--gate']) == 1


def _band(samples, spread):
    """(N, H, T, 5) quantiles: the observed target +/- spread * [2, 1, 0, 1, 2]."""
    y = np.stack([s.target_Y for s in samples])
    offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    return y[..., None] + spread * offsets


def test_band_scores_and_gate_check():
    samples = _windows()
    q = (0.05, 0.25, 0.5, 0.75, 0.95)
    pred = _pred(samples, 'perfect')
    rep = evaluate.build_report(evaluate.forecasts(samples, pred, SCALERS, band=_band(samples, 5.0), quantiles=q))
    band = rep['all']['band']
    assert band['levels'] == [0.05, 0.95] and band['inner_levels'] == [0.25, 0.75]
    assert band['coverage_90'] == [1.0] * 14 and band['coverage_50'] == [1.0] * 14
    assert band['rise_peak_in_90'] == 1.0
    assert band['width_90_pct'][0] == pytest.approx(20 / 110 * 100, abs=0.1)   # (y+10 - (y-10)) / y
    assert 'band' in rep['stations']['A']
    assert any(c['check'] == 'forecast range holds observations' and c['passed'] for c in rep['gate']['checks'])
    assert 'Forecast range' in evaluate.to_markdown(rep)

    # A band shifted off the observations covers nothing and fails the gate.
    off = _band(samples, 1.0) + 50.0
    bad = evaluate.build_report(evaluate.forecasts(samples, pred, SCALERS, band=off, quantiles=q))
    assert bad['all']['band']['coverage_90'] == [0.0] * 14
    failed = {c['check'] for c in bad['gate']['checks'] if not c['passed']}
    assert failed == {'forecast range holds observations'}


def test_forecasts_accept_a_two_output_prediction_dict():
    samples = _windows()
    q = (0.05, 0.25, 0.5, 0.75, 0.95)
    pred = {'persistence_plus_delta': _pred(samples, 'perfect'), 'flow_quantiles': _band(samples, 5.0)}
    fc = evaluate.forecasts(samples, pred, SCALERS, quantiles=q)
    np.testing.assert_allclose(fc['model'], fc['obs'])
    assert fc['band'].shape == (3, 14, 5)
    # Point-only reports are unchanged: no band section, no band gate check.
    plain = evaluate.build_report(evaluate.forecasts(samples, pred['persistence_plus_delta'], SCALERS))
    assert 'band' not in plain['all']
    assert all(c['check'] != 'forecast range holds observations' for c in plain['gate']['checks'])
