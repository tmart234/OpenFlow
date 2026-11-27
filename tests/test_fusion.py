"""Offline tests for Phase 3 lead-weighted routing + ML fusion and archive scoring."""

import json
import os

import numpy as np
import pandas as pd
import pytest

import flow_forecast
import fusion
import ml_inference
import routing
import score_archive as sa
from test_flow_forecast import GRID, INOW, NOW, _scenario, fake_nwps  # noqa: F401


def test_choose_tau_hours_adapts_to_mode_lag_and_rfc():
    assert fusion.choose_tau_hours('nws') == fusion.DEFAULT_TAU_NWS_H
    assert fusion.choose_tau_hours('recession') == fusion.DEFAULT_TAU_RECESSION_H
    t_no_rfc = fusion.choose_tau_hours('routing', min_lag_hours=24, has_rfc=False)
    t_rfc = fusion.choose_tau_hours('routing', min_lag_hours=24, has_rfc=True)
    assert t_rfc > t_no_rfc >= 36.0


def test_blend_forecasts_anchors_at_now_transitions_to_ml_and_widens_on_disagreement():
    grid = pd.date_range('2026-06-01T00:00Z', periods=10 * 24, freq='1h')
    inow = 24
    # Routing stays flat at 500 cfs; ML forecasts a steady rise to 1500 cfs.
    r_c = np.full(len(grid), 500.0)
    r_lo = np.full(len(grid), 450.0)
    r_hi = np.full(len(grid), 550.0)
    ef = routing.EnsembleForecast(central=r_c, low=r_lo, high=r_hi, members=r_c[None])

    dates = pd.date_range('2026-06-02', periods=9, name='Date')
    ml_agree = pd.DataFrame({
        'Date': dates, 'min_cfs': 480.0, 'max_cfs': 520.0,
        'min_q05_cfs': 430.0, 'max_q05_cfs': 470.0,
        'min_q95_cfs': 530.0, 'max_q95_cfs': 570.0,
    })
    ml_disagree = pd.DataFrame({
        'Date': dates, 'min_cfs': 1400.0, 'max_cfs': 1600.0,
        'min_q05_cfs': 1250.0, 'max_q05_cfs': 1450.0,
        'min_q95_cfs': 1550.0, 'max_q95_cfs': 1750.0,
    })

    f_agree = fusion.blend_forecasts(grid, inow, ef, ml_agree, 'UTC', tau_hours=24.0)
    f_dis = fusion.blend_forecasts(grid, inow, ef, ml_disagree, 'UTC', tau_hours=24.0)

    # At h = 0 (now), fused equals current observation (500 cfs).
    assert f_dis['central'][inow] == pytest.approx(500.0, rel=1e-3)
    # By end of forecast (~192h >> tau=24h), fused approaches ML (~1500 cfs).
    assert f_dis['central'][-1] > 1400.0
    # At mid-lead (~24h, where w ~ 0.37), disagreement widens the fused band.
    mid = inow + 24
    width_agree = np.log1p(f_agree['high'][mid]) - np.log1p(f_agree['low'][mid])
    width_dis = np.log1p(f_dis['high'][mid]) - np.log1p(f_dis['low'][mid])
    assert width_dis > width_agree * 1.5
    assert (f_dis['low'][inow:] <= f_dis['central'][inow:] + 1e-6).all()
    assert (f_dis['central'][inow:] <= f_dis['high'][inow:] + 1e-6).all()


def test_flow_forecast_writes_fused_outputs_and_score_archive_scores_them(tmp_path, fake_nwps, monkeypatch):
    payloads, truth = _scenario()
    fake_nwps(payloads)

    dates = pd.date_range(NOW.tz_convert('America/Denver').normalize().tz_localize(None), periods=8, name='Date')
    fake_ml = {
        'daily': pd.DataFrame({
            'Date': dates,
            'min_cfs': np.full(8, 700.0), 'max_cfs': np.full(8, 900.0),
            'min_q05_cfs': np.full(8, 550.0), 'max_q05_cfs': np.full(8, 750.0),
            'min_q95_cfs': np.full(8, 850.0), 'max_q95_cfs': np.full(8, 1050.0),
        }),
        'info': {'anchor': str(dates[0].date()), 'trained_station': True},
        'version': 'test-fusion',
        'verified': None,
    }
    monkeypatch.setattr(ml_inference, 'forecast', lambda site_id, model_dir, art=None: fake_ml)

    out_dir = tmp_path / 'archive' / '2026-11-02'
    summary = flow_forecast.run('taos-box', days=6, out_dir=str(out_dir),
                                make_chart=False, model_dir='dummy_model')
    assert summary['fused'] is not None
    assert summary['fused']['tau_hours'] > 0
    assert len(summary['fused']['daily']) == 6

    csv = pd.read_csv(summary['files']['csv'])
    for col in ('fused_cfs', 'fused_low_cfs', 'fused_high_cfs'):
        assert col in csv.columns
        assert csv[col].notna().any()

    txt = flow_forecast.format_summary(summary)
    assert 'Fused (routing + ML' in txt

    # Score the archived directory with score_archive.py and verify 'fused' is scored!
    fcs = sa.load_archive(str(tmp_path))
    assert len(fcs) == 1 and 'fused' in fcs[0]['hourly'].columns
    obs = pd.Series(truth, index=GRID)
    rows = pd.DataFrame(sa.score_one(fcs[0], obs))
    assert 'fused' in set(rows[rows.kind == 'daily']['source'])
    md = sa.summarize(rows)
    assert '| Lead day | routing | nws | ml | fused | forecasts |' in md
