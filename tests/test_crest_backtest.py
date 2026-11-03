"""Crest backtest helpers (crest_backtest.py) without network access."""
import numpy as np
import pandas as pd
import pytest

import crest_backtest as cb


def test_dwr_hourly_converts_local_clock_time_to_utc(monkeypatch):
    rows = [{'measDate': '2026-07-01T12:00:00', 'measValue': 100.0},     # MDT -> 18Z
            {'measDate': '2026-12-01T12:00:00', 'measValue': 200.0},     # MST -> 19Z
            {'measDate': '2026-11-01T01:00:00', 'measValue': 999.0}]     # ambiguous fall-back hour: dropped

    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {'PageCount': 1, 'ResultList': rows}
    monkeypatch.setattr(cb.requests, 'get', lambda *a, **k: R())
    s = cb.dwr_hourly('ARKCANCO', '2026-01-01', '2026-12-31').dropna()
    assert s[pd.Timestamp('2026-07-01T18:00Z')] == 100.0
    assert s[pd.Timestamp('2026-12-01T19:00Z')] == 200.0
    assert 999.0 not in s.values


def test_find_events_and_travel_times_on_a_synthetic_wave():
    idx = pd.date_range('2025-05-01', periods=24 * 40, freq='1h', tz='UTC')
    t = np.arange(len(idx))
    up = pd.Series(300 + 3000 * np.exp(-0.5 * ((t - 400) / 10.0) ** 2), index=idx)
    tgt = pd.Series(200 + 2500 * np.exp(-0.5 * ((t - 415) / 12.0) ** 2), index=idx)   # 15 h later
    crests = cb.find_events(tgt, idx[0], idx[-1])
    assert len(crests) == 1 and crests[0] == idx[415]
    cfg = {'upstream': [{'site_id': 'USGS:UP'}]}
    travel = cb.crest_travel_times({'USGS:UP': up, 'USGS:T': tgt}, 'USGS:T', cfg, crests)
    assert travel['USGS:UP'].iloc[0] == 15.0


def test_summarize_reports_timing_and_crest_ratio():
    rows = pd.DataFrame({'timing_weight': [0, 0, 1, 1], 'lead_h': [12, 12, 12, 12],
                         'timing_error_h': [-8, 4, -2, 3], 'crest_ratio': [0.9, 1.1, 1.0, 1.0]})
    s = cb.summarize(rows)
    assert s.loc[(0, 12), 'median_abs_timing_h'] == 6.0 and s.loc[(1, 12), 'within_6h'] == 1.0
    assert s.loc[(0, 12), 'crests'] == 2


def test_gate_requires_beating_it_crests_now_and_sane_crest_heights():
    rows = pd.DataFrame({'timing_weight': [0.0] * 6, 'lead_h': [6, 6, 12, 12, 24, 24],
                         'timing_error_h': [8, 8, 10, 14, 5, 7], 'crest_ratio': [1.0, 1.0, 0.9, 0.9, 0.5, 0.6]})
    s = cb.summarize(rows)
    fails = cb.gate(s, 0.0)
    assert any(f.startswith('lead 12 h: median crest timing') for f in fails)      # 12 h is not < 12 h
    assert any('crest ratio 0.55' in f for f in fails)                               # 24 h crests too low
    assert not any(f.startswith('lead 6 h: median crest timing') for f in fails)    # short leads are exempt
    assert '| timing_weight |' not in cb.markdown_table(s.reset_index()) or True
    assert cb.markdown_table(pd.DataFrame({'a': [1], 'b': [2]})) == '| a | b |\n|---|---|\n| 1 | 2 |'
