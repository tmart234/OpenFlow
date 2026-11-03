"""Scoring helpers of the routing backtest (backtest_routing.py); no network."""
import numpy as np
import pandas as pd

import backtest_routing as bt


def _rows():
    issued = pd.to_datetime(['2026-07-01T18:00Z', '2026-07-03T18:00Z', '2026-12-01T18:00Z'])
    rows = []
    for t, obs, fc in zip(issued, ([100, 100], [100, 200], [400, 400]), ([110, 100], [100, 180], [400, 380])):
        for lead in (1, 2):
            rows.append({'issued': t, 'lead': lead, 'obs': obs[lead - 1], 'persistence': 100.0 if t.month == 7 else 400.0,
                         'anchored': fc[lead - 1], 'direct': 2.0 * fc[lead - 1]})
    return pd.DataFrame(rows)


def test_score_splits_irrigation_season_and_reports_median_errors():
    s = bt.score(_rows(), ['direct', 'anchored'])
    assert set(s.index.get_level_values('season')) == {'Apr-Oct', 'Nov-Mar'}
    assert s.loc[('Apr-Oct', 1), 'err_anchored'] == 5.0          # median of 10% and 0%
    assert s.loc[('Apr-Oct', 1), 'err_direct'] == 110.0          # median of 120% and 100%
    assert s.loc[('Nov-Mar', 2), 'bias_anchored'] == -5.0


def test_rise_peaks_scores_only_rising_windows():
    out = bt.rise_peaks(_rows(), ['direct', 'anchored'])
    assert out['windows'] == 1                                    # only Jul 3 doubles
    assert out['anchored'] == 0.9 and out['direct'] == 1.8 and out['persistence'] == 0.5


def test_daily_to_hourly_puts_daily_means_at_local_noon():
    d = pd.Series([100.0, 200.0], index=pd.to_datetime(['2026-07-01', '2026-07-02']))
    h = bt.daily_to_hourly(d, 'America/Denver')
    assert h.index[0] == pd.Timestamp('2026-07-01T18:00Z') and h.iloc[0] == 100.0
    assert h.loc['2026-07-02T06:00Z'] == 150.0 and np.isfinite(h).all()
