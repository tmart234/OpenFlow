"""Scoring archived forecasts (score_archive.py) on a synthetic archive; no network."""
import json
import os

import numpy as np
import pandas as pd

import score_archive as sa

ISSUED = pd.Timestamp('2026-11-02T03:00Z')     # Nov 1, 8 PM MST


def _write_day(root, routing_shift_h=0, nws=True):
    d = os.path.join(root, 'archive', '2026-11-02')
    os.makedirs(d, exist_ok=True)
    t = pd.date_range(ISSUED - pd.Timedelta(days=1), ISSUED + pd.Timedelta(days=6), freq='1h')
    crest = ISSUED + pd.Timedelta(hours=60)
    shape = lambda shift: 500 + 3000 * np.exp(-0.5 * (((t - crest) / pd.Timedelta(hours=1) - shift) / 10.0) ** 2)
    pd.DataFrame({'time_utc': t.strftime('%Y-%m-%dT%H:%MZ'),
                  'forecast_cfs': shape(routing_shift_h),
                  'nws_official_cfs': shape(0) * 1.1 if nws else np.nan}).to_csv(
        os.path.join(d, 'taos-box_forecast.csv'), index=False)
    ml = [{'date': str(day.date()), 'min_cfs': 600, 'max_cfs': 800}
          for day in pd.date_range('2026-11-02', periods=6)]
    with open(os.path.join(d, 'taos-box_forecast.json'), 'w') as f:
        json.dump({'site_id': 'USGS:08276500', 'timezone': 'America/Denver',
                   'latest_observation': {'time_utc': ISSUED.isoformat()}, 'ml': {'daily': ml}}, f)
    obs_t = pd.date_range(ISSUED - pd.Timedelta(days=5), ISSUED + pd.Timedelta(days=6), freq='1h')
    obs = pd.Series(500 + 3000 * np.exp(-0.5 * (((obs_t - crest) / pd.Timedelta(hours=1)) / 10.0) ** 2), index=obs_t)
    return obs


def test_scores_daily_errors_and_crest_timing_per_source(tmp_path):
    obs = _write_day(str(tmp_path), routing_shift_h=-8)          # routing 8 h early, NWS on time but 10% high
    fcs = sa.load_archive(str(tmp_path))
    assert len(fcs) == 1 and fcs[0]['site_id'] == 'USGS:08276500'
    rows = pd.DataFrame(sa.score_one(fcs[0], obs))
    crest = rows[rows.kind == 'crest'].set_index('source')
    assert crest.loc['routing', 'timing_error_h'] == -8 and crest.loc['nws', 'timing_error_h'] == 0
    assert abs(crest.loc['nws', 'crest_ratio'] - 1.1) < 0.01 and np.isnan(crest.loc['ml', 'timing_error_h'])
    daily = rows[rows.kind == 'daily']
    assert set(daily.source) == {'routing', 'nws', 'ml'} and daily.lead_day.max() <= 5
    md = sa.summarize(rows)
    assert 'Lower Taos Box' in md and '| median timing error | 8 h | 0 h | — |' in md


def test_summarize_handles_an_empty_archive():
    assert 'No archived forecasts' in sa.summarize(pd.DataFrame())
