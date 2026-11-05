"""
live_forcing.py: decoder inputs rebuilt from archived forecasts, in the units
and scaling training uses, and the live-forcing score. No network: the
archive table is synthetic.
"""

import numpy as np
import pandas as pd
import pytest

import evaluate
import live_forcing
import windowing
from test_windowing import _make_station_frame

SITE = 'USGS:08276500'      # in basin_zones.json with local / mid / far zones
D = windowing.DECODER_FEATURES
IDENT = {'mean': 0.0, 'scale': 1.0, 'transform': 'identity'}
LOG = {'mean': 0.0, 'scale': 1.0, 'transform': 'log1p'}
SCALERS = {'TMIN': IDENT, 'TMAX': IDENT, 'precipitation': LOG, 'Max Flow': IDENT, 'Min Flow': IDENT,
           'upstream_flow': IDENT, **{c: LOG for c in windowing.BASIN_PRECIP_FEATURES}}


def _samples():
    frame = _make_station_frame(SITE, 1, 1, n_days=90)
    samples = windowing.build_windows(frame)
    return samples[:3]


def _table(samples, drop_lead=None):
    rows = []
    for s in samples:
        issue = s.anchor_date + pd.Timedelta(days=windowing.ENCODER_DAYS - 1)
        for k in range(1, 8):
            if k == drop_lead and s is samples[0]:
                continue
            date = issue + pd.Timedelta(days=k)
            rows.append({'site': SITE, 'kind': 'station', 'date': date, 'lead': k,
                         'tmin_c': 1.5, 'tmax_c': 10.0, 'tmean_c': 5.0, 'precip_mm': 2.0})
            for zone in ('local', 'mid', 'far'):
                rows.append({'site': SITE, 'kind': zone, 'date': date, 'lead': k,
                             'tmin_c': 5.0, 'tmax_c': 15.0, 'tmean_c': 10.0, 'precip_mm': 4.0})
    return pd.DataFrame(rows).drop_duplicates(['site', 'kind', 'date', 'lead'])


def test_archived_forecasts_replace_lead_days_1_to_7_in_training_units():
    samples = _samples()
    x, _ = windowing.stack(samples)
    live, ok = live_forcing.archived_inputs(samples, x, SCALERS, _table(samples))
    assert ok.all()
    dec = live['decoder_input']
    assert np.allclose(dec[:, :7, D.index('TMIN')], 15.0)          # tenths of deg C, like GHCND
    assert np.allclose(dec[:, :7, D.index('TMAX')], 100.0)
    assert np.allclose(dec[:, :7, D.index('precipitation')], np.log1p(2.0))
    known = x['decoder_input'][:, :7, D.index('basin_precip_known')] > 0.5
    rain = dec[:, :7, D.index('rain_local')]
    assert np.allclose(rain[known], np.log1p(4.0))                  # 10 deg C -> all rain
    assert np.allclose(dec[:, :7, D.index('snow_local')][known], 0.0)
    np.testing.assert_array_equal(dec[:, 7:], x['decoder_input'][:, 7:])   # days 8-14 untouched
    untouched = [i for i, c in enumerate(D) if c not in live_forcing.POINT_COLUMNS + live_forcing.ZONE_COLUMNS]
    np.testing.assert_array_equal(dec[..., untouched], x['decoder_input'][..., untouched])


def test_windows_with_an_incomplete_archive_are_flagged_and_left_alone():
    samples = _samples()
    x, _ = windowing.stack(samples)
    live, ok = live_forcing.archived_inputs(samples, x, SCALERS, _table(samples, drop_lead=4))
    assert ok.tolist() == [False, True, True]
    np.testing.assert_array_equal(live['decoder_input'][0], x['decoder_input'][0])


class _Persistence:
    """Stub model: predicts the persistence anchor whatever the weather says."""

    def predict(self, x, verbose=0, batch_size=None):
        return np.repeat(x['persistence_input'][:, None, :], windowing.DECODER_DAYS, axis=1)


def test_score_and_report_section():
    samples = _samples()
    result = live_forcing.score(_Persistence(), samples, SCALERS, _table(samples))
    assert result['windows'] == 3 and result['leads'] == list(range(1, 8))
    e = result['error_pct']
    assert e['perfect'] == e['live'] == e['persistence']             # weather-blind model
    assert set(result['stations']) == {SITE}
    md = live_forcing.to_markdown(result)
    assert '**Live forcing**' in md and '**Real forecasts**' in md

    report = {'leads': list(range(1, 15)), 'all': {'windows': 0}, 'stations': {},
              'gate': {'passed': True, 'checks': []}, 'live_forcing': result}
    assert '**Live forcing**' in evaluate.to_markdown(report)


def test_no_complete_archive_gives_none():
    samples = _samples()
    assert live_forcing.score(_Persistence(), samples, SCALERS, _table(samples).iloc[:0]) is None
