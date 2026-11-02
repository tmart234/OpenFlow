import numpy as np
import pandas as pd
import pytest

from data import nwps


def _payload(obs=None, fc=None, issued='2026-11-02T14:15:00Z', units='kcfs'):
    def block(points, issued_time):
        return {'issuedTime': issued_time, 'primaryUnits': 'ft', 'secondaryUnits': units,
                'data': [{'validTime': t, 'primary': st, 'secondary': q} for t, st, q in points]}
    return {'observed': block(obs or [], '2026-11-02T17:30:00Z'),
            'forecast': block(fc or [], issued)}


def test_parse_series_converts_kcfs_to_cfs_and_missing_to_nan():
    p = _payload(obs=[('2026-11-02T16:00:00Z', 4.9, 1.26),
                      ('2026-11-02T17:00:00Z', -999, -999),
                      ('2026-11-02T15:00:00Z', 4.8, 1.2)])
    df = nwps.parse_series(p, 'observed')
    assert list(df['flow_cfs'].iloc[:2]) == [1200.0, 1260.0]   # sorted by time
    assert np.isnan(df['flow_cfs'].iloc[2])
    assert np.isnan(df['stage_ft'].iloc[2])
    assert str(df['time'].dt.tz) == 'UTC'


def test_parse_series_respects_cfs_units():
    p = _payload(obs=[('2026-11-02T16:00:00Z', 2.0, 350.0)], units='cfs')
    assert nwps.parse_series(p, 'observed')['flow_cfs'].iloc[0] == 350.0


def test_parse_series_empty_forecast_returns_empty_frame():
    df = nwps.parse_series(_payload(obs=[('2026-11-02T16:00:00Z', 1, 1)]), 'forecast')
    assert df.empty
    assert list(df.columns) == ['time', 'flow_cfs', 'stage_ft']


def test_parse_series_rejects_unknown_kind():
    with pytest.raises(ValueError):
        nwps.parse_series({}, 'nowcast')


def test_forecast_issued_time_none_when_not_current():
    assert nwps.forecast_issued_time(_payload()) is None
    p = _payload(fc=[('2026-11-03T00:00:00Z', 5, 3.0)], issued='0001-01-01T00:00:00Z')
    assert nwps.forecast_issued_time(p) is None
    p = _payload(fc=[('2026-11-03T00:00:00Z', 5, 3.0)])
    assert nwps.forecast_issued_time(p) == pd.Timestamp('2026-11-02T14:15:00Z')


def test_flood_thresholds_skip_undefined_categories():
    gauge = {'flood': {'categories': {
        'action': {'stage': 9, 'flow': 7750}, 'minor': {'stage': 10, 'flow': 9800},
        'moderate': {'stage': -999, 'flow': -999}, 'major': {'stage': 15, 'flow': 25000}}}}
    assert nwps.flood_thresholds_cfs(gauge) == {'action': 7750.0, 'minor': 9800.0, 'major': 25000.0}
    assert nwps.flood_thresholds_cfs(None) == {}


def test_hourly_resamples_and_interpolates_interior_gaps():
    p = _payload(obs=[('2026-11-02T00:00:00Z', 1, 1.0), ('2026-11-02T00:30:00Z', 1, 1.2),
                      ('2026-11-02T03:00:00Z', 1, 1.6)])
    h = nwps.hourly(nwps.parse_series(p, 'observed'))
    assert len(h) == 4
    assert h.iloc[0] == pytest.approx(1100.0)       # mean within the hour
    assert h.iloc[3] == pytest.approx(1600.0)
    assert h.iloc[1] == pytest.approx(1100 + (1600 - 1100) / 3)


def test_hourly_of_empty_frame_has_datetime_index():
    h = nwps.hourly(nwps.parse_series(_payload(), 'forecast'))
    assert h.empty
    assert isinstance(h.index, pd.DatetimeIndex)
