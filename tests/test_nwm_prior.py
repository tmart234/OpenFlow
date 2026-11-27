"""Offline tests for Phase 3 NWS RFC + NOAA NWM v3 forecast prior."""

import numpy as np
import pandas as pd
import pytest

import windowing
from data import get_nwm_prior, nwps


def test_parse_nwm_reach_series_handles_cfs_and_cms():
    payload_cfs = {
        'mediumRangeBlend': {
            'series': {
                'units': 'ft3/s',
                'data': [
                    {'validTime': '2026-11-27T12:00:00Z', 'flow': 250.0},
                    {'validTime': '2026-11-27T18:00:00Z', 'flow': 350.0},
                    {'validTime': '2026-11-28T12:00:00Z', 'flow': 400.0},
                ],
            }
        }
    }
    s = get_nwm_prior.parse_nwm_reach_series(payload_cfs, tz='UTC')
    assert s.loc[pd.Timestamp('2026-11-27')] == pytest.approx(300.0)
    assert s.loc[pd.Timestamp('2026-11-28')] == pytest.approx(400.0)

    payload_cms = {
        'mediumRangeBlend': {
            'series': {
                'units': 'm3/s',
                'data': [{'validTime': '2026-11-27T12:00:00Z', 'flow': 10.0}],
            }
        }
    }
    s2 = get_nwm_prior.parse_nwm_reach_series(payload_cms, tz='UTC')
    assert s2.loc[pd.Timestamp('2026-11-27')] == pytest.approx(10.0 * get_nwm_prior.M3S_TO_CFS)


def test_fetch_live_prior_prefers_rfc_and_extends_with_ratio_aligned_nwm(monkeypatch):
    # RFC covers days 1-3 at 200 cfs; NWM reads 2x higher (400 cfs on overlap, 600 cfs on days 4-5).
    # Ratio alignment rescales NWM tail to 300 cfs on days 4-5!
    rfc_payload = {
        'forecast': {
            'secondaryUnits': 'kcfs',
            'data': [
                {'validTime': f'2026-11-0{d}T12:00:00Z', 'primary': 2.0, 'secondary': 0.2}
                for d in (1, 2, 3)
            ],
        }
    }
    nwm_series = pd.Series(
        [400.0, 400.0, 400.0, 600.0, 600.0],
        index=pd.date_range('2026-11-01', periods=5, name='Date'),
    )
    monkeypatch.setattr(nwps, 'get_stageflow', lambda lid: rfc_payload)
    monkeypatch.setattr(get_nwm_prior, 'resolve_reach_id', lambda *a, **k: 12345)
    monkeypatch.setattr(get_nwm_prior, 'fetch_nwm_live_daily', lambda rid, tz='UTC': nwm_series)

    df = get_nwm_prior.fetch_live_prior('USGS:08276500', nws_lid='TJBN5', tz='UTC').set_index('Date')
    assert df.loc['2026-11-01', 'prior_flow'] == pytest.approx(200.0)
    assert df.loc['2026-11-03', 'prior_flow'] == pytest.approx(200.0)
    assert df.loc['2026-11-04', 'prior_flow'] == pytest.approx(300.0)
    assert df.loc['2026-11-05', 'prior_flow'] == pytest.approx(300.0)


def test_prior_decoder_window_holds_past_known_days_and_computes_change():
    future = np.array([1.2, 1.5, 1.8, 2.0, 2.2])
    obs = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    vals, known, chg = windowing.prior_decoder_window(future, obs, last_value=1.0, last_observed=1.0, known_days=3)
    np.testing.assert_allclose(vals, [1.2, 1.5, 1.8, 1.8, 1.8])
    np.testing.assert_array_equal(known, [1.0, 1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(chg, [0.2, 0.5, 0.8, 0.8, 0.8])

    # Completely unobserved -> all zeros cleanly.
    v0, k0, c0 = windowing.prior_decoder_window(future, np.zeros(5), last_value=0.0, last_observed=0.0)
    assert (v0 == 0).all() and (k0 == 0).all() and (c0 == 0).all()
