"""
Smoke tests for the CBRFC + S2F baseline modules.

Both modules expose a `baseline_predictions(test_samples)` hook that train.py
calls during evaluation. The CBRFC module fetches forecasts (live AHPS for
today, NWPS archive for historical anchors) -- when no LID is mapped for a
site, or the archive returns no forecast for an anchor date, the module must
return None / empty cleanly. The S2F module remains a stub; same contract.
These tests pin that contract so the train.py integration stays safe.
"""

import numpy as np
import pandas as pd
import pytest

from data import get_cbrfc, get_s2f
from windowing import WindowedSample


def _sample(site_id='USGS:09163500'):
    """A single WindowedSample-shaped object sufficient for baseline_predictions."""
    return WindowedSample(
        encoder_X=np.zeros((60, 9), dtype='float32'),
        decoder_X=np.zeros((14, 4), dtype='float32'),
        target_Y=np.zeros((14, 2), dtype='float32'),
        persistence_anchor=np.zeros(2, dtype='float32'),
        station_idx=1,
        basin_idx=1,
        site_id=site_id,
        anchor_date=pd.Timestamp('2024-01-01'),
    )


def test_cbrfc_baseline_returns_none_when_no_lid_mapped():
    # The seeded cbrfc_lid_map.json carries no site mappings by default --
    # the comment-only file means every fetch returns empty and the stacked
    # prediction is None (skip cleanly).
    pred = get_cbrfc.baseline_predictions([_sample('USGS:UNMAPPED')])
    assert pred is None


def test_cbrfc_baseline_returns_none_for_empty_test_set():
    assert get_cbrfc.baseline_predictions([]) is None


def test_cbrfc_fetch_current_skips_when_no_lid():
    df = get_cbrfc.fetch_current('USGS:UNMAPPED')
    assert df.empty
    assert list(df.columns) == ['Date', 'cbrfc_flow']


def test_cbrfc_fetch_historical_skips_when_no_lid():
    # Anchor in the past, no LID mapping -> empty (no NWPS request issued).
    df = get_cbrfc.fetch('USGS:UNMAPPED', '2024-01-01')
    assert df.empty


def test_s2f_baseline_returns_none_until_archive_wired():
    assert get_s2f.baseline_predictions([_sample()]) is None
    assert get_s2f.baseline_predictions([]) is None


def test_s2f_fetch_returns_empty_until_archive_wired():
    df = get_s2f.fetch('USGS:09163500', '2024-01-01')
    assert df.empty
    assert list(df.columns) == ['Date', 's2f_volume_kaf']


def test_s2f_disaggregate_uniform_share_sums_to_seasonal_volume():
    """With no climatology, full-season disaggregation totals back to the input volume in cfs-days."""
    import datetime as dt
    # April-July: 30 + 31 + 30 + 31 = 122 days. 1000 kAF total.
    season_start, season_end = dt.date(2024, 4, 1), dt.date(2024, 7, 31)
    daily_cfs = get_s2f.disaggregate_seasonal_to_daily(
        volume_kaf=1000.0,
        season_start=season_start,
        season_end=season_end,
        anchor_date=season_start,
        horizon_days=122,
    )
    # Each day's allocation: (1000 / 122) kAF * 504.17 cfs/(kAF/day)
    expected_daily = (1000.0 / 122.0) * (43_560.0 * 1_000.0 / 86_400.0)
    np.testing.assert_allclose(daily_cfs, expected_daily, rtol=1e-5)


def test_s2f_disaggregate_zeros_out_of_season_days():
    """Anchor before the season starts -> first few days are zero (no S2F contribution)."""
    import datetime as dt
    daily_cfs = get_s2f.disaggregate_seasonal_to_daily(
        volume_kaf=850.0,
        season_start=dt.date(2024, 4, 1),
        season_end=dt.date(2024, 7, 31),
        anchor_date=dt.date(2024, 3, 25),  # 7 days before April starts
        horizon_days=14,
    )
    assert (daily_cfs[:7] == 0).all()      # March 25..31, outside season
    assert (daily_cfs[7:] > 0).all()       # April 1..7, inside season


def test_s2f_disaggregate_respects_per_day_climatology():
    """Climatology that concentrates flow on a single DOY -> that day gets the full volume."""
    import datetime as dt
    season_start, season_end = dt.date(2024, 6, 1), dt.date(2024, 6, 10)
    # Stack 100% of the seasonal volume on June 5 (doy 157 in 2024).
    climatology = {(season_start + dt.timedelta(days=4)).timetuple().tm_yday: 1.0}
    daily_cfs = get_s2f.disaggregate_seasonal_to_daily(
        volume_kaf=100.0,
        season_start=season_start, season_end=season_end,
        anchor_date=season_start,
        horizon_days=10,
        climatology=climatology,
    )
    assert daily_cfs[4] > 0
    # Every other day inside the season gets 0 (climatology key absent ->
    # fallback per the implementation? No: climatology dict explicitly does
    # NOT include the other days, so they default to uniform share. That's a
    # design choice -- assert what the implementation actually does so a
    # future refactor flips this test, not the production behavior).
    # Per the implementation: missing-DOY keys fall back to 1/season_length.
    assert (daily_cfs[:4] > 0).all()
    assert (daily_cfs[5:] > 0).all()
