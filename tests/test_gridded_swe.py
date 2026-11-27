"""Offline tests for Phase 3 elevation-band gridded SWE."""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

import combine_data
import normalize_data
from data import get_gridded_swe


def test_select_elevation_band_points_picks_low_mid_high_tertiles():
    pts = [{'lat': 38.0 + 0.1 * i, 'lon': -106.0} for i in range(9)]
    elevs = [1800, 2100, 2300, 2700, 2900, 3100, 3400, 3700, 4000]
    selected = get_gridded_swe.select_elevation_band_points(pts, elevs)
    assert set(selected) == {'low', 'mid', 'high'}
    # Median of each tertile: index 1 (38.1), index 4 (38.4), index 7 (38.7).
    assert selected['low'][0] == pytest.approx(38.1)
    assert selected['mid'][0] == pytest.approx(38.4)
    assert selected['high'][0] == pytest.approx(38.7)


def test_assemble_band_swe_computes_melt_rate():
    dates = pd.date_range('2025-05-01', periods=5, freq='D')
    bands = {
        'low': pd.Series([10.0, 5.0, 0.0, 0.0, 0.0], index=dates),
        'mid': pd.Series([120.0, 110.0, 95.0, 80.0, 60.0], index=dates),
        'high': pd.Series([400.0, 405.0, 400.0, 385.0, 360.0], index=dates),
    }
    df = get_gridded_swe.assemble_band_swe(bands)
    assert list(df.columns) == ['Date', *get_gridded_swe.SWE_COLUMNS]
    # Negative delta_swe = active snowmelt.
    assert df.loc[2, 'delta_swe_mid'] == pytest.approx(-15.0)
    assert df.loc[4, 'delta_swe_high'] == pytest.approx(-25.0)


def test_gridded_swe_merges_and_normalizes_with_linear_melt_rate():
    dates = pd.date_range('2025-05-01', periods=15, freq='D')
    noaa = pd.DataFrame({'Date': dates, 'TMIN': 20.0, 'TMAX': 150.0, 'precipitation': 0.0})
    flow = pd.DataFrame({'Date': dates, 'Min Flow': 400.0, 'Max Flow': 500.0})
    gswe = pd.DataFrame({
        'Date': dates,
        'swe_low': np.linspace(20, 0, 15),
        'swe_mid': np.linspace(200, 100, 15),
        'swe_high': np.linspace(500, 420, 15),
        'delta_swe_mid': np.full(15, -7.1),
        'delta_swe_high': np.full(15, -5.7),
    })
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST', datetime(2025, 5, 1), datetime(2025, 5, 15),
        gridded_swe_data=gswe)
    assert (merged['gridded_swe_observed'] == 1).all()
    norm = normalize_data.normalize_data(merged)
    assert norm is not None
    # delta_swe can be negative -> must never use log1p!
    assert 'delta_swe_mid' not in normalize_data.LOG_TRANSFORM_COLUMNS
    assert 'delta_swe_high' not in normalize_data.LOG_TRANSFORM_COLUMNS
    assert np.isfinite(norm[get_gridded_swe.SWE_COLUMNS].to_numpy()).all()
