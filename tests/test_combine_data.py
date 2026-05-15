import numpy as np
import pandas as pd
from datetime import datetime

import combine_data


def _make_frames():
    """20 days of flow with a 3-day interior gap; 20 days of complete temp."""
    dates = pd.date_range('2022-01-01', '2022-01-20', freq='D')
    flow = pd.DataFrame({
        'Date': dates,
        'Min Flow': np.arange(20, dtype=float) + 100,
        'Max Flow': np.arange(20, dtype=float) + 200,
    })
    # Punch a 3-day interior gap (within MAX_GAP_DAYS).
    flow.loc[5:7, ['Min Flow', 'Max Flow']] = np.nan
    noaa = pd.DataFrame({
        'Date': dates,
        'TMIN': np.arange(20, dtype=float),
        'TMAX': np.arange(20, dtype=float) + 10,
    })
    return noaa, flow


def test_merge_produces_regular_daily_index():
    noaa, flow = _make_frames()
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST', datetime(2022, 1, 1), datetime(2022, 1, 20))
    dates = pd.to_datetime(merged['Date'])
    # Every consecutive row is exactly one day apart.
    assert (dates.diff().dropna() == pd.Timedelta(days=1)).all()
    assert 'site_id' in merged.columns
    assert (merged['site_id'] == 'USGS:TEST').all()


def test_merge_interpolates_short_gaps_with_no_nan():
    noaa, flow = _make_frames()
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST', datetime(2022, 1, 1), datetime(2022, 1, 20))
    for col in combine_data.CORE_COLUMNS:
        assert not merged[col].isnull().any()
    # The 3-day gap is short enough to interpolate, so all 20 days survive.
    assert len(merged) == 20


def test_merge_drops_long_gaps_instead_of_filling():
    noaa, flow = _make_frames()
    # Widen the gap well beyond MAX_GAP_DAYS.
    flow.loc[3:15, ['Min Flow', 'Max Flow']] = np.nan
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST', datetime(2022, 1, 1), datetime(2022, 1, 20))
    # Long gap is not interpolated; those rows are dropped, no NaN remains.
    assert not merged[combine_data.CORE_COLUMNS].isnull().any().any()
    assert len(merged) < 20


def test_merge_does_not_pooled_mean_fill_edges():
    # The daily index runs past the data; trailing all-missing days must be
    # dropped, NOT back-filled with a column mean (the old, wrong behavior).
    noaa, flow = _make_frames()
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST', datetime(2022, 1, 1), datetime(2022, 1, 25))
    assert pd.to_datetime(merged['Date']).max() == pd.Timestamp('2022-01-20')


def test_merge_collapses_duplicate_dates():
    noaa, flow = _make_frames()
    # Duplicate a date in the flow frame; merge should collapse it, not error.
    flow = pd.concat([flow, flow.iloc[[0]]], ignore_index=True)
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST', datetime(2022, 1, 1), datetime(2022, 1, 20))
    assert pd.to_datetime(merged['Date']).is_unique


def test_merge_returns_empty_when_a_source_is_all_missing():
    noaa, flow = _make_frames()
    flow['Min Flow'] = np.nan
    flow['Max Flow'] = np.nan
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST', datetime(2022, 1, 1), datetime(2022, 1, 20))
    assert merged.empty


def test_merge_attaches_interpolated_swe_when_provided():
    noaa, flow = _make_frames()
    dates = pd.date_range('2022-01-01', '2022-01-20', freq='D')
    # Sparse SWE -- only every 3rd day. Interior gaps fill via interpolation.
    swe = pd.DataFrame({'Date': dates[::3],
                        'SWE': [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0]})
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST',
        datetime(2022, 1, 1), datetime(2022, 1, 20),
        swe_data=swe)
    assert 'SWE' in merged.columns
    assert not merged['SWE'].isnull().any()
    # Interior rows (up to the last known SWE date) lie between the sparse
    # known points spanning 4..10; trailing rows past the last known SWE
    # date default to 0 (documented behavior, see merge_dataframes).
    interior = merged.loc[pd.to_datetime(merged['Date']) <= pd.Timestamp('2022-01-19'), 'SWE']
    assert interior.max() <= 10.0 and interior.min() >= 4.0
    trailing = merged.loc[pd.to_datetime(merged['Date']) > pd.Timestamp('2022-01-19'), 'SWE']
    assert (trailing == 0.0).all()


def test_merge_defaults_swe_to_zero_when_not_provided():
    noaa, flow = _make_frames()
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST', datetime(2022, 1, 1), datetime(2022, 1, 20))
    assert 'SWE' in merged.columns
    assert (merged['SWE'] == 0.0).all()
