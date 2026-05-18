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


def test_merge_attaches_interpolated_soil_moisture_when_provided():
    noaa, flow = _make_frames()
    dates = pd.date_range('2022-01-01', '2022-01-20', freq='D')
    # Sparse SMAP retrievals -- only every 3rd day, like a real revisit cadence.
    sm = pd.DataFrame({'Date': dates[::3],
                       'soil_moisture': [0.15, 0.20, 0.25, 0.30, 0.25, 0.20, 0.15]})
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST',
        datetime(2022, 1, 1), datetime(2022, 1, 20),
        sm_data=sm)
    assert 'soil_moisture' in merged.columns
    assert not merged['soil_moisture'].isnull().any()
    # Every value must lie within the observed range -- ffill/bfill carries
    # the edge values to the trailing days, not 0.
    sm_min, sm_max = 0.15, 0.30
    assert merged['soil_moisture'].max() <= sm_max
    assert merged['soil_moisture'].min() >= sm_min


def test_merge_soil_moisture_indicator_flags_real_observations():
    noaa, flow = _make_frames()
    dates = pd.date_range('2022-01-01', '2022-01-20', freq='D')
    # SMAP retrievals on a sparse subset of days -- the rest get interpolated
    # or ffilled / bfilled.
    sm = pd.DataFrame({'Date': dates[::3],
                       'soil_moisture': [0.15, 0.20, 0.25, 0.30, 0.25, 0.20, 0.15]})
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST',
        datetime(2022, 1, 1), datetime(2022, 1, 20),
        sm_data=sm)
    assert 'sm_observed' in merged.columns
    # The indicator is 1 on real-or-interpolated rows (everything from the
    # first observation to the last observation, since MAX_SM_GAP_DAYS=30
    # exceeds the 3-day spacing) and 0 on rows imputed via ffill/bfill.
    rows = merged.set_index(pd.to_datetime(merged['Date']))
    # Jan 19 is the last sparse observation; Jan 20 is ffill / bfill territory.
    assert rows.loc['2022-01-19', 'sm_observed'] == 1
    assert rows.loc['2022-01-20', 'sm_observed'] == 0


def test_merge_soil_moisture_falls_back_to_site_median_not_zero():
    # If a station has SOME observations but nothing extending to a portion of
    # the window, the missing rows should imputed via ffill/bfill (which here
    # leaves no gap), then the median if a gap remained -- never 0.
    noaa, flow = _make_frames()
    dates = pd.date_range('2022-01-01', '2022-01-20', freq='D')
    # Only two observations, well within the window. Long stretches before and
    # after these get ffilled / bfilled to those known values, NOT to 0.
    sm = pd.DataFrame({'Date': [dates[5], dates[10]],
                       'soil_moisture': [0.30, 0.40]})
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST',
        datetime(2022, 1, 1), datetime(2022, 1, 20),
        sm_data=sm)
    # Leading days bfill to 0.30; trailing days ffill from 0.40. Neither is 0.
    assert (merged['soil_moisture'] > 0).all()
    # sm_observed: the two real observations + the interior rows between them
    # (interpolated within MAX_SM_GAP_DAYS) are flagged as observed (1). Days
    # before the first or after the last observation are imputed via ffill/
    # bfill and flagged not-observed (0).
    rows = merged.set_index(pd.to_datetime(merged['Date']))
    assert rows.loc['2022-01-01', 'sm_observed'] == 0   # before first obs
    assert rows.loc['2022-01-06', 'sm_observed'] == 1   # first obs
    assert rows.loc['2022-01-08', 'sm_observed'] == 1   # interior interpolated
    assert rows.loc['2022-01-11', 'sm_observed'] == 1   # second obs
    assert rows.loc['2022-01-20', 'sm_observed'] == 0   # after last obs


def test_merge_defaults_soil_moisture_to_zero_only_when_no_observations():
    noaa, flow = _make_frames()
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST', datetime(2022, 1, 1), datetime(2022, 1, 20))
    assert 'soil_moisture' in merged.columns
    # With no SMAP data at all, fall back to 0 (last-resort default).
    assert (merged['soil_moisture'] == 0.0).all()
    # And the indicator correctly says "none of these are real observations".
    assert (merged['sm_observed'] == 0).all()


def test_merge_attaches_drought_index_with_ffill():
    noaa, flow = _make_frames()
    dates = pd.date_range('2022-01-01', '2022-01-20', freq='D')
    # USDM weekly snapshots on Jan 4 and Jan 11.
    drought = pd.DataFrame({
        'Date': [dates[3], dates[10]],
        'drought_index': [100.0, 250.0],
    })
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST',
        datetime(2022, 1, 1), datetime(2022, 1, 20),
        drought_data=drought)
    assert 'drought_index' in merged.columns
    rows = merged.set_index(pd.to_datetime(merged['Date']))
    # Pre-first-snapshot days have no value to ffill from -> 0 (the default).
    assert rows.loc['2022-01-01', 'drought_index'] == 0.0
    # The snapshot day takes the first value.
    assert rows.loc['2022-01-04', 'drought_index'] == 100.0
    # ffilled through the week.
    assert rows.loc['2022-01-10', 'drought_index'] == 100.0
    # Next snapshot kicks in.
    assert rows.loc['2022-01-11', 'drought_index'] == 250.0
    assert rows.loc['2022-01-20', 'drought_index'] == 250.0


def test_merge_defaults_drought_index_to_zero_when_not_provided():
    noaa, flow = _make_frames()
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST', datetime(2022, 1, 1), datetime(2022, 1, 20))
    assert 'drought_index' in merged.columns
    assert (merged['drought_index'] == 0.0).all()


def test_merge_attaches_reservoir_storage_and_release():
    noaa, flow = _make_frames()
    dates = pd.date_range('2022-01-01', '2022-01-20', freq='D')
    reservoir = pd.DataFrame({
        'Date': dates[::5],  # every 5 days
        'reservoir_storage': [1000.0, 1100.0, 1050.0, 1000.0],
        'reservoir_release': [50.0, 60.0, 55.0, 50.0],
    })
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST',
        datetime(2022, 1, 1), datetime(2022, 1, 20),
        reservoir_data=reservoir)
    assert 'reservoir_storage' in merged.columns
    assert 'reservoir_release' in merged.columns
    assert 'reservoir_observed' in merged.columns
    # Observed where the reservoir series provided a value (post-interpolation).
    assert merged['reservoir_observed'].sum() > 0
    # Storage stays in the observed range after ffill / bfill.
    s_min, s_max = 1000.0, 1100.0
    assert merged['reservoir_storage'].max() <= s_max
    assert merged['reservoir_storage'].min() >= s_min


def test_merge_defaults_reservoir_to_zero_when_unregulated():
    # Unregulated station: no reservoir_data passed -> storage / release default
    # to 0 and reservoir_observed stays 0.
    noaa, flow = _make_frames()
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST', datetime(2022, 1, 1), datetime(2022, 1, 20))
    assert (merged['reservoir_storage'] == 0.0).all()
    assert (merged['reservoir_release'] == 0.0).all()
    assert (merged['reservoir_observed'] == 0).all()


def test_merge_records_huc8_when_provided():
    noaa, flow = _make_frames()
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST', datetime(2022, 1, 1), datetime(2022, 1, 20),
        huc8='14010001')
    assert 'huc8' in merged.columns
    assert (merged['huc8'] == '14010001').all()


def test_merge_huc8_defaults_to_empty_when_not_provided():
    noaa, flow = _make_frames()
    merged = combine_data.merge_dataframes(
        noaa, flow, 'USGS:TEST', datetime(2022, 1, 1), datetime(2022, 1, 20))
    assert 'huc8' in merged.columns
    assert (merged['huc8'] == '').all()
