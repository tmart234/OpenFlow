"""Unit tests for the incremental per-station Feature Store (data/feature_store.py)."""

from datetime import datetime
import json
import os

import numpy as np
import pandas as pd
import pytest

import combine_data
from data import feature_store


def _make_station_frame(site_id: str, start: datetime, end: datetime, offset: float = 0.0) -> pd.DataFrame:
    dates = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq='D')
    n = len(dates)
    base = np.arange(n, dtype=float) + 100.0 + offset
    return pd.DataFrame({
        'Date': dates,
        'site_id': site_id,
        'huc8': '10190004',
        'Min Flow': base * 0.9,
        'Max Flow': base * 1.1,
        'TMIN': 10.0,
        'TMAX': 150.0,
        'precipitation': 0.5,
        'SWE': 2.0,
        'soil_moisture': 0.0,
        'sm_observed': 0,
        'drought_index': 50.0,
        'reservoir_storage': 0.0,
        'reservoir_release': 0.0,
        'reservoir_observed': 0,
        'upstream_flow': base * 0.6,
        'upstream_observed': 1,
        'upstream_known_days': 0,
        'rain_local': 0.5,
        'rain_mid': 0.0,
        'rain_far': 0.0,
        'snow_local': 0.1,
        'snow_mid': 0.0,
        'snow_far': 0.0,
        'basin_precip_observed': 1,
    })


def test_cold_miss_then_hit_avoids_refetch(tmp_path):
    cache_dir = str(tmp_path / 'cache')
    calls = []

    def fetch(site_id, start, end):
        calls.append((site_id, pd.Timestamp(start).date(), pd.Timestamp(end).date()))
        return _make_station_frame(site_id, start, end)

    df1, status1 = feature_store.load_or_update_station(
        'USGS:06719505', datetime(2024, 1, 1), datetime(2024, 3, 1), fetch, cache_dir,
        fingerprint='fp-1',
    )
    assert status1 == 'rebuilt'
    assert len(calls) == 1
    assert len(df1) == 61

    # Exact same or narrower window hits cache with zero fetches.
    df2, status2 = feature_store.load_or_update_station(
        'USGS:06719505', datetime(2024, 1, 15), datetime(2024, 2, 15), fetch, cache_dir,
        fingerprint='fp-1',
    )
    assert status2 == 'hit'
    assert len(calls) == 1
    assert df2['Date'].min() == pd.Timestamp('2024-01-15')
    assert df2['Date'].max() == pd.Timestamp('2024-02-15')


def test_incremental_refresh_refetches_only_trailing_window_and_updates_revisions(tmp_path):
    cache_dir = str(tmp_path / 'cache')
    calls = []

    def fetch(site_id, start, end):
        calls.append((pd.Timestamp(start).date(), pd.Timestamp(end).date()))
        # Return shifted values on the second call to confirm provisional revision update.
        offset = 500.0 if len(calls) == 2 else 0.0
        return _make_station_frame(site_id, start, end, offset=offset)

    feature_store.load_or_update_station(
        'USGS:06719505', datetime(2024, 1, 1), datetime(2024, 3, 1), fetch, cache_dir,
        refresh_days=14, fingerprint='fp-1',
    )
    df, status = feature_store.load_or_update_station(
        'USGS:06719505', datetime(2024, 1, 1), datetime(2024, 3, 10), fetch, cache_dir,
        refresh_days=14, fingerprint='fp-1',
    )
    assert status == 'incremental'
    assert len(calls) == 2
    # Second fetch only requested trailing ~14 days, not from Jan 1.
    assert calls[1][0] == pd.Timestamp('2024-02-16').date()
    assert calls[1][1] == pd.Timestamp('2024-03-10').date()
    assert len(df) == 70
    assert df['Date'].is_unique and df['Date'].is_monotonic_increasing
    # Early January kept cached values (offset=0); recent March got revised values (offset=500).
    assert df.loc[df['Date'] == pd.Timestamp('2024-01-10'), 'Max Flow'].iloc[0] < 200.0
    assert df.loc[df['Date'] == pd.Timestamp('2024-03-05'), 'Max Flow'].iloc[0] > 500.0


def test_backfill_prepends_older_history_without_refetching_existing_years(tmp_path):
    cache_dir = str(tmp_path / 'cache')
    calls = []

    def fetch(site_id, start, end):
        calls.append((pd.Timestamp(start).date(), pd.Timestamp(end).date()))
        return _make_station_frame(site_id, start, end)

    feature_store.load_or_update_station(
        'USGS:06719505', datetime(2020, 1, 1), datetime(2024, 1, 1), fetch, cache_dir,
        refresh_days=14, fingerprint='fp-1',
    )
    df, status = feature_store.load_or_update_station(
        'USGS:06719505', datetime(2015, 1, 1), datetime(2024, 1, 1), fetch, cache_dir,
        refresh_days=14, fingerprint='fp-1',
    )
    assert status == 'backfill'
    assert len(calls) == 2
    # Backfill ended shortly past 2020-01-01, never re-fetching 2021..2024.
    assert calls[1][0] == pd.Timestamp('2015-01-01').date()
    assert calls[1][1] == pd.Timestamp('2020-01-15').date()
    assert df['Date'].min() == pd.Timestamp('2015-01-01')
    assert df['Date'].max() == pd.Timestamp('2024-01-01')
    assert df['Date'].is_unique


def test_fingerprint_change_invalidates_only_affected_station(tmp_path, monkeypatch):
    cache_dir = str(tmp_path / 'cache')
    fp_a = feature_store.station_fingerprint('USGS:06719505', weather_source='gridmet')
    monkeypatch.setenv('OPENFLOW_DISABLE_SMAP', '1')
    fp_b = feature_store.station_fingerprint('USGS:06719505', weather_source='gridmet')
    assert fp_a != fp_b

    calls = []

    def fetch(site_id, start, end):
        calls.append(site_id)
        return _make_station_frame(site_id, start, end)

    _, s1 = feature_store.load_or_update_station(
        'USGS:06719505', datetime(2024, 1, 1), datetime(2024, 2, 1), fetch, cache_dir,
        fingerprint=fp_a,
    )
    _, s2 = feature_store.load_or_update_station(
        'USGS:06719505', datetime(2024, 1, 1), datetime(2024, 2, 1), fetch, cache_dir,
        fingerprint=fp_b,
    )
    assert s1 == 'rebuilt' and s2 == 'rebuilt'
    assert len(calls) == 2


def test_csv_gz_fallback_when_parquet_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_store, '_has_parquet', lambda: False)
    cache_dir = str(tmp_path / 'cache')
    df1, s1 = feature_store.load_or_update_station(
        'USGS:06719505', datetime(2024, 1, 1), datetime(2024, 1, 20),
        lambda s, a, b: _make_station_frame(s, a, b), cache_dir, fingerprint='fp-csv',
    )
    assert s1 == 'rebuilt'
    assert os.path.exists(os.path.join(cache_dir, 'stations', 'USGS_06719505.csv.gz'))
    df2, s2 = feature_store.load_or_update_station(
        'USGS:06719505', datetime(2024, 1, 1), datetime(2024, 1, 20),
        lambda s, a, b: (_ for _ in ()).throw(AssertionError("should not fetch")),
        cache_dir, fingerprint='fp-csv',
    )
    assert s2 == 'hit'
    assert len(df2) == len(df1)


def test_combine_data_main_uses_feature_store_when_cache_dir_set(tmp_path, monkeypatch):
    base = str(tmp_path / 'repo')
    os.makedirs(os.path.join(base, 'openFlowML'), exist_ok=True)
    cache_dir = str(tmp_path / 'cache')
    calls = []

    monkeypatch.setattr(combine_data, 'get_base_path', lambda: base)
    monkeypatch.setattr(combine_data, 'get_site_ids', lambda: ['USGS:06719505'])

    def fake_fetch(site_id, start, end):
        calls.append((site_id, pd.Timestamp(start).date(), pd.Timestamp(end).date()))
        return _make_station_frame(site_id, start, end)

    monkeypatch.setattr(combine_data, 'fetch_station_merged', fake_fetch)

    out1 = combine_data.main(training_num_years=1, cache_dir=cache_dir)
    assert out1 is not None and not out1.empty
    assert len(calls) == 1
    assert os.path.exists(os.path.join(cache_dir, 'manifest.json'))
    with open(os.path.join(cache_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    assert 'USGS:06719505' in manifest['stations']

    # Second run on the same day hits cache with zero calls to fetch_station_merged.
    out2 = combine_data.main(training_num_years=1, cache_dir=cache_dir)
    assert out2 is not None and len(out2) == len(out1)
    assert len(calls) == 1
