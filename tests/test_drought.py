"""
Tests for get_drought.main / get_drought helpers (USDM intensity-by-HUC).

USDM is a clean REST endpoint; we mock with requests_mock and don't hit the
network. The HUC lookup is patched out per test so we can isolate the USDM
parsing + forward-fill logic from the WBD ArcGIS dependency.
"""

from datetime import datetime

import pandas as pd
import pytest
import requests_mock as _rm

from data import get_drought


@pytest.fixture
def mock_api():
    with _rm.Mocker() as m:
        yield m


def test_intensity_weights_collapse_categories_correctly():
    # 100% D2 == weight 3 == index 300; D0..D4 weights are ordinal (1..5).
    assert get_drought._record_to_index(
        {'D0': 0, 'D1': 0, 'D2': 100, 'D3': 0, 'D4': 0}) == 300.0
    # 50% D1 + 50% D3 == 0.5*2 + 0.5*4 = 3 -- but expressed as percent it's
    # 50 + 50 -> 50*2 + 50*4 = 300.
    assert get_drought._record_to_index(
        {'D0': 0, 'D1': 50, 'D2': 0, 'D3': 50, 'D4': 0}) == 300.0
    # No drought anywhere -> 0.
    assert get_drought._record_to_index(
        {'D0': 0, 'D1': 0, 'D2': 0, 'D3': 0, 'D4': 0}) == 0.0
    # Maximum: 100% D4 -> 500.
    assert get_drought._record_to_index(
        {'D0': 0, 'D1': 0, 'D2': 0, 'D3': 0, 'D4': 100}) == 500.0


def test_intensity_treats_missing_categories_as_zero():
    # USDM occasionally omits zero-category fields; missing == 0, not error.
    assert get_drought._record_to_index({'D4': 100}) == 500.0
    assert get_drought._record_to_index({}) == 0.0


def test_format_mdy_strips_zero_padding():
    # USDM API requires M/D/YYYY (NOT MM/DD/YYYY).
    assert get_drought._format_mdy('2024-01-05') == '1/5/2024'
    assert get_drought._format_mdy(datetime(2024, 12, 31)) == '12/31/2024'


def test_get_drought_weekly_parses_records(mock_api):
    mock_api.get(
        get_drought.USDM_URL,
        json=[
            {'MapDate': '2024-01-02', 'D0': 100, 'D1': 0, 'D2': 0, 'D3': 0, 'D4': 0},
            {'MapDate': '2024-01-09', 'D0': 0,   'D1': 50, 'D2': 50, 'D3': 0, 'D4': 0},
        ],
    )
    recs = get_drought.get_drought_weekly('14010001', '2024-01-01', '2024-01-15')
    assert len(recs) == 2


def test_get_drought_returns_empty_when_huc_lookup_fails(monkeypatch):
    monkeypatch.setattr(get_drought.get_swe, 'get_huc_id', lambda *a, **kw: None)
    df = get_drought.get_drought(40.0, -106.0, '2024-01-01', '2024-01-15')
    assert df.empty
    assert list(df.columns) == ['Date', 'drought_index']


def test_get_drought_forward_fills_weekly_to_daily(mock_api, monkeypatch):
    monkeypatch.setattr(get_drought.get_swe, 'get_huc_id', lambda *a, **kw: '14010001')
    mock_api.get(
        get_drought.USDM_URL,
        json=[
            {'MapDate': '2024-01-02', 'D0': 100, 'D1': 0, 'D2': 0, 'D3': 0, 'D4': 0},
            {'MapDate': '2024-01-09', 'D0': 0,   'D1': 50, 'D2': 50, 'D3': 0, 'D4': 0},
        ],
    )
    df = get_drought.get_drought(40.0, -106.0, '2024-01-01', '2024-01-15')
    # Daily reindex over Jan 1..15 = 15 rows.
    assert len(df) == 15
    rows = df.set_index('Date')
    # Days before the first weekly snapshot stay NaN (we used ffill only --
    # the spine fills those with 0).
    assert pd.isna(rows.loc['2024-01-01', 'drought_index'])
    # Days from Jan 2 through Jan 8 inherit the first snapshot's intensity
    # (100% D0 -> weight 1 -> 100).
    assert rows.loc['2024-01-02', 'drought_index'] == 100.0
    assert rows.loc['2024-01-08', 'drought_index'] == 100.0
    # Days from Jan 9 onward inherit the second snapshot (50*2 + 50*3 = 250).
    assert rows.loc['2024-01-09', 'drought_index'] == 250.0
    assert rows.loc['2024-01-15', 'drought_index'] == 250.0


def test_get_drought_returns_empty_when_api_returns_nothing(mock_api, monkeypatch):
    monkeypatch.setattr(get_drought.get_swe, 'get_huc_id', lambda *a, **kw: '14010001')
    mock_api.get(get_drought.USDM_URL, json=[])
    df = get_drought.get_drought(40.0, -106.0, '2024-01-01', '2024-01-15')
    assert df.empty


def test_get_drought_tolerates_alternate_date_keys(mock_api, monkeypatch):
    monkeypatch.setattr(get_drought.get_swe, 'get_huc_id', lambda *a, **kw: '14010001')
    mock_api.get(
        get_drought.USDM_URL,
        json=[
            # USDM has been observed to return ValidStart instead of MapDate.
            {'ValidStart': '20240102', 'D0': 100, 'D1': 0, 'D2': 0, 'D3': 0, 'D4': 0},
        ],
    )
    df = get_drought.get_drought(40.0, -106.0, '2024-01-01', '2024-01-15')
    assert len(df) == 15
    assert df.set_index('Date').loc['2024-01-02', 'drought_index'] == 100.0
