"""
Unit tests for the CBRFC NWPS historical archive path in get_cbrfc.py.

These tests do NOT make live network calls -- they monkey-patch
data_utils.request_with_retry to return canned NWPS payloads, so they run in
the default `pytest -m "not network"` selection.
"""

import json
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from data import get_cbrfc


@pytest.fixture(autouse=True)
def _reset_lid_cache():
    """Clear the module-level LID-table cache between tests."""
    if hasattr(get_cbrfc._load_lid_table, '_cache'):
        delattr(get_cbrfc._load_lid_table, '_cache')
    yield
    if hasattr(get_cbrfc._load_lid_table, '_cache'):
        delattr(get_cbrfc._load_lid_table, '_cache')


def _seed_lid_map(monkeypatch, mapping):
    """Stub _load_lid_table so the tests don't depend on the on-disk JSON."""
    monkeypatch.setattr(get_cbrfc, '_load_lid_table', lambda: mapping)


def _mock_response(payload, monkeypatch):
    """Make data_utils.request_with_retry return a Response-shaped object whose .json() yields `payload`."""
    response = MagicMock()
    response.json.return_value = payload
    response.text = json.dumps(payload)
    monkeypatch.setattr(get_cbrfc.data_utils, 'request_with_retry',
                        lambda *args, **kwargs: response)


def test_nwps_historical_parses_daily_forecast(monkeypatch):
    _seed_lid_map(monkeypatch, {'USGS:09163500': 'CRSC2'})
    payload = {
        'data': [
            {'validTime': '2024-01-02T00:00:00Z', 'primary': 100.0},
            {'validTime': '2024-01-02T12:00:00Z', 'primary': 120.0},  # collapses to mean
            {'validTime': '2024-01-03T00:00:00Z', 'primary': 150.0},
            {'validTime': '2024-01-04T00:00:00Z', 'primary': 140.0},
        ]
    }
    _mock_response(payload, monkeypatch)

    df = get_cbrfc.fetch('USGS:09163500', '2024-01-01', horizon_days=14)
    assert list(df.columns) == ['Date', 'cbrfc_flow']
    # Sub-daily values for 2024-01-02 collapse to the daily mean of 110.0.
    by_date = dict(zip(df['Date'], df['cbrfc_flow']))
    assert by_date['2024-01-02'] == pytest.approx(110.0)
    assert by_date['2024-01-03'] == pytest.approx(150.0)
    assert by_date['2024-01-04'] == pytest.approx(140.0)


def test_nwps_historical_honors_horizon_cap(monkeypatch):
    _seed_lid_map(monkeypatch, {'USGS:09163500': 'CRSC2'})
    # 20 distinct days; horizon_days=5 should keep only the first 5.
    payload = {
        'data': [
            {'validTime': f'2024-01-{d:02d}T00:00:00Z', 'primary': float(d)}
            for d in range(2, 22)
        ]
    }
    _mock_response(payload, monkeypatch)

    df = get_cbrfc.fetch('USGS:09163500', '2024-01-01', horizon_days=5)
    assert len(df) == 5


def test_nwps_historical_returns_empty_when_lid_missing(monkeypatch):
    _seed_lid_map(monkeypatch, {})  # no mapping
    # The fetch should short-circuit before issuing a request.
    sentinel = MagicMock(side_effect=AssertionError("request_with_retry must not be called"))
    monkeypatch.setattr(get_cbrfc.data_utils, 'request_with_retry', sentinel)

    df = get_cbrfc.fetch('USGS:UNMAPPED', '2024-01-01')
    assert df.empty


def test_nwps_historical_returns_empty_on_request_failure(monkeypatch):
    _seed_lid_map(monkeypatch, {'USGS:09163500': 'CRSC2'})
    monkeypatch.setattr(get_cbrfc.data_utils, 'request_with_retry',
                        lambda *args, **kwargs: None)

    df = get_cbrfc.fetch('USGS:09163500', '2024-01-01')
    assert df.empty


def test_nwps_historical_tolerates_alt_schema_wrapper(monkeypatch):
    """NWPS occasionally wraps the data array under `forecast` instead of top-level `data`."""
    _seed_lid_map(monkeypatch, {'USGS:09163500': 'CRSC2'})
    payload = {
        'forecast': {
            'data': [
                {'validTime': '2024-01-02T00:00:00Z', 'primary': 200.0},
            ]
        }
    }
    _mock_response(payload, monkeypatch)

    df = get_cbrfc.fetch('USGS:09163500', '2024-01-01')
    assert len(df) == 1
    assert df.iloc[0]['cbrfc_flow'] == pytest.approx(200.0)


def test_fetch_today_uses_ahps_not_nwps(monkeypatch):
    """For anchor_date >= today, fetch() must use the AHPS live path."""
    _seed_lid_map(monkeypatch, {'USGS:09163500': 'CRSC2'})
    calls = []

    def _fake_request(url, **kwargs):
        calls.append(url)
        response = MagicMock()
        response.text = '<root><forecast></forecast></root>'  # empty AHPS XML
        return response

    monkeypatch.setattr(get_cbrfc.data_utils, 'request_with_retry', _fake_request)
    get_cbrfc.fetch('USGS:09163500', date.today())
    assert calls and calls[0] == get_cbrfc.AHPS_FORECAST_URL


def test_lid_map_file_skips_metadata_keys():
    """The on-disk cbrfc_lid_map.json keeps a leading underscore comment row; loader must ignore it."""
    table = get_cbrfc._load_lid_table()
    assert all(not k.startswith('_') for k in table)
