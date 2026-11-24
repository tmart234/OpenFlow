"""Tests for get_noaa dead-station rejection and realized-coverage checks (no network)."""

import pandas as pd
import pytest

from data import get_noaa


def test_find_station_rejects_dead_station_even_when_alive_at_start(monkeypatch):
    """A station that died years before endStr must not pass (regression for start_date_obj bug)."""
    meta = {
        'DEAD_NEAR': {'datacoverage': 0.99, 'mindate': '1990-01-01', 'maxdate': '2012-06-30'},
        'ALIVE_FAR': {'datacoverage': 0.95, 'mindate': '2000-01-01', 'maxdate': '2026-11-10'},
    }
    monkeypatch.setattr(get_noaa, 'get_station_metadata', lambda st: meta.get(st[0]))
    monkeypatch.setattr(get_noaa, 'check_fields', lambda fields, sid, s, e: True)

    chosen = get_noaa.find_station_with_recent_data(
        [('DEAD_NEAR', 5.0), ('ALIVE_FAR', 15.0)],
        '2011-11-25', ['TMIN', 'TMAX', 'PRCP'], '2026-11-21',
    )
    assert chosen == ('ALIVE_FAR', 15.0)


def test_realized_coverage_rejects_short_dead_station_against_expected_days():
    """213 non-NaN rows in a 5,476-day window is 3.9% coverage, not 100%."""
    df = pd.DataFrame({
        'Date': pd.date_range('2011-11-25', periods=213),
        'TMIN': [10.0] * 213,
        'TMAX': [150.0] * 213,
    })
    assert get_noaa._realized_coverage_ok(df, ('TMIN', 'TMAX'), 0.7, expected_days=5476) is False
    assert get_noaa._realized_coverage_ok(df, ('TMIN', 'TMAX'), 0.7, expected_days=250) is True
