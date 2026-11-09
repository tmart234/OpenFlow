"""NOAA station fetch: failures come back as empty frames, never crash the caller."""

import pandas as pd

from data import get_noaa


def test_metadata_failure_returns_an_empty_frame(monkeypatch):
    # NCEI rate limiting (HTTP 429) -> no metadata. This used to return {}
    # and crash the realized-coverage check ('dict' has no attribute 'empty').
    monkeypatch.setattr(get_noaa, 'get_station_metadata', lambda sid: None)
    out = get_noaa.fetch_temperature_data('USC00051294', '2026-10-01', '2026-11-08')
    assert isinstance(out, pd.DataFrame) and out.empty
    assert get_noaa._realized_coverage_ok(out, ['TMIN', 'TMAX'], 0.5) is False


def test_coverage_check_rejects_non_frames():
    assert get_noaa._realized_coverage_ok({}, ['TMIN'], 0.5) is False
    assert get_noaa._realized_coverage_ok(None, ['TMIN'], 0.5) is False
