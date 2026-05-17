"""
Tests for the consolidated SMAP soil-moisture fetcher (nasa_moisture.main).

We mock at the earthaccess + polygon-lookup boundaries so the test runs offline
and deterministically; the on-disk extraction is exercised by writing a tiny
real HDF5 file and reading it back through _extract_polygon_mean.
"""

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

# h5py is part of requirements-data.txt; skip if not installed.
h5py = pytest.importorskip('h5py')

from data import nasa_moisture


# Polygon big enough to contain the synthetic grids used below.
_POLYGON = [(-106.5, 39.5), (-104.5, 39.5), (-104.5, 41.5), (-106.5, 41.5)]


def _write_smap_h5(path, sm_am, sm_pm=None):
    """Write a minimal SMAP-shaped HDF5 file at `path`."""
    lat = np.array([
        [40.0, 40.0, 40.0],
        [40.5, 40.5, 40.5],
        [41.0, 41.0, 41.0],
    ])
    lon = np.array([
        [-106.0, -105.5, -105.0],
        [-106.0, -105.5, -105.0],
        [-106.0, -105.5, -105.0],
    ])
    with h5py.File(path, 'w') as f:
        am = f.create_group('Soil_Moisture_Retrieval_Data_AM')
        am.create_dataset('soil_moisture', data=sm_am)
        am.create_dataset('latitude', data=lat)
        am.create_dataset('longitude', data=lon)
        if sm_pm is not None:
            pm = f.create_group('Soil_Moisture_Retrieval_Data_PM')
            pm.create_dataset('soil_moisture', data=sm_pm)
            pm.create_dataset('latitude', data=lat)
            pm.create_dataset('longitude', data=lon)


class _FakeGranule:
    """Stand-in for an earthaccess granule with the bits nasa_moisture uses."""

    def __init__(self, filename):
        self._filename = filename

    def data_links(self):
        return [f"https://example.test/{self._filename}"]


def test_extract_polygon_mean_averages_valid_pixels_only(tmp_path):
    path = str(tmp_path / 'sm.h5')
    sm = np.array([
        [-9999.0, 0.25, 0.30],
        [0.20, 0.35, 0.40],
        [-9999.0, 0.45, 0.50],
    ])
    _write_smap_h5(path, sm)
    value = nasa_moisture._extract_polygon_mean(path, _POLYGON)
    expected = (0.25 + 0.30 + 0.20 + 0.35 + 0.40 + 0.45 + 0.50) / 7
    assert value == pytest.approx(expected)


def test_extract_polygon_mean_returns_none_when_all_fill(tmp_path):
    path = str(tmp_path / 'sm.h5')
    sm = np.full((3, 3), -9999.0)
    _write_smap_h5(path, sm)
    assert nasa_moisture._extract_polygon_mean(path, _POLYGON) is None


def test_extract_polygon_mean_combines_am_and_pm(tmp_path):
    path = str(tmp_path / 'sm.h5')
    sm_am = np.full((3, 3), 0.20)
    sm_pm = np.full((3, 3), 0.40)
    _write_smap_h5(path, sm_am, sm_pm=sm_pm)
    # 9 AM pixels at 0.2 + 9 PM pixels at 0.4 -> grand mean of 0.30.
    assert nasa_moisture._extract_polygon_mean(path, _POLYGON) == pytest.approx(0.30)


def test_extract_polygon_mean_clips_to_valid_range(tmp_path):
    path = str(tmp_path / 'sm.h5')
    sm = np.array([
        [0.30, 0.40, 5.0],     # 5.0 > VALID_MAX -- rejected
        [0.20, -0.10, 0.50],   # -0.10 < VALID_MIN -- rejected
        [0.10, 0.20, 0.30],
    ])
    _write_smap_h5(path, sm)
    value = nasa_moisture._extract_polygon_mean(path, _POLYGON)
    kept = [0.30, 0.40, 0.20, 0.50, 0.10, 0.20, 0.30]
    assert value == pytest.approx(sum(kept) / len(kept))


def test_granule_date_parses_from_filename():
    g = _FakeGranule('SMAP_L3_SM_P_E_20240115_R18290_001.h5')
    assert nasa_moisture._granule_date(g) == date(2024, 1, 15)


def test_granule_date_returns_none_when_no_pattern():
    g = _FakeGranule('arbitrary_filename.h5')
    assert nasa_moisture._granule_date(g) is None


def test_main_returns_empty_when_polygon_lookup_fails(monkeypatch):
    monkeypatch.setattr(nasa_moisture, '_get_huc8_polygon', lambda lat, lon: None)
    out = nasa_moisture.main(40.0, -105.0, '2024-01-01', '2024-01-03')
    assert out.empty
    assert list(out.columns) == ['Date', 'soil_moisture']


def test_main_returns_empty_when_auth_fails(monkeypatch):
    monkeypatch.setattr(nasa_moisture, '_get_huc8_polygon', lambda lat, lon: _POLYGON)
    monkeypatch.setattr(nasa_moisture, '_login_earthdata', lambda: None)
    out = nasa_moisture.main(40.0, -105.0, '2024-01-01', '2024-01-03')
    assert out.empty


def test_main_returns_empty_when_no_granules(monkeypatch):
    monkeypatch.setattr(nasa_moisture, '_get_huc8_polygon', lambda lat, lon: _POLYGON)
    monkeypatch.setattr(nasa_moisture, '_login_earthdata', lambda: object())
    monkeypatch.setattr(nasa_moisture, '_search_granules',
                        lambda poly, s, e: [])
    out = nasa_moisture.main(40.0, -105.0, '2024-01-01', '2024-01-03')
    assert out.empty


def test_main_end_to_end_with_mocked_search_and_download(monkeypatch, tmp_path):
    # Two distinct dates, each with a known mean. Build the granule fixtures.
    g1_path = str(tmp_path / 'SMAP_L3_SM_P_E_20240115_R18290_001.h5')
    g2_path = str(tmp_path / 'SMAP_L3_SM_P_E_20240116_R18290_001.h5')
    _write_smap_h5(g1_path, np.full((3, 3), 0.30))
    _write_smap_h5(g2_path, np.full((3, 3), 0.50))

    g1 = _FakeGranule('SMAP_L3_SM_P_E_20240115_R18290_001.h5')
    g2 = _FakeGranule('SMAP_L3_SM_P_E_20240116_R18290_001.h5')

    download_map = {id(g1): g1_path, id(g2): g2_path}

    monkeypatch.setattr(nasa_moisture, '_get_huc8_polygon', lambda lat, lon: _POLYGON)
    monkeypatch.setattr(nasa_moisture, '_login_earthdata', lambda: object())
    monkeypatch.setattr(nasa_moisture, '_search_granules',
                        lambda poly, s, e: [g1, g2])

    def fake_download(granule, tmpdir):
        # Copy fixture into the temp dir so the real cleanup path runs cleanly.
        import shutil
        src = download_map[id(granule)]
        dst = f"{tmpdir}/{src.rsplit('/', 1)[-1]}"
        shutil.copy(src, dst)
        return dst

    monkeypatch.setattr(nasa_moisture, '_download_granule', fake_download)

    out = nasa_moisture.main(40.0, -105.0, '2024-01-15', '2024-01-16')
    assert list(out.columns) == ['Date', 'soil_moisture']
    assert len(out) == 2
    assert out.iloc[0]['Date'] == '2024-01-15'
    assert out.iloc[0]['soil_moisture'] == pytest.approx(0.30)
    assert out.iloc[1]['Date'] == '2024-01-16'
    assert out.iloc[1]['soil_moisture'] == pytest.approx(0.50)


def test_main_collapses_multiple_granules_on_same_date(monkeypatch, tmp_path):
    # Two granules tagged with the same date -- the daily series collapses
    # them via mean (single row, averaged value).
    a_path = str(tmp_path / 'SMAP_L3_SM_P_E_20240120_pass_a_001.h5')
    b_path = str(tmp_path / 'SMAP_L3_SM_P_E_20240120_pass_b_001.h5')
    _write_smap_h5(a_path, np.full((3, 3), 0.20))
    _write_smap_h5(b_path, np.full((3, 3), 0.40))

    g_a = _FakeGranule(a_path.rsplit('/', 1)[-1])
    g_b = _FakeGranule(b_path.rsplit('/', 1)[-1])
    download_map = {id(g_a): a_path, id(g_b): b_path}

    monkeypatch.setattr(nasa_moisture, '_get_huc8_polygon', lambda lat, lon: _POLYGON)
    monkeypatch.setattr(nasa_moisture, '_login_earthdata', lambda: object())
    monkeypatch.setattr(nasa_moisture, '_search_granules',
                        lambda poly, s, e: [g_a, g_b])

    def fake_download(granule, tmpdir):
        import shutil
        src = download_map[id(granule)]
        dst = f"{tmpdir}/{src.rsplit('/', 1)[-1]}"
        shutil.copy(src, dst)
        return dst

    monkeypatch.setattr(nasa_moisture, '_download_granule', fake_download)

    out = nasa_moisture.main(40.0, -105.0, '2024-01-20', '2024-01-20')
    assert len(out) == 1
    assert out.iloc[0]['Date'] == '2024-01-20'
    assert out.iloc[0]['soil_moisture'] == pytest.approx(0.30)


def test_main_skips_granules_that_fail_download(monkeypatch, tmp_path):
    good_path = str(tmp_path / 'SMAP_L3_SM_P_E_20240117_R001_001.h5')
    _write_smap_h5(good_path, np.full((3, 3), 0.25))

    g_bad = _FakeGranule('SMAP_L3_SM_P_E_20240115_R000_001.h5')
    g_good = _FakeGranule('SMAP_L3_SM_P_E_20240117_R001_001.h5')

    monkeypatch.setattr(nasa_moisture, '_get_huc8_polygon', lambda lat, lon: _POLYGON)
    monkeypatch.setattr(nasa_moisture, '_login_earthdata', lambda: object())
    monkeypatch.setattr(nasa_moisture, '_search_granules',
                        lambda poly, s, e: [g_bad, g_good])

    def fake_download(granule, tmpdir):
        if granule is g_bad:
            return None
        import shutil
        dst = f"{tmpdir}/{good_path.rsplit('/', 1)[-1]}"
        shutil.copy(good_path, dst)
        return dst

    monkeypatch.setattr(nasa_moisture, '_download_granule', fake_download)

    out = nasa_moisture.main(40.0, -105.0, '2024-01-15', '2024-01-17')
    assert len(out) == 1
    assert out.iloc[0]['Date'] == '2024-01-17'


def test_to_date_accepts_string_datetime_and_date():
    assert nasa_moisture._to_date('2024-01-15') == date(2024, 1, 15)
    assert nasa_moisture._to_date(datetime(2024, 1, 15, 12, 0)) == date(2024, 1, 15)
    assert nasa_moisture._to_date(date(2024, 1, 15)) == date(2024, 1, 15)
