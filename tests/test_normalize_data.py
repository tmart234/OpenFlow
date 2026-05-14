import json
import numpy as np
import pandas as pd

import normalize_data


def _make_combined():
    """Two stations, 10 daily rows each, all core columns present."""
    dates = pd.date_range('2021-01-01', periods=10, freq='D')
    return pd.DataFrame({
        'Date': list(dates) + list(dates),
        'site_id': ['USGS:A'] * 10 + ['DWR:B'] * 10,
        'Min Flow': np.linspace(10, 30, 20),
        'Max Flow': np.linspace(20, 50, 20),
        'TMIN': np.linspace(-5, 15, 20),
        'TMAX': np.linspace(5, 25, 20),
    })


def test_day_of_year_features_are_cyclical():
    df = pd.DataFrame({'Date': pd.to_datetime(['2021-01-01', '2021-12-31', '2021-07-02'])})
    out = normalize_data.add_day_of_year_features(df)
    # Jan 1 -> angle 0 -> sin 0, cos 1
    assert np.isclose(out['doy_sin'].iloc[0], 0.0, atol=1e-9)
    assert np.isclose(out['doy_cos'].iloc[0], 1.0, atol=1e-9)
    # The features always lie on the unit circle.
    assert np.allclose(out['doy_sin'] ** 2 + out['doy_cos'] ** 2, 1.0)


def test_day_of_year_is_continuous_across_year_boundary():
    # Dec 31 and the following Jan 1 must be adjacent in feature space.
    df = pd.DataFrame({'Date': pd.to_datetime(['2021-12-31', '2022-01-01'])})
    out = normalize_data.add_day_of_year_features(df)
    dec31 = np.array([out['doy_sin'].iloc[0], out['doy_cos'].iloc[0]])
    jan01 = np.array([out['doy_sin'].iloc[1], out['doy_cos'].iloc[1]])
    assert np.linalg.norm(dec31 - jan01) < 0.05


def test_day_of_year_handles_leap_year():
    # Dec 31 of a leap year is day 366; the angle should still wrap near 2*pi.
    df = pd.DataFrame({'Date': pd.to_datetime(['2020-12-31'])})
    out = normalize_data.add_day_of_year_features(df)
    assert np.isclose(out['doy_sin'].iloc[0], 0.0, atol=2e-2)
    assert out['doy_cos'].iloc[0] > 0.99


def test_station_index_reserves_zero_for_unknown():
    idx = normalize_data.build_station_index(['USGS:A', 'DWR:B', 'USGS:A'])
    assert 0 not in idx.values()
    assert sorted(idx.values()) == [1, 2]


def test_normalize_persists_artifacts_and_z_scores(tmp_path):
    df = _make_combined()
    out = normalize_data.normalize_data(df, artifacts_dir=str(tmp_path))
    assert out is not None

    scalers = json.loads((tmp_path / 'scalers.json').read_text())
    station_index = json.loads((tmp_path / 'station_index.json').read_text())
    assert set(scalers) == {'TMIN', 'TMAX', 'Min Flow', 'Max Flow'}
    assert set(station_index) == {'USGS:A', 'DWR:B'}

    # Normalized numeric columns are z-scores: ~0 mean, ~1 std.
    for col in normalize_data.NUMERIC_COLUMNS:
        assert abs(out[col].mean()) < 1e-6
        assert abs(out[col].std(ddof=0) - 1.0) < 1e-6

    # Integer station index present; Date + site_id preserved for Phase 3.
    assert set(out['station_idx'].unique()) <= {1, 2}
    assert 'Date' in out.columns and 'site_id' in out.columns


def test_apply_scalers_round_trips():
    df = _make_combined()
    scalers = normalize_data.fit_scalers(df, ['Min Flow'])
    scaled = normalize_data.apply_scalers(df.copy(), scalers)
    params = scalers['Min Flow']
    restored = scaled['Min Flow'] * params['scale'] + params['mean']
    assert np.allclose(restored, df['Min Flow'])


def test_normalize_missing_column_returns_none():
    df = _make_combined().drop(columns=['TMIN'])
    assert normalize_data.normalize_data(df) is None
