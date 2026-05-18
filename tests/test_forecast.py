import pytest
import requests_mock

import data.get_forecast as get_forecast


def test_open_meteo_parses_daily_min_max_and_precipitation():
    with requests_mock.Mocker() as m:
        m.get('https://api.open-meteo.com/v1/forecast', json={
            "daily": {
                "time": ["2026-05-15", "2026-05-16", "2026-05-17"],
                "temperature_2m_max": [72.0, 75.0, 70.0],
                "temperature_2m_min": [45.0, 48.0, 42.0],
                "precipitation_sum": [0.0, 3.5, 0.2],
            }
        })
        df = get_forecast.get_open_meteo_forecast(39.0, -106.0, days=3)
        assert list(df.columns) == ['Date', 'TMIN', 'TMAX', 'precipitation']
        assert len(df) == 3
        assert df.loc[0, 'TMIN'] == 45.0
        assert df.loc[0, 'TMAX'] == 72.0
        assert df.loc[1, 'precipitation'] == 3.5


def test_open_meteo_defaults_precipitation_to_zero_when_absent():
    """An older Open-Meteo deployment that doesn't return precipitation_sum still works."""
    with requests_mock.Mocker() as m:
        m.get('https://api.open-meteo.com/v1/forecast', json={
            "daily": {
                "time": ["2026-05-15", "2026-05-16"],
                "temperature_2m_max": [72.0, 75.0],
                "temperature_2m_min": [45.0, 48.0],
            }
        })
        df = get_forecast.get_open_meteo_forecast(39.0, -106.0, days=2)
        assert 'precipitation' in df.columns
        assert (df['precipitation'] == 0.0).all()


def test_open_meteo_returns_empty_on_http_error():
    with requests_mock.Mocker() as m:
        m.get('https://api.open-meteo.com/v1/forecast', status_code=500)
        df = get_forecast.get_open_meteo_forecast(39.0, -106.0, days=3)
        assert df.empty


def test_open_meteo_returns_empty_on_malformed_payload():
    with requests_mock.Mocker() as m:
        m.get('https://api.open-meteo.com/v1/forecast', json={"daily": {}})
        assert get_forecast.get_open_meteo_forecast(39.0, -106.0, days=3).empty


def test_open_meteo_rejects_out_of_range_horizon():
    with pytest.raises(ValueError):
        get_forecast.get_open_meteo_forecast(39.0, -106.0, days=20)
    with pytest.raises(ValueError):
        get_forecast.get_open_meteo_forecast(39.0, -106.0, days=0)


def test_get_forecast_delegates_to_open_meteo():
    with requests_mock.Mocker() as m:
        m.get('https://api.open-meteo.com/v1/forecast', json={
            "daily": {
                "time": ["2026-05-15"],
                "temperature_2m_max": [80.0],
                "temperature_2m_min": [55.0],
            }
        })
        df = get_forecast.get_forecast(39.0, -106.0, days=1)
        assert len(df) == 1 and df.loc[0, 'TMAX'] == 80.0
