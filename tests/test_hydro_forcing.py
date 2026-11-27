"""Offline tests for Phase 1 hydro forcing, gridMET energy, and extended Open-Meteo forecasts."""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from data import get_basin_precip, get_forecast, get_hydro_forcing


def test_summarize_hourly_frame_computes_storm_intensity_and_soil_moisture():
    # Day 1: intense 2-hour monsoon cloudburst (20 mm + 15 mm) -> max1h=20, max3h=35, wet=2
    # Day 2: gentle drizzle (0.2 mm/h all day) -> max1h=0.2, max3h=0.6, wet=0
    t = pd.date_range('2025-08-01 00:00', periods=48, freq='1h')
    precip = np.zeros(48)
    precip[14] = 20.0
    precip[15] = 15.0
    precip[24:] = 0.2
    df = pd.DataFrame({
        'time': t,
        'precip_mm': precip,
        'sm_surface': np.concatenate([np.full(24, 0.18), np.full(24, 0.32)]),
        'sm_rootzone': np.concatenate([np.full(24, 0.22), np.full(24, 0.28)]),
    })
    daily = get_hydro_forcing.summarize_hourly_frame(df).set_index('Date')
    assert daily.loc['2025-08-01', 'precip_max1h'] == pytest.approx(20.0)
    assert daily.loc['2025-08-01', 'precip_max3h'] == pytest.approx(35.0)
    assert daily.loc['2025-08-01', 'precip_wet_hours'] == pytest.approx(2.0)
    assert daily.loc['2025-08-01', 'sm_surface'] == pytest.approx(0.18)
    assert daily.loc['2025-08-01', 'sm_rootzone'] == pytest.approx(0.22)

    assert daily.loc['2025-08-02', 'precip_max1h'] == pytest.approx(0.2)
    assert daily.loc['2025-08-02', 'precip_max3h'] == pytest.approx(0.6)
    assert daily.loc['2025-08-02', 'precip_wet_hours'] == pytest.approx(0.0)
    assert daily.loc['2025-08-02', 'sm_surface'] == pytest.approx(0.32)


def test_hydro_source_resolution_and_nldas_fallback(monkeypatch):
    monkeypatch.delenv('OPENFLOW_HYDRO_SOURCE', raising=False)
    monkeypatch.delenv('EARTHDATA_USERNAME', raising=False)
    monkeypatch.delenv('EARTHDATA_PASSWORD', raising=False)
    assert get_hydro_forcing.hydro_source() == 'era5'
    assert get_hydro_forcing.hydro_source('auto') == 'era5'
    monkeypatch.setenv('EARTHDATA_USERNAME', 'u')
    monkeypatch.setenv('EARTHDATA_PASSWORD', 'p')
    assert get_hydro_forcing.hydro_source('auto') == 'nldas'

    # If NLDAS returns empty, get_hydro_forcing falls back cleanly to ERA5-Land.
    expected = pd.DataFrame({
        'Date': [pd.Timestamp('2025-07-01')],
        'precip_max1h': [5.0], 'precip_max3h': [8.0], 'precip_wet_hours': [2.0],
        'sm_surface': [0.25], 'sm_rootzone': [0.27],
    })
    monkeypatch.setattr(get_hydro_forcing, 'fetch_nldas_daily', lambda *a, **k: get_hydro_forcing._empty())
    monkeypatch.setattr(get_hydro_forcing, 'fetch_era5_daily', lambda *a, **k: expected)
    out = get_hydro_forcing.get_hydro_forcing(38.5, -106.0, '2025-07-01', '2025-07-02', source_override='nldas')
    assert len(out) == 1 and out['sm_rootzone'].iloc[0] == pytest.approx(0.27)


def test_effective_abstraction_scales_with_soil_moisture_and_disaggregates_hourly():
    # Dry soil absorbs much more rain than saturated soil.
    dry_abs = get_basin_precip.effective_abstraction_mm(0.08)
    mid_abs = get_basin_precip.effective_abstraction_mm(None)
    wet_abs = get_basin_precip.effective_abstraction_mm(0.40)
    assert dry_abs > mid_abs > wet_abs
    assert mid_abs == pytest.approx(get_basin_precip.ABSTRACTION_MM_PER_DAY)

    # Hourly disaggregation concentrates a daily storm total into the storm hours
    # while conserving the basin total (after abstraction).
    ring = [[-106.0, 36.0], [-105.0, 36.0], [-105.0, 37.0], [-106.0, 37.0], [-106.0, 36.0]]
    zcfg = {
        'zones': {'local': {'area_km2': 500.0, 'rings': [ring]},
                  'mid': {'area_km2': 0.0, 'rings': []}, 'far': {'area_km2': 0.0, 'rings': []}},
        'ungauged': {'area_km2': 500.0, 'rings': [ring]},
        'points': [{'lat': 36.5, 'lon': -105.5, 'zone': 'local', 'ungauged': True, 'w': 500.0}],
    }
    grid = pd.date_range('2025-08-01T00:00Z', '2025-08-02T23:00Z', freq='1h')
    obs = pd.DataFrame({
        'rain_ungauged': [25.0, 0.0],
        'window_start': [pd.Timestamp('2025-08-01T00:00Z'), pd.Timestamp('2025-08-02T00:00Z')],
        'source': ['gridmet', 'gridmet'],
    }, index=pd.DatetimeIndex(['2025-08-01', '2025-08-02']))
    # All rain fell during 2 hours (18:00-19:00 UTC) on Aug 1.
    hourly_past = pd.Series(0.0, index=grid)
    hourly_past.loc['2025-08-01T18:00Z'] = 10.0
    hourly_past.loc['2025-08-01T19:00Z'] = 10.0

    flux_flat, info_flat = get_basin_precip.routing_rain_flux(zcfg, grid, obs, {}, soil_moisture=0.25)
    flux_hr, info_hr = get_basin_precip.routing_rain_flux(
        zcfg, grid, obs, {}, hourly_past=hourly_past, soil_moisture=0.25)
    # Total volume matches, but peak hourly flux is much sharper when disaggregated!
    assert flux_hr.sum() == pytest.approx(flux_flat.sum(), rel=1e-6)
    assert flux_hr.max() > 5.0 * flux_flat.max()
    assert 'hourly-disagg' in info_hr['sources']
    assert info_hr['soil_moisture'] == pytest.approx(0.25)


def test_gridmet_point_energy_converts_srad_to_mj_m2_day(monkeypatch):
    dates = pd.DatetimeIndex(['2025-06-01', '2025-06-02'])
    la, lo = np.array([38.5, 38.4]), np.array([-106.1, -106.0])

    def fake(key, *a, **k):
        v = {'srad': 250.0, 'pet': 6.5, 'vpd': 1.8}[key]
        return dates, la, lo, np.full((2, 2, 2), v)

    monkeypatch.setattr(get_basin_precip, '_gridmet_subset', fake)
    df = get_basin_precip.gridmet_point_energy(38.45, -106.05, datetime(2025, 6, 1), datetime(2025, 6, 2))
    assert list(df.columns) == ['Date', 'srad', 'pet', 'vpd']
    assert df['srad'].iloc[0] == pytest.approx(250.0 * 0.0864)
    assert df['pet'].iloc[0] == pytest.approx(6.5)
    assert df['vpd'].iloc[0] == pytest.approx(1.8)


def test_open_meteo_extended_and_ensemble_parsers(monkeypatch):
    class FakeResp:
        def __init__(self, payload):
            self._p = payload

        def json(self):
            return self._p

    ext_payload = {
        'daily': {
            'time': ['2026-07-01', '2026-07-02'],
            'temperature_2m_min': [5.0, 6.0],
            'temperature_2m_max': [22.0, 24.0],
            'precipitation_sum': [12.0, 0.0],
            'shortwave_radiation_sum': [24.5, 28.0],
            'et0_fao_evapotranspiration': [5.2, 6.1],
        },
        'hourly': {
            'time': [f'2026-07-01T{h:02d}:00' for h in range(24)] + [f'2026-07-02T{h:02d}:00' for h in range(24)],
            'precipitation': [0.0] * 14 + [8.0, 4.0] + [0.0] * 32,
            'vapour_pressure_deficit': [1.5] * 24 + [2.2] * 24,
        },
    }
    monkeypatch.setattr(get_forecast.data_utils, 'request_with_retry', lambda *a, **k: FakeResp(ext_payload))
    ext = get_forecast.get_open_meteo_extended_forecast(38.5, -106.0, days=2)
    assert ext.loc[0, 'srad'] == pytest.approx(24.5)
    assert ext.loc[0, 'pet'] == pytest.approx(5.2)
    assert ext.loc[0, 'vpd'] == pytest.approx(1.5)
    assert ext.loc[0, 'precip_max1h'] == pytest.approx(8.0)
    assert ext.loc[0, 'precip_max3h'] == pytest.approx(12.0)
    assert ext.loc[0, 'precip_wet_hours'] == pytest.approx(2.0)

    ens_payload = {
        'daily': {
            'time': ['2026-07-01', '2026-07-02'],
            'precipitation_sum_member01': [0.0, 1.0],
            'precipitation_sum_member02': [10.0, 1.0],
            'precipitation_sum_member03': [20.0, 1.0],
            'temperature_2m_max_member01': [20.0, 25.0],
            'temperature_2m_max_member02': [22.0, 25.0],
            'temperature_2m_max_member03': [24.0, 25.0],
        }
    }
    monkeypatch.setattr(get_forecast.data_utils, 'request_with_retry', lambda *a, **k: FakeResp(ens_payload))
    ens = get_forecast.get_open_meteo_ensemble(38.5, -106.0, days=2)
    assert ens.loc[0, 'precip_ens_mean'] == pytest.approx(10.0)
    assert ens.loc[0, 'precip_ens_std'] > 5.0
    assert ens.loc[0, 'precip_ens_p90'] > 15.0
    assert ens.loc[0, 'temp_ens_std'] > 10.0  # tenths of deg C!
    assert ens.loc[1, 'precip_ens_std'] == pytest.approx(0.0)
