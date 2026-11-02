"""
ML inference assembly (ml_inference.prepare_inputs) against synthetic data
sources, plus an end-to-end predict with a tiny model when tensorflow is
installed. Checks the things that silently break forecasts: window shapes,
the NOAA-lag bridge, the upstream hold rule, the 7-day basin-precip mask,
station/basin index lookup, and scaling/inversion.
"""

import numpy as np
import pandas as pd
import pytest

import combine_data
import ml_inference
import windowing
from data import get_basin_precip, get_upstream, nwps

TODAY = pd.Timestamp('2026-11-02')
SITE = 'USGS:TEST'


def _artifacts():
    numeric = ['Min Flow', 'Max Flow', 'TMIN', 'TMAX', 'precipitation', 'SWE', 'soil_moisture',
               'drought_index', 'reservoir_storage', 'reservoir_release', 'upstream_flow',
               *windowing.BASIN_PRECIP_FEATURES]
    logs = {'Min Flow', 'Max Flow', 'precipitation', 'upstream_flow', *windowing.BASIN_PRECIP_FEATURES}
    scalers = {c: {'mean': 1.0, 'scale': 2.0, 'transform': 'log1p' if c in logs else 'identity'}
               for c in numeric}
    return {
        'scalers': scalers,
        'station_index': {SITE: 1},
        'basin_index': {'14010001': 1},
        'training_config': {'encoder_days': 60, 'decoder_days': 14,
                            'encoder_features': windowing.ENCODER_FEATURES,
                            'decoder_features': windowing.DECODER_FEATURES,
                            'target_features': windowing.TARGET_FEATURES,
                            'upstream_known_days': {SITE: 3}, 'basin_precip_known_days': 7,
                            'num_stations': 1, 'num_basins': 1},
        'registry': {SITE: {'timezone': 'UTC', 'forecast_days': 2, 'upstream': [
            {'site_id': 'USGS:UP', 'nws_lid': 'UPL', 'lag_days': 1, 'weight': 1.0}]}},
        'version': 'test-model',
    }


@pytest.fixture
def fake_sources(monkeypatch):
    days = pd.date_range(TODAY - pd.Timedelta(days=95), TODAY)
    rng = np.random.default_rng(0)
    flow = pd.DataFrame({'Date': days, 'Min Flow': 300 + rng.random(len(days)) * 50,
                         'Max Flow': 400 + rng.random(len(days)) * 50})
    noaa_days = days[days <= TODAY - pd.Timedelta(days=3)]          # 3-day NOAA lag
    bp_days = days[days <= TODAY - pd.Timedelta(days=2)]            # gridMET lag
    fetched = {
        'noaa_data': pd.DataFrame({'Date': noaa_days, 'TMIN': -20.0, 'TMAX': 120.0,
                                   'precipitation': 0.5}),
        'swe_data': pd.DataFrame(columns=['Date', 'SWE']),
        'sm_data': pd.DataFrame(columns=['Date', 'soil_moisture']),
        'drought_data': pd.DataFrame(columns=['Date', 'drought_index']),
        'reservoir_data': pd.DataFrame(columns=['Date', 'reservoir_storage', 'reservoir_release']),
        'huc8': '14010001',
        'upstream_data': pd.DataFrame({'Date': days, 'upstream_flow': 200.0}),
        'upstream_known_days': 3,
        'basin_precip_data': pd.DataFrame({'Date': bp_days, **{c: 1.0 for c in
                                                                get_basin_precip.FEATURE_COLUMNS}}),
    }
    monkeypatch.setattr(ml_inference, '_coords', lambda site: {'lat': 38.0, 'lon': -106.0})
    monkeypatch.setattr(ml_inference, '_flow', lambda site, s, e: flow)
    monkeypatch.setattr(combine_data, 'fetch_and_process_data', lambda *a, **k: fetched)
    wx_days = pd.date_range(TODAY, TODAY + pd.Timedelta(days=6), name='Date')
    monkeypatch.setattr(ml_inference, 'forecast_weather', lambda lat, lon, tz: pd.DataFrame(
        {'TMIN': 0.0, 'TMAX': 100.0, 'precipitation': 2.0}, index=wx_days))
    zone_days = pd.date_range(TODAY - pd.Timedelta(days=59), TODAY + pd.Timedelta(days=14), name='Date')
    zones = pd.DataFrame({c: 3.0 for c in get_basin_precip.FEATURE_COLUMNS}, index=zone_days)
    zones['basin_precip_observed'] = (zones.index <= TODAY + pd.Timedelta(days=7)).astype(int)
    monkeypatch.setattr(get_basin_precip, 'inference_daily', lambda *a, **k: zones)
    monkeypatch.setattr(get_upstream, 'fetch_daily_mean',
                        lambda site, s, e: pd.Series(250.0, index=days))
    monkeypatch.setattr(nwps, 'get_stageflow', lambda lid: None)
    return {'flow': flow}


def test_prepare_inputs_builds_model_ready_windows(fake_sources):
    prep = ml_inference.prepare_inputs(SITE, _artifacts(), today=TODAY)
    x = prep['inputs']
    assert x['encoder_input'].shape == (1, 60, len(windowing.ENCODER_FEATURES))
    assert x['decoder_input'].shape == (1, 14, len(windowing.DECODER_FEATURES))
    assert np.isfinite(x['encoder_input']).all() and np.isfinite(x['decoder_input']).all()
    assert x['station_input'][0] == 1 and x['basin_input'][0] == 1
    # NOAA lag bridged so the encoder ends today and the forecast starts tomorrow.
    assert prep['anchor'] == TODAY
    assert prep['dates'][0] == TODAY + pd.Timedelta(days=1)
    assert prep['info']['bridged_weather_days'] == ['2026-10-31', '2026-11-01', '2026-11-02']
    dec = x['decoder_input'][0]
    # Upstream: lag 1 day -> only tomorrow's routed value is known (no RFC forecast).
    up_known = dec[:, windowing.DECODER_FEATURES.index('upstream_known')]
    np.testing.assert_array_equal(up_known, [1] + [0] * 13)
    # Basin precip: NWS QPF covers through today + 7 -> 7 known decoder days.
    bp_known = dec[:, windowing.DECODER_FEATURES.index('basin_precip_known')]
    np.testing.assert_array_equal(bp_known, [1] * 7 + [0] * 7)
    rain_col = windowing.DECODER_FEATURES.index('rain_local')
    assert dec[0, rain_col] == pytest.approx((np.log1p(3.0) - 1.0) / 2.0)
    assert (dec[7:, rain_col] == 0).all()
    # Persistence anchor = last encoder day's scaled flow.
    t_idx = [windowing.ENCODER_FEATURES.index(c) for c in windowing.TARGET_FEATURES]
    np.testing.assert_allclose(x['persistence_input'][0], x['encoder_input'][0, -1, t_idx])


def test_prepare_inputs_unknown_station_uses_basin_fallback(fake_sources):
    art = _artifacts()
    art['station_index'] = {}
    prep = ml_inference.prepare_inputs(SITE, art, today=TODAY)
    assert prep['inputs']['station_input'][0] == 0
    assert prep['info']['trained_station'] is False


def test_load_artifacts_rejects_non_model_dir(tmp_path):
    with pytest.raises(ml_inference.InferenceError):
        ml_inference.load_artifacts(str(tmp_path), with_model=False)


def test_predict_end_to_end_with_tiny_model(fake_sources):
    pytest.importorskip('tensorflow')
    import model as model_mod
    art = _artifacts()
    art['model'] = model_mod.build_encoder_decoder(
        num_stations=1, num_basins=1, encoder_features=len(windowing.ENCODER_FEATURES),
        decoder_features=len(windowing.DECODER_FEATURES),
        target_features=len(windowing.TARGET_FEATURES), encoder_days=60, decoder_days=14,
        lstm_units=4, embedding_dim=2, dropout=0.0, recurrent_dropout=0.0, station_dropout=0.0)
    prep = ml_inference.prepare_inputs(SITE, art, today=TODAY)
    daily = ml_inference.predict(art, prep)
    assert list(daily.columns) == ['Date', 'min_cfs', 'max_cfs']
    assert len(daily) == 14 and (daily['min_cfs'] <= daily['max_cfs']).all()
    assert (daily['min_cfs'] > 0).all()
