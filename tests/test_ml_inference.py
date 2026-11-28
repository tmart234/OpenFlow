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
    monkeypatch.setattr(ml_inference, '_flow', lambda site, s, e, **kw: flow)
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
    up = dec[:, windowing.DECODER_FEATURES.index('upstream_flow')]
    last_up = x['encoder_input'][0, -1, windowing.ENCODER_FEATURES.index('upstream_flow')]
    chg = dec[:, windowing.DECODER_FEATURES.index('upstream_change')]
    assert chg[0] == pytest.approx(up[0] - last_up, rel=1e-4) and (chg[1:] == 0).all()
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


def test_predict_adds_the_forecast_range_for_a_band_model(fake_sources):
    pytest.importorskip('tensorflow')
    import model as model_mod
    art = _artifacts()
    art['training_config'] = dict(art.get('training_config') or {}, quantiles=list(model_mod.DEFAULT_QUANTILES))
    art['model'] = model_mod.build_encoder_decoder(
        num_stations=1, num_basins=1, encoder_features=len(windowing.ENCODER_FEATURES),
        decoder_features=len(windowing.DECODER_FEATURES),
        target_features=len(windowing.TARGET_FEATURES), encoder_days=60, decoder_days=14,
        lstm_units=4, embedding_dim=2, dropout=0.0, recurrent_dropout=0.0, station_dropout=0.0,
        quantiles=model_mod.DEFAULT_QUANTILES)
    prep = ml_inference.prepare_inputs(SITE, art, today=TODAY)
    daily = ml_inference.predict(art, prep)
    cols = ml_inference.band_columns(model_mod.DEFAULT_QUANTILES)
    assert cols == ['min_q05_cfs', 'min_q25_cfs', 'min_q75_cfs', 'min_q95_cfs',
                    'max_q05_cfs', 'max_q25_cfs', 'max_q75_cfs', 'max_q95_cfs']
    assert list(daily.columns) == ['Date', 'min_cfs', 'max_cfs'] + cols
    for side in ('min', 'max'):
        q = daily[[c for c in cols if c.startswith(side)]].to_numpy()
        assert (np.diff(q, axis=1) >= -1e-6).all()          # quantiles never cross
        assert (q > 0).all()


def test_flow_is_topped_up_with_nwps_days_after_the_archive(monkeypatch):
    from data import get_CODWR_flow
    archive = pd.DataFrame({'Date': ['2026-10-31', '2026-11-01'], 'Min Flow': [800.0, 850.0],
                            'Max Flow': [900.0, 950.0]})
    monkeypatch.setattr(get_CODWR_flow, 'get_historical_data', lambda *a: archive)
    obs_t = pd.date_range('2026-11-01T07:00Z', '2026-11-02T19:00Z', freq='1h')
    flows = np.linspace(900, 3300, len(obs_t))
    payload = {'observed': {'secondaryUnits': 'kcfs', 'data': [
        {'validTime': t.strftime('%Y-%m-%dT%H:%M:%SZ'), 'primary': 1.0, 'secondary': q / 1000}
        for t, q in zip(obs_t, flows)]}}
    monkeypatch.setattr(nwps, 'get_stageflow', lambda lid: payload)
    df = ml_inference._flow('DWR:CONMOGCO', pd.Timestamp('2026-10-01'), TODAY, nws_lid='MOGC2',
                            tz='America/Denver')
    assert list(df['Date']) == ['2026-10-31', '2026-11-01', '2026-11-02']
    # DWR training data is one daily mean per day (Min == Max); the top-up matches.
    today = obs_t[obs_t.tz_convert('America/Denver').date == pd.Timestamp('2026-11-02').date()]
    want = flows[-len(today):].mean()
    assert df.iloc[-1]['Max Flow'] == pytest.approx(want, rel=0.01) == df.iloc[-1]['Min Flow']
    assert df.iloc[1]['Max Flow'] == 950.0                             # archive wins where present


def test_gridmet_point_weather_is_in_training_units(monkeypatch):
    dates = pd.DatetimeIndex(['2026-10-30', '2026-10-31'])
    la, lo = np.array([38.80, 38.75]), np.array([-109.25, -109.20])

    def fake(key, *a, **k):
        v = {'tmmn': 273.15 + 1.5, 'tmmx': 273.15 + 12.0, 'pr': 4.0}[key]
        return dates, la, lo, np.full((2, 2, 2), v)
    monkeypatch.setattr(get_basin_precip, '_gridmet_subset', fake)
    w = ml_inference.gridmet_point_weather(38.79, -109.2, dates[0], dates[-1])
    assert list(w.columns) == ['Date', 'TMIN', 'TMAX', 'precipitation']
    assert w['TMIN'].iloc[0] == pytest.approx(15.0) and w['TMAX'].iloc[0] == pytest.approx(120.0)
    assert w['precipitation'].iloc[0] == 4.0


def test_verified_skill_reads_the_station_from_the_quality_report():
    art = {'quality': {'stations': {SITE: {
        'windows': 784, 'issued': ['2024-08-01', '2026-10-17'],
        'error_pct': {'model': [4.0 + d for d in range(14)], 'persistence': [3.0 + 2 * d for d in range(14)]},
        'rises': {'model': {'peak_ratio': 0.85}}}}}}
    v = ml_inference.verified_skill(art, SITE)
    assert v['model_error_pct'] == [4.0, 6.0, 10.0] and v['persistence_error_pct'] == [3.0, 7.0, 15.0]
    assert v['rise_peak_ratio'] == 0.85 and v['windows'] == 784
    assert ml_inference.verified_skill(art, 'USGS:OTHER') is None
    assert ml_inference.verified_skill({'quality': None}, SITE) is None


def test_splice_keeps_the_training_scale_and_rescales_nwps_days():
    days = pd.date_range('2026-10-01', periods=20)
    archive = pd.Series(50.0, index=days[:15])                   # DWR: half of NWPS
    nwps_days = pd.Series(100.0, index=days)                     # NWPS obs + forecast
    nwps_days.iloc[-3:] = 300.0                                  # RFC forecast rise
    out = ml_inference.splice_on_training_scale(archive, nwps_days)
    assert (out[days[:15]] == 50.0).all()                        # archive wins where present
    assert out[days[15]] == pytest.approx(50.0)                  # NWPS rescaled by 0.5
    assert out[days[-1]] == pytest.approx(150.0)                 # the forecast rise keeps its shape
    assert ml_inference.splice_on_training_scale(pd.Series(dtype=float), nwps_days).equals(nwps_days)


def test_flow_rescales_nwps_topup_when_rating_curve_offset_exists(monkeypatch):
    """Regression for Conejos (DWR:CONMOGCO vs NWPS MOGC2 reading +48% higher)."""
    from data import get_CODWR_flow
    days = pd.date_range('2026-10-25', '2026-11-01')
    archive = pd.DataFrame({'Date': days.strftime('%Y-%m-%d'),
                            'Min Flow': [400.0] * len(days),
                            'Max Flow': [400.0] * len(days)})
    monkeypatch.setattr(get_CODWR_flow, 'get_historical_data', lambda *a: archive)
    # NWPS reads 600 cfs (+50% shift) across all overlap days and on 2026-11-02.
    obs_t = pd.date_range('2026-10-27T12:00Z', '2026-11-02T19:00Z', freq='6h')
    payload = {'observed': {'secondaryUnits': 'kcfs', 'data': [
        {'validTime': t.strftime('%Y-%m-%dT%H:%M:%SZ'), 'primary': 1.0, 'secondary': 0.600}
        for t in obs_t]}}
    monkeypatch.setattr(nwps, 'get_stageflow', lambda lid: payload)
    df = ml_inference._flow('DWR:CONMOGCO', pd.Timestamp('2026-10-01'), TODAY, nws_lid='MOGC2',
                            tz='America/Denver')
    assert df.iloc[-1]['Date'] == '2026-11-02'
    assert df.iloc[-1]['Min Flow'] == pytest.approx(400.0, rel=0.01)
    assert df.iloc[-1]['Max Flow'] == pytest.approx(400.0, rel=0.01)


def test_predict_swaps_quantile_bands_when_min_exceeds_max():
    """Swapping min_cfs/max_cfs when lo > hi must swap min_q*/max_q* too so quantiles never cross."""
    art = _artifacts()
    art['training_config'] = dict(art['training_config'], quantiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    # Synthetic model whose Min Flow output exceeds Max Flow on day 0.
    class FakeModel:
        def predict(self, x, verbose=0):
            pt = np.zeros((1, 14, 2), dtype=np.float32)
            pt[0, 0, 0] = 2.0  # Min Flow higher than Max Flow!
            pt[0, 0, 1] = 1.0
            band = np.zeros((1, 14, 2, 5), dtype=np.float32)
            for j in range(2):
                for k, off in enumerate([-0.4, -0.2, 0.0, 0.2, 0.4]):
                    band[0, :, j, k] = pt[0, :, j] + off
            return {'persistence_plus_delta': pt, 'flow_quantiles': band}

    art['model'] = FakeModel()
    prep = {'inputs': {}, 'dates': pd.date_range(TODAY + pd.Timedelta(days=1), periods=14, name='Date')}
    df = ml_inference.predict(art, prep)
    # Day 0 was swapped: min_cfs is the lower branch, and min_q05 <= min_cfs <= min_q95, same for max.
    assert df.loc[0, 'min_cfs'] < df.loc[0, 'max_cfs']
    assert df.loc[0, 'min_q05_cfs'] <= df.loc[0, 'min_cfs'] <= df.loc[0, 'min_q95_cfs']
    assert df.loc[0, 'max_q05_cfs'] <= df.loc[0, 'max_cfs'] <= df.loc[0, 'max_q95_cfs']


def test_coords_falls_back_to_cached_coordinates_on_outage(monkeypatch):
    monkeypatch.setattr(ml_inference, 'get_usgs_coordinates', lambda sid: None)
    c = ml_inference._coords('USGS:08276500')
    assert c['lat'] == pytest.approx(36.3200333)
    with pytest.raises(ml_inference.InferenceError):
        ml_inference._coords('USGS:99999999')


def test_mirror_disabled_sources_guards_constant_zero_scalers(monkeypatch):
    import os
    monkeypatch.delenv('OPENFLOW_DISABLE_DROUGHT', raising=False)
    scalers = {'drought_index': {'mean': 0.0, 'scale': 1.0, 'transform': 'identity'}}
    ml_inference._mirror_disabled_sources({'disabled_sources': []}, scalers)
    assert os.environ.get('OPENFLOW_DISABLE_DROUGHT') == '1'
    monkeypatch.delenv('OPENFLOW_DISABLE_DROUGHT', raising=False)
    live_scalers = {'drought_index': {'mean': 120.0, 'scale': 95.0, 'transform': 'identity'}}
    ml_inference._mirror_disabled_sources({'disabled_sources': []}, live_scalers)
    assert os.environ.get('OPENFLOW_DISABLE_DROUGHT') is None


def test_prepare_inputs_mass_balance_hold_upstream_change(fake_sources):
    art = _artifacts()
    art['training_config'] = dict(art['training_config'], upstream_change_mode='mass_balance_hold')
    prep = ml_inference.prepare_inputs(SITE, art, today=TODAY)
    dec = prep['inputs']['decoder_input'][0]
    chg = dec[:, windowing.DECODER_FEATURES.index('upstream_change')]
    # Upstream rises from 200 -> 250 cfs (+50 cfs) on day 1, held across days 2..14.
    assert chg[0] > 0
    np.testing.assert_allclose(chg[1:], chg[0], rtol=1e-6)


def test_prepare_inputs_forwards_weather_source_and_defaults_legacy_noaa(fake_sources, monkeypatch):
    seen = []
    orig = combine_data.fetch_and_process_data

    def spy(*a, **k):
        seen.append(k.get('weather_source_override'))
        return orig(*a, **k)

    monkeypatch.setattr(combine_data, 'fetch_and_process_data', spy)
    ml_inference.prepare_inputs(SITE, _artifacts(), today=TODAY)
    art_grid = _artifacts()
    art_grid['training_config'] = dict(art_grid['training_config'], weather_source='gridmet')
    ml_inference.prepare_inputs(SITE, art_grid, today=TODAY)
    assert seen == ['noaa', 'gridmet']


def test_prepare_inputs_supports_legacy_23_15_schema(fake_sources):
    """Published releases like model-219 (23 enc / 15 dec) must run unchanged."""
    legacy_enc = ['Min Flow', 'Max Flow', 'TMIN', 'TMAX', 'precipitation',
                  'SWE', 'soil_moisture', 'sm_observed', 'drought_index',
                  'reservoir_storage', 'reservoir_release', 'reservoir_observed',
                  'upstream_flow', 'upstream_observed',
                  *windowing.BASIN_PRECIP_FEATURES, 'basin_precip_observed',
                  'doy_sin', 'doy_cos']
    legacy_dec = ['TMIN', 'TMAX', 'precipitation',
                  'upstream_flow', 'upstream_known', 'upstream_change',
                  *windowing.BASIN_PRECIP_FEATURES, 'basin_precip_known',
                  'doy_sin', 'doy_cos']
    art = _artifacts()
    art['training_config'] = dict(art['training_config'],
                                  encoder_features=legacy_enc,
                                  decoder_features=legacy_dec)
    prep = ml_inference.prepare_inputs(SITE, art, today=TODAY)
    assert prep['inputs']['encoder_input'].shape == (1, 60, 23)
    assert prep['inputs']['decoder_input'].shape == (1, 14, 15)
    assert np.isfinite(prep['inputs']['encoder_input']).all()
    assert np.isfinite(prep['inputs']['decoder_input']).all()


def test_prepare_inputs_populates_phase1_to_3_decoder_features(fake_sources, monkeypatch):
    from data import get_nwm_prior
    art = _artifacts()
    for c in (*windowing.ENERGY_FEATURES, *windowing.INTENSITY_FEATURES,
              *windowing.HYDRO_SM_FEATURES, *windowing.GRIDDED_SWE_FEATURES, 'prior_flow'):
        art['scalers'][c] = {
            'mean': 2.0, 'scale': 1.5,
            'transform': 'log1p' if c in ('precip_max1h', 'precip_max3h', 'prior_flow') else 'identity',
        }
    wx_days = pd.date_range(TODAY, TODAY + pd.Timedelta(days=14), name='Date')
    monkeypatch.setattr(ml_inference, 'forecast_weather', lambda lat, lon, tz: pd.DataFrame({
        'TMIN': 0.0, 'TMAX': 100.0, 'precipitation': 5.0,
        'srad': 22.0, 'pet': 4.5, 'vpd': 1.3,
        'precip_max1h': 3.0, 'precip_max3h': 5.0, 'precip_wet_hours': 2.0,
        'precip_ens_std': 2.0, 'precip_ens_p90': 9.0, 'temp_ens_std': 15.0,
    }, index=wx_days))
    prior_days = pd.date_range(TODAY - pd.Timedelta(days=5), TODAY + pd.Timedelta(days=10), name='Date')
    monkeypatch.setattr(get_nwm_prior, 'fetch_live_prior',
                        lambda *a, **k: pd.DataFrame({'Date': prior_days, 'prior_flow': np.linspace(300, 600, len(prior_days))}))

    prep = ml_inference.prepare_inputs(SITE, art, today=TODAY)
    dec = prep['inputs']['decoder_input'][0]
    D = windowing.DECODER_FEATURES
    assert (dec[:, D.index('energy_known')] == 1.0).all()
    assert (dec[:7, D.index('precip_intensity_known')] == 1.0).all()
    assert (dec[7:, D.index('precip_intensity_known')] == 0.0).all()
    assert (dec[:, D.index('ens_spread_known')] == 1.0).all()
    assert (dec[:, D.index('forcing_is_forecast')] == 1.0).all()
    assert dec[0, D.index('lead_day_norm')] == pytest.approx(1.0 / 14.0)
    assert (dec[:10, D.index('prior_known')] == 1.0).all()
    assert (dec[10:, D.index('prior_known')] == 0.0).all()
    assert dec[5, D.index('prior_change')] > 0.0


def test_prepare_inputs_zeros_real_forecast_flags_when_trained_without_reforecasts(fake_sources, monkeypatch):
    art = _artifacts()
    art['training_config'] = dict(art['training_config'], real_forecast_training=False)
    wx_days = pd.date_range(TODAY, TODAY + pd.Timedelta(days=14), name='Date')
    monkeypatch.setattr(ml_inference, 'forecast_weather', lambda lat, lon, tz: pd.DataFrame({
        'TMIN': 0.0, 'TMAX': 100.0, 'precipitation': 5.0,
        'precip_ens_std': 3.0, 'precip_ens_p90': 12.0, 'temp_ens_std': 20.0,
    }, index=wx_days))

    prep = ml_inference.prepare_inputs(SITE, art, today=TODAY)
    dec = prep['inputs']['decoder_input'][0]
    D = windowing.DECODER_FEATURES
    assert (dec[:, D.index('forcing_is_forecast')] == 0.0).all()
    assert (dec[:, D.index('ens_spread_known')] == 0.0).all()
    for col in windowing.ENSEMBLE_SPREAD_FEATURES:
        assert (dec[:, D.index(col)] == 0.0).all()


def test_prepare_inputs_restores_disabled_source_env_vars(fake_sources, monkeypatch):
    import os
    monkeypatch.delenv('OPENFLOW_DISABLE_SMAP', raising=False)
    monkeypatch.delenv('OPENFLOW_DISABLE_DROUGHT', raising=False)
    monkeypatch.delenv('OPENFLOW_DISABLE_RESERVOIR', raising=False)

    seen_env = {}
    orig = combine_data.fetch_and_process_data

    def spy(*a, **k):
        for name in ('SMAP', 'DROUGHT', 'RESERVOIR'):
            seen_env[name] = os.environ.get(f'OPENFLOW_DISABLE_{name}')
        return orig(*a, **k)

    monkeypatch.setattr(combine_data, 'fetch_and_process_data', spy)
    art = _artifacts()
    art['training_config'] = dict(art['training_config'], disabled_sources=['SMAP', 'DROUGHT', 'RESERVOIR'])
    ml_inference.prepare_inputs(SITE, art, today=TODAY)
    assert seen_env == {'SMAP': '1', 'DROUGHT': '1', 'RESERVOIR': '1'}
    assert 'OPENFLOW_DISABLE_SMAP' not in os.environ
    assert 'OPENFLOW_DISABLE_DROUGHT' not in os.environ
    assert 'OPENFLOW_DISABLE_RESERVOIR' not in os.environ
