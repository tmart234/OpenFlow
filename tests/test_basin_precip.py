import io

import numpy as np
import pandas as pd
import pytest

from data import basin_zones, get_basin_precip as gbp

RING = [[-106.0, 36.0], [-105.0, 36.0], [-105.0, 37.0], [-106.0, 37.0], [-106.0, 36.0]]
FAR = [[-107.0, 37.0], [-106.0, 37.0], [-106.0, 38.0], [-107.0, 38.0], [-107.0, 37.0]]
CFG = {'zones': {'local': {'area_km2': 10000, 'rings': [RING]},
                 'mid': {'area_km2': 0, 'rings': []},
                 'far': {'area_km2': 10000, 'rings': [FAR]}},
       'ungauged': {'area_km2': 10000, 'rings': [RING]},
       'points': [{'lat': 36.5, 'lon': -105.5, 'zone': 'local', 'ungauged': True, 'w': 1.0, 'nws': 'A/1,1'},
                  {'lat': 37.5, 'lon': -106.5, 'zone': 'far', 'ungauged': False, 'w': 1.0, 'nws': 'A/2,2'}]}


def _netcdf3(var, days, lat, lon, raw, scale=0.1, offset=0.0):
    from scipy.io import netcdf_file
    buf = io.BytesIO()
    f = netcdf_file(buf, 'w')
    f.createDimension('day', len(days))
    f.createDimension('lat', len(lat))
    f.createDimension('lon', len(lon))
    for name, vals in (('day', days), ('lat', lat), ('lon', lon)):
        v = f.createVariable(name, 'f8', (name,))
        v[:] = vals
    v = f.createVariable(var, 'i2', ('day', 'lat', 'lon'))
    v[:] = raw
    v.scale_factor = scale
    v.add_offset = offset
    v._FillValue = np.int16(32767)
    f.flush()
    content = buf.getvalue()
    f.close()
    return content


def test_parse_gridmet_netcdf_applies_scaling_and_fill():
    days = np.array([46313.0, 46314.0])
    lat, lon = np.array([36.75, 36.25]), np.array([-105.75, -105.25])
    raw = np.array([[[10, 20], [30, 32767]], [[0, 5], [5, 5]]], dtype=np.int16)
    dates, la, lo, vals = gbp.parse_gridmet_netcdf(_netcdf3('precipitation_amount', days, lat, lon, raw),
                                                   'precipitation_amount')
    assert dates[0] == pd.Timestamp('2026-10-20')
    assert vals.shape == (2, 2, 2)
    assert vals[0, 0, 1] == pytest.approx(2.0)
    assert np.isnan(vals[0, 1, 1])


def test_gridmet_zone_daily_partitions_rain_and_snow(monkeypatch):
    lat = np.array([37.5, 36.5])                   # far row, local row
    lon = np.array([-106.5, -105.5])
    dates = pd.DatetimeIndex(['2026-10-21', '2026-10-22'])
    pr = np.array([[[10.0, 0.0], [0.0, 20.0]], [[4.0, 0.0], [0.0, 0.0]]])
    # far cell is cold (-5 C: all snow), local cell warm (+10 C: all rain)
    tk = np.array([[[268.15, 0], [0, 283.15]], [[268.15, 0], [0, 283.15]]])

    def fake(var_key, *a, **k):
        return dates, lat, lon, (pr if var_key == 'pr' else tk)
    monkeypatch.setattr(gbp, '_gridmet_subset', fake)
    out = gbp.gridmet_zone_daily(CFG, '2026-10-21', '2026-10-22')
    assert out.loc['2026-10-21', 'rain_local'] == pytest.approx(20.0)
    assert out.loc['2026-10-21', 'snow_local'] == pytest.approx(0.0)
    assert out.loc['2026-10-21', 'snow_far'] == pytest.approx(10.0)
    assert out.loc['2026-10-21', 'rain_far'] == pytest.approx(0.0)
    assert (out['rain_mid'] == 0).all()             # empty zone
    assert out.loc['2026-10-21', 'rain_ungauged'] == pytest.approx(20.0)
    assert out.loc['2026-10-21', 'tmean_far'] == pytest.approx(-5.0)


def test_get_zone_precip_daily_degrades_to_empty(monkeypatch):
    assert gbp.get_zone_precip_daily('USGS:NOPE', '2026-01-01', '2026-01-02', zones={}).empty
    monkeypatch.setattr(gbp, 'gridmet_zone_daily', lambda *a: (_ for _ in ()).throw(RuntimeError('down')))
    out = gbp.get_zone_precip_daily('X', '2026-01-01', '2026-01-02', zones={'X': CFG})
    assert out.empty and list(out.columns) == ['Date'] + gbp.FEATURE_COLUMNS
    monkeypatch.setenv('OPENFLOW_DISABLE_BASIN_PRECIP', '1')
    assert gbp.get_zone_precip_daily('X', '2026-01-01', '2026-01-02', zones={'X': CFG}).empty


def test_hrap_projection_round_trips():
    lon, lat = np.array([-105.75, -120.0, -80.0]), np.array([36.3, 45.0, 30.0])
    x, y = gbp.lonlat_to_hrap(lon, lat)
    lo, la = gbp.hrap_to_lonlat(x, y)
    np.testing.assert_allclose(lo, lon, atol=1e-9)
    np.testing.assert_allclose(la, lat, atol=1e-9)


def test_stage4_zone_precip_from_synthetic_file():
    h5py = pytest.importorskip('h5py')
    cx, cy = gbp.lonlat_to_hrap(np.array([-107.5, -104.5, -107.5, -104.5]),
                                np.array([35.5, 35.5, 38.5, 38.5]))
    xs = np.arange(cx.min(), cx.max(), 4762.5)
    ys = np.arange(cy.min(), cy.max(), 4762.5)[::-1]
    X, Y = np.meshgrid(xs, ys)
    lon, lat = gbp.hrap_to_lonlat(X, Y)
    obs = np.where(lat < 37.0, 1.0, 2.0).astype('float32')   # inches: local 1", far 2"
    buf = io.BytesIO()
    with h5py.File(buf, 'w') as f:
        f['x'] = xs
        f['y'] = ys
        f['observation'] = obs
    out = gbp.stage4_zone_precip(CFG, buf.getvalue())
    assert out['local'] == pytest.approx(25.4, rel=0.02)
    assert out['far'] == pytest.approx(50.8, rel=0.02)
    assert out['mid'] == 0.0


def test_observed_daily_fills_gap_with_stage4_using_persisted_temperature():
    days = pd.DatetimeIndex(['2026-10-30', '2026-10-31'], name='Date')
    gm = pd.DataFrame({c: [1.0, 2.0] for c in gbp.FEATURE_COLUMNS}, index=days)
    gm['rain_ungauged'] = [1.0, 2.0]
    gm['precip_ungauged'] = [1.0, 2.0]
    for a in gbp.AREAS:
        gm[f'tmean_{a}'] = [5.0, 1.0]               # last day 1 C -> 50 % rain
    calls = []

    def fake_stage4(cfg, end_day):
        calls.append(pd.Timestamp(end_day))
        return {'local': 20.0, 'mid': 0.0, 'far': 10.0, 'ungauged': 8.0}
    out = gbp.observed_daily(CFG, '2026-10-30', '2026-11-03', gridmet=gm, stage4=fake_stage4)
    assert list(out.index) == list(pd.date_range('2026-10-30', '2026-11-02'))
    assert calls == list(pd.date_range('2026-11-02', '2026-11-03'))   # 24 h ending 12Z next day
    row = out.loc['2026-11-01']
    assert row['source'] == 'stage4'
    assert row['rain_local'] == pytest.approx(10.0) and row['snow_local'] == pytest.approx(10.0)
    assert row['rain_ungauged'] == pytest.approx(4.0)
    assert row['window_start'] == pd.Timestamp('2026-11-01T12:00Z')
    assert out.loc['2026-10-31', 'window_start'] == pd.Timestamp('2026-10-31T00:00Z')


def _gridpoint_payload():
    return {'properties': {
        'quantitativePrecipitation': {'values': [
            {'validTime': '2026-11-03T00:00:00+00:00/PT6H', 'value': 6.0},
            {'validTime': '2026-11-03T06:00:00+00:00/PT6H', 'value': 0.0},
            {'validTime': '2026-11-03T12:00:00+00:00/PT6H', 'value': None}]},
        'temperature': {'values': [
            {'validTime': '2026-11-03T00:00:00+00:00/PT3H', 'value': 5.0},
            {'validTime': '2026-11-03T03:00:00+00:00/P1DT2H', 'value': -5.0}]}}}


def test_iso_duration_hours():
    assert gbp.iso_duration_hours('PT6H') == 6
    assert gbp.iso_duration_hours('P1DT2H') == 26
    assert gbp.iso_duration_hours('P7DT13H') == 181
    with pytest.raises(ValueError):
        gbp.iso_duration_hours('six hours')


def test_parse_nws_gridpoint_spreads_qpf_and_holds_temperature():
    df = gbp.parse_nws_gridpoint(_gridpoint_payload())
    assert len(df) == 12
    assert df['precip_mm'].iloc[:6].tolist() == [1.0] * 6
    assert df['precip_mm'].iloc[6:].sum() == 0.0
    assert df['temp_c'].iloc[0] == 5.0 and df['temp_c'].iloc[5] == -5.0


def test_points_hourly_and_daily_zone_partition():
    df = gbp.parse_nws_gridpoint(_gridpoint_payload())
    pts = {0: df, 1: df.assign(precip_mm=df['precip_mm'] * 2)}
    hourly = gbp.points_hourly_zones(CFG, pts)
    # first 3 hours warm (5 C, all rain), next 3 cold (-5 C, all snow)
    assert hourly['rain_local'].iloc[:3].tolist() == [1.0] * 3
    assert hourly['snow_local'].iloc[3:6].tolist() == [1.0] * 3
    assert hourly['rain_ungauged'].iloc[0] == 1.0
    daily = gbp.points_daily_zones(CFG, pts, 'UTC')
    total_local = daily.loc['2026-11-03', 'rain_local'] + daily.loc['2026-11-03', 'snow_local']
    assert total_local == pytest.approx(6.0)
    assert daily.loc['2026-11-03', 'snow_far'] + daily.loc['2026-11-03', 'rain_far'] == pytest.approx(12.0)


def test_inference_daily_assembles_encoder_and_masks_decoder():
    today = pd.Timestamp('2026-11-02')
    obs_days = pd.date_range('2026-10-25', '2026-11-01', name='Date')
    observed = pd.DataFrame({c: 1.0 for c in gbp.FEATURE_COLUMNS}, index=obs_days)
    observed['source'] = 'gridmet'
    hours = pd.date_range('2026-11-02T12:00Z', periods=24 * 10, freq='1h')
    point = pd.DataFrame({'precip_mm': 0.5, 'temp_c': 10.0}, index=hours)
    fpts = {0: point, 1: point}                     # both zones' sample points
    out = gbp.inference_daily('X', today=today, encoder_days=10, decoder_days=14, tz='UTC',
                              zones={'X': CFG}, observed=observed, forecast_points=fpts)
    assert len(out) == 24
    assert out.loc['2026-10-24', gbp.OBSERVED_COLUMN] == 0          # before gridMET rows
    assert out.loc['2026-10-30', 'source'] == 'gridmet'
    assert out.loc['2026-11-02', 'source'] == 'nws'                 # today's forecast hours
    assert out.loc['2026-11-05', 'rain_local'] == pytest.approx(12.0)
    assert out.loc['2026-11-09', gbp.OBSERVED_COLUMN] == 1          # today + 7
    assert out.loc['2026-11-10', gbp.OBSERVED_COLUMN] == 0          # beyond QPF horizon
    assert out.loc['2026-11-10', gbp.FEATURE_COLUMNS].isna().all()


def test_routing_rain_flux_spreads_days_and_never_overwrites_observations():
    grid = pd.date_range('2026-11-01T00:00Z', periods=72, freq='1h')
    observed = pd.DataFrame({'rain_ungauged': [24.0], 'source': ['stage4'],
                             'window_start': [pd.Timestamp('2026-11-01T12:00Z')]},
                            index=pd.DatetimeIndex(['2026-11-01']))
    hours = pd.date_range('2026-11-01T00:00Z', periods=72, freq='1h')
    fpts = {0: pd.DataFrame({'precip_mm': 2.0, 'temp_c': 10.0}, index=hours)}
    flux, info = gbp.routing_rain_flux(CFG, grid, observed, fpts, abstraction_mm_per_day=0)
    rate = info['rate_mm_per_hour']
    assert rate['2026-11-01T12:00Z'] == pytest.approx(1.0)
    assert rate['2026-11-02T11:00Z'] == pytest.approx(1.0)
    assert rate['2026-11-01T03:00Z'] == pytest.approx(0.0)   # before obs: forecast not allowed
    assert rate['2026-11-02T13:00Z'] == pytest.approx(2.0)   # after obs: forecast
    assert info['sources'] == ['stage4', 'nws']
    assert flux[12] == pytest.approx(basin_zones.mm_per_hour_to_cfs(1.0, 10000))


def test_zone_totals():
    days = pd.date_range('2026-10-20', '2026-11-01', name='Date')
    observed = pd.DataFrame({c: 1.0 for c in gbp.FEATURE_COLUMNS}, index=days)
    t = gbp.zone_totals(CFG, observed, None, '2026-11-02', 'UTC', days=7)
    assert t['past']['rain_local'] == 6.0 and t['past_through'] == '2026-11-01'
    assert t['next'] == {}


def test_routing_rain_flux_initial_abstraction_drops_light_rain():
    grid = pd.date_range('2026-11-01T00:00Z', periods=48, freq='1h')
    hours = grid
    precip = np.r_[np.full(24, 3.0 / 24), np.full(24, 30.0 / 24)]   # 3 mm day, then 30 mm day
    fpts = {0: pd.DataFrame({'precip_mm': precip, 'temp_c': 10.0}, index=hours)}
    flux, info = gbp.routing_rain_flux(CFG, grid, None, fpts, abstraction_mm_per_day=5.0)
    assert (flux[:24] == 0).all()                                     # light day: no runoff
    full = basin_zones.mm_per_hour_to_cfs(30.0 / 24, 10000)
    assert flux[30] == pytest.approx(full * (1 - 5.0 / 30.0))         # storm day: 25 of 30 mm
    assert info['rate_mm_per_hour'].iloc[0] == pytest.approx(3.0 / 24)   # raw rate reported
