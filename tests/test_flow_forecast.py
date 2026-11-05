"""
End-to-end test of the upstream-routing forecast CLI against synthetic NWPS
payloads: a known routing (lags, attenuation, weights) generates the target
gauge's history, upstream gauges carry an RFC forecast of a new flood pulse,
and the CLI must calibrate on the history and put the forecast peak where
the known routing says it will be.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

import flow_forecast
import routing
from data import get_basin_precip, nwps

NOW = pd.Timestamp('2026-11-02T17:00Z')
GRID = pd.date_range(NOW - pd.Timedelta(days=30), NOW + pd.Timedelta(days=9), freq='1h')
INOW = GRID.get_loc(NOW)


def _pulse(center, height, width, base):
    t = (GRID - center) / pd.Timedelta(hours=1)
    return base + height * np.exp(-0.5 * (np.asarray(t, float) / width) ** 2)


def _block(times, flows, issued=None, every=1):
    return {'issuedTime': issued or '2026-11-02T17:30:00Z', 'secondaryUnits': 'kcfs',
            'data': [{'validTime': t.strftime('%Y-%m-%dT%H:%M:%SZ'), 'primary': 5.0,
                      'secondary': q / 1000.0} for t, q in list(zip(times, flows))[::every]]}


def _scenario(quiet=False):
    if quiet:
        a = np.full(len(GRID), 400.0)
        b = np.full(len(GRID), 300.0)
        a[INOW + 1:] += 2500 * np.exp(-0.5 * ((np.arange(len(GRID) - INOW - 1) - 40) / 15.0) ** 2)
    else:
        a = (_pulse(NOW - pd.Timedelta(days=10), 3000, 20, 200)
             + _pulse(NOW + pd.Timedelta(hours=60), 3000, 18, 0))
        b = (_pulse(NOW - pd.Timedelta(days=12), 2000, 30, 300)
             + _pulse(NOW + pd.Timedelta(hours=20), 2000, 12, 0))
    truth = 150 + 1.3 * routing.route(a, 30, 24) + 0.7 * routing.route(b, 48, 72)
    obs = slice(0, INOW + 1)
    fc_times = GRID[INOW + 1:INOW + 1 + 5 * 24]
    payloads = {
        'TJBN5': {'observed': _block(GRID[obs], truth[obs]), 'forecast': {'data': []}},
        'ALAC2': {'observed': _block(GRID[obs], a[obs]),
                  'forecast': _block(fc_times, a[INOW + 1:INOW + 1 + 5 * 24],
                                     issued='2026-11-02T14:15:00Z', every=6)},
        'MOGC2': {'observed': _block(GRID[obs], b[obs]),
                  'forecast': _block(fc_times, b[INOW + 1:INOW + 1 + 5 * 24],
                                     issued='2026-11-02T14:15:00Z', every=6)},
    }
    return payloads, truth


@pytest.fixture
def fake_nwps(monkeypatch):
    def install(payloads, observed=None, forecast_points=None):
        monkeypatch.setattr(nwps, 'get_stageflow', lambda lid: payloads.get(lid))
        # Basin rain: no network; empty unless a test supplies data.
        monkeypatch.setattr(get_basin_precip, 'observed_daily',
                            lambda cfg, start, today, **kw: observed if observed is not None
                            else pd.DataFrame())
        monkeypatch.setattr(get_basin_precip, 'nws_points_hourly',
                            lambda cfg: forecast_points or {})
        import backtest_routing
        monkeypatch.setattr(backtest_routing, 'usgs_hourly', lambda site, a, b: pd.Series(dtype=float))
        monkeypatch.setattr(nwps, 'get_gauge', lambda lid: {'flood': {'categories': {
            'action': {'flow': 7750}, 'minor': {'flow': 9800}}}})
    return install


def test_cli_calibrates_and_places_the_peak_where_routing_says(tmp_path, fake_nwps):
    payloads, truth = _scenario()
    fake_nwps(payloads)
    s = flow_forecast.run('taos-box', days=7, out_dir=str(tmp_path))

    assert s['routing']['mode'] == 'calibrated'
    best = s['routing']['fits'][0]
    assert abs(best['lags_hours'][0] - 30) <= 6
    true_peak_t = GRID[INOW + 1 + int(np.argmax(truth[INOW + 1:]))]
    got_peak_t = pd.Timestamp(s['peak']['time_utc'])
    assert abs((got_peak_t - true_peak_t) / pd.Timedelta(hours=1)) <= 6
    assert s['peak']['cfs'] == pytest.approx(truth[INOW + 1:].max(), rel=0.1)
    assert s['peak']['band_high_cfs'] >= s['peak']['cfs']

    assert len(s['daily']) == 7
    assert s['daily'][0]['date'] == '2026-11-02'
    assert s['flood_thresholds_cfs'] == {'action': 7750.0, 'minor': 9800.0}
    assert {u['nws_lid'] for u in s['upstream']} == {'ALAC2', 'MOGC2'}

    for key in ('csv', 'json', 'png'):
        assert os.path.getsize(s['files'][key]) > 0
    csv = pd.read_csv(s['files']['csv'])
    assert {'time_utc', 'time_local', 'observed_cfs', 'forecast_cfs',
            'band_low_cfs', 'band_high_cfs', 'nws_official_cfs'} <= set(csv.columns)
    assert csv['forecast_cfs'].notna().any() and csv['observed_cfs'].notna().any()
    with open(s['files']['json']) as f:
        assert json.load(f)['site_id'] == 'USGS:08276500'


def test_cli_falls_back_to_registry_defaults_when_window_is_quiet(tmp_path, fake_nwps):
    payloads, _ = _scenario(quiet=True)
    fake_nwps(payloads)
    s = flow_forecast.run('taos-box', days=5, out_dir=str(tmp_path), make_chart=False)
    assert s['routing']['mode'] == 'registry-default'
    assert all(f['source'] == 'default' for f in s['routing']['fits'])
    assert 'png' not in s['files']
    # The upstream pulse still arrives: forecast peak well above the base flow.
    assert s['peak']['cfs'] > 1.5 * s['latest_observation']['cfs']


def test_cli_draws_official_forecast_when_present(tmp_path, fake_nwps):
    payloads, truth = _scenario()
    official_t = GRID[INOW + 1:INOW + 49]
    payloads['TJBN5']['forecast'] = _block(official_t, truth[INOW + 1:INOW + 49] * 1.1,
                                           issued='2026-11-02T15:00:00Z', every=6)
    fake_nwps(payloads)
    s = flow_forecast.run('taos-box', days=3, out_dir=str(tmp_path))
    assert s['nws_official_forecast_issued_utc'].startswith('2026-11-02T15:00')
    csv = pd.read_csv(s['files']['csv'])
    assert csv['nws_official_cfs'].notna().sum() >= 8


def test_main_reports_unknown_site_with_exit_code_2(capsys):
    assert flow_forecast.main(['--site', 'nowhere-creek', '-q']) == 2
    assert 'Unknown station' in capsys.readouterr().err


def test_main_prints_summary(tmp_path, fake_nwps, capsys):
    payloads, _ = _scenario()
    fake_nwps(payloads)
    assert flow_forecast.main(['--site', 'TJBN5', '--out-dir', str(tmp_path),
                               '--no-chart', '-q']) == 0
    out = capsys.readouterr().out
    assert 'peak:' in out and 'Rio Grande at Alamosa' in out


def _rain_zones():
    """A tiny station zone config: one ungauged local zone with one sample point."""
    ring = [[-106.0, 36.0], [-105.0, 36.0], [-105.0, 37.0], [-106.0, 37.0], [-106.0, 36.0]]
    return {'USGS:08276500': {
        'zones': {'local': {'area_km2': 1000.0, 'rings': [ring]},
                  'mid': {'area_km2': 0, 'rings': []}, 'far': {'area_km2': 0, 'rings': []}},
        'ungauged': {'area_km2': 1000.0, 'rings': [ring]},
        'points': [{'lat': 36.5, 'lon': -105.5, 'zone': 'local', 'ungauged': True, 'w': 1000.0,
                    'nws': 'ABQ/1,1'}]}}


def test_cli_rain_term_and_rain_panel(tmp_path, fake_nwps):
    payloads, _ = _scenario()
    days = pd.date_range(NOW.normalize().tz_localize(None) - pd.Timedelta(days=31),
                         NOW.normalize().tz_localize(None) - pd.Timedelta(days=1))
    rain = np.zeros(len(days))
    rain[-12:-9] = [20.0, 35.0, 10.0]                 # a storm ~10 days ago
    observed = pd.DataFrame({c: 0.0 for c in get_basin_precip.FEATURE_COLUMNS}, index=days)
    observed['rain_local'] = rain
    observed['rain_ungauged'] = rain
    observed['precip_ungauged'] = rain
    observed['source'] = 'gridmet'
    observed['window_start'] = days.tz_localize('UTC')
    # 12 mm of forecast rain tomorrow (local), nothing else.
    hours = pd.date_range(NOW, periods=96, freq='1h')
    precip = np.zeros(96)
    precip[24:36] = 1.0
    fpts = {0: pd.DataFrame({'precip_mm': precip, 'temp_c': 5.0}, index=hours)}
    fake_nwps(payloads, observed=observed, forecast_points=fpts)
    s = flow_forecast.run('taos-box', days=5, out_dir=str(tmp_path), zones=_rain_zones())
    br = s['basin_rain']
    assert br['rain_term'] in ('calibrated', 'default')
    assert br['ungauged_area_km2'] == 1000.0
    assert br['ungauged_rain_mm']['forecast'] == pytest.approx(12.0, abs=0.5)
    assert br['zone_totals_mm']['next']['rain_local'] == pytest.approx(12.0, abs=0.5)
    assert 0 <= br['runoff_ratio'] <= 1
    assert os.path.getsize(s['files']['png']) > 0


def test_cli_without_rain_reports_none(tmp_path, fake_nwps):
    payloads, _ = _scenario()
    fake_nwps(payloads)
    s = flow_forecast.run('taos-box', days=3, out_dir=str(tmp_path), make_chart=False,
                          use_rain=False)
    assert s['basin_rain'] is None


def _conejos_payload(with_forecast=True):
    obs_t = GRID[:INOW + 1]
    flow = 800 + 2500 * np.exp(-0.5 * ((np.arange(len(obs_t)) - len(obs_t) + 3) / 10.0) ** 2)
    payload = {'observed': _block(obs_t, flow), 'forecast': {'data': []}}
    if with_forecast:
        ft = GRID[INOW + 1:INOW + 1 + 5 * 24]
        # Rises a little to ~3,400, then recedes, still above the 800 base at its end.
        tt = np.arange(len(ft))
        fflow = np.where(tt < 4, 3300 + 25 * tt, 900 + 2500 * np.exp(-(tt - 4) / 30.0))
        payload['forecast'] = _block(ft, fflow, issued='2026-11-02T18:48:00Z', every=6)
    return payload


def test_station_without_upstream_uses_the_nws_forecast_then_recession(tmp_path, fake_nwps):
    fake_nwps({'MOGC2': _conejos_payload(), 'DNRC2': _conejos_payload(False)})
    s = flow_forecast.run('conejos', days=7, out_dir=str(tmp_path))
    assert s['mode'] == 'nws' and s['routing'] is None
    assert s['nws_forecast_end_utc'] is not None
    assert s['peak']['band_high_cfs'] is None and s['daily'][0]['band_low_cfs'] is None
    assert s['peak']['cfs'] == pytest.approx(3400, rel=0.05)
    # Past the end of the NWS forecast the flow keeps receding.
    assert s['daily'][-1]['max_cfs'] < s['daily'][3]['max_cfs']
    assert os.path.getsize(s['files']['png']) > 0
    out = flow_forecast.format_summary(s)
    assert 'mode:   nws' in out and '—' in out


def test_station_with_nothing_current_falls_back_to_recession(tmp_path, fake_nwps):
    fake_nwps({'MOGC2': _conejos_payload(with_forecast=False), 'DNRC2': _conejos_payload(False)})
    s = flow_forecast.run('conejos', days=5, out_dir=str(tmp_path), make_chart=False)
    assert s['mode'] == 'recession'
    assert not s['peak']['ahead']
    assert s['daily'][-1]['mean_cfs'] < s['latest_observation']['cfs']
    assert 'none ahead' in flow_forecast.format_summary(s)


def test_rain_panel_is_opt_in(tmp_path, fake_nwps, monkeypatch):
    payloads, _ = _scenario()
    fake_nwps(payloads)
    seen = []
    real = flow_forecast.charts.plot_forecast
    monkeypatch.setattr(flow_forecast.charts, 'plot_forecast',
                        lambda *a, **k: seen.append(k.get('rain')) or real(*a, **k))
    flow_forecast.run('taos-box', days=3, out_dir=str(tmp_path), zones=_rain_zones())
    assert seen == [None]                                   # default: no rain panel


def test_rain_term_is_never_an_instant_step():
    assert min(flow_forecast.RAIN_K_GRID) >= 12


def _irrigation_scenario():
    """Summer: 20% of the Alamosa/Mogote supply gets past the ditches to Lobatos."""
    payloads, _ = _scenario()
    a = (_pulse(NOW - pd.Timedelta(days=10), 1500, 20, 300) + _pulse(NOW + pd.Timedelta(hours=60), 300, 18, 0))
    b = _pulse(NOW - pd.Timedelta(days=12), 800, 30, 200)
    supply_at_lobatos = 1.3 * routing.route(a, 6, 24) + 0.7 * routing.route(b, 24, 72)
    lob = 0.2 * supply_at_lobatos
    truth = 120 + 1.2 * routing.route(lob, 24, 12)
    obs = slice(0, INOW + 1)
    fc_times = GRID[INOW + 1:INOW + 1 + 5 * 24]
    payloads['TJBN5']['observed'] = _block(GRID[obs], truth[obs])
    for lid, x in (('ALAC2', a), ('MOGC2', b)):
        payloads[lid] = {'observed': _block(GRID[obs], x[obs]),
                         'forecast': _block(fc_times, x[INOW + 1:INOW + 1 + 5 * 24],
                                            issued='2026-11-02T14:15:00Z', every=6)}
    payloads['LOBC2'] = {'observed': _block(GRID[obs], lob[obs]), 'forecast': {'data': []}}
    return payloads, truth


def test_irrigation_season_routes_through_the_lobatos_anchor(tmp_path, fake_nwps):
    payloads, truth = _irrigation_scenario()
    fake_nwps(payloads)
    s = flow_forecast.run('taos-box', days=5, out_dir=str(tmp_path), use_rain=False)
    assert s['routing']['routed_from'] == 'anchor'
    assert s['anchor']['nws_lid'] == 'LOBC2'
    assert s['anchor']['pass_through_ratio'] == pytest.approx(0.2, abs=0.03)
    future_max = truth[INOW + 1:INOW + 1 + 5 * 24].max()
    assert s['peak']['cfs'] == pytest.approx(future_max, rel=0.15) or not s['peak']['ahead']
    assert max(d['max_cfs'] for d in s['daily']) <= 1.15 * max(future_max, truth[INOW])


def test_missing_anchor_gauge_falls_back_to_direct_routing(tmp_path, fake_nwps):
    payloads, _ = _irrigation_scenario()
    payloads.pop('LOBC2')
    fake_nwps(payloads)
    s = flow_forecast.run('taos-box', days=5, out_dir=str(tmp_path), use_rain=False, make_chart=False)
    assert s['routing']['routed_from'] == 'upstream' and s['anchor'] is None


def test_routed_forecast_is_pure_and_matches_the_cli_core():
    payloads, truth = _irrigation_scenario()
    cfg = flow_forecast.get_upstream.load_registry()['USGS:08276500']
    a = np.array([q['secondary'] for q in payloads['ALAC2']['observed']['data']]) * 1000
    tgt = np.full(len(GRID), np.nan)
    tgt[:INOW + 1] = truth[:INOW + 1]
    ins = []
    for lid in ('ALAC2', 'MOGC2'):
        obs = pd.Series([q['secondary'] * 1000 for q in payloads[lid]['observed']['data']], index=GRID[:INOW + 1])
        fc = pd.Series([q['secondary'] * 1000 for q in payloads[lid]['forecast']['data']],
                       index=pd.DatetimeIndex([q['validTime'] for q in payloads[lid]['forecast']['data']]))
        ins.append(routing.build_input(GRID, obs, fc)[0])
    lob = np.full(len(GRID), np.nan)
    lob[:INOW + 1] = [q['secondary'] * 1000 for q in payloads['LOBC2']['observed']['data']]
    res = flow_forecast.routed_forecast(cfg, tgt, INOW, ins, anchor_obs=lob)
    assert res['anchor']['ratio'] == pytest.approx(0.2, abs=0.03)
    assert res['gauges'] == [cfg['anchor']] and len(res['ef'].central) == len(GRID)
    direct = flow_forecast.routed_forecast({k: v for k, v in cfg.items() if k != 'anchor'}, tgt, INOW, ins)
    assert direct['anchor'] is None and direct['gauges'] == cfg['upstream']


def _cerro_scenario():
    """Taos routed from Cerro (fed by Lobatos), with a storm on Rio Pueblo de Taos just before now."""
    payloads, _ = _irrigation_scenario()
    lob = np.array([q['secondary'] * 1000 for q in payloads['LOBC2']['observed']['data']])
    a = np.array([q['secondary'] * 1000 for q in payloads['ALAC2']['observed']['data']])
    a_full = (_pulse(NOW - pd.Timedelta(days=10), 1500, 20, 300) + _pulse(NOW + pd.Timedelta(hours=60), 300, 18, 0))
    b_full = _pulse(NOW - pd.Timedelta(days=12), 800, 30, 200)
    lob_full = 0.2 * (1.3 * routing.route(a_full, 6, 24) + 0.7 * routing.route(b_full, 24, 72))
    assert np.allclose(lob_full[:INOW + 1], lob, rtol=1e-3) and np.allclose(a_full[:INOW + 1], a, rtol=1e-3)
    cerro = 40 + routing.route(lob_full, 18, 3)
    pueblo = _pulse(NOW - pd.Timedelta(hours=6), 600, 4, 25)
    red = np.full(len(GRID), 50.0)
    truth = 100 + routing.route(cerro, 10, 3) + routing.route(pueblo, 2, 3) + routing.route(red, 5, 3)
    obs = slice(0, INOW + 1)
    payloads['TJBN5']['observed'] = _block(GRID[obs], truth[obs])
    payloads['CERN5'] = {'observed': _block(GRID[obs], cerro[obs]), 'forecast': {'data': []}}
    payloads['TBCN5'] = {'observed': _block(GRID[obs], pueblo[obs]), 'forecast': {'data': []}}
    return payloads, truth, pd.Series(red[obs], index=GRID[obs])


def test_taos_routes_from_cerro_plus_tributaries(tmp_path, fake_nwps, monkeypatch):
    payloads, truth, red = _cerro_scenario()
    fake_nwps(payloads)
    import backtest_routing
    monkeypatch.setattr(backtest_routing, 'usgs_hourly', lambda site, a, b: red if site == '08266820' else pd.Series(dtype=float))
    s = flow_forecast.run('taos-box', days=5, out_dir=str(tmp_path), use_rain=False)
    assert s['routing']['routed_from'] == 'routing anchor'
    assert s['routing_anchor']['nws_lid'] == 'CERN5' and s['routing_anchor']['used']
    assert [t['name'].split(' ')[0] for t in s['tributaries']] == ['Rio', 'Red']
    fc = pd.read_csv(tmp_path / 'taos-box_forecast.csv')
    fc = fc[fc['forecast_cfs'].notna()]
    want = pd.Series(truth, index=GRID.strftime('%Y-%m-%dT%H:%MZ')).reindex(fc['time_utc']).to_numpy()
    # The tributary storm water already measured reaches Taos in the next hours
    # (after that the tributary's own future is a generic recession).
    np.testing.assert_allclose(fc['forecast_cfs'].to_numpy()[:3], want[:3], rtol=0.06)
    assert np.all(np.diff(fc['forecast_cfs'].to_numpy()[:10]) < 0)
    # The next Lobatos wave: Lobatos -> Cerro refitted on the window.
    assert s['peak']['cfs'] == pytest.approx(truth[INOW + 1:INOW + 1 + 5 * 24].max(), rel=0.1)
    assert abs((pd.Timestamp(s['peak']['time_utc']) - GRID[INOW + 1 + int(np.argmax(truth[INOW + 1:INOW + 1 + 5 * 24]))])
               / pd.Timedelta(hours=1)) <= 6


def test_routed_forecast_tributaries_are_mass_balance_not_fitted():
    payloads, truth, red = _cerro_scenario()
    cfg = flow_forecast.get_upstream.load_registry()['USGS:08276500']
    ser = lambda lid: np.array([q['secondary'] * 1000 for q in payloads[lid]['observed']['data']])
    tgt = np.full(len(GRID), np.nan)
    tgt[:INOW + 1] = truth[:INOW + 1]
    ins = []
    for lid in ('ALAC2', 'MOGC2'):
        obs = pd.Series(ser(lid), index=GRID[:INOW + 1])
        fc = pd.Series([q['secondary'] * 1000 for q in payloads[lid]['forecast']['data']],
                       index=pd.DatetimeIndex([q['validTime'] for q in payloads[lid]['forecast']['data']]))
        ins.append(routing.build_input(GRID, obs, fc)[0])
    pad = lambda x: np.r_[x, np.full(len(GRID) - len(x), np.nan)]
    tribs = [routing.build_input(GRID, pd.Series(ser('TBCN5'), index=GRID[:INOW + 1]))[0],
             routing.build_input(GRID, red)[0]]
    res = flow_forecast.routed_forecast(cfg, tgt, INOW, ins, anchor_obs=pad(ser('LOBC2')),
                                        near_obs=pad(ser('CERN5')), tributary_inputs=tribs)
    assert res['gauges'] == [cfg['routing']['anchor']] and res['tributaries']
    # Registry: flow-dependent Lobatos -> Cerro travel time, not refitted.
    fa = cfg['routing']['anchor']['from_anchor']
    assert not res['near_anchor']['from_anchor_fitted']
    assert res['near_anchor']['from_anchor_lag_hours'] == fa['lag_hours']
    # from_anchor.refit: refitted on the window (this scenario uses a fixed 18 h lag).
    import copy
    cfg_refit = copy.deepcopy(cfg)
    cfg_refit['routing']['anchor']['from_anchor'] = {'lag_hours': 12, 'k_hours': 3, 'refit': True}
    refit = flow_forecast.routed_forecast(cfg_refit, tgt, INOW, ins, anchor_obs=pad(ser('LOBC2')),
                                          near_obs=pad(ser('CERN5')), tributary_inputs=tribs)
    assert refit['near_anchor']['from_anchor_fitted']
    assert abs(refit['near_anchor']['from_anchor_lag_hours'] - 18) <= 4
    # Observed hours are kept exactly; the routed fit only explains Cerro's part.
    np.testing.assert_allclose(res['ef'].central[:INOW + 1], truth[:INOW + 1], rtol=1e-6)
    assert res['fits'][0].coefs[0] == pytest.approx(1.0, abs=0.1)
    with pytest.raises(ValueError):
        flow_forecast.routed_forecast(cfg, tgt, INOW, ins, near_obs=pad(ser('CERN5')), tributary_inputs=tribs[:1])


def test_skill_note_quotes_backtested_crest_timing(tmp_path):
    path = str(tmp_path / 'skill.json')
    with open(path, 'w') as f:
        json.dump({'USGS:09180000': {'period': ['2015-01-01', '2026-11-01'], 'crests': 60, 'rain': True,
                                     'leads': {'12': {'median_abs_timing_h': 6.0, 'median_crest_ratio': 0.88},
                                               '24': {'median_abs_timing_h': 8.5, 'median_crest_ratio': 0.91},
                                               '6': {'median_abs_timing_h': 5.0, 'median_crest_ratio': 0.9}}}}, f)
    note = flow_forecast.skill_note(flow_forecast.routing_skill('USGS:09180000', path))
    assert note.startswith('Routing crest timing, backtested on 60 floods (2015-2026)')
    assert '6 h at 12 h ahead, 8 h at 24 h ahead' in note and '0.88-0.91x' in note
    assert flow_forecast.routing_skill('USGS:OTHER', path) is None and flow_forecast.skill_note(None) is None


def test_skill_note_switches_to_rain_driven_crests_in_a_storm():
    skill = {'period': ['2015-01-01', '2026-11-01'], 'crests': 60,
             'leads': {'12': {'median_abs_timing_h': 9, 'median_timing_h': 8, 'median_crest_ratio': 0.85}},
             'rain_driven': {'crests': 30, 'leads': {'12': {'median_abs_timing_h': 16, 'median_timing_h': 15,
                                                             'median_crest_ratio': 0.8}}},
             'other': {'crests': 30, 'leads': {'12': {'median_abs_timing_h': 4, 'median_timing_h': 2,
                                                       'median_crest_ratio': 0.95}}}}
    storm = flow_forecast.skill_note(skill, storm=True)
    assert '30 rain-driven floods' in storm and '16 h at 12 h ahead, usually late' in storm
    assert 'NWS forecast models the storm' in storm
    calm = flow_forecast.skill_note(skill, storm=False)
    assert 'without heavy local rain' in calm and '4 h at 12 h ahead;' in calm and 'NWS' not in calm


def test_storm_test_looks_at_the_72_hours_before_the_crest():
    rate = pd.Series(np.zeros(400))
    rate.iloc[100:104] = 4.0                                  # 16 mm, hours 100-103
    assert flow_forecast.storm_mm(rate, 150) == pytest.approx(16.0)
    assert flow_forecast.storm_mm(rate, 200) == 0.0           # > 72 h before this crest
    assert flow_forecast.storm_mm(rate, 101) == pytest.approx(8.0)


def test_cli_takes_upstream_observations_from_usgs_when_nwps_has_only_the_forecast(tmp_path, fake_nwps, monkeypatch):
    """NWPS publishes only the RFC forecast for some gauges (e.g. ALTC2); USGS IV has the readings."""
    payloads, truth = _scenario()
    mogc2_obs = nwps.hourly(nwps.parse_series(payloads['MOGC2'], 'observed'))
    payloads['MOGC2'] = dict(payloads['MOGC2'], observed={'data': []})
    fake_nwps(payloads)
    import backtest_routing
    monkeypatch.setattr(backtest_routing, 'usgs_hourly',
                        lambda site, a, b: mogc2_obs if site == '08246500' else pd.Series(dtype=float))
    s = flow_forecast.run('taos-box', days=7, out_dir=str(tmp_path), make_chart=False)
    mog = next(u for u in s['upstream'] if u['nws_lid'] == 'MOGC2')
    assert mog['latest_cfs'] == pytest.approx(float(mogc2_obs.dropna().iloc[-1]))
    assert mog['forecast_peak_cfs'] is not None                   # the NWPS forecast is kept
    assert s['routing']['mode'] == 'calibrated'
