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
from data import nwps

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
    def install(payloads):
        monkeypatch.setattr(nwps, 'get_stageflow', lambda lid: payloads.get(lid))
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
