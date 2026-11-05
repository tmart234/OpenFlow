from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from data import get_upstream


TAOS = 'USGS:08276500'


def test_registry_has_a_complete_taos_box_entry():
    reg = get_upstream.load_registry()
    assert not any(k.startswith('_') for k in reg)
    cfg = reg[TAOS]
    assert cfg['nws_lid'] == 'TJBN5'
    assert cfg['timezone'] == 'America/Denver'
    assert cfg['upstream'], "Taos Box needs upstream gauges"
    for g in cfg['upstream']:
        assert g['site_id'].split(':')[0] in ('DWR', 'USGS')
        for key in ('nws_lid', 'lag_days', 'weight', 'lag_hours', 'k_hours'):
            assert key in g, (g['name'], key)
        assert g['weight'] > 0 and g['lag_days'] >= 0


def test_every_registry_station_is_in_the_training_site_list():
    # A registered station that isn't trained on would silently never get a
    # station embedding; keep the two lists in step.
    import combine_data
    sites = set(combine_data.get_site_ids())
    for site_id in get_upstream.load_registry():
        assert site_id in sites


@pytest.mark.parametrize('name', ['taos-box', 'Lower-Taos-Box', 'USGS:08276500',
                                  '08276500', 'tjbn5'])
def test_resolve_site_accepts_aliases_ids_and_lid(name):
    site_id, cfg = get_upstream.resolve_site(name)
    assert site_id == TAOS
    assert cfg['nws_lid'] == 'TJBN5'


def test_resolve_site_unknown_lists_known_stations():
    with pytest.raises(KeyError, match='taos-box'):
        get_upstream.resolve_site('nowhere-creek')


def _cfg():
    return {'forecast_days': 5, 'upstream': [
        {'site_id': 'DWR:A', 'lag_days': 2, 'weight': 1.0},
        {'site_id': 'DWR:B', 'lag_days': 5, 'weight': 0.5}]}


def test_known_days_is_min_lag_plus_forecast_capped():
    assert get_upstream.known_days(_cfg(), 14) == 7
    assert get_upstream.known_days(_cfg(), 6) == 6
    assert get_upstream.known_days(None, 14) == 0
    assert get_upstream.known_days({'upstream': []}, 14) == 0


def test_routed_daily_shifts_weights_and_requires_all_gauges():
    days = pd.date_range('2022-01-01', periods=10, freq='D')
    a = pd.Series(np.arange(10, dtype=float) * 10, index=days)
    b = pd.Series(np.full(10, 100.0), index=days)
    out = get_upstream.routed_daily({'DWR:A': a, 'DWR:B': b}, _cfg())
    # Jan 6 = A(Jan 4) * 1.0 + B(Jan 1) * 0.5 = 30 + 50
    assert out.loc['2022-01-06'] == pytest.approx(80.0)
    # Before B's lagged series starts there is no complete sum.
    assert np.isnan(out.loc['2022-01-05'])
    assert out.name == 'upstream_flow'


def test_routed_daily_empty_when_a_gauge_is_missing():
    days = pd.date_range('2022-01-01', periods=5, freq='D')
    out = get_upstream.routed_daily({'DWR:A': pd.Series(1.0, index=days)}, _cfg())
    assert out.empty


def test_interpolate_short_gaps_leaves_long_runs_missing():
    idx = pd.date_range('2022-01-01', periods=20, freq='D')
    v = np.arange(20, dtype=float)
    v[2:4] = np.nan       # 2-day gap -> filled
    v[8:18] = np.nan      # 10-day gap -> left alone entirely
    out = get_upstream.interpolate_short_gaps(pd.Series(v, index=idx), 7)
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[8:18].isna().all()


def test_get_upstream_routes_fetched_series(monkeypatch):
    calls = []

    def fake_fetch(site_id, start, end):
        calls.append((site_id, start))
        idx = pd.date_range('2021-12-01', '2022-02-01', freq='D')
        return pd.Series({'DWR:RIOALACO': 100.0, 'DWR:CONMOGCO': 50.0}.get(site_id, 30.0), index=idx)

    monkeypatch.setattr(get_upstream, 'fetch_daily_mean', fake_fetch)
    df = get_upstream.get_upstream(TAOS, datetime(2022, 1, 1), datetime(2022, 1, 31))
    assert list(df.columns) == ['Date', 'upstream_flow']
    assert len(df) == 31
    # Taos is anchored at Lobatos: the routed value is the (lagged) anchor flow.
    assert {c[0] for c in calls} == {'DWR:RIOALACO', 'DWR:CONMOGCO', 'DWR:RIOLOBCO'}
    assert df['upstream_flow'].iloc[-1] == pytest.approx(30.0)
    reg = get_upstream.load_registry()
    cfg = dict(reg[TAOS])
    cfg.pop('anchor')
    monkeypatch.setattr(get_upstream, 'load_registry', lambda path=None: {**reg, TAOS: cfg})
    df = get_upstream.get_upstream(TAOS, datetime(2022, 1, 1), datetime(2022, 1, 31))
    w = {g['site_id']: g['weight'] for g in cfg['upstream']}
    assert df['upstream_flow'].iloc[-1] == pytest.approx(100 * w['DWR:RIOALACO'] + 50 * w['DWR:CONMOGCO'])
    # History is fetched far enough back to cover the longest lag.
    assert all(start <= datetime(2022, 1, 1) - pd.Timedelta(days=5) for _, start in calls)


def test_get_upstream_empty_for_unregistered_station_or_failed_fetch(monkeypatch):
    assert get_upstream.get_upstream('USGS:09999999', datetime(2022, 1, 1), datetime(2022, 1, 5)).empty
    monkeypatch.setattr(get_upstream, 'fetch_daily_mean', lambda *a: pd.Series(dtype=float))
    assert get_upstream.get_upstream(TAOS, datetime(2022, 1, 1), datetime(2022, 1, 5)).empty


def test_get_upstream_disabled_by_env(monkeypatch):
    monkeypatch.setenv('OPENFLOW_DISABLE_UPSTREAM', '1')
    monkeypatch.setattr(get_upstream, 'fetch_daily_mean',
                        lambda *a: pytest.fail('should not fetch when disabled'))
    assert get_upstream.get_upstream(TAOS, datetime(2022, 1, 1), datetime(2022, 1, 5)).empty


def test_fetch_daily_mean_dispatches_on_prefix(monkeypatch):
    from data import get_CODWR_flow, get_flow
    monkeypatch.setattr(get_CODWR_flow, 'get_historical_data', lambda *a: pd.DataFrame(
        {'Date': ['2022-01-01'], 'Min Flow': [42.0], 'Max Flow': [42.0]}))
    monkeypatch.setattr(get_flow, 'get_daily_mean_flow', lambda *a: pd.DataFrame(
        {'Date': ['2022-01-01'], 'flow': [7.0]}))
    assert get_upstream.fetch_daily_mean('DWR:X', None, None).iloc[0] == 42.0
    assert get_upstream.fetch_daily_mean('USGS:1', None, None).iloc[0] == 7.0
    with pytest.raises(ValueError):
        get_upstream.fetch_daily_mean('NOAA:1', None, None)


def _anchor_cfg():
    return {'upstream': [{'site_id': 'A', 'lag_days': 2, 'weight': 1.0}],
            'anchor': {'site_id': 'L', 'lag_days': 1, 'weight': 1.0}}


def test_anchored_routing_is_the_lagged_anchor_where_observed():
    days = pd.date_range('2026-06-01', periods=10)
    series = {'A': pd.Series(200.0, index=days), 'L': pd.Series(np.arange(10) + 20.0, index=days)}
    out = get_upstream.routed_daily(series, _anchor_cfg())
    # Lobatos shifted 1 day; the 200-cfs supply never shows up while the anchor has data.
    assert out.loc['2026-06-02'] == 20.0 and out.loc['2026-06-11'] == 29.0
    assert out.name == 'upstream_flow'


def test_anchored_routing_scales_future_supply_changes_by_pass_through():
    days = pd.date_range('2026-06-01', periods=10)
    supply_up = pd.Series(100.0, index=pd.date_range('2026-06-01', periods=14))
    supply_up.iloc[-3:] = 300.0                  # RFC forecast: +200 cfs upstream
    series = {'A': supply_up, 'L': pd.Series(25.0, index=days)}   # 25% passes the ditches
    out = get_upstream.routed_daily(series, _anchor_cfg())
    # Anchor ends 2026-06-11 (10 obs + 1 day lag); supply (lag 2) runs to 06-16.
    assert out.loc['2026-06-11'] == 25.0
    assert out.loc['2026-06-13'] == pytest.approx(25.0)                 # supply unchanged
    assert out.loc['2026-06-14'] == pytest.approx(25.0 + 0.25 * 200)    # scaled rise
    assert out.index.max() == pd.Timestamp('2026-06-16')


def test_anchored_routing_needs_the_anchor_series():
    days = pd.date_range('2026-06-01', periods=5)
    out = get_upstream.routed_daily({'A': pd.Series(1.0, index=days)}, _anchor_cfg())
    assert out.empty
    assert get_upstream.gauge_ids(_anchor_cfg()) == ['A', 'L']
    assert get_upstream.max_lag_days(_anchor_cfg()) == 2


def test_taos_box_is_anchored_at_lobatos():
    cfg = get_upstream.upstream_config('USGS:08276500')
    assert cfg['anchor']['site_id'] == 'DWR:RIOLOBCO' and cfg['anchor']['lag_days'] == 1


def test_passed_change_diverts_small_rises_and_passes_floods():
    pc = get_upstream.passed_change
    assert pc(200.0, 0.2) == pytest.approx(40.0)                     # within ditch capacity
    assert pc(-1000.0, 0.2) == pytest.approx(-200.0)                 # falls at the ratio
    assert pc(2500.0, 0.2) == pytest.approx(0.2 * 500 + 0.5 * 2000)  # flood beyond capacity
    assert pc(2500.0, 0.9) == pytest.approx(0.9 * 2500)              # winter: ratio already high
    assert pc(2500.0, 0.2, {'divert_capacity_cfs': 1000, 'flood_pass_through': 1.0}) == \
        pytest.approx(0.2 * 1000 + 1500)
