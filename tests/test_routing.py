import numpy as np
import pandas as pd
import pytest

import routing


def _pulse(n=600, center=150, width=20, base=100.0, height=2000.0):
    t = np.arange(n)
    return base + height * np.exp(-0.5 * ((t - center) / width) ** 2)


def test_linear_reservoir_conserves_volume_and_attenuates():
    x = _pulse(n=2000, center=200)
    y = routing.linear_reservoir(x, 24)
    assert y.max() < x.max()                       # attenuated
    assert np.argmax(y) > np.argmax(x)             # delayed
    assert y.sum() == pytest.approx(x.sum(), rel=1e-3)   # mass conserved


def test_linear_reservoir_k1_is_identity():
    x = _pulse()
    np.testing.assert_array_equal(routing.linear_reservoir(x, 1), x)


def test_shift_pads_front_with_first_value():
    np.testing.assert_array_equal(routing.shift(np.array([1., 2., 3., 4.]), 2), [1., 1., 1., 2.])
    np.testing.assert_array_equal(routing.shift(np.array([1., 2.]), 5), [1., 1.])


def test_fit_recovers_known_lag_attenuation_and_coefficients():
    a = _pulse(center=150)
    b = _pulse(center=250, height=800, base=50)
    target = 120 + 1.2 * routing.route(a, 30, 12) + 0.6 * routing.route(b, 48, 24)
    fits = routing.fit(target, [a, b], lag_grid=range(0, 73, 6), k_grid=(1, 6, 12, 24, 48))
    best = fits[0]
    assert best.lags == [30, 48]
    assert best.ks == [12, 24]
    assert best.coefs == pytest.approx([1.2, 0.6], rel=1e-6)
    assert best.intercept == pytest.approx(120, abs=1e-6)
    assert best.rmse < 1e-6
    assert [f.rmse for f in fits] == sorted(f.rmse for f in fits)


def test_fit_with_three_inputs_uses_coordinate_descent_and_finds_truth():
    xs = [_pulse(center=c, height=h) for c, h in ((120, 1500), (200, 900), (300, 600))]
    target = sum(c * routing.route(x, lag, k) for x, c, lag, k in
                 zip(xs, (1.0, 0.8, 0.5), (12, 24, 36), (6, 12, 24)))
    fits = routing.fit(target, xs, lag_grid=range(0, 49, 6), k_grid=(1, 6, 12, 24))
    assert fits[0].lags == [12, 24, 36]
    assert fits[0].rmse < 1e-6


def test_fit_rejects_negative_contributions():
    a = _pulse(center=150)
    b = _pulse(center=300)
    target = 2000 - 0.5 * routing.route(b, 12, 6) + routing.route(a, 12, 6)
    for f in routing.fit(target, [a, b], lag_grid=range(0, 25, 6), k_grid=(1, 6)):
        assert min(f.coefs) >= 0


def test_fit_validates_inputs():
    with pytest.raises(ValueError):
        routing.fit(np.ones(100), [])
    with pytest.raises(ValueError):
        routing.fit(np.ones(100), [np.ones(99)])
    with pytest.raises(ValueError):
        routing.fit(np.ones(100), [np.array([np.nan] * 100)])


def test_is_informative_needs_variation():
    assert routing.is_informative(_pulse())
    assert not routing.is_informative(np.full(500, 300.0))
    assert not routing.is_informative(np.ones(10))


def test_default_fit_only_fits_the_intercept():
    a = _pulse()
    target = 75 + routing.route(a, 24, 12)
    f = routing.default_fit(target, [a], lags=[24], ks=[12], weights=[1.0])
    assert f.source == 'default'
    assert f.intercept == pytest.approx(75)
    assert f.rmse == pytest.approx(0, abs=1e-6)


def test_extend_recession_decays_toward_floor():
    v = np.concatenate([np.linspace(2000, 1500, 48), np.full(100, np.nan)])
    out = routing.extend_recession(v, 47, floor=200.0)
    tail = out[48:]
    assert (np.diff(tail) < 0).all()
    assert tail.min() > 200.0


def test_extend_recession_forces_decay_when_still_rising():
    v = np.concatenate([np.linspace(1000, 2000, 48), np.full(50, np.nan)])
    out = routing.extend_recession(v, 47, floor=100.0)
    assert out[48] < 2000.0
    assert (np.diff(out[48:]) < 0).all()


def _hourly(start, values):
    idx = pd.date_range(start, periods=len(values), freq='1h', tz='UTC')
    return pd.Series(values, index=idx, dtype=float)


def test_build_input_joins_observed_forecast_and_recession():
    grid = pd.date_range('2026-11-01T00:00Z', periods=72, freq='1h')
    obs = _hourly('2026-11-01T00:00Z', np.full(24, 500.0))
    fc = _hourly('2026-11-02T06:00Z', np.linspace(800, 2000, 18))
    values, info = routing.build_input(grid, obs, fc)
    assert info['last_observed'] == pd.Timestamp('2026-11-01T23:00Z')
    assert info['forecast_end'] == pd.Timestamp('2026-11-02T23:00Z')
    assert values[0] == 500.0
    assert 500.0 < values[26] < 800.0              # interpolated handoff
    assert values[47] == pytest.approx(2000.0)     # end of forecast
    assert values[60] < 2000.0                     # recession afterwards
    assert np.isfinite(values).all()


def test_build_input_without_forecast_recedes_from_last_observation():
    grid = pd.date_range('2026-11-01T00:00Z', periods=48, freq='1h')
    obs = _hourly('2026-11-01T00:00Z', np.linspace(900, 1000, 24))
    values, info = routing.build_input(grid, obs, None)
    assert info['forecast_end'] is None
    # No forecast: decay from the last observation toward the recent minimum
    # (the default floor), never below it.
    assert 900.0 < values[30] < 1000.0
    assert values[-1] >= 900.0


def test_build_input_requires_observations():
    grid = pd.date_range('2026-11-01T00:00Z', periods=10, freq='1h')
    with pytest.raises(ValueError):
        routing.build_input(grid, pd.Series(dtype=float), None)


def test_forecast_anchors_to_latest_obs_and_band_brackets_central():
    a = _pulse(center=300)
    truth = 150 + routing.route(a, 24, 12)
    now = 200
    fits = routing.fit(truth[:now + 1] + 40, [a[:now + 1]],    # biased calibration
                       lag_grid=range(0, 49, 6), k_grid=(1, 6, 12, 24))
    ef = routing.forecast(fits, [a], truth, now, bias_tau_hours=24)
    np.testing.assert_array_equal(ef.central[:now + 1], truth[:now + 1])
    assert (ef.low <= ef.central + 1e-9).all()
    assert (ef.high >= ef.central - 1e-9).all()
    assert (ef.low >= 0).all()
    # Bias toward the latest observation decays: far out, the model shape rules.
    peak_t = int(np.argmax(ef.central))
    assert abs(peak_t - int(np.argmax(truth))) <= 6


def test_daily_summary_and_peak():
    times = pd.date_range('2026-11-02T07:00Z', periods=48, freq='1h')
    central = np.concatenate([np.full(24, 1000.0), np.linspace(1000, 3000, 24)])
    s = routing.daily_summary(times, central, central - 100, central + 100, 'America/Denver')
    assert list(s.columns) == ['min_cfs', 'mean_cfs', 'max_cfs', 'band_low_cfs', 'band_high_cfs']
    assert s.index[0].isoformat() == '2026-11-02'
    t, v, i = routing.peak(times, central)
    assert v == 3000.0 and i == 47 and t == times[-1]


def test_add_input_keeps_upstream_fit_and_explains_the_residual():
    up = _pulse(center=150)
    rain = np.zeros(600)
    rain[300:306] = 5000.0                       # a burst of local rain flux, after the pulse
    target = 100 + 1.1 * routing.route(up, 24, 12) + 0.05 * routing.route(rain, 12, 24)
    base = routing.fit(target, [up], lag_grid=range(0, 49, 6), k_grid=(1, 6, 12, 24, 48))
    fits = routing.add_input(target, [up], base, rain, lag_grid=range(0, 49, 6),
                             k_grid=(1, 6, 12, 24, 48), max_coef=1.0)
    best = fits[0]
    # Upstream travel time AND weight stay exactly as stage 1 found them.
    assert best.lags[0] == base[0].lags[0] and best.ks[0] == base[0].ks[0]
    assert best.coefs[0] == base[0].coefs[0]
    assert best.lags[1] == 12 and best.ks[1] == 24
    assert best.coefs[1] == pytest.approx(0.05, rel=0.25)
    assert best.rmse < 0.5 * base[0].rmse
    assert len(best.predict([up, rain])) == 600


def test_add_input_bounds_the_extra_coefficient():
    up = _pulse(center=150)
    rain = np.zeros(600)
    rain[300:306] = 10.0
    target = routing.route(up, 6, 6) + 50 * routing.route(rain, 6, 6)   # "ratio" 50 > 1
    base = routing.fit(target, [up], lag_grid=(0, 6, 12), k_grid=(1, 6))
    fits = routing.add_input(target, [up], base, rain, lag_grid=(0, 6), k_grid=(1, 6), max_coef=1.0)
    assert all(f.coefs[-1] <= 1.0 for f in fits)


def test_fit_coef_bounds_reject_implausible_weights():
    up = _pulse(center=150, height=300)
    target = 3.0 * routing.route(up, 12, 6)          # a gauge "tripling" downstream
    free = routing.fit(target, [up], lag_grid=(0, 6, 12), k_grid=(1, 6))
    assert free[0].coefs[0] == pytest.approx(3.0)
    bounded = routing.fit(target, [up], lag_grid=(0, 6, 12), k_grid=(1, 6), coef_bounds=(0.25, 2.5))
    assert all(f.coefs[0] <= 2.5 for f in bounded)


def test_anchored_input_continues_with_the_pass_through_share_of_supply_changes():
    supply = np.r_[np.full(200, 400.0), np.full(100, 700.0)]       # +300 cfs after hour 199
    anchor = np.full(300, np.nan)
    anchor[:200] = 100.0                                              # 25% passes the ditches
    anchor[50:53] = np.nan                                            # a short outage
    values, info = routing.anchored_input(anchor, supply)
    assert info['ratio'] == pytest.approx(0.25) and info['last_index'] == 199
    assert np.isfinite(values).all() and values[51] == pytest.approx(100.0)
    assert values[250] == pytest.approx(100.0 + 0.25 * 300)


def test_anchored_input_passes_more_of_a_rise_beyond_ditch_capacity():
    supply = np.r_[np.full(200, 400.0), np.full(50, 4400.0)]         # +4,000 cfs flood
    anchor = np.r_[np.full(200, 60.0), np.full(50, np.nan)]          # 15% passes now
    values, info = routing.anchored_input(anchor, supply)
    r = info['ratio']
    want = 60 + r * 500 + 0.5 * 3500                                  # capacity 500, flood share 0.5
    assert values[-1] == pytest.approx(want)
    loose, _ = routing.anchored_input(anchor, supply, anchor_cfg={'divert_capacity_cfs': 1e9})
    assert loose[-1] == pytest.approx(60 + r * 4000)                 # pure ratio without capacity
    with pytest.raises(ValueError):
        routing.anchored_input(np.full(10, np.nan), np.zeros(10))


def test_find_crests_picks_floods_not_noise():
    rng = np.random.default_rng(0)
    x = 300 + rng.normal(0, 5, 1200)
    x[200:260] += 3000 * np.exp(-0.5 * ((np.arange(60) - 30) / 8.0) ** 2)    # flood crest at ~230
    x[700:760] += 1200 * np.exp(-0.5 * ((np.arange(60) - 30) / 8.0) ** 2)    # smaller flood at ~730
    x[1000:1010] += 60                                                     # a bump, not a flood
    crests = routing.find_crests(x)
    assert len(crests) == 2 and abs(crests[0] - 230) <= 2 and abs(crests[1] - 730) <= 2


def test_crest_timing_errors_sign_and_window():
    pred = np.zeros(200)
    pred[110] = 5.0                       # predicted 10 h after the crest at 100
    assert routing.crest_timing_errors(pred, [100]).tolist() == [10.0]
    pred2 = np.zeros(200)
    pred2[90] = 5.0
    assert routing.crest_timing_errors(pred2, [100]).tolist() == [-10.0]


def test_fit_with_timing_weight_prefers_the_crest_timed_lag():
    # The target follows the upstream input 12 h later at the crests, but a
    # simultaneous local rise (same storm) makes a 0 h lag fit ordinary hours
    # about as well.
    n = 900
    t = np.arange(n)
    up = 200 + 2500 * np.exp(-0.5 * ((t - 300) / 10.0) ** 2) + 2500 * np.exp(-0.5 * ((t - 650) / 10.0) ** 2)
    local = 1500 * (np.exp(-0.5 * ((t - 296) / 14.0) ** 2) + np.exp(-0.5 * ((t - 646) / 14.0) ** 2))
    target = 100 + routing.route(up, 12, 1) + local
    grids = [(list(range(0, 25, 2)), [1, 6])]
    plain = routing.fit(target, [up], grids=grids)[0]
    timed = routing.fit(target, [up], grids=grids, timing_weight=2.0)[0]
    err = lambda f: np.abs(routing.crest_timing_errors(f.predict([up]), routing.find_crests(target))).mean()
    assert timed.timing_mae_hours is not None and err(timed) <= err(plain)
    assert timed.to_dict()['timing_mae_hours'] == timed.timing_mae_hours


def test_anchored_input_fixed_ratio_passes_changes_one_to_one():
    supply = np.r_[np.full(200, 1000.0), np.full(50, 2000.0)]        # Lobatos +1,000 cfs
    anchor = np.r_[np.full(200, 700.0), np.full(50, np.nan)]          # Cerro: mean ratio 0.7
    values, info = routing.anchored_input(anchor, supply, anchor_cfg={'fixed_ratio': 1.0})
    assert info['ratio'] == 1.0 and values[-1] == pytest.approx(1700.0)


def test_variable_shift_moves_high_water_faster_than_low_water():
    # A rise from 300 to 3,000 cfs over hours 50-80, then steady.
    y = np.r_[np.full(50, 300.0), np.linspace(300, 3000, 31), np.full(40, 3000.0)]
    out = routing.variable_shift(y, 9, 0.3, 1000.0)
    lag_low, lag_high = 9 * 0.3 ** -0.3, 9 * 3.0 ** -0.3        # ~12.9 h and ~6.5 h
    assert out[int(lag_low) + 49] == pytest.approx(300.0)        # low water: slow
    assert out[80 + int(np.ceil(lag_high))] == pytest.approx(3000.0)   # high water: fast
    assert routing.shift(y, 9)[80 + int(np.ceil(lag_high))] < 3000.0   # a fixed 9 h lag is later
    # A jump into low water cannot overtake it: the front arrives at the slow
    # water's time (a steep front), not 6.5 h after it left.
    step = routing.variable_shift(np.r_[np.full(50, 300.0), np.full(50, 3000.0)], 9, 0.3, 1000.0)
    front = int(np.argmax(step > 1000))
    assert 49 + lag_low - 1 <= front <= 49 + lag_low + 1
    assert np.all(np.diff(out) >= -1e-9) and np.all(np.diff(step) >= -1e-9)


def test_route_with_zero_exponent_is_the_fixed_lag():
    x = np.r_[np.full(30, 100.0), np.linspace(100, 900, 20), np.full(30, 900.0)]
    np.testing.assert_allclose(routing.route(x, 7, 3, (0.0, 1000.0)), routing.route(x, 7, 3))


def test_fit_recovers_a_flow_dependent_travel_time():
    t = np.arange(600)
    x = 300 + 2500 * np.exp(-0.5 * ((t - 200) / 15.0) ** 2) + 600 * np.exp(-0.5 * ((t - 420) / 12.0) ** 2)
    cel = (0.3, 1000.0)
    target = 80 + routing.route(x, 9, 1, cel)
    fits = routing.fit(target, [x], lag_grid=range(0, 21), k_grid=(1, 3), celerity=[cel])
    assert fits[0].lags == [9] and fits[0].celerity == [cel]
    np.testing.assert_allclose(fits[0].predict([x]), target, atol=1.0)
    fixed = routing.fit(target, [x], lag_grid=range(0, 21), k_grid=(1, 3))
    assert fixed[0].rmse > 5 * max(fits[0].rmse, 1e-6)


def test_blend_forecast_shifts_a_stale_rfc_onto_the_latest_reading():
    # Alamosa, Nov 5 2026: read 2,409 cfs at 8 AM while the RFC issued the
    # evening before still had ~2,815 for that hour.
    t = pd.date_range('2026-11-05T05:00Z', periods=4, freq='6h')
    fc = pd.Series([2960.0, 2670.0, 2450.0, 2170.0], index=t)
    obs = pd.Series([2512.0, 2409.0], index=pd.DatetimeIndex(['2026-11-05T13:00Z', '2026-11-05T15:00Z']))
    out = routing.blend_forecast(obs, fc, tau_hours=24.0)
    at_obs = np.interp(obs.index[-1].value, out.index.asi8, out.to_numpy())
    assert at_obs == pytest.approx(2409.0, abs=60)                 # starts at the reading, not 2,815
    off = out - fc
    assert off.iloc[-1] < 0 and abs(off.iloc[-1]) < abs(off.iloc[1])   # the shift fades
    # A forecast issued after the last reading is left alone.
    later = pd.Series([2000.0, 1900.0], index=pd.date_range('2026-11-05T18:00Z', periods=2, freq='6h'))
    pd.testing.assert_series_equal(routing.blend_forecast(obs, later), later)


def test_build_input_without_blend_keeps_the_rfc_level():
    grid = pd.date_range('2026-11-05T00:00Z', periods=24, freq='1h')
    obs = pd.Series(2400.0, index=grid[:10])
    fc = pd.Series([2800.0, 2700.0, 2600.0], index=pd.DatetimeIndex(['2026-11-05T06:00Z', '2026-11-05T12:00Z',
                                                                    '2026-11-05T18:00Z']))
    plain, _ = routing.build_input(grid, obs, fc)
    blended, _ = routing.build_input(grid, obs, fc, blend_tau_hours=24.0)
    assert plain[12] == pytest.approx(2700.0)
    assert blended[12] < 2500.0
