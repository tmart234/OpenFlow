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
