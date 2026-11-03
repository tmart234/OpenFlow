import numpy as np
import pandas as pd
import pytest

pytest.importorskip('matplotlib')

import charts  # noqa: E402

PNG_MAGIC = b'\x89PNG\r\n\x1a\n'


def _hourly(start, n):
    return pd.date_range(start, periods=n, freq='1h', tz='UTC')


def test_plot_forecast_writes_png_and_returns_peak(tmp_path):
    obs_t = _hourly('2026-10-29T17:00Z', 97)
    fc_t = _hourly('2026-11-02T17:00Z', 24 * 6)
    central = 1200 + 2800 * np.exp(-0.5 * ((np.arange(len(fc_t)) - 99) / 20.0) ** 2)
    out = tmp_path / 'f.png'
    info = charts.plot_forecast(
        str(out), title='Test', subtitle='sub', footnote='line 1\nline 2',
        obs_times=obs_t, obs_values=np.linspace(4000, 1200, len(obs_t)),
        fc_times=fc_t, fc_central=central, fc_low=central * 0.9, fc_high=central * 1.1,
        now=fc_t[0], tz='America/Denver', thresholds={'action': 7750.0, 'minor': 9800.0})
    assert out.read_bytes()[:8] == PNG_MAGIC
    # Peak is reported in local (naive) time: index 99 -> Nov 6 20:00 UTC = 1 PM MST.
    assert info['peak_value'] == pytest.approx(central.max())
    assert info['peak_time'] == pd.Timestamp('2026-11-06 13:00')


def test_plot_forecast_handles_daily_series_and_official_overlay(tmp_path):
    days = pd.date_range('2024-05-01', periods=14, freq='D')
    out = tmp_path / 'daily.png'
    info = charts.plot_forecast(
        str(out), title='Daily', obs_times=days[:7], obs_values=np.arange(7) * 100.0,
        fc_times=days[6:], fc_central=np.linspace(600, 1500, 8),
        official_times=days[6:], official_values=np.linspace(650, 1400, 8), now=days[6])
    assert out.stat().st_size > 0
    assert info['peak_time'] == days[-1]


def test_plot_forecast_without_forecast_has_no_peak(tmp_path):
    t = _hourly('2026-11-01T00:00Z', 48)
    info = charts.plot_forecast(str(tmp_path / 'o.png'), title='Obs only',
                                obs_times=t, obs_values=np.full(48, 500.0))
    assert info == {'peak_time': None, 'peak_value': None, 'callout': None}


def test_plot_hindcast_writes_png(tmp_path):
    hist = pd.date_range('2024-04-01', periods=30, freq='D')
    fut = pd.date_range('2024-05-01', periods=14, freq='D')
    info = charts.plot_hindcast(
        str(tmp_path / 'h.png'), title='Hindcast', history_dates=hist,
        history_flow=np.linspace(200, 800, 30), future_dates=fut,
        observed_min=np.linspace(800, 1500, 14), observed_max=np.linspace(900, 1700, 14),
        predicted_min=np.linspace(780, 1400, 14), predicted_max=np.linspace(880, 1600, 14))
    assert (tmp_path / 'h.png').read_bytes()[:8] == PNG_MAGIC
    assert info['peak_value'] == pytest.approx((1400 + 1600) / 2)


def test_nice_ceiling_and_rounding():
    assert charts._nice_ceiling(4730) == 5000
    assert charts._nice_ceiling(0) == 1.0
    assert charts._round_cfs(3932.0) == 3900
    assert charts._round_cfs(456.0) == 460


def test_plot_forecast_with_rain_panel(tmp_path):
    fc_t = _hourly('2026-11-02T17:00Z', 24 * 5)
    central = 1000 + 500 * np.sin(np.arange(len(fc_t)) / 20.0) ** 2
    days = pd.date_range('2026-10-30', '2026-11-06')
    rain = {'dates': days, 'zones': {'local': np.arange(8.0), 'mid': np.ones(8), 'far': np.zeros(8)},
            'forecast_from': pd.Timestamp('2026-11-03')}
    out = tmp_path / 'r.png'
    info = charts.plot_forecast(str(out), title='Rain', fc_times=fc_t, fc_central=central,
                                now=fc_t[0], tz='America/Denver', rain=rain,
                                footnote='a\nb\nc\nd')
    assert out.read_bytes()[:8] == PNG_MAGIC
    assert info['peak_value'] == pytest.approx(central.max())


def test_falling_river_calls_out_the_end_value_not_a_peak(tmp_path):
    fc_t = _hourly('2026-11-02T19:00Z', 24 * 6 + 1)          # ends at local midnight
    central = np.linspace(607, 360, len(fc_t))
    info = charts.plot_forecast(str(tmp_path / 'f.png'), title='Falling', fc_times=fc_t,
                                fc_central=central, now=fc_t[0], tz='America/Denver')
    assert info['callout'] == 'end'
    assert info['peak_value'] == pytest.approx(360)


def test_rising_river_calls_out_the_peak(tmp_path):
    fc_t = _hourly('2026-11-02T19:00Z', 48)
    central = np.r_[np.linspace(1000, 3000, 24), np.linspace(3000, 2000, 24)]
    info = charts.plot_forecast(str(tmp_path / 'r.png'), title='Rising', fc_times=fc_t,
                                fc_central=central, now=fc_t[0], tz='America/Denver')
    assert info['callout'] == 'peak' and info['peak_value'] == pytest.approx(3000)
