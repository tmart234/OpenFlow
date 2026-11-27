"""
Lead-weighted hourly fusion of OpenFlow upstream routing and daily ML forecasts.

The two forecasters have complementary skill profiles (docs/VERIFICATION.md):
  - Hourly routing (`routing.py`): excels over hours 0-36 because it tracks
    waves already observed at upstream gauges with kinematic travel times,
    but reverts to simple exponential recession past the upstream/RFC window.
  - Daily ML (`ml_inference.py`): excels over days 2-7+ because it uses basin
    weather forecasts, radiation/PET snowmelt energy, soil moisture, and
    calibrated forecast quantiles, but is daily-resolution only.

This module blends them in log1p-flow space with a smooth exponential lead
weight `w(h) = exp(-h / tau_hours)` (anchored at `now` with `w(0) = 1`),
and computes the fused predictive band using the mixture law of total
variance in log space so disagreement between routing and ML naturally
widens the uncertainty interval.
"""

from typing import Optional

import numpy as np
import pandas as pd

import charts
import routing

DEFAULT_TAU_ROUTING_H = 24.0
DEFAULT_TAU_RFC_BONUS_H = 18.0
DEFAULT_TAU_NWS_H = 48.0
DEFAULT_TAU_RECESSION_H = 12.0


def choose_tau_hours(mode: str, min_lag_hours: Optional[float] = None,
                     has_rfc: bool = False) -> float:
    """Adaptive e-folding transition timescale (hours) from routing/NWS to ML."""
    if mode == 'nws':
        return DEFAULT_TAU_NWS_H
    if mode == 'recession':
        return DEFAULT_TAU_RECESSION_H
    base = max(18.0, float(min_lag_hours or 12.0) + 12.0)
    return base + (DEFAULT_TAU_RFC_BONUS_H if has_rfc else 0.0)


def _ml_hourly_on_grid(ml_daily: pd.DataFrame, grid: pd.DatetimeIndex,
                       inow: int, obs_now: float, tz: str):
    """Interpolate daily ML midpoint + 90% range (or daily min/max) onto UTC `grid[inow:]`."""
    if ml_daily is None or ml_daily.empty:
        return None
    df = ml_daily.copy()
    dates = pd.DatetimeIndex(pd.to_datetime(df['Date'])).normalize()
    mid = 0.5 * (df['min_cfs'].to_numpy(dtype=float) + df['max_cfs'].to_numpy(dtype=float))
    has_q = all(k in df.columns for k in ('min_q05_cfs', 'max_q05_cfs', 'min_q95_cfs', 'max_q95_cfs'))
    if has_q:
        lo = 0.5 * (df['min_q05_cfs'].to_numpy(dtype=float) + df['max_q05_cfs'].to_numpy(dtype=float))
        hi = 0.5 * (df['min_q95_cfs'].to_numpy(dtype=float) + df['max_q95_cfs'].to_numpy(dtype=float))
    else:
        lo = df['min_cfs'].to_numpy(dtype=float)
        hi = df['max_cfs'].to_numpy(dtype=float)
    lo = np.maximum(lo, 0.0)
    hi = np.maximum(hi, lo)

    now_local = grid[inow].tz_convert(tz).tz_localize(None)
    bt_mid, m_mid, _ = charts.ml_band(dates, mid, mid, now_local, obs_now)
    bt_rng, m_lo, m_hi = charts.ml_band(dates, lo, hi, now_local, obs_now)
    idx_utc = bt_mid.tz_localize(tz, ambiguous='NaT', nonexistent='shift_forward').tz_convert('UTC')
    valid = idx_utc.notna()
    s_mid = pd.Series(m_mid[valid], index=idx_utc[valid]).groupby(level=0).mean()
    s_lo = pd.Series(m_lo[valid], index=idx_utc[valid]).groupby(level=0).mean()
    s_hi = pd.Series(m_hi[valid], index=idx_utc[valid]).groupby(level=0).mean()

    fc_grid = grid[inow:]
    out_mid = s_mid.reindex(fc_grid).ffill().bfill().to_numpy(dtype=float)
    out_lo = s_lo.reindex(fc_grid).ffill().bfill().to_numpy(dtype=float)
    out_hi = s_hi.reindex(fc_grid).ffill().bfill().to_numpy(dtype=float)
    return out_mid, np.minimum(out_lo, out_mid), np.maximum(out_hi, out_mid)


def blend_forecasts(grid: pd.DatetimeIndex,
                    inow: int,
                    routing_ef: routing.EnsembleForecast,
                    ml_daily: pd.DataFrame,
                    tz: str,
                    *,
                    mode: str = 'routing',
                    min_lag_hours: Optional[float] = None,
                    has_rfc: bool = False,
                    tau_hours: Optional[float] = None,
                    has_routing_band: bool = True) -> Optional[dict]:
    """
    Blend hourly `routing_ef` and daily `ml_daily` on `grid[inow:]`.

    Returns None when `ml_daily` is empty; otherwise returns:
      {
        'central': ndarray[len(grid)],
        'low':     ndarray[len(grid)],
        'high':    ndarray[len(grid)],
        'tau_hours': float,
        'weights': ndarray[len(grid) - inow],   # routing weight per forecast hour
        'daily':   DataFrame[min_cfs, mean_cfs, max_cfs, band_low_cfs, band_high_cfs],
        'peak':    {'time_utc', 'time_local', 'cfs', 'ahead', 'end_cfs',
                    'band_low_cfs', 'band_high_cfs'},
      }
    """
    if ml_daily is None or ml_daily.empty:
        return None
    obs_now = float(routing_ef.central[inow])
    if not np.isfinite(obs_now):
        return None
    ml_interp = _ml_hourly_on_grid(ml_daily, grid, inow, obs_now, tz)
    if ml_interp is None:
        return None
    ml_mid, ml_lo, ml_hi = ml_interp

    tau = float(tau_hours) if tau_hours is not None else choose_tau_hours(mode, min_lag_hours, has_rfc)
    tau = max(tau, 1.0)
    lead_h = np.arange(len(grid) - inow, dtype=float)
    w = np.exp(-lead_h / tau)

    r_mid = np.maximum(np.asarray(routing_ef.central[inow:], dtype=float), 0.0)
    r_lo = np.maximum(np.asarray(routing_ef.low[inow:], dtype=float), 0.0)
    r_hi = np.maximum(np.asarray(routing_ef.high[inow:], dtype=float), r_mid)

    lr = np.log1p(r_mid)
    lm = np.log1p(np.maximum(ml_mid, 0.0))
    lf = w * lr + (1.0 - w) * lm

    # Mixture-variance half-widths in log1p space (asymmetric lower/upper).
    dr_lo = np.maximum(lr - np.log1p(r_lo), 0.0) if has_routing_band else np.zeros_like(lr)
    dr_hi = np.maximum(np.log1p(r_hi) - lr, 0.0) if has_routing_band else np.zeros_like(lr)
    dm_lo = np.maximum(lm - np.log1p(np.maximum(ml_lo, 0.0)), 0.0)
    dm_hi = np.maximum(np.log1p(np.maximum(ml_hi, 0.0)) - lm, 0.0)
    disagree_sq = w * (1.0 - w) * (lr - lm) ** 2
    df_lo = np.sqrt(w * dr_lo ** 2 + (1.0 - w) * dm_lo ** 2 + disagree_sq)
    df_hi = np.sqrt(w * dr_hi ** 2 + (1.0 - w) * dm_hi ** 2 + disagree_sq)

    fc_central = np.maximum(np.expm1(lf), 0.0)
    fc_low = np.maximum(np.expm1(lf - df_lo), 0.0)
    fc_high = np.maximum(np.expm1(lf + df_hi), fc_central)

    central = np.asarray(routing_ef.central, dtype=float).copy()
    low = np.asarray(routing_ef.low, dtype=float).copy()
    high = np.asarray(routing_ef.high, dtype=float).copy()
    central[inow:] = fc_central
    low[inow:] = fc_low
    high[inow:] = fc_high

    peak_t, peak_v, peak_i = routing.peak(grid, central, min(inow + 1, len(grid) - 1))
    daily = routing.daily_summary(grid[inow:], central[inow:], low[inow:], high[inow:], tz)
    peak = {
        'time_utc': peak_t.isoformat(),
        'time_local': peak_t.tz_convert(tz).isoformat(),
        'cfs': round(float(peak_v)),
        'ahead': bool(peak_v > 1.02 * obs_now),
        'end_cfs': round(float(central[-1])),
        'band_low_cfs': round(float(low[peak_i])),
        'band_high_cfs': round(float(high[inow:].max())),
    }
    return {
        'central': central,
        'low': low,
        'high': high,
        'tau_hours': round(tau, 1),
        'weights': w,
        'daily': daily,
        'peak': peak,
    }
