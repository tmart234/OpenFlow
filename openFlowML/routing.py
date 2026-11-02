"""
Upstream-routing streamflow forecaster (hourly, physically structured).

This is the "what is already in the river upstream" forecast. It answers the
question a single-gauge ML model cannot: when a flood wave is 100+ miles
upstream, when does it reach the target gauge and how big is it when it gets
there?

Model, per target gauge with upstream gauges i = 1..n:

    Q_target(t) = b + sum_i c_i * R_i(Q_i)(t)
    R_i(x)      = linear_reservoir(x, K_i) shifted later by lag_i hours

  - lag_i (hours) is the pure travel time, K_i (hours) the attenuation of a
    linear reservoir (bigger K = flatter, later, more spread-out wave).
  - c_i absorbs gains/losses along the reach (diversions, seepage, side
    tributaries); b absorbs base inflow between the gauges (springs,
    ungauged tributaries) or a constant diversion loss when negative.

Calibration: a grid search over (lag_i, K_i), with c_i and b solved by least
squares at every grid point, against recent hourly observations (NWPS keeps
~30 days). Calibrating on the most recent weeks means the parameters reflect
current conditions (irrigation diversions on/off, wet vs dry channel). When
the calibration window has too little upstream variation for the travel
times to be identifiable, the caller falls back to registry defaults
(`default_fit`).

Forecast: upstream inputs are observations up to "now", then the official
NWS/RFC forecast for each upstream gauge, then an exponential recession past
the end of that forecast. The best few parameter sets form an ensemble; the
forecast is bias-corrected to the latest target observation (correction
decays with lead time) and the band is the ensemble spread widened by the
calibration RMSE.

Pure numpy/pandas; no network. See data/nwps.py for the fetchers and
flow_forecast.py for the CLI that wires it all together.
"""

from dataclasses import dataclass, field
import itertools
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_LAG_GRID = tuple(range(0, 97, 6))          # hours
DEFAULT_K_GRID = (1, 6, 12, 24, 48, 72, 96, 120)   # hours
WARMUP_HOURS = 72
# Recession past the end of an upstream forecast decays at least this fast
# (e-folding time, hours) toward the recent minimum flow.
MAX_RECESSION_TAU_HOURS = 120.0
# Minimum relative variability ((p95 - p5) / median) an upstream series needs
# in the calibration window for travel times to be identifiable.
MIN_RELATIVE_RANGE = 0.5


# --------------------------------------------------------------------------
# Routing primitives
# --------------------------------------------------------------------------

def linear_reservoir(x: np.ndarray, k_hours: float) -> np.ndarray:
    """
    Route an hourly series through a linear reservoir with time constant
    k_hours (explicit Euler, dt = 1 h). k <= 1 is a pass-through. The state
    starts at x[0], i.e. steady state at the beginning of the series.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    k = max(float(k_hours), 1.0)
    if k == 1.0:
        return x.copy()
    y = np.empty_like(x)
    s = x[0]
    a = 1.0 / k
    for i, xi in enumerate(x):
        s = s + (xi - s) * a
        y[i] = s
    return y


def shift(x: np.ndarray, lag_hours: int) -> np.ndarray:
    """Delay a series by lag_hours, padding the front with its first value."""
    x = np.asarray(x, dtype=float)
    lag = int(lag_hours)
    if lag <= 0 or x.size == 0:
        return x.copy()
    if lag >= x.size:
        return np.full_like(x, x[0])
    return np.concatenate([np.full(lag, x[0]), x[:-lag]])


def route(x: np.ndarray, lag_hours: int, k_hours: float) -> np.ndarray:
    """Attenuate (linear reservoir K) then translate (lag) an hourly series."""
    return shift(linear_reservoir(x, k_hours), lag_hours)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

@dataclass
class RoutingFit:
    lags: List[int]
    ks: List[float]
    coefs: List[float]
    intercept: float
    rmse: float
    source: str = 'calibrated'   # 'calibrated' | 'default'

    def predict(self, inputs: Sequence[np.ndarray]) -> np.ndarray:
        out = np.full(len(inputs[0]), float(self.intercept))
        for x, lag, k, c in zip(inputs, self.lags, self.ks, self.coefs):
            out = out + c * route(x, lag, k)
        return out

    def to_dict(self) -> dict:
        return {'lags_hours': list(map(int, self.lags)),
                'k_hours': list(map(float, self.ks)),
                'coefs': list(map(float, self.coefs)),
                'intercept': float(self.intercept),
                'rmse': float(self.rmse), 'source': self.source}


def _solve(target, routed, rows):
    """Least-squares c_i, b on `rows`; returns (coefs, intercept, rmse)."""
    X = np.column_stack([r[rows] for r in routed] + [np.ones(rows.sum())])
    y = target[rows]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    rmse = float(np.sqrt(np.mean((X @ coef - y) ** 2)))
    return coef[:-1], float(coef[-1]), rmse


def is_informative(x: np.ndarray, min_relative_range: float = MIN_RELATIVE_RANGE) -> bool:
    """True when a calibration-window series varies enough to fit travel times."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 48:
        return False
    med = np.median(x)
    if med <= 0:
        return False
    return (np.percentile(x, 95) - np.percentile(x, 5)) / med >= min_relative_range


def fit(target: np.ndarray,
        inputs: Sequence[np.ndarray],
        lag_grid: Sequence[int] = DEFAULT_LAG_GRID,
        k_grid: Sequence[float] = DEFAULT_K_GRID,
        top_n: int = 5,
        warmup_hours: int = WARMUP_HOURS,
        sweeps: int = 3,
        coef_bounds: Optional[Tuple[float, float]] = None) -> List[RoutingFit]:
    """
    Calibrate (lag_i, K_i) by grid search and (c_i, b) by least squares.

    Exhaustive for <= 2 upstream inputs; coordinate descent (one input's
    (lag, K) at a time, `sweeps` passes) beyond that so cost stays linear in
    the number of inputs. Parameter sets with any negative c_i are rejected
    (an upstream gauge cannot contribute negative flow), as are sets with a
    coefficient outside `coef_bounds` when given -- e.g. (0.25, 2.5): a gauge
    "tripling" on its way downstream means the fit is absorbing something
    else (local tributary inflow) and should not be trusted. Returns up to top_n
    fits, best (lowest RMSE) first. Inputs and target must be hourly arrays
    of equal length with no NaN in the inputs; NaN target hours are ignored.
    """
    target = np.asarray(target, dtype=float)
    inputs = [np.asarray(x, dtype=float) for x in inputs]
    n = len(target)
    if not inputs:
        raise ValueError("need at least one upstream input")
    if any(len(x) != n for x in inputs):
        raise ValueError("inputs and target must have equal length")
    if any(not np.isfinite(x).all() for x in inputs):
        raise ValueError("upstream inputs must be gap-free (interpolate first)")
    rows = np.isfinite(target)
    rows[:min(warmup_hours, n)] = False
    if rows.sum() < 24:
        raise ValueError("not enough target observations to calibrate")

    combos = list(itertools.product(lag_grid, k_grid))
    cache: Dict[tuple, np.ndarray] = {}

    def routed(i, lag, k):
        key = (i, lag, k)
        if key not in cache:
            cache[key] = route(inputs[i], lag, k)
        return cache[key]

    results: Dict[tuple, RoutingFit] = {}

    def evaluate(params):
        if params in results:
            return results[params]
        r = [routed(i, lag, k) for i, (lag, k) in enumerate(params)]
        coefs, b, rmse = _solve(target, r, rows)
        res = None
        lo, hi = coef_bounds if coef_bounds else (0.0, np.inf)
        if (coefs >= lo).all() and (coefs <= hi).all():
            res = RoutingFit([p[0] for p in params], [p[1] for p in params],
                             coefs.tolist(), b, rmse)
        results[params] = res
        return res

    if len(inputs) <= 2:
        for params in itertools.product(combos, repeat=len(inputs)):
            evaluate(tuple(params))
    else:
        current = [(lag_grid[len(lag_grid) // 3], k_grid[len(k_grid) // 3])] * len(inputs)
        for _ in range(sweeps):
            for i in range(len(inputs)):
                best = None
                for c in combos:
                    trial = list(current)
                    trial[i] = c
                    res = evaluate(tuple(trial))
                    if res is not None and (best is None or res.rmse < best[0]):
                        best = (res.rmse, c)
                if best is not None:
                    current[i] = best[1]

    fits = sorted((f for f in results.values() if f is not None), key=lambda f: f.rmse)
    return fits[:top_n]


def add_input(target: np.ndarray, inputs: Sequence[np.ndarray], base_fits: Sequence[RoutingFit],
              extra: np.ndarray, lag_grid: Sequence[int] = DEFAULT_LAG_GRID,
              k_grid: Sequence[float] = DEFAULT_K_GRID, max_coef: Optional[float] = None,
              warmup_hours: int = WARMUP_HOURS) -> List[RoutingFit]:
    """
    Second-stage fit of an extra input (the rain term) on the RESIDUAL of
    each base fit: the upstream travel times AND weights stay exactly as the
    upstream hydrographs identified them, and the extra input only explains
    what they could not. Re-solving everything jointly lets rain "steal"
    weight from an upstream gauge whenever both rise in the same storm.
    Searches the extra input's (lag, K); solves its coefficient (bounded to
    [0, max_coef]) plus an intercept adjustment. A base fit with no valid
    extra term keeps a zero coefficient.
    """
    target = np.asarray(target, dtype=float)
    extra = np.asarray(extra, dtype=float)
    rows = np.isfinite(target)
    rows[:min(warmup_hours, len(target))] = False
    extra_routed = {(lag, k): route(extra, lag, k) for lag in lag_grid for k in k_grid}
    out = []
    for base in base_fits:
        resid = target - base.predict([np.asarray(x, float) for x in inputs])
        best = None
        for (lag, k), r in extra_routed.items():
            coefs, b, rmse = _solve(resid, [r], rows)
            c = float(coefs[0])
            if c < 0 or (max_coef is not None and c > max_coef):
                continue
            if best is None or rmse < best.rmse:
                best = RoutingFit(list(base.lags) + [int(lag)], list(base.ks) + [float(k)],
                                  list(base.coefs) + [c], base.intercept + b, rmse, source=base.source)
        if best is None:
            best = RoutingFit(list(base.lags) + [int(lag_grid[0])], list(base.ks) + [float(k_grid[0])],
                              list(base.coefs) + [0.0], base.intercept, base.rmse, source=base.source)
        out.append(best)
    return sorted(out, key=lambda f: f.rmse)


def default_fit(target: np.ndarray, inputs: Sequence[np.ndarray],
                lags: Sequence[int], ks: Sequence[float],
                weights: Sequence[float], warmup_hours: int = WARMUP_HOURS) -> RoutingFit:
    """
    Registry-default routing: fixed lag/K/weight per upstream gauge; only the
    intercept is fitted (mean residual), so the result is anchored to the
    target's recent base flow. Used when the calibration window is too quiet
    to identify travel times.
    """
    f = RoutingFit(list(map(int, lags)), list(map(float, ks)),
                   list(map(float, weights)), 0.0, float('nan'), source='default')
    target = np.asarray(target, dtype=float)
    pred = f.predict([np.asarray(x, dtype=float) for x in inputs])
    rows = np.isfinite(target)
    rows[:min(warmup_hours, len(target))] = False
    if rows.sum() == 0:
        rows = np.isfinite(target)
    resid = target[rows] - pred[rows]
    f.intercept = float(np.mean(resid)) if resid.size else 0.0
    f.rmse = float(np.sqrt(np.mean((resid - f.intercept) ** 2))) if resid.size else float('nan')
    return f


# --------------------------------------------------------------------------
# Building forecast inputs
# --------------------------------------------------------------------------

def extend_recession(values: np.ndarray, last_idx: int, floor: float,
                     max_tau_hours: float = MAX_RECESSION_TAU_HOURS) -> np.ndarray:
    """
    Fill values[last_idx+1:] with an exponential recession toward `floor`,
    using the decay rate of the 24 h before last_idx (forced to decay at
    least as fast as e-folding time max_tau_hours, so a series that is still
    rising at the end of its forecast does not run away).
    """
    out = np.asarray(values, dtype=float).copy()
    n = len(out)
    if last_idx >= n - 1:
        return out
    last = out[last_idx]
    prev = out[max(last_idx - 24, 0)]
    span = max(last_idx - max(last_idx - 24, 0), 1)
    a, b = max(last - floor, 1.0), max(prev - floor, 1.0)
    rate = np.log(a / b) / span
    rate = min(rate, -1.0 / max_tau_hours)
    steps = np.arange(1, n - last_idx)
    out[last_idx + 1:] = floor + (last - floor) * np.exp(rate * steps)
    return out


def build_input(grid: pd.DatetimeIndex, observed: pd.Series,
                forecast: Optional[pd.Series] = None,
                floor: Optional[float] = None) -> np.ndarray:
    """
    Assemble one upstream gauge's hourly input on `grid` (UTC): observations
    up to the last observed hour, then the RFC forecast (if any), then an
    exponential recession. Interior gaps are interpolated; leading hours
    before the first observation take the first observed value.

    Returns (values, info) where info records the last observed time and the
    last forecast time actually used.
    """
    obs = observed.dropna() if observed is not None else pd.Series(dtype=float)
    if obs.empty:
        raise ValueError("upstream gauge has no observations")
    last_obs_t = obs.index.max()
    s = obs.reindex(grid)
    fc_end = None
    if forecast is not None and not forecast.dropna().empty:
        fc = forecast.dropna()
        fc = fc[fc.index > last_obs_t]
        if not fc.empty:
            # Join the forecast onto the last observation so the handoff hour
            # interpolates instead of stepping.
            joined = pd.concat([obs.iloc[-1:], fc])
            joined = joined[~joined.index.duplicated(keep='last')]
            joined = joined.reindex(joined.index.union(grid)).interpolate(
                method='time', limit_area='inside').reindex(grid)
            s = s.where(grid <= last_obs_t, joined)
            fc_end = fc.index.max()
    s = s.interpolate(method='time', limit_area='inside')
    values = s.to_numpy(dtype=float, copy=True)
    finite = np.where(np.isfinite(values))[0]
    first, last = finite.min(), finite.max()
    values[:first] = values[first]
    if floor is None:
        floor = float(np.nanmin(obs.to_numpy()))
    values = extend_recession(values, last, floor)
    return values, {'last_observed': last_obs_t, 'forecast_end': fc_end}


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------

@dataclass
class EnsembleForecast:
    central: np.ndarray
    low: np.ndarray
    high: np.ndarray
    members: np.ndarray
    fits: List[RoutingFit] = field(default_factory=list)


def forecast(fits: Sequence[RoutingFit], inputs: Sequence[np.ndarray],
             target_obs: np.ndarray, now_idx: int,
             bias_tau_hours: float = 24.0, anchor_hours: int = 1) -> EnsembleForecast:
    """
    Run each fit over the full (obs + forecast) inputs, bias-correct every
    member to the mean of the last `anchor_hours` target observations (the
    correction decays with e-folding time bias_tau_hours), and form the band.

    Hours <= now_idx carry the observations themselves. The band is the
    ensemble min/max widened by the best fit's calibration RMSE, ramping in
    over the first 24 h of lead time; it never goes below zero.
    """
    target_obs = np.asarray(target_obs, dtype=float)
    n = len(inputs[0])
    t = np.arange(n) - now_idx
    decay = np.where(t <= 0, 1.0, np.exp(-np.maximum(t, 0) / bias_tau_hours))
    lo = max(0, now_idx - anchor_hours + 1)
    anchor = np.nanmean(target_obs[lo:now_idx + 1])
    members = []
    for f in fits:
        pred = f.predict(inputs)
        bias = anchor - np.mean(pred[lo:now_idx + 1])
        m = pred + bias * decay
        m[:now_idx + 1] = target_obs[:now_idx + 1]
        members.append(np.maximum(m, 0.0))
    members = np.array(members)
    central = members[0]
    rmse = fits[0].rmse if np.isfinite(fits[0].rmse) else 0.0
    ramp = np.clip(t / 24.0, 0.0, 1.0)
    low = np.maximum(members.min(axis=0) - rmse * ramp, 0.0)
    high = members.max(axis=0) + rmse * ramp
    return EnsembleForecast(central=central, low=low, high=high,
                            members=members, fits=list(fits))


def daily_summary(times: pd.DatetimeIndex, central: np.ndarray, low: np.ndarray,
                  high: np.ndarray, tz: str) -> pd.DataFrame:
    """Per local calendar day: min / mean / max of the central forecast plus band extremes."""
    df = pd.DataFrame({'central': central, 'low': low, 'high': high},
                      index=times.tz_convert(tz))
    g = df.groupby(df.index.date)
    out = pd.DataFrame({
        'min_cfs': g['central'].min(), 'mean_cfs': g['central'].mean(),
        'max_cfs': g['central'].max(), 'band_low_cfs': g['low'].min(),
        'band_high_cfs': g['high'].max(),
    })
    out.index.name = 'date'
    return out


def peak(times: pd.DatetimeIndex, values: np.ndarray, start_idx: int = 0):
    """(time, value, index) of the maximum at or after start_idx."""
    i = start_idx + int(np.nanargmax(values[start_idx:]))
    return times[i], float(values[i]), i
