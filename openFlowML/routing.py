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


# Flow-dependent travel time (kinematic wave): the lag at flow q is
# lag_ref * (q / q_ref) ** -exponent, with q / q_ref clipped to this range.
CELERITY_FLOW_RATIO = (0.1, 10.0)


def variable_shift(y: np.ndarray, lag_hours: float, exponent: float, q_ref: float) -> np.ndarray:
    """
    Delay a series by a flow-dependent travel time: water passing at hour s
    (flow y[s]) arrives lag_hours * (y[s] / q_ref) ** -exponent hours later.
    A flood wave moves faster than low water (kinematic wave; exponent 0.4
    for a wide channel under Manning's equation): the Rio Grande takes ~12 h
    from Cerro to Taos Junction Bridge at 300 cfs and ~4-6 h at 3,500 cfs.
    Water may not overtake: arrival times are made non-decreasing, so a
    fast rise behind slow water steepens into a front. Hours before the
    first arrival take y[0]; hours after the last arrival hold y[-1].
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y.copy()
    s = np.arange(n, dtype=float)
    ratio = np.clip(y / float(q_ref), *CELERITY_FLOW_RATIO)
    arrival = np.maximum.accumulate(s + max(float(lag_hours), 0.0) * ratio ** (-float(exponent)))
    arrival = arrival + s * 1e-9                       # strictly increasing for interp
    return np.interp(s, arrival, y, left=y[0], right=y[-1])


def route(x: np.ndarray, lag_hours: int, k_hours: float,
          celerity: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Attenuate (linear reservoir K) then translate (lag) an hourly series.
    celerity = (exponent, q_ref): the lag is lag_hours at q_ref and varies
    with flow (variable_shift); None or exponent 0 is a fixed lag.
    """
    y = linear_reservoir(x, k_hours)
    if celerity and float(celerity[0]):
        return variable_shift(y, lag_hours, float(celerity[0]), float(celerity[1]))
    return shift(y, lag_hours)


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
    # Mean absolute crest timing error (h) on the calibration window's crests,
    # when fit() was asked to score timing (None otherwise).
    timing_mae_hours: Optional[float] = None
    # Per input: (exponent, q_ref) for a flow-dependent lag (route), or None.
    celerity: Optional[List[Optional[Tuple[float, float]]]] = None

    def _celerity(self, i: int):
        return self.celerity[i] if self.celerity and i < len(self.celerity) else None

    def predict(self, inputs: Sequence[np.ndarray]) -> np.ndarray:
        out = np.full(len(inputs[0]), float(self.intercept))
        for i, (x, lag, k, c) in enumerate(zip(inputs, self.lags, self.ks, self.coefs)):
            out = out + c * route(x, lag, k, self._celerity(i))
        return out

    def to_dict(self) -> dict:
        d = {'lags_hours': list(map(int, self.lags)),
             'k_hours': list(map(float, self.ks)),
             'coefs': list(map(float, self.coefs)),
             'intercept': float(self.intercept),
             'rmse': float(self.rmse), 'source': self.source,
             'timing_mae_hours': self.timing_mae_hours}
        if self.celerity and any(self.celerity):
            d['celerity'] = [list(c) if c else None for c in self.celerity]
        return d


# --------------------------------------------------------------------------
# Crests
# --------------------------------------------------------------------------

CREST_MIN_RATIO = 1.5        # crest >= 1.5x the lowest flow of the 3 days before
CREST_MIN_RISE_CFS = 150.0   # ...and at least this much above it
CREST_FALL_RATIO = 0.75      # ...and the flow falls below 75% of it within 3 days
CREST_SEPARATION_HOURS = 96  # one crest per 4 days
CREST_WINDOW_HOURS = 48      # a routed crest within +-48 h of the observed one


def find_crests(values: np.ndarray, min_ratio: float = CREST_MIN_RATIO,
                min_rise: float = CREST_MIN_RISE_CFS,
                separation: int = CREST_SEPARATION_HOURS,
                base_window: int = 72, fall_ratio: float = CREST_FALL_RATIO,
                fall_window: int = 72) -> np.ndarray:
    """
    Indices of flood crests in an hourly flow series: local maxima of the
    3-hour mean, at least `separation` hours apart, that rose at least
    `min_ratio` x and `min_rise` cfs above the lowest flow of the preceding
    `base_window` hours and fall below `fall_ratio` x within `fall_window`
    hours after. Sharp rain / flood-wave crests only: a snowmelt plateau that
    builds over weeks does not qualify, and its "crest time" would be noise.
    NaN hours never count as crests.
    """
    from scipy.signal import find_peaks
    x = pd.Series(np.asarray(values, dtype=float)).rolling(3, center=True, min_periods=1).mean()
    base = x.rolling(base_window, min_periods=12).min().shift(1)
    after = x[::-1].rolling(fall_window, min_periods=12).min()[::-1].shift(-1)
    filled = x.fillna(-np.inf).to_numpy()
    peaks, _ = find_peaks(filled, distance=max(1, int(separation)))
    keep = [int(i) for i in peaks
            if np.isfinite(x.iloc[i]) and np.isfinite(base.iloc[i]) and base.iloc[i] > 0
            and x.iloc[i] >= min_ratio * base.iloc[i] and x.iloc[i] - base.iloc[i] >= min_rise
            and np.isfinite(after.iloc[i]) and after.iloc[i] <= fall_ratio * x.iloc[i]]
    return np.asarray(keep, dtype=int)


def crest_timing_errors(pred: np.ndarray, crest_idx: Sequence[int],
                        window: int = CREST_WINDOW_HOURS) -> np.ndarray:
    """Hours from each observed crest to the predicted maximum within +-window (positive = late)."""
    pred = np.asarray(pred, dtype=float)
    out = []
    for k in crest_idx:
        lo, hi = max(0, k - window), min(len(pred), k + window + 1)
        seg = pred[lo:hi]
        if seg.size == 0 or not np.isfinite(seg).any():
            continue
        out.append(lo + int(np.nanargmax(seg)) - k)
    return np.asarray(out, dtype=float)


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
        coef_bounds: Optional[Tuple[float, float]] = None,
        grids: Optional[Sequence[Tuple[Sequence[int], Sequence[float]]]] = None,
        timing_weight: float = 0.0,
        celerity: Optional[Sequence[Optional[Tuple[float, float]]]] = None) -> List[RoutingFit]:
    """
    Calibrate (lag_i, K_i) by grid search and (c_i, b) by least squares.

    Exhaustive for <= 2 upstream inputs; coordinate descent (one input's
    (lag, K) at a time, `sweeps` passes) beyond that so cost stays linear in
    the number of inputs. `grids` gives each input its own (lag_grid,
    k_grid) -- e.g. a window around a registry prior -- instead of the shared
    lag_grid / k_grid. Parameter sets with any negative c_i are rejected
    (an upstream gauge cannot contribute negative flow), as are sets with a
    coefficient outside `coef_bounds` when given -- e.g. (0.25, 2.5): a gauge
    "tripling" on its way downstream means the fit is absorbing something
    else (local tributary inflow) and should not be trusted. Returns up to top_n
    fits, best first. Inputs and target must be hourly arrays of equal
    length with no NaN in the inputs; NaN target hours are ignored.

    Ranking: by RMSE, or, with timing_weight > 0 and flood crests in the
    target (find_crests), by RMSE / std(target) + timing_weight * mean
    |crest timing error| / 12 h. A pure RMSE fit is dominated by the many
    ordinary hours; in a rain flood every gauge rises at once, so it can pick
    a travel time of ~0 h and put the crest hours early.

    celerity: per input, (exponent, q_ref) for a flow-dependent lag, or
    None; the lag grid is then the lag at q_ref (route).
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

    crest_idx = find_crests(np.where(rows, target, np.nan)) if timing_weight > 0 else np.array([], int)
    scale = float(np.nanstd(target[rows])) or 1.0

    if grids is None:
        grids = [(lag_grid, k_grid)] * len(inputs)
    if len(grids) != len(inputs):
        raise ValueError("need one (lag_grid, k_grid) per input")
    combos_per_input = [list(itertools.product(lg, kg)) for lg, kg in grids]
    cache: Dict[tuple, np.ndarray] = {}

    cel = list(celerity) if celerity is not None else [None] * len(inputs)
    if len(cel) != len(inputs):
        raise ValueError("need one celerity entry per input")

    def routed(i, lag, k):
        key = (i, lag, k)
        if key not in cache:
            cache[key] = route(inputs[i], lag, k, cel[i])
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
                             coefs.tolist(), b, rmse, celerity=cel if any(cel) else None)
            if len(crest_idx):
                pred = b + sum(c * ri for c, ri in zip(coefs, r))
                err = crest_timing_errors(pred, crest_idx)
                res.timing_mae_hours = float(np.mean(np.abs(err))) if err.size else None
        results[params] = res
        return res

    def score(f: RoutingFit) -> float:
        if f.timing_mae_hours is None:
            return f.rmse / scale
        return f.rmse / scale + timing_weight * f.timing_mae_hours / 12.0

    if len(inputs) <= 2:
        for params in itertools.product(*combos_per_input):
            evaluate(tuple(params))
    else:
        current = [(lg[len(lg) // 3], kg[len(kg) // 3]) for lg, kg in grids]
        for _ in range(sweeps):
            for i in range(len(inputs)):
                best = None
                for c in combos_per_input[i]:
                    trial = list(current)
                    trial[i] = c
                    res = evaluate(tuple(trial))
                    if res is not None and (best is None or score(res) < best[0]):
                        best = (score(res), c)
                if best is not None:
                    current[i] = best[1]

    fits = sorted((f for f in results.values() if f is not None), key=score)
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
                                  list(base.coefs) + [c], base.intercept + b, rmse, source=base.source,
                                  celerity=_extend(base.celerity, len(base.lags)))
        if best is None:
            best = RoutingFit(list(base.lags) + [int(lag_grid[0])], list(base.ks) + [float(k_grid[0])],
                              list(base.coefs) + [0.0], base.intercept, base.rmse, source=base.source,
                              celerity=_extend(base.celerity, len(base.lags)))
        out.append(best)
    return sorted(out, key=lambda f: f.rmse)


def _extend(celerity, n_base: int):
    """A fit's celerity list with a fixed-lag (None) entry for one more input."""
    if not celerity:
        return None
    return list(celerity) + [None] * (n_base + 1 - len(celerity))


def default_fit(target: np.ndarray, inputs: Sequence[np.ndarray],
                lags: Sequence[int], ks: Sequence[float],
                weights: Sequence[float], warmup_hours: int = WARMUP_HOURS,
                celerity: Optional[Sequence[Optional[Tuple[float, float]]]] = None) -> RoutingFit:
    """
    Registry-default routing: fixed lag/K/weight per upstream gauge; only the
    intercept is fitted (mean residual), so the result is anchored to the
    target's recent base flow. Used when the calibration window is too quiet
    to identify travel times.
    """
    f = RoutingFit(list(map(int, lags)), list(map(float, ks)),
                   list(map(float, weights)), 0.0, float('nan'), source='default',
                   celerity=list(celerity) if celerity is not None and any(celerity) else None)
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


# An RFC forecast is hours old by the time it is used; where it disagrees with
# the gauge's latest reading, it is shifted onto that reading and the shift
# decays with this time constant (blend_forecast).
FORECAST_BLEND_TAU_HOURS = 24.0


def blend_forecast(observed: pd.Series, forecast: pd.Series,
                   tau_hours: float = FORECAST_BLEND_TAU_HOURS) -> pd.Series:
    """
    Shift an RFC forecast onto the gauge's latest observation: offset =
    observed - forecast at the last observed time (the forecast interpolated
    there), applied as offset * exp(-hours since then / tau_hours), floored
    at 0. Only when the forecast spans the last observation; a forecast that
    starts after it is returned unchanged.

    Nov 5 2026, 8 AM: Rio Grande at Alamosa read 2,409 cfs and was falling
    ~50 cfs/h; the RFC forecast issued 14 h earlier still had 2,670 cfs for
    11 AM and stayed ~400 cfs high. Taken as is, the handoff stepped the input
    up 260 cfs and the extra water routed into a higher, flatter Taos Box crest.
    """
    obs, fc = observed.dropna(), forecast.dropna()
    if obs.empty or fc.empty or not tau_hours:
        return forecast
    t0 = obs.index.max()
    if not (fc.index.min() <= t0 <= fc.index.max()):
        return forecast
    at = float(np.interp(t0.value, fc.index.asi8, fc.to_numpy(dtype=float)))
    hours = np.clip((fc.index - t0) / pd.Timedelta(hours=1), 0.0, None)
    return (fc + (float(obs.iloc[-1]) - at) * np.exp(-np.asarray(hours) / tau_hours)).clip(lower=0.0)


def build_input(grid: pd.DatetimeIndex, observed: pd.Series,
                forecast: Optional[pd.Series] = None,
                floor: Optional[float] = None,
                blend_tau_hours: Optional[float] = None) -> np.ndarray:
    """
    Assemble one upstream gauge's hourly input on `grid` (UTC): observations
    up to the last observed hour, then the RFC forecast (if any), then an
    exponential recession. Interior gaps are interpolated; leading hours
    before the first observation take the first observed value.
    blend_tau_hours: shift the forecast onto the last observation first
    (blend_forecast).

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
        if blend_tau_hours:
            fc = blend_forecast(obs, fc, blend_tau_hours)
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


# Anchor gauges (see anchored_input).
ANCHOR_RATIO_HOURS = 7 * 24
ANCHOR_RATIO_MAX = 1.5


def anchored_input(anchor_obs: np.ndarray, supply: np.ndarray,
                   ratio_hours: int = ANCHOR_RATIO_HOURS,
                   max_ratio: float = ANCHOR_RATIO_MAX,
                   anchor_cfg: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
    """
    Hourly input for an anchor gauge: a gauge below the diversions between
    the upstream gauges and the target (Lobatos for the Taos Box). The hourly
    twin of get_upstream.anchored_daily.

    anchor_obs: the anchor's flow on the grid, NaN where not observed.
    supply: the upstream gauges' flow routed to the anchor's clock, gap-free
    (observations, then RFC forecasts, then a recession).

    Observed hours keep the anchor (interior gaps interpolated; leading hours
    take the first observation). After its last observation the anchor
    continues as anchor_last + passed_change(supply - supply_last), floored
    at 0. r = mean(anchor) / mean(supply) over the last `ratio_hours` hours
    both exist, clipped to [0, max_ratio], is the share of an upstream change
    currently getting past the diversions (about 0.1-0.4 in irrigation
    season, about 0.9 in winter); the ditches' capacity is limited, so the
    part of a big rise beyond it passes at a higher rate
    (get_upstream.passed_change; anchor_cfg overrides its defaults).
    anchor_cfg['fixed_ratio'] replaces r where nothing is diverted between
    the supply and the anchor.

    Returns (values, info) with info = {'ratio', 'last_index'}.
    """
    a = pd.Series(np.asarray(anchor_obs, dtype=float))
    supply = np.asarray(supply, dtype=float)
    if len(a) != len(supply):
        raise ValueError("anchor and supply must have equal length")
    finite = np.where(np.isfinite(a.to_numpy()))[0]
    if finite.size == 0:
        raise ValueError("anchor gauge has no observations")
    first, last = int(finite.min()), int(finite.max())
    values = a.interpolate(limit_area='inside').to_numpy(dtype=float, copy=True)
    values[:first] = values[first]
    lo = max(first, last - ratio_hours + 1)
    window = np.isfinite(values[lo:last + 1]) & np.isfinite(supply[lo:last + 1])
    mean_supply = float(np.mean(supply[lo:last + 1][window])) if window.any() else 0.0
    ratio = (float(np.clip(np.mean(values[lo:last + 1][window]) / mean_supply, 0.0, max_ratio))
             if mean_supply > 0 else 0.0)
    if anchor_cfg and anchor_cfg.get('fixed_ratio') is not None:
        # No diversions between the supply and this anchor (e.g. Lobatos ->
        # Cerro): changes pass at a fixed rate; the gains between them are
        # base flow, so the mean ratio would understate a rise.
        ratio = float(anchor_cfg['fixed_ratio'])
    from data.get_upstream import passed_change
    values[last + 1:] = np.maximum(
        values[last] + passed_change(supply[last + 1:] - supply[last], ratio, anchor_cfg), 0.0)
    return values, {'ratio': ratio, 'last_index': last}


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
