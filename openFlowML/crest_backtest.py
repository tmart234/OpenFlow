"""
Crest-timing backtest for the routing forecast (flow_forecast.routed_forecast).

For past floods at a station, this finds each crest in the target's hourly
record (routing.find_crests). It then issues forecasts at fixed lead times
before the crest and scores, for each forecast:

  timing error   forecast crest time - observed crest time (h; + = late)
  crest ratio    forecast crest / observed crest

Each forecast is calibrated on the 28 days before it is issued, like the live
CLI. Upstream gauges get their observed future as a perfect RFC forecast, so
the score isolates routing and timing skill. Anchor gauges and tributaries
(no RFC forecasts) are known only up to the issue time; tributaries then
recede, as live. Rain is off unless --rain:
then the routing rain term gets observed gridMET basin rain, a perfect rain
forecast.

It also measures crest travel times: for every target crest, the lag back to
each upstream gauge's crest in the preceding days. The medians are what the
registry's lag priors should be.

Hourly data:
  USGS:<id>  USGS IV, fetched by year
  DWR:<abbr> Colorado DWR telemetrytimeserieshour (hourly back to 2008+;
             timestamps in local clock time, MST/MDT)
Both are cached under ~/.cache/openflow/hourly.

  python crest_backtest.py --site dolores --start 2015-01-01 --end 2026-11-01
  python crest_backtest.py --site taos-box --timing-weight 0 1 2 --leads 12 24 48 72
"""

import argparse
import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests

import flow_forecast as ff
import routing
from backtest_routing import usgs_hourly
from data import basin_zones, get_basin_precip, get_upstream

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.expanduser(os.getenv('OPENFLOW_HOURLY_CACHE', '~/.cache/openflow/hourly'))
DWR_HOURLY_URL = 'https://dwr.state.co.us/Rest/GET/api/v2/telemetrystations/telemetrytimeserieshour/'
DWR_TZ = 'America/Denver'                     # DWR telemetry stamps local clock time (MST/MDT)
CALIB_DAYS = 28
# A crest is "rain-driven" when at least STORM_MM fell on the station's
# ungauged area in the STORM_WINDOW_H hours before it (gridMET).
STORM_WINDOW_H = 72
STORM_MM = 10.0
HORIZON_DAYS = 5
DEFAULT_LEADS = (6, 12, 24, 48)


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def dwr_hourly(abbrev: str, start, end) -> pd.Series:
    """Hourly discharge (cfs, UTC index) from Colorado DWR telemetry, paged."""
    parts, page = [], 1
    while True:
        r = requests.get(DWR_HOURLY_URL, params={
            'abbrev': abbrev, 'parameter': 'DISCHRG', 'pageSize': 50000, 'pageIndex': page,
            'startDate': pd.Timestamp(start).strftime('%m/%d/%Y'),
            'endDate': pd.Timestamp(end).strftime('%m/%d/%Y')}, timeout=300)
        if r.status_code == 404:              # DWR answers 404 for an empty range
            break
        r.raise_for_status()
        j = r.json()
        rows = j.get('ResultList', [])
        if rows:
            df = pd.DataFrame(rows)
            # The repeated fall-back hour is ambiguous: drop it; shift the skipped spring hour.
            t = pd.DatetimeIndex(pd.to_datetime(df['measDate'])).tz_localize(
                DWR_TZ, ambiguous='NaT', nonexistent='shift_forward').tz_convert('UTC')
            keep = ~t.isna()
            parts.append(pd.Series(pd.to_numeric(df['measValue'], errors='coerce').to_numpy()[keep],
                                   index=t[keep]))
        if page >= int(j.get('PageCount', 1) or 1):
            break
        page += 1
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated()].resample('1h').mean()


def gauge_hourly(site_id: str, start, end) -> pd.Series:
    """Hourly flow for a registry site id, cached per site and calendar year."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    parts = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        path = os.path.join(CACHE_DIR, f"{site_id.replace(':', '_')}_{year}.csv")
        current = year >= pd.Timestamp.now().year
        if os.path.exists(path) and not current:
            s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        else:
            a, b = pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31')
            prefix, sid = site_id.split(':', 1)
            s = dwr_hourly(sid, a, b) if prefix == 'DWR' else usgs_hourly(sid, a, b)
            if not current:
                s.rename('cfs').to_csv(path)
        if len(s):
            s.index = pd.DatetimeIndex(s.index).tz_convert('UTC') if s.index.tz else \
                pd.DatetimeIndex(s.index).tz_localize('UTC')
            parts.append(s)
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated()]
    return s[(s.index >= _utc(start)) & (s.index <= _utc(end))]


def routing_gauges(cfg: dict) -> List[dict]:
    """The routing anchor (cfg['routing']['anchor']) and tributaries, if any."""
    rc = cfg.get('routing') or {}
    return ([rc['anchor']] if rc.get('anchor') else []) + list(rc.get('tributaries') or [])


def load(cfg: dict, site_id: str, start, end) -> Dict[str, pd.Series]:
    """Hourly series for the target, every upstream gauge, the anchor and the routing gauges."""
    ids = [site_id] + [g['site_id'] for g in cfg['upstream']] + \
          ([cfg['anchor']['site_id']] if cfg.get('anchor') else []) + \
          [g['site_id'] for g in routing_gauges(cfg)]
    pad = pd.Timedelta(days=CALIB_DAYS + 5)
    return {i: gauge_hourly(i, pd.Timestamp(start) - pad, pd.Timestamp(end) + pad) for i in ids}


def crest_travel_times(data: Dict[str, pd.Series], site_id: str, cfg: dict,
                       crests: Sequence[pd.Timestamp], max_lag_hours: int = 144) -> pd.DataFrame:
    """Per target crest, hours back to each upstream / anchor gauge's highest flow before it."""
    gauges = list(cfg['upstream']) + ([cfg['anchor']] if cfg.get('anchor') else []) + routing_gauges(cfg)
    rows = []
    for t in crests:
        row = {'crest': t}
        for g in gauges:
            s = data[g['site_id']]
            w = s[(s.index > t - pd.Timedelta(hours=max_lag_hours)) & (s.index <= t)].dropna()
            row[g['site_id']] = (t - w.idxmax()) / pd.Timedelta(hours=1) if len(w) > 24 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def find_events(target: pd.Series, start, end, max_events: int = 60) -> List[pd.Timestamp]:
    """Crest times of the target within [start, end], largest rises first, capped."""
    s = target[(target.index >= _utc(start)) & (target.index <= _utc(end))]
    s = s.resample('1h').mean().interpolate(limit=6, limit_area='inside')
    idx = routing.find_crests(s.to_numpy(dtype=float))
    base = s.rolling(72, min_periods=12).min().shift(1)
    rise = (s.iloc[idx] - base.iloc[idx]).sort_values(ascending=False)
    return sorted(rise.index[:max_events])


def run_one(cfg: dict, site_id: str, data: Dict[str, pd.Series], crest: pd.Timestamp, lead_h: int,
            timing_weight: float, rain: Optional[dict] = None) -> Optional[dict]:
    issue = crest - pd.Timedelta(hours=lead_h)
    grid = pd.date_range(issue - pd.Timedelta(days=CALIB_DAYS), issue + pd.Timedelta(days=HORIZON_DAYS), freq='1h')
    inow = grid.get_loc(issue)
    truth = data[site_id].reindex(grid).interpolate(limit=6, limit_area='inside').to_numpy(float, copy=True)
    if not np.isfinite(truth[max(0, inow - 6):inow + 1]).any():
        return None
    tgt = truth.copy()
    tgt[inow + 1:] = np.nan
    ins = []
    for g in cfg['upstream']:
        x = data[g['site_id']].reindex(grid).interpolate(limit_direction='both').to_numpy(float, copy=True)
        if not np.isfinite(x).all():
            return None
        ins.append(x)
    def known_until_issue(g):
        x = data[g['site_id']].reindex(grid).interpolate(limit=6, limit_area='inside').to_numpy(float, copy=True)
        x[inow + 1:] = np.nan
        return x
    anchor = known_until_issue(cfg['anchor']) if cfg.get('anchor') else None
    kw, storm_mm = {}, np.nan
    rc = cfg.get('routing') or {}
    if rc.get('anchor'):
        kw['near_obs'] = known_until_issue(rc['anchor'])
    if rc.get('tributaries'):
        # No RFC forecasts for these: observed up to the issue time, then a
        # recession, as live (routing.build_input).
        tribs = []
        for g in rc['tributaries']:
            obs = data[g['site_id']]
            obs = obs[(obs.index >= grid[0]) & (obs.index <= issue)].dropna()
            if obs.empty:
                return None
            tribs.append(routing.build_input(grid, obs)[0])
        kw['tributary_inputs'] = tribs
    if rain is not None:
        flux, info = get_basin_precip.routing_rain_flux(rain['cfg'], grid, rain['observed'], None)
        kw.update(rain_flux=flux, rain_calib_mm=float(info['rate_mm_per_hour'].iloc[:inow + 1].sum()))
        k0 = grid.get_loc(crest)
        storm_mm = float(info['rate_mm_per_hour'].iloc[max(0, k0 - STORM_WINDOW_H):k0 + 1].sum())
    try:
        res = ff.routed_forecast(cfg, tgt, inow, ins, anchor_obs=anchor, timing_weight=timing_weight, **kw)
    except Exception as e:                  # one bad window must not stop the backtest
        logger.warning("%s %s lead %d: %s", site_id, crest, lead_h, e)
        return None
    cen = res['ef'].central
    # This crest's forecast: the highest forecast flow from the issue time to
    # 48 h after the observed crest (a later, separate wave in the 5-day
    # window is not this crest).
    k = grid.get_loc(crest)
    hi = min(len(cen), k + routing.CREST_WINDOW_HOURS + 1)
    j = inow + 1 + int(np.nanargmax(cen[inow + 1:hi]))
    return {'crest': crest, 'lead_h': lead_h, 'timing_weight': timing_weight,
            'obs_crest_cfs': float(truth[k]), 'fc_crest_cfs': float(cen[j]),
            'timing_error_h': float((grid[j] - crest) / pd.Timedelta(hours=1)),
            'crest_ratio': float(cen[j] / truth[k]) if truth[k] > 0 else np.nan,
            'calibrated': bool(res['calibrated']), 'rain': rain is not None,
            'storm_mm': storm_mm,
            'lags_h': res['fits'][0].lags[:len(res['gauges'])],
            'routed_from': ('routing anchor' if res.get('near_anchor') is not None
                            else 'anchor' if res['anchor'] is not None else 'upstream')}


def backtest(site: str, start, end, leads: Sequence[int] = DEFAULT_LEADS,
             timing_weights: Sequence[float] = (0.0,), max_events: int = 60,
             with_rain: bool = False) -> pd.DataFrame:
    site_id, cfg = get_upstream.resolve_site(site)
    data = load(cfg, site_id, start, end)
    rain = None
    if with_rain:
        zcfg = basin_zones.station_zones(site_id)
        if zcfg is not None:
            pad = pd.Timedelta(days=CALIB_DAYS + 5)
            a, b = pd.Timestamp(start) - pad, pd.Timestamp(end) + pad
            path = os.path.join(CACHE_DIR, f"rain_{site_id.replace(':', '_')}_{a:%Y%m%d}_{b:%Y%m%d}.pkl")
            if os.path.exists(path):
                observed = pd.read_pickle(path)
            else:
                observed = get_basin_precip.observed_daily(zcfg, a, b, stage4=lambda *x, **k: None)
                os.makedirs(CACHE_DIR, exist_ok=True)
                observed.to_pickle(path)
            rain = {'cfg': zcfg, 'observed': observed}
    crests = find_events(data[site_id], start, end, max_events)
    logger.info("%s: %d crests", site_id, len(crests))
    rows = []
    for crest in crests:
        for lead in leads:
            for w in timing_weights:
                r = run_one(cfg, site_id, data, crest, lead, w, rain)
                if r is not None:
                    rows.append(r)
    out = pd.DataFrame(rows)
    out.attrs['travel'] = crest_travel_times(data, site_id, cfg, crests)
    return out


EVENT_BEFORE_H, EVENT_AFTER_H = 5 * 24, 3 * 24
CAL_LAG_GRID = tuple(range(0, 97, 3))
CAL_K_GRID = (1, 3, 6, 12, 24, 48)
CAL_RAIN_LAG_GRID = tuple(range(0, 49, 3))
CAL_RAIN_K_GRID = (3, 6, 12, 24, 48, 96)


def _event_windows(data, site_id, cfg, crests, rain=None):
    """Per crest: hourly target, routing inputs (the anchor alone when the station has one), rain flux."""
    gauges = [cfg['anchor']] if cfg.get('anchor') else list(cfg['upstream'])
    out = []
    for c in crests:
        grid = pd.date_range(c - pd.Timedelta(hours=EVENT_BEFORE_H), c + pd.Timedelta(hours=EVENT_AFTER_H), freq='1h')
        y = data[site_id].reindex(grid).interpolate(limit=6, limit_area='inside').to_numpy(float, copy=True)
        xs = [data[g['site_id']].reindex(grid).interpolate(limit_direction='both').to_numpy(float, copy=True)
              for g in gauges]
        if not np.isfinite(y).sum() > 0.8 * len(y) or any(not np.isfinite(x).all() for x in xs):
            continue
        f = None
        if rain is not None:
            f, _ = get_basin_precip.routing_rain_flux(rain['cfg'], grid, rain['observed'], None)
        out.append({'crest': c, 'grid': grid, 'y': y, 'xs': xs, 'rain': f, 'k': grid.get_loc(c)})
    return out, gauges


def _pooled_fit(events, routed_cols, warm=72, coef_bounds=(0.25, 2.5), max_rain=1.0, n_up=1):
    """Least squares over all events with one intercept per event; None when a bound is violated."""
    rows_X, rows_y = [], []
    n_ev = len(events)
    for e_i, (e, cols) in enumerate(zip(events, routed_cols)):
        ok = np.isfinite(e['y'])
        ok[:warm] = False
        dummies = np.zeros((ok.sum(), n_ev))
        dummies[:, e_i] = 1.0
        rows_X.append(np.column_stack([c[ok] for c in cols] + [dummies]))
        rows_y.append(e['y'][ok])
    X, y = np.vstack(rows_X), np.concatenate(rows_y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    nc = len(routed_cols[0])
    up, rain_c = coef[:n_up], coef[n_up:nc]
    if (up < coef_bounds[0]).any() or (up > coef_bounds[1]).any() or (rain_c < 0).any() or (rain_c > max_rain).any():
        return None
    pred = X @ coef
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return coef, rmse


def _timing_mae(events, routed_cols, coef, n_up_rain):
    errs = []
    for e_i, (e, cols) in enumerate(zip(events, routed_cols)):
        pred = sum(c * col for c, col in zip(coef[:n_up_rain], cols))
        errs += list(routing.crest_timing_errors(pred, [e['k']]))
    return float(np.mean(np.abs(errs))) if errs else np.nan


def calibrate_events(events, gauges, timing_weight: float = 1.0) -> dict:
    """
    Upstream (or anchor) travel time / smoothing / weight and the rain term's
    lag / smoothing / runoff ratio, fitted on a set of flood windows at once
    (one baseflow intercept per flood). Candidates are ranked by RMSE / std +
    timing_weight * mean |crest timing error| / 12 h. Coordinate descent over
    the upstream gauges, then the rain term on top, then one more upstream
    pass with the rain term in place.
    """
    n_up = len(gauges)
    scale = float(np.nanstd(np.concatenate([e['y'] for e in events])))
    params = [(int(g.get('lag_hours', 24 * g.get('lag_days', 1))), float(g.get('k_hours', 12))) for g in gauges]
    rain_p = None
    has_rain = all(e['rain'] is not None for e in events)

    def cols_for(up_params, rp):
        out = []
        for e in events:
            c = [routing.route(x, lag, k) for x, (lag, k) in zip(e['xs'], up_params)]
            if rp is not None:
                c.append(routing.route(e['rain'], rp[0], rp[1]))
            out.append(c)
        return out

    def score(up_params, rp):
        cols = cols_for(up_params, rp)
        res = _pooled_fit(events, cols, n_up=n_up)
        if res is None:
            return np.inf, None
        coef, rmse = res
        t = _timing_mae(events, cols, coef, n_up + (rp is not None))
        return rmse / scale + timing_weight * (t if np.isfinite(t) else 24.0) / 12.0, (coef, rmse, t)

    best = (np.inf, None)
    for sweep in range(2):
        for i in range(n_up):
            for lag in CAL_LAG_GRID:
                for k in CAL_K_GRID:
                    trial = list(params)
                    trial[i] = (lag, k)
                    sc, info = score(trial, rain_p)
                    if sc < best[0]:
                        best, params = (sc, info), trial
        if has_rain and sweep == 0:
            for lag in CAL_RAIN_LAG_GRID:
                for k in CAL_RAIN_K_GRID:
                    sc, info = score(params, (lag, k))
                    if sc < best[0]:
                        best, rain_p = (sc, info), (lag, k)
    coef, rmse, t = best[1]
    out = {'upstream': [{'site_id': g['site_id'], 'lag_hours': lag, 'k_hours': k, 'weight': round(float(c), 3)}
                        for g, (lag, k), c in zip(gauges, params, coef[:n_up])],
           'rmse_cfs': round(rmse, 1), 'timing_mae_h': round(t, 1), 'events': len(events)}
    if rain_p is not None:
        out['rain'] = {'lag_hours': rain_p[0], 'k_hours': rain_p[1], 'runoff_ratio': round(float(coef[n_up]), 4)}
    return out


def apply_calibration(cfg: dict, cal: dict) -> dict:
    """A registry entry with the event-fitted parameters in place and refitting off."""
    import copy
    new = copy.deepcopy(cfg)
    target = [new['anchor']] if new.get('anchor') else new['upstream']
    for g, u in zip(target, cal['upstream']):
        g.update(lag_hours=u['lag_hours'], k_hours=u['k_hours'], weight=u['weight'])
    if 'rain' in cal:
        new.setdefault('rain', {}).update(cal['rain'])
    new['routing'] = {**(new.get('routing') or {}), 'calibrate': False}
    return new


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Per timing weight and lead: crests, median |timing error|, bias, share within 6 h, crest ratio."""
    g = rows.groupby(['timing_weight', 'lead_h'])
    return pd.DataFrame({
        'crests': g.size(),
        # "It crests now": the naive forecast's |timing error| is the lead time itself.
        'peaks_now_abs_h': g['lead_h'].first().astype(float),
        'median_abs_timing_h': g['timing_error_h'].apply(lambda e: np.median(np.abs(e))),
        'median_timing_h': g['timing_error_h'].median(),
        'within_6h': g['timing_error_h'].apply(lambda e: np.mean(np.abs(e) <= 6)),
        'median_crest_ratio': g['crest_ratio'].median(),
    }).round(2)


def regime_summaries(rows: pd.DataFrame, timing_weight: float) -> dict:
    """summarize() for rain-driven crests (storm_mm >= STORM_MM) and for the rest."""
    if 'storm_mm' not in rows or rows['storm_mm'].isna().all():
        return {}
    out = {}
    for name, mask in (('rain_driven', rows['storm_mm'] >= STORM_MM), ('other', rows['storm_mm'] < STORM_MM)):
        sub = rows[mask]
        if len(sub) and (sub['timing_weight'] == timing_weight).any():
            out[name] = summarize(sub).xs(timing_weight, level='timing_weight')
    return out


def write_skill(path: str, site_id: str, summary: pd.DataFrame, start, end, crests: int, rain: bool,
                by_regime: Optional[dict] = None) -> None:
    """Merge one station's crest scores into the skill JSON the forecast charts quote."""
    import json
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    def leads_of(summ):
        return {str(int(lead)): {'median_abs_timing_h': float(r['median_abs_timing_h']),
                                 'median_timing_h': float(r['median_timing_h']),
                                 'within_6h': float(r['within_6h']),
                                 'median_crest_ratio': float(r['median_crest_ratio'])}
                for lead, r in summ.iterrows()}
    entry = {
        'period': [str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())],
        'crests': int(crests), 'rain': bool(rain),
        'updated': str(pd.Timestamp.now(tz='UTC').date()),
        'leads': leads_of(summary),
    }
    for regime, summ in (by_regime or {}).items():
        entry[regime] = {'crests': int(summ['crests'].max()), 'leads': leads_of(summ)}
    data[site_id] = entry
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join(lines)


# Quality gate (--gate), per site, on the default timing weight:
#  - from 12 h of lead on, the median |crest timing error| must beat the
#    naive "it crests now" forecast (whose error is the lead time itself);
#  - the median forecast/observed crest ratio must stay within 0.7-1.3.
GATE_MIN_LEAD_H = 12
GATE_CREST_RATIO = (0.7, 1.3)


def gate(summary: pd.DataFrame, timing_weight: float) -> List[str]:
    """Failed checks (empty = pass) for one timing weight's rows of summarize()."""
    fails = []
    s = summary.xs(timing_weight, level='timing_weight')
    for lead, r in s.iterrows():
        if lead >= GATE_MIN_LEAD_H and r['median_abs_timing_h'] >= r['peaks_now_abs_h']:
            fails.append(f"lead {lead} h: median crest timing error {r['median_abs_timing_h']} h "
                         f"is no better than 'it crests now' ({r['peaks_now_abs_h']} h)")
        lo, hi = GATE_CREST_RATIO
        if not lo <= r['median_crest_ratio'] <= hi:
            fails.append(f"lead {lead} h: median crest ratio {r['median_crest_ratio']} outside {lo}-{hi}")
    return fails


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--site', required=True)
    p.add_argument('--start', default='2015-01-01')
    p.add_argument('--end', default=str(pd.Timestamp.now().date()))
    p.add_argument('--leads', type=int, nargs='+', default=list(DEFAULT_LEADS))
    p.add_argument('--timing-weight', type=float, nargs='+', default=[ff.TIMING_WEIGHT])
    p.add_argument('--max-events', type=int, default=60)
    p.add_argument('--rain', action='store_true', help='rain term on, with observed gridMET rain')
    p.add_argument('--gate', action='store_true', help='exit 1 when the quality gate fails')
    p.add_argument('--markdown', help='append a markdown summary here (e.g. $GITHUB_STEP_SUMMARY)')
    p.add_argument('--skill-json', help='merge this station\'s scores into this JSON file '
                                        '(data/routing_skill.json, quoted on the forecast charts)')
    p.add_argument('--out', help='CSV of every scored forecast')
    args = p.parse_args(argv)
    rows = backtest(args.site, args.start, args.end, args.leads, args.timing_weight, args.max_events,
                    with_rain=args.rain)
    if rows.empty:
        print("No crests scored")
        return 1
    if args.out:
        rows.to_csv(args.out, index=False)
    summary = summarize(rows)
    print(summary.to_string())
    fails = gate(summary, args.timing_weight[0])
    if args.skill_json:
        site_id, _ = get_upstream.resolve_site(args.site)
        write_skill(args.skill_json, site_id, summary.xs(args.timing_weight[0], level='timing_weight'),
                    args.start, args.end, rows['crest'].nunique(), args.rain,
                    by_regime=regime_summaries(rows, args.timing_weight[0]))
    if args.markdown:
        with open(args.markdown, 'a') as f:
            f.write(f"### Crest timing backtest: {args.site} ({args.start} to {args.end}, rain {'on' if args.rain else 'off'})\n\n")
            f.write(markdown_table(summary.reset_index()) + '\n\n')
            f.write('**Gate: ' + ('PASSED' if not fails else 'FAILED') + '**\n\n' + ''.join(f'- {x}\n' for x in fails) + '\n')
    travel = rows.attrs['travel']
    print('\nCrest travel times to the target (median hours, from', len(travel), 'crests):')
    for col in travel.columns[1:]:
        print(f"  {col}: {travel[col].median():.0f} h (25-75%: {travel[col].quantile(.25):.0f}-{travel[col].quantile(.75):.0f})")
    print('\nGate:', 'PASSED' if not fails else 'FAILED\n  ' + '\n  '.join(fails))
    return 1 if (args.gate and fails) else 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
