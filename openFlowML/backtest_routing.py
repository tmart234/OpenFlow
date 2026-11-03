"""
Backtest the routing forecast (flow_forecast.routed_forecast) on history.

For each issue time it calibrates on the previous `calib_days`, forecasts the
next `days`, and scores daily-mean flow against what the target gauge did,
next to persistence ("tomorrow = today"). For a station with an anchor gauge
it scores both ways: upstream gauges routed directly, and routed through the
anchor (Lobatos for the Taos Box).

Data:
- Target: USGS hourly (IV).
- Upstream and anchor gauges: daily means placed at local noon and
  interpolated hourly. CO DWR keeps hourly data for only about 30 days.
- Upstream gauges get their observed future as a perfect RFC forecast. The
  anchor has no RFC forecast, so it is known only up to the issue time (minus
  12 h, because today's daily mean is not complete).
- --rain adds the rain term with gridMET as a perfect rain forecast.

  python backtest_routing.py --site taos-box --start 2025-04-15 --end 2026-10-27 --rain
"""

import argparse
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

import flow_forecast as ff
from data import basin_zones, get_basin_precip, get_upstream

logger = logging.getLogger(__name__)

USGS_IV_URL = "https://nwis.waterservices.usgs.gov/nwis/iv/"
TZ_OFFSETS = {'MST': '-07:00', 'MDT': '-06:00', 'PST': '-08:00', 'PDT': '-07:00',
              'CST': '-06:00', 'CDT': '-05:00'}


def usgs_hourly(site: str, start, end) -> pd.Series:
    """Hourly-mean discharge (cfs, UTC index) from the USGS IV service, fetched by calendar year."""
    parts = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        a = max(pd.Timestamp(start), pd.Timestamp(f'{y}-01-01'))
        b = min(pd.Timestamp(end), pd.Timestamp(f'{y}-12-31'))
        r = requests.get(USGS_IV_URL, params={'sites': site, 'parameterCd': '00060', 'siteStatus': 'all',
                                              'startDT': f'{a:%Y-%m-%d}', 'endDT': f'{b:%Y-%m-%d}',
                                              'format': 'rdb'}, timeout=300)
        r.raise_for_status()
        rows = [l.split('\t') for l in r.text.splitlines() if l.startswith('USGS')]
        df = pd.DataFrame([(x[2], x[3], x[4]) for x in rows if len(x) > 4], columns=['t', 'tz', 'q'])
        if df.empty:
            continue
        df['q'] = pd.to_numeric(df['q'], errors='coerce')
        df['t'] = pd.to_datetime(df['t'] + df['tz'].map(TZ_OFFSETS).fillna('+00:00'), utc=True,
                                 format='%Y-%m-%d %H:%M%z')
        parts.append(df.set_index('t')['q'])
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated()].resample('1h').mean()


def daily_to_hourly(daily: pd.Series, tz: str) -> pd.Series:
    """Daily means placed at local noon and interpolated to an hourly UTC series."""
    s = daily.copy()
    s.index = (pd.DatetimeIndex(s.index) + pd.Timedelta(hours=12)).tz_localize(tz).tz_convert('UTC')
    return s.resample('1h').mean().interpolate(limit_area='inside')


def score(rows: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    """Median absolute % error (and median signed %) by season and lead day."""
    df = rows.copy()
    df['season'] = np.where(pd.DatetimeIndex(df['issued']).month.isin(range(4, 11)), 'Apr-Oct', 'Nov-Mar')
    cols = {}
    for m in ['persistence'] + methods:
        df[f'err_{m}'] = (df[m] / df['obs'] - 1).abs() * 100
        cols[f'err_{m}'] = 'median'
    for m in methods:
        df[f'bias_{m}'] = (df[m] / df['obs'] - 1) * 100
        cols[f'bias_{m}'] = 'median'
    return df.groupby(['season', 'lead']).agg(cols).round(1)


def rise_peaks(rows: pd.DataFrame, methods: List[str], factor: float = 1.3) -> Dict[str, float]:
    """Median forecast/observed peak ratio over issue times whose observed peak is >= factor x now."""
    g = rows.pivot_table(index='issued', columns='lead', values=['obs', 'persistence'] + methods)
    peak = g['obs'].max(axis=1)
    rise = peak >= factor * g['persistence'].iloc[:, 0]
    out = {'windows': int(rise.sum())}
    for m in ['persistence'] + methods:
        out[m] = round(float((g[m].max(axis=1)[rise] / peak[rise]).median()), 2) if rise.any() else None
    return out


def run(site: str, start, end, every_days: int = 2, days: int = 4, calib_days: int = 28,
        rain: bool = False, hour_utc: int = 18) -> pd.DataFrame:
    site_id, cfg = get_upstream.resolve_site(site)
    tz = cfg.get('timezone', 'UTC')
    fetch_start = pd.Timestamp(start) - pd.Timedelta(days=calib_days + 5)
    fetch_end = pd.Timestamp(end) + pd.Timedelta(days=days + 2)
    if not site_id.startswith('USGS:'):
        raise ValueError(f"{site_id}: the backtest needs a USGS target gauge (hourly IV archive)")
    target = usgs_hourly(site_id.split(':', 1)[1], fetch_start, fetch_end)
    up = {g['site_id']: daily_to_hourly(get_upstream.fetch_daily_mean(g['site_id'], fetch_start, fetch_end), tz)
          for g in cfg['upstream']}
    anchor = (daily_to_hourly(get_upstream.fetch_daily_mean(cfg['anchor']['site_id'], fetch_start, fetch_end), tz)
              if cfg.get('anchor') else None)
    zcfg = basin_zones.station_zones(site_id) if rain else None
    rain_obs = (get_basin_precip.observed_daily(zcfg, fetch_start, fetch_end, stage4=lambda *a, **k: None)
                if zcfg is not None else None)
    variants = {'direct': {k: v for k, v in cfg.items() if k != 'anchor'}}
    if anchor is not None:
        variants['anchored'] = cfg
    rows = []
    issues = pd.date_range(pd.Timestamp(start) + pd.Timedelta(hours=hour_utc),
                           pd.Timestamp(end) + pd.Timedelta(hours=hour_utc), freq=f'{every_days}D', tz='UTC')
    for t in issues:
        grid = pd.date_range(t - pd.Timedelta(days=calib_days), t + pd.Timedelta(days=days + 1), freq='1h')
        inow = grid.get_loc(t)
        full = target.reindex(grid).to_numpy(float)
        if np.isfinite(full[inow - 72:inow + 1]).sum() < 48:
            continue
        tgt = full.copy()
        tgt[inow + 1:] = np.nan
        ins = [up[g['site_id']].reindex(grid).interpolate(limit_direction='both').to_numpy(float)
               for g in cfg['upstream']]
        if any(not np.isfinite(x).all() for x in ins):
            continue
        kw = {}
        if rain_obs is not None and len(rain_obs):
            flux, info = get_basin_precip.routing_rain_flux(zcfg, grid, rain_obs, None)
            kw = {'rain_flux': flux, 'rain_calib_mm': float(info['rate_mm_per_hour'].iloc[:inow + 1].sum())}
        preds = {}
        for name, c in variants.items():
            a = None
            if name == 'anchored':
                a = anchor.reindex(grid).to_numpy(float).copy()
                a[grid > t - pd.Timedelta(hours=12)] = np.nan
            try:
                preds[name] = ff.routed_forecast(c, tgt, inow, ins, anchor_obs=a, **kw)['ef'].central
            except Exception as e:  # one bad window must not stop the backtest
                logger.warning("%s at %s failed: %s", name, t, e)
        if len(preds) != len(variants):
            continue
        local = grid.tz_convert(tz)
        q0 = float(np.nanmean(tgt[inow - 2:inow + 1]))
        for lead in range(1, days + 1):
            day = (t.tz_convert(tz).normalize() + pd.Timedelta(days=lead)).date()
            m = local.date == day
            obs = float(np.nanmean(full[m])) if m.sum() >= 20 else np.nan
            if np.isfinite(obs):
                rows.append({'issued': t, 'lead': lead, 'obs': obs, 'persistence': q0,
                             **{k: float(np.mean(v[m])) for k, v in preds.items()}})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--site', default='taos-box')
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--every-days', type=int, default=2)
    p.add_argument('--days', type=int, default=4)
    p.add_argument('--rain', action='store_true')
    p.add_argument('--out', help='CSV of every scored forecast day')
    args = p.parse_args(argv)
    rows = run(args.site, args.start, args.end, every_days=args.every_days, days=args.days, rain=args.rain)
    if args.out:
        rows.to_csv(args.out, index=False)
    methods = [m for m in ('direct', 'anchored') if m in rows]
    print(score(rows, methods).to_string())
    print('rising windows (forecast peak / observed peak):', rise_peaks(rows, methods))
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
