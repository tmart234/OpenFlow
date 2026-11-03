"""
Score archived forecasts against what the river did: OpenFlow routing, the
NWS official forecast and the OpenFlow ML model, side by side.

NWS keeps no archive of its past river forecasts, so the only way to tell
which source to trust at a site is to save each day's forecasts (the Flow
Forecast workflow commits them to the `forecast-archive` branch, under
archive/YYYY-MM-DD/) and score them once the days have happened.

The forecast CSV and JSON files that flow_forecast.py writes carry:
  routing  forecast_cfs (hourly)
  NWS      nws_official_cfs (hourly; only where NWS forecasts the gauge)
  ML       ml.daily min/max in the JSON (daily; its midpoint is scored)

Per source, river and lead day it reports:
  error    median absolute % error of the daily mean flow
  crest    for each sharp crest observed within a forecast's window
           (routing.find_crests): the forecast's highest flow within 48 h of it.
           Hourly sources get a timing error; every source gets a
           forecast/observed crest ratio.

  python score_archive.py --archive <dir with archive/> [--days 60]
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import routing
from data import get_upstream

SOURCES = ('routing', 'nws', 'ml')


def load_archive(root: str, since: Optional[pd.Timestamp] = None) -> List[dict]:
    """Every archived forecast under root/archive/*/ as {site_id, issued, hourly, ml_daily, tz}."""
    out = []
    for js in sorted(glob.glob(os.path.join(root, 'archive', '*', '*_forecast.json'))):
        with open(js) as f:
            s = json.load(f)
        issued = pd.Timestamp(s['latest_observation']['time_utc'])
        if since is not None and issued < since:
            continue
        csv = pd.read_csv(js[:-5] + '.csv')
        t = pd.to_datetime(csv['time_utc'], utc=True)
        hourly = pd.DataFrame({'routing': csv['forecast_cfs'].to_numpy(),
                               'nws': csv['nws_official_cfs'].to_numpy()}, index=t)
        hourly = hourly[hourly.index > issued]
        ml = (s.get('ml') or {}).get('daily') or []
        ml_daily = pd.Series({pd.Timestamp(d['date']): (d['min_cfs'] + d['max_cfs']) / 2 for d in ml}, dtype=float)
        out.append({'site_id': s['site_id'], 'issued': issued, 'hourly': hourly,
                    'ml_daily': ml_daily, 'tz': s.get('timezone', 'UTC')})
    return out


def score_one(fc: dict, obs: pd.Series, days: int = 5) -> List[dict]:
    """Rows of (source, lead_day, pct_error) and (source, crest, ...) for one archived forecast."""
    rows = []
    tz = fc['tz']
    obs_local = obs.tz_convert(tz)
    obs_daily = obs_local.groupby(obs_local.index.date).mean()
    obs_daily.index = pd.DatetimeIndex(obs_daily.index)
    issue_day = fc['issued'].tz_convert(tz).normalize().tz_localize(None)
    hourly_local = fc['hourly'].tz_convert(tz)
    fc_daily = hourly_local.groupby(hourly_local.index.date).mean()
    fc_daily.index = pd.DatetimeIndex(fc_daily.index)
    for lead in range(1, days + 1):
        day = issue_day + pd.Timedelta(days=lead)
        o = obs_daily.get(day, np.nan)
        if not np.isfinite(o) or o <= 0:
            continue
        values = {'routing': fc_daily['routing'].get(day, np.nan), 'nws': fc_daily['nws'].get(day, np.nan),
                  'ml': fc['ml_daily'].get(day, np.nan)}
        for src, v in values.items():
            if np.isfinite(v):
                rows.append({'site_id': fc['site_id'], 'issued': fc['issued'], 'kind': 'daily',
                             'source': src, 'lead_day': lead, 'pct_error': abs(v / o - 1) * 100})
    # Crests observed inside the forecast window.
    window = obs[(obs.index > fc['issued']) & (obs.index <= fc['hourly'].index.max())]
    if len(window) > 72:
        span = obs[(obs.index > fc['issued'] - pd.Timedelta(days=4)) & (obs.index <= window.index.max())]
        idx = routing.find_crests(span.to_numpy(dtype=float))
        for i in idx:
            ct = span.index[i]
            if ct <= fc['issued']:
                continue
            near = (fc['hourly'].index >= ct - pd.Timedelta(hours=routing.CREST_WINDOW_HOURS)) & \
                   (fc['hourly'].index <= ct + pd.Timedelta(hours=routing.CREST_WINDOW_HOURS))
            for src in ('routing', 'nws'):
                seg = fc['hourly'].loc[near, src].dropna()
                if len(seg):
                    rows.append({'site_id': fc['site_id'], 'issued': fc['issued'], 'kind': 'crest', 'source': src,
                                 'lead_h': (ct - fc['issued']) / pd.Timedelta(hours=1),
                                 'timing_error_h': (seg.idxmax() - ct) / pd.Timedelta(hours=1),
                                 'crest_ratio': seg.max() / span.iloc[i]})
            day = ct.tz_convert(tz).normalize().tz_localize(None)
            v = fc['ml_daily'].get(day, np.nan)
            if np.isfinite(v):
                rows.append({'site_id': fc['site_id'], 'issued': fc['issued'], 'kind': 'crest', 'source': 'ml',
                             'lead_h': (ct - fc['issued']) / pd.Timedelta(hours=1),
                             'timing_error_h': np.nan, 'crest_ratio': v / span.iloc[i]})
    return rows


def summarize(rows: pd.DataFrame) -> str:
    """Markdown: per river, the daily error by lead day and the crest scores, one column per source."""
    if rows.empty:
        return 'No archived forecasts with observations to score yet.\n'
    reg = get_upstream.load_registry()
    lines = ['# Archived forecast scores', '',
             'Median absolute % error of the daily mean flow (lower is better); crests: median |timing error| '
             '(hourly sources) and forecast/observed crest ratio.', '']
    for site, g in rows.groupby('site_id'):
        name = (reg.get(site) or {}).get('short_name', site)
        d = g[g['kind'] == 'daily']
        lines += [f'## {name}', '', '| Lead day | ' + ' | '.join(SOURCES) + ' | forecasts |', '|---|' + '---|' * (len(SOURCES) + 1)]
        for lead, gl in d.groupby('lead_day'):
            cells = [f"{gl.loc[gl.source == s, 'pct_error'].median():.1f}" if (gl.source == s).any() else '—'
                     for s in SOURCES]
            lines.append(f'| {int(lead)} | ' + ' | '.join(cells) + f" | {gl['issued'].nunique()} |")
        c = g[g['kind'] == 'crest']
        if len(c):
            lines += ['', '| Crests | ' + ' | '.join(SOURCES) + ' |', '|---|' + '---|' * len(SOURCES)]
            t = [f"{c.loc[c.source == s, 'timing_error_h'].abs().median():.0f} h" if c.loc[c.source == s, 'timing_error_h'].notna().any() else '—'
                 for s in SOURCES]
            r = [f"{c.loc[c.source == s, 'crest_ratio'].median():.2f}" if (c.source == s).any() else '—' for s in SOURCES]
            lines += ['| median timing error | ' + ' | '.join(t) + ' |', '| crest ratio | ' + ' | '.join(r) + ' |',
                      f"| scored (forecast, crest) pairs | {len(c[c.source == 'routing'])} | {len(c[c.source == 'nws'])} | {len(c[c.source == 'ml'])} |"]
        lines.append('')
    return '\n'.join(lines)


def observations(site_id: str, start, end) -> pd.Series:
    """Hourly observed flow (UTC) for scoring: the same archives the crest backtest uses."""
    import crest_backtest
    return crest_backtest.gauge_hourly(site_id, start, end)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--archive', required=True, help='directory holding archive/YYYY-MM-DD/')
    p.add_argument('--days', type=int, default=60, help='score forecasts issued in the last N days')
    p.add_argument('--out', help='write the markdown here')
    args = p.parse_args(argv)
    since = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=args.days)
    fcs = load_archive(args.archive, since)
    rows = []
    obs_cache: Dict[str, pd.Series] = {}
    for fc in fcs:
        if fc['site_id'] not in obs_cache:
            obs_cache[fc['site_id']] = observations(fc['site_id'], since - pd.Timedelta(days=5),
                                                    pd.Timestamp.now(tz='UTC'))
        rows += score_one(fc, obs_cache[fc['site_id']])
    md = summarize(pd.DataFrame(rows))
    if args.out:
        with open(args.out, 'w') as f:
            f.write(md)
    print(md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
