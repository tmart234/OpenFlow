"""
OpenFlow upstream-routing flow forecast (CLI).

    python openFlowML/flow_forecast.py --site taos-box
    python openFlowML/flow_forecast.py --site USGS:08276500 --days 7 --out-dir forecasts/

For a station listed in data/upstream_gauges.json this:
  1. pulls ~30 days of hourly observations at the target gauge and each
     upstream gauge, plus the current NWS River Forecast Center (RFC)
     forecast at every gauge that has one (NOAA NWPS API, data/nwps.py);
  2. calibrates travel time + attenuation from the upstream gauges to the
     target on the last --calib-days of observations (routing.py), falling
     back to the registry defaults when that window is too quiet;
     rain on the UNGAUGED part of the basin (basin_zones.json: gridMET, then
     the NWS Stage IV analysis for the last 1-2 days, then NWS gridpoint QPF
     for the forecast) is a third input: a local-inflow term fitted on top
     of the upstream travel times when the window had enough rain (runoff
     ratio bounded to [0, 1]), else taken from the registry's `rain` defaults;
  3. routes observed + RFC-forecast upstream flow to the target, bias-
     corrected to the latest observation, with an ensemble band;
  4. writes <slug>_forecast.csv (hourly), <slug>_forecast.json (peak, daily
     table, fit, data provenance) and <slug>_forecast.png (chart with the
     peak called out, the official NWS forecast when one is in effect, and
     NWS flood-category flows when the forecast approaches them).

This works with no trained ML model: it is the physically structured
"what is already in the river upstream" forecast. It is not an official NWS
product; when the RFC issues its own forecast for the target gauge it is
drawn on the chart for comparison.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charts  # noqa: E402
import routing  # noqa: E402
from data import basin_zones, get_basin_precip, get_upstream, nwps  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DAYS = 7
DEFAULT_CALIB_DAYS = 28
# Error correction to the latest observation decays with this e-folding time;
# ~ the reach travel time (1-2 days) is the hydrologic rule of thumb.
DEFAULT_BIAS_TAU_HOURS = 48.0
CHART_HISTORY_DAYS = 4
MIN_CALIB_HOURS = 7 * 24
# Basin-mean rain (mm) on the ungauged area within the calibration window
# needed to calibrate the rain term rather than use registry defaults.
MIN_CALIB_RAIN_MM = 10.0
DEFAULT_RAIN = {'lag_hours': 12, 'k_hours': 48, 'runoff_ratio': 0.03}


class ForecastError(RuntimeError):
    pass


def _gauge_series(lid: str) -> dict:
    payload = nwps.get_stageflow(lid)
    if payload is None:
        raise ForecastError(f"NWPS returned nothing for gauge {lid}")
    obs = nwps.hourly(nwps.parse_series(payload, 'observed'))
    fc = nwps.hourly(nwps.parse_series(payload, 'forecast'))
    return {'obs': obs, 'fc': fc, 'issued': nwps.forecast_issued_time(payload)}


def _status(lid: str, series: dict, name: str) -> dict:
    obs = series['obs'].dropna()
    fc = series['fc'].dropna()
    out = {'name': name, 'nws_lid': lid,
           'latest_time_utc': obs.index.max().isoformat() if not obs.empty else None,
           'latest_cfs': float(obs.iloc[-1]) if not obs.empty else None,
           'forecast_issued_utc': series['issued'].isoformat() if series['issued'] is not None else None,
           'forecast_peak_cfs': None, 'forecast_peak_time_utc': None}
    if not fc.empty:
        out['forecast_peak_cfs'] = float(fc.max())
        out['forecast_peak_time_utc'] = fc.idxmax().isoformat()
    return out


def _default_ensemble(cfg, target, inputs):
    gauges = cfg['upstream']
    lags = np.array([g.get('lag_hours', 24 * g.get('lag_days', 1)) for g in gauges])
    ks = [g.get('k_hours', 12) for g in gauges]
    weights = np.array([g.get('weight', 1.0) for g in gauges])
    fits = []
    for dl in (0, -12, 12):
        for ws in (1.0, 0.8, 1.2):
            fits.append(routing.default_fit(target, inputs, np.maximum(lags + dl, 0),
                                            ks, weights * ws))
    return fits


def _rain_input(site_id: str, grid: pd.DatetimeIndex, inow: int, tz: str, zones=None):
    """
    Hourly rain flux (cfs) on the station's ungauged area over `grid`, plus
    per-zone basin rain totals for the summary. None when the station has no
    basin zones.
    """
    zcfg = basin_zones.station_zones(site_id, zones)
    if zcfg is None:
        return None
    now = grid[inow]
    today = now.tz_convert(tz).tz_localize(None).normalize()
    start = (grid[0] - pd.Timedelta(days=1)).tz_convert(None).normalize()
    observed = get_basin_precip.observed_daily(zcfg, start, today)
    forecast_points = get_basin_precip.nws_points_hourly(zcfg)
    totals = get_basin_precip.zone_totals(zcfg, observed, forecast_points, today, tz)
    out = {'flux': None, 'info': {}, 'totals': totals, 'observed': observed,
           'forecast_points': forecast_points, 'cfg': zcfg}
    if not (zcfg.get('ungauged') or {}).get('area_km2'):
        return out
    flux, info = get_basin_precip.routing_rain_flux(zcfg, grid, observed, forecast_points)
    rate = info.pop('rate_mm_per_hour')
    info['calib_mm'] = round(float(rate.iloc[:inow + 1].sum()), 1)
    info['next_mm'] = round(float(rate.iloc[inow + 1:].sum()), 1)
    info['past7_mm'] = round(float(rate.iloc[max(0, inow - 167):inow + 1].sum()), 1)
    out.update(flux=flux, info=info)
    return out


def _rain_panel(rain: dict, tz: str, start, end) -> dict:
    """Daily basin rain by zone (gridMET past + NWS forecast) for the chart's lower panel."""
    cols = [f'rain_{z}' for z in basin_zones.ZONES]
    empty = pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name='Date'), dtype=float)
    obs = rain.get('observed')
    past = obs[cols].astype(float) if obs is not None and len(obs) else empty
    fut = (get_basin_precip.points_daily_zones(rain['cfg'], rain['forecast_points'], tz)[cols]
           if rain.get('forecast_points') else empty)
    if len(past):
        fut = fut[fut.index > past.index.max()]
    both = pd.concat([d for d in (past, fut) if len(d)]).sort_index() if (len(past) or len(fut)) else empty
    both = both[(both.index >= start.normalize()) & (both.index <= end)]
    if both.empty:
        return None
    return {'dates': both.index, 'zones': {z: both[f'rain_{z}'].to_numpy(dtype=float) for z in basin_zones.ZONES},
            'forecast_from': fut.index.min() if len(fut) else None}


def run(site: str, days: int = DEFAULT_DAYS, calib_days: int = DEFAULT_CALIB_DAYS,
        out_dir: str = '.', make_chart: bool = True, registry=None,
        bias_tau_hours: float = DEFAULT_BIAS_TAU_HOURS, use_rain: bool = True,
        zones=None) -> dict:
    site_id, cfg = get_upstream.resolve_site(site, registry)
    if not cfg.get('upstream'):
        raise ForecastError(f"{site_id} has no upstream gauges in the registry")
    tz = cfg.get('timezone', 'UTC')
    name = cfg.get('name', site_id)
    short = cfg.get('short_name', name)
    slug = (cfg.get('aliases') or [site_id.replace(':', '_')])[0]

    target = _gauge_series(cfg['nws_lid'])
    target_obs = target['obs'].dropna()
    if target_obs.empty:
        raise ForecastError(f"No recent observations at {cfg['nws_lid']}")
    try:
        thresholds = nwps.flood_thresholds_cfs(nwps.get_gauge(cfg['nws_lid']))
    except Exception as e:  # metadata is decoration; never fail the forecast on it
        logger.warning("Flood thresholds unavailable: %s", e)
        thresholds = {}

    now = target_obs.index.max().floor('h')
    now_local = now.tz_convert(tz)
    end = (now_local.normalize() + pd.Timedelta(days=days)).tz_convert('UTC')
    grid = pd.date_range(now - pd.Timedelta(days=calib_days), end, freq='1h')
    inow = grid.get_loc(now)

    inputs, upstream_status = [], []
    for g in cfg['upstream']:
        s = _gauge_series(g['nws_lid'])
        if s['obs'].dropna().empty:
            raise ForecastError(f"No recent observations at upstream gauge {g['nws_lid']}")
        values, info = routing.build_input(grid, s['obs'], s['fc'])
        inputs.append(values)
        st = _status(g['nws_lid'], s, g.get('name', g['nws_lid']))
        st['forecast_end_utc'] = info['forecast_end'].isoformat() if info['forecast_end'] is not None else None
        upstream_status.append(st)

    context_status = []
    for g in cfg.get('context_gauges', []):
        try:
            context_status.append(_status(g['nws_lid'], _gauge_series(g['nws_lid']), g.get('name', g['nws_lid'])))
        except Exception as e:
            logger.warning("Context gauge %s unavailable: %s", g.get('nws_lid'), e)

    rain = None
    if use_rain:
        try:
            rain = _rain_input(site_id, grid, inow, tz, zones)
        except Exception as e:  # rain is an add-on; never fail the forecast on it
            logger.warning("Basin rain unavailable: %s", e)
    rain_flux = rain['flux'] if rain and rain.get('flux') is not None else None
    rain_cfg = {**DEFAULT_RAIN, **{k: v for k, v in cfg.get('rain', {}).items() if not k.startswith('_')}}
    rain_fit = rain_flux is not None and rain['info']['calib_mm'] >= MIN_CALIB_RAIN_MM

    tgt = target["obs"].reindex(grid).to_numpy(dtype=float, copy=True)
    tgt[inow + 1:] = np.nan
    cal_inputs = [x[:inow + 1] for x in inputs]
    calibrated = (np.isfinite(tgt[:inow + 1]).sum() >= MIN_CALIB_HOURS
                  and all(routing.is_informative(x) for x in cal_inputs))
    fits = []
    if calibrated:
        try:
            fits = routing.fit(tgt[:inow + 1], cal_inputs)
            if fits and rain_fit:
                # Two-stage: upstream travel times first, then the rain term
                # on top (runoff ratio bounded to [0, 1]).
                fits = routing.add_input(tgt[:inow + 1], cal_inputs, fits,
                                         rain_flux[:inow + 1], max_coef=1.0)
        except ValueError as e:
            logger.warning("Calibration failed (%s); using registry defaults", e)
    rain_mode = 'off' if rain_flux is None else ('calibrated' if (fits and rain_fit) else 'default')
    if not fits:
        calibrated = False
        fits = _default_ensemble(cfg, tgt[:inow + 1], cal_inputs)
    if rain_mode == 'default':
        for f in fits:
            f.lags.append(int(rain_cfg['lag_hours']))
            f.ks.append(float(rain_cfg['k_hours']))
            f.coefs.append(float(rain_cfg['runoff_ratio']))
    all_inputs = inputs + ([rain_flux] if rain_flux is not None else [])

    ef = routing.forecast(fits, all_inputs, tgt, inow, bias_tau_hours=bias_tau_hours)
    peak_t, peak_v, peak_i = routing.peak(grid, ef.central, min(inow + 1, len(grid) - 1))
    daily = routing.daily_summary(grid[inow:], ef.central[inow:], ef.low[inow:], ef.high[inow:], tz)
    daily = daily.iloc[:days]

    official = target['fc'].dropna()
    official = official[(official.index >= now - pd.Timedelta(days=1)) & (official.index <= end)]

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{slug}_forecast")
    start_out = max(0, inow - CHART_HISTORY_DAYS * 24)
    times = grid[start_out:]
    csv = pd.DataFrame({
        'time_utc': times.strftime('%Y-%m-%dT%H:%MZ'),
        'time_local': times.tz_convert(tz).strftime('%Y-%m-%d %H:%M %Z'),
        'observed_cfs': np.round(tgt[start_out:], 0),
        'forecast_cfs': np.where(np.arange(start_out, len(grid)) > inow,
                                 np.round(ef.central[start_out:], 0), np.nan),
        'band_low_cfs': np.where(np.arange(start_out, len(grid)) > inow,
                                 np.round(ef.low[start_out:], 0), np.nan),
        'band_high_cfs': np.where(np.arange(start_out, len(grid)) > inow,
                                  np.round(ef.high[start_out:], 0), np.nan),
        'nws_official_cfs': np.round(official.reindex(times).to_numpy(dtype=float), 0),
    })
    csv.to_csv(base + '.csv', index=False)

    fit_desc = [f.to_dict() for f in fits]
    issued = [s['forecast_issued_utc'] for s in upstream_status if s['forecast_issued_utc']]
    summary = {
        'site_id': site_id, 'name': name, 'reach': cfg.get('reach'),
        'nws_lid': cfg['nws_lid'], 'timezone': tz,
        'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'latest_observation': {'time_utc': now.isoformat(), 'time_local': now_local.isoformat(),
                               'cfs': float(target_obs.iloc[-1])},
        'peak': {'time_utc': peak_t.isoformat(), 'time_local': peak_t.tz_convert(tz).isoformat(),
                 'cfs': round(peak_v), 'band_low_cfs': round(float(ef.low[peak_i])),
                 'band_high_cfs': round(float(ef.high[inow:].max()))},
        'daily': [{'date': str(d), **{k: round(float(v)) for k, v in row.items()}}
                  for d, row in daily.iterrows()],
        'routing': {'mode': 'calibrated' if calibrated else 'registry-default',
                    'calibration_days': calib_days, 'fits': fit_desc},
        'upstream': upstream_status, 'context_gauges': context_status,
        'nws_official_forecast_issued_utc': (target['issued'].isoformat()
                                             if target['issued'] is not None else None),
        'flood_thresholds_cfs': thresholds,
        'basin_rain': None,
        'files': {'csv': base + '.csv', 'json': base + '.json'},
    }
    if rain is not None:
        info = rain.get('info') or {}
        summary['basin_rain'] = {
            'zone_totals_mm': rain['totals'],
            'ungauged_area_km2': info.get('area_km2'),
            'ungauged_rain_mm': {'calibration_window': info.get('calib_mm'),
                                 'past_7_days': info.get('past7_mm'),
                                 'forecast': info.get('next_mm')},
            'sources': info.get('sources'), 'missing_hours': info.get('missing_hours'),
            'rain_term': rain_mode,
            'runoff_ratio': (round(float(fits[0].coefs[-1]), 3) if rain_mode != 'off' else None),
        }

    if make_chart:
        first, last = daily.index[0], daily.index[-1]
        up_names = ' + '.join(g.get('name', g['nws_lid']).split(',')[0] for g in cfg['upstream'])
        tz_abbr = now_local.strftime('%Z')
        issued_txt = (f"issued {pd.Timestamp(min(issued)).tz_convert(tz).strftime('%b %-d %-I %p %Z')}"
                      if issued else 'no current upstream RFC forecast: observed flow + recession only')
        best = fits[0]
        mode = (f"calibrated on the last {calib_days} days (RMSE ≈ {best.rmse:,.0f} cfs; "
                f"lags {'/'.join(str(int(l)) for l in best.lags)} h)") if calibrated else \
            "registry-default travel times (recent window too quiet to calibrate)"
        rain_line = ''
        br = summary['basin_rain']
        if br and br['rain_term'] != 'off':
            u = br['ungauged_rain_mm']
            rain_line = (f"\nLocal rain term ({br['rain_term']}, runoff ratio {br['runoff_ratio']:.3f}): "
                         f"{u['past_7_days']:.0f} mm fell on the {br['ungauged_area_km2']:,.0f} km² ungauged area "
                         f"in the last 7 days; NWS forecasts {u['forecast']:.0f} mm more.")
        footnote = (f"OpenFlow upstream-routing forecast — not an official NWS product"
                    f"{'; NWS official forecast shown for comparison' if not official.empty else ' (no NWS forecast is current for this gauge)'}.\n"
                    f"Routed from NWS RFC forecasts for {up_names} ({issued_txt}), {mode}.\n"
                    f"Band = parameter ensemble ± calibration error. Rain beyond the NWS forecast window is not included."
                    f"{rain_line}")
        title = f"{short}: flow forecast {pd.Timestamp(first):%b %-d}–{pd.Timestamp(last):%b %-d, %Y}"
        subtitle = f"{name} · USGS {site_id.split(':')[-1]} / NWS {cfg['nws_lid']} · times {tz_abbr}"
        charts.plot_forecast(
            base + '.png', title=title, subtitle=subtitle, footnote=footnote,
            obs_times=grid[start_out:inow + 1], obs_values=tgt[start_out:inow + 1],
            fc_times=grid[inow:], fc_central=ef.central[inow:],
            fc_low=ef.low[inow:], fc_high=ef.high[inow:],
            official_times=official.index if not official.empty else None,
            official_values=official.to_numpy() if not official.empty else None,
            now=now, tz=tz, thresholds=thresholds,
            obs_label=f"Observed (USGS {site_id.split(':')[-1]})",
            rain=(_rain_panel(rain, tz, grid[start_out].tz_convert(tz).tz_localize(None),
                              end.tz_convert(tz).tz_localize(None))
                  if rain is not None else None))
        summary['files']['png'] = base + '.png'

    with open(base + '.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def format_summary(s: dict) -> str:
    tz = s['timezone']
    peak_local = pd.Timestamp(s['peak']['time_utc']).tz_convert(tz)
    lines = [
        f"{s['name']} ({s['nws_lid']})",
        f"  latest: {s['latest_observation']['cfs']:,.0f} cfs at "
        f"{pd.Timestamp(s['latest_observation']['time_utc']).tz_convert(tz):%a %b %-d %-I %p %Z}",
        f"  peak:   ~{s['peak']['cfs']:,} cfs {peak_local:%a %b %-d ~%-I %p %Z} "
        f"(band up to {s['peak']['band_high_cfs']:,} cfs)",
        f"  routing: {s['routing']['mode']}",
        "", f"  {'day':<12}{'min':>7}{'mean':>7}{'max':>7}   band",
    ]
    for d in s['daily']:
        day = pd.Timestamp(d['date'])
        lines.append(f"  {day:%a %b %d}  {d['min_cfs']:>6,} {d['mean_cfs']:>6,} {d['max_cfs']:>6,}"
                     f"   {d['band_low_cfs']:,}–{d['band_high_cfs']:,}")
    lines.append("")
    lines.append("  upstream now:")
    for u in s['upstream'] + s['context_gauges']:
        fc = (f", RFC peak {u['forecast_peak_cfs']:,.0f} cfs "
              f"{pd.Timestamp(u['forecast_peak_time_utc']).tz_convert(tz):%a %-I %p}"
              if u.get('forecast_peak_cfs') else '')
        latest = f"{u['latest_cfs']:,.0f} cfs" if u.get('latest_cfs') is not None else 'n/a'
        lines.append(f"    {u['name']:<44} {latest}{fc}")
    br = s.get('basin_rain')
    if br:
        t = br['zone_totals_mm']
        lines.append(f"  basin rain, zone mean mm (past 7 d through {t.get('past_through')} | next 7 d NWS):")
        for z in basin_zones.ZONES:
            p, n = t.get('past', {}), t.get('next', {})
            lines.append(f"    {z:<6} rain {p.get(f'rain_{z}', float('nan')):5.1f} | {n.get(f'rain_{z}', float('nan')):5.1f}"
                         f"   snow (water eq.) {p.get(f'snow_{z}', float('nan')):5.1f} | {n.get(f'snow_{z}', float('nan')):5.1f}")
        if br['rain_term'] != 'off':
            lines.append(f"  local rain term: {br['rain_term']}, runoff ratio {br['runoff_ratio']}, "
                         f"ungauged area {br['ungauged_area_km2']:,.0f} km2")
    if s.get('nws_official_forecast_issued_utc'):
        lines.append(f"  NWS official forecast for this gauge issued {s['nws_official_forecast_issued_utc']}")
    for k, v in s['files'].items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--site', required=True,
                        help="Station: alias (taos-box), site id (USGS:08276500), or NWS LID (TJBN5)")
    parser.add_argument('--days', type=int, default=DEFAULT_DAYS,
                        help='Local calendar days to forecast, starting today (default 7)')
    parser.add_argument('--calib-days', type=int, default=DEFAULT_CALIB_DAYS,
                        help='Days of recent hourly observations to calibrate routing on (default 28; NWPS keeps ~30)')
    parser.add_argument('--bias-tau-hours', type=float, default=DEFAULT_BIAS_TAU_HOURS,
                        help='E-folding time of the correction to the latest observation (default 48)')
    parser.add_argument('--out-dir', default='.', help='Where to write the CSV/JSON/PNG')
    parser.add_argument('--no-chart', action='store_true', help='Skip the PNG chart')
    parser.add_argument('--no-rain', action='store_true',
                        help='Skip basin rain (gridMET + NWS QPF) and the local rain term')
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        summary = run(args.site, days=args.days, calib_days=args.calib_days,
                      out_dir=args.out_dir, make_chart=not args.no_chart,
                      bias_tau_hours=args.bias_tau_hours, use_rain=not args.no_rain)
    except (ForecastError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(format_summary(summary))
    return 0


if __name__ == '__main__':
    sys.exit(main())
