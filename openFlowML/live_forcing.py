"""
Score a model on the weather forecasts it would really have had.

quality_report (evaluate.py) scores held-out windows with the *observed*
weather, basin rain/snow and upstream flow as the decoder's "forecast" --
perfect forcing, which overstates live skill. This re-scores the same windows
with the decoder weather on lead days 1-7 replaced by the forecasts actually
issued for those days:

  source    Open-Meteo Previous Runs API, GFS (gfs_seamless); its precipitation
            archive starts in 2024, so only windows issued since then count
  points    each station's gauge (TMIN, TMAX, point precipitation) and the
            heaviest sample point of each of its rain zones (zone rain/snow)
  units     exactly as training converts observations: TMIN/TMAX in tenths
            of deg C, precipitation in mm, zone rain/snow split by
            basin_zones.liquid_fraction, all scaled with scalers.json

Days 8-14 keep the observed values: the decoder runs forward in time, so they
cannot change the forecasts for days 1-7, which are the ones scored. Upstream
inputs stay observed (NWS keeps no archive of its river forecasts), so the
result is still a little optimistic where upstream gauges are registered.

train.py runs this on its held-out ensemble (before the final refit) when
OPENFLOW_LIVE_FORCING=1 and adds a section to quality_report. Standalone:

  python openFlowML/live_forcing.py --model-dir <dir> --data-dir <dir> [--start YYYY-MM-DD]

<model-dir> must hold a model that did not train on the scored dates: a
released model is refit on every year, so score it with score_archive.py
(the live forecast archive) instead.
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import windowing

logger = logging.getLogger(__name__)

API_URL = 'https://previous-runs-api.open-meteo.com/v1/forecast'
API_LEADS = tuple(range(1, 8))
ARCHIVE_START = '2024-01-01'          # GFS precipitation in the archive starts here
POINT_COLUMNS = ['TMIN', 'TMAX', 'precipitation']
ZONE_COLUMNS = list(windowing.BASIN_PRECIP_FEATURES)
ZONES = ('local', 'mid', 'far')


# --------------------------------------------------------------------------
# Archive
# --------------------------------------------------------------------------

def _points(sites: Iterable[str]):
    from data import basin_zones
    import ml_inference
    zones = basin_zones.load_zones()
    for site in sites:
        c = ml_inference._coords(site)
        yield {'site': site, 'kind': 'station', 'lat': c['lat'], 'lon': c['lon']}
        for zone in ZONES:
            pts = [p for p in (zones.get(site) or {}).get('points', []) if p['zone'] == zone]
            if pts:
                p = max(pts, key=lambda p: p['w'])
                yield {'site': site, 'kind': zone, 'lat': p['lat'], 'lon': p['lon']}


def _request(pt: dict, start: str, end: str, cache: Optional[str]) -> dict:
    import requests
    path = None
    if cache:
        path = os.path.join(cache, f"{pt['site'].replace(':', '_')}_{pt['kind']}_{start}_{end}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    hourly = ([f'precipitation_previous_day{k}' for k in API_LEADS]
              + [f'temperature_2m_previous_day{k}' for k in API_LEADS])
    params = {'latitude': pt['lat'], 'longitude': pt['lon'], 'hourly': ','.join(hourly),
              'start_date': start, 'end_date': end, 'models': 'gfs_seamless', 'timezone': 'America/Denver'}
    for attempt in range(12):
        try:
            d = requests.get(API_URL, params=params, timeout=180).json()
        except Exception as e:                       # timeouts, resets
            logger.warning("Open-Meteo retry %s %s: %s", pt['site'], pt['kind'], e)
            time.sleep(15 * (attempt + 1))
            continue
        if d.get('error'):
            if 'limit' in str(d.get('reason', '')).lower():
                raise RuntimeError(f"Open-Meteo: {d.get('reason')}")
            time.sleep(15 * (attempt + 1))           # "too many concurrent requests"
            continue
        if path:
            os.makedirs(cache, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(d, f)
        return d
    raise RuntimeError(f"no archive answer for {pt['site']} {pt['kind']}")


def daily_from_hourly(d: dict) -> pd.DataFrame:
    """DataFrame[date, lead, tmin_c, tmax_c, tmean_c, precip_mm] from one archive answer."""
    h = pd.DataFrame(d['hourly'])
    h['date'] = pd.to_datetime(h['time']).dt.normalize()
    rows = []
    for k in API_LEADS:
        g = h.groupby('date').agg(precip_mm=(f'precipitation_previous_day{k}', lambda s: s.sum(min_count=20)),
                                  tmin_c=(f'temperature_2m_previous_day{k}', 'min'),
                                  tmax_c=(f'temperature_2m_previous_day{k}', 'max'),
                                  tmean_c=(f'temperature_2m_previous_day{k}', 'mean'),
                                  n=(f'temperature_2m_previous_day{k}', 'count'))
        g = g[g['n'] >= 20].drop(columns='n').reset_index()
        g['lead'] = k
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def fetch_table(sites: Sequence[str], start, end, cache: Optional[str] = None) -> pd.DataFrame:
    """Archived daily forecasts, DataFrame[site, kind, date, lead, tmin_c, tmax_c, tmean_c, precip_mm]."""
    start = max(pd.Timestamp(start), pd.Timestamp(ARCHIVE_START))
    end = pd.Timestamp(end)
    frames = []
    for pt in _points(sites):
        for y in range(start.year, end.year + 1):     # one request per point and year
            a = max(start, pd.Timestamp(f'{y}-01-01')).strftime('%Y-%m-%d')
            b = min(end, pd.Timestamp(f'{y}-12-31')).strftime('%Y-%m-%d')
            f = daily_from_hourly(_request(pt, a, b, cache))
            frames.append(f.assign(site=pt['site'], kind=pt['kind']))
        logger.info("Archived forecasts: %s %s", pt['site'], pt['kind'])
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Inputs and scoring
# --------------------------------------------------------------------------

def _scale(raw, params):
    raw = np.asarray(raw, dtype=float)
    x = np.log1p(np.maximum(raw, 0.0)) if params.get('transform') == 'log1p' else raw
    return (x - params['mean']) / params['scale']


def _lookup(table: pd.DataFrame) -> Dict[Tuple[str, pd.Timestamp, int], Dict[str, float]]:
    """{(site, date, lead): {decoder column: raw value}}."""
    from data import basin_zones
    out: Dict = {}
    st = table[table['kind'] == 'station']
    for r in st.itertuples():
        out.setdefault((r.site, pd.Timestamp(r.date), int(r.lead)), {}).update(
            {'TMIN': r.tmin_c * 10.0, 'TMAX': r.tmax_c * 10.0, 'precipitation': r.precip_mm})
    zn = table[table['kind'].isin(ZONES)]
    frac = basin_zones.liquid_fraction(zn['tmean_c'].to_numpy(dtype=float))
    zn = zn.assign(rain=zn['precip_mm'] * frac, snow=zn['precip_mm'] * (1 - frac))
    for r in zn.itertuples():
        out.setdefault((r.site, pd.Timestamp(r.date), int(r.lead)), {}).update(
            {f'rain_{r.kind}': r.rain, f'snow_{r.kind}': r.snow})
    return out


def archived_inputs(samples: Sequence, x: dict, scalers: dict, table: pd.DataFrame,
                    leads: int = 7) -> Tuple[dict, np.ndarray]:
    """
    Copy of the stacked inputs `x` with the decoder weather on lead days
    1..leads replaced by the archived forecasts, and a boolean mask of the
    windows whose archive is complete (incomplete windows are left as is).
    Zone columns change only on days with basin_precip_known = 1 and only for
    zones the station has.
    """
    from data import basin_zones
    feats = windowing.DECODER_FEATURES
    known_col = feats.index('basin_precip_known')
    zones = basin_zones.load_zones()
    look = _lookup(table)
    dec = np.array(x['decoder_input'], dtype='float32', copy=True)
    ok = np.ones(len(samples), dtype=bool)
    for i, s in enumerate(samples):
        cfg = zones.get(s.site_id) or {}
        have = {z for z, v in (cfg.get('zones') or {}).items() if v.get('area_km2', 0) > 0}
        cols = POINT_COLUMNS + [c for c in ZONE_COLUMNS if c.split('_')[1] in have]
        issue = s.anchor_date + pd.Timedelta(days=windowing.ENCODER_DAYS - 1)
        new = {}
        for k in range(1, leads + 1):
            v = look.get((s.site_id, issue + pd.Timedelta(days=k), k))
            if v is None or any(not np.isfinite(v.get(c, np.nan)) for c in cols):
                ok[i] = False
                break
            new[k] = v
        if not ok[i]:
            continue
        for k, v in new.items():
            for c in cols:
                if c in ZONE_COLUMNS and dec[i, k - 1, known_col] < 0.5:
                    continue
                dec[i, k - 1, feats.index(c)] = _scale(v[c], scalers[c])
    out = dict(x)
    out['decoder_input'] = dec
    return out, ok


def score(model, samples: Sequence, scalers: dict, table: pd.DataFrame, leads: int = 7) -> Optional[dict]:
    """
    Median absolute % error of the daily maximum on lead days 1..leads, on the
    windows with a complete archive: the model with perfect forcing, the model
    with the archived forecasts, and persistence; pooled and per station.
    None when no window has a complete archive.
    """
    import evaluate

    def point(pred):                     # model.point_forecast, without importing tensorflow
        return pred['persistence_plus_delta'] if isinstance(pred, dict) else pred
    samples = list(samples)
    x, _ = windowing.stack(samples)
    live_x, ok = archived_inputs(samples, x, scalers, table, leads)
    if not ok.any():
        return None
    sub = [s for s, keep in zip(samples, ok) if keep]
    xs = {k: v[ok] for k, v in x.items()}
    ls = {k: v[ok] for k, v in live_x.items()}
    perfect = evaluate.forecasts(sub, point(model.predict(xs, verbose=0, batch_size=512)), scalers)
    live = evaluate.forecasts(sub, point(model.predict(ls, verbose=0, batch_size=512)), scalers)

    def med(fc, key, mask=None):
        m = slice(None) if mask is None else mask
        err = np.abs(fc[key][m] / np.maximum(fc['obs'][m], 1e-6) - 1)[:, :leads]
        return [round(float(v) * 100, 2) for v in np.median(err, axis=0)]

    out = {'windows': int(ok.sum()), 'leads': list(range(1, leads + 1)),
           'issued': [str(pd.Timestamp(perfect['issued'].min()).date()),
                      str(pd.Timestamp(perfect['issued'].max()).date())],
           'source': 'Open-Meteo Previous Runs API (gfs_seamless)',
           'error_pct': {'perfect': med(perfect, 'model'), 'live': med(live, 'model'),
                         'persistence': med(perfect, 'persistence')},
           'stations': {}}
    for site in sorted(set(perfect['site'])):
        m = perfect['site'] == site
        out['stations'][site] = {'windows': int(m.sum()),
                                 'mean_error_pct': {k: round(float(np.mean(med(f, key, m))), 2)
                                                    for k, f, key in (('perfect', perfect, 'model'),
                                                                      ('live', live, 'model'),
                                                                      ('persistence', perfect, 'persistence'))}}
    return out


def to_markdown(result: dict, names: Optional[dict] = None) -> str:
    names = names or {}
    e = result['error_pct']
    leads = result['leads']
    lines = ['**Live forcing** (held-out windows issued '
             f"{result['issued'][0]}–{result['issued'][1]}, {result['windows']} with a complete forecast archive). "
             'The same model scored with the weather forecasts actually issued for lead days 1–7 '
             f"({result['source']}) instead of the observed weather. "
             'Median absolute % error of the daily maximum:', '',
             '| | ' + ' | '.join(f'Day {d}' for d in leads) + ' | Mean |', '|---|' + '---|' * (len(leads) + 1)]
    for label, key in (('Observed weather (as above)', 'perfect'), ('**Real forecasts**', 'live'),
                       ('Persistence', 'persistence')):
        lines.append(f'| {label} | ' + ' | '.join(f'{v:.1f}' for v in e[key]) + f' | {np.mean(e[key]):.1f} |')
    lines += ['', '| Station | Windows | Observed weather | Real forecasts | Persistence |', '|---|---|---|---|---|']
    for site, s in result['stations'].items():
        m = s['mean_error_pct']
        lines.append(f"| {names.get(site, site)} | {s['windows']} | {m['perfect']:.1f} | {m['live']:.1f} | "
                     f"{m['persistence']:.1f} |")
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--model-dir', required=True, help='trained-model directory (lstm_model.h5 + JSONs)')
    parser.add_argument('--data-dir', required=True, help='directory with openFlowML/normalized_data.csv')
    parser.add_argument('--start', default=None, help='first issue date to score (default: the test split)')
    parser.add_argument('--cache', default=None, help='directory for cached archive answers')
    args = parser.parse_args(argv)
    import ml_inference
    art = ml_inference.load_artifacts(args.model_dir, with_model=True)
    data = pd.read_csv(os.path.join(args.data_dir, 'openFlowML', 'normalized_data.csv'), parse_dates=['Date'])
    samples = windowing.build_windows(data)
    if args.start:
        start = pd.Timestamp(args.start)
        samples = [s for s in samples
                   if s.anchor_date + pd.Timedelta(days=windowing.ENCODER_DAYS - 1) >= start]
    else:
        samples = windowing.chronological_split(samples, 0.15, 0.15, windowing.DECODER_DAYS).test
    first = min(s.anchor_date for s in samples) + pd.Timedelta(days=windowing.ENCODER_DAYS)
    last = max(s.anchor_date for s in samples) + pd.Timedelta(days=windowing.ENCODER_DAYS + 7)
    table = fetch_table(sorted({s.site_id for s in samples}), first, last, args.cache)
    result = score(art['model'], samples, art['scalers'], table)
    if result is None:
        print('No window has a complete forecast archive.')
        return 1
    print(to_markdown(result))
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    raise SystemExit(main())
