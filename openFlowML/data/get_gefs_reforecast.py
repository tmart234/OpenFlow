"""
Real archived weather forecasts for training the ML decoder (Phase 2).

Training the decoder on observed weather assumes perfect foresight (r = 1.0),
which causes ~19-25% accuracy degradation on live GFS forecasts where day 1-7
precipitation correlation is only r = 0.29-0.48 (docs/MODEL_DESIGN.md).

This module fetches and caches per-station historical weather forecasts keyed
by `(issue_date, lead_day)` from two zero-auth public archives:

  1. Open-Meteo Previous Runs API (2024-present, fast JSON, lead days 1-7):
       https://previous-runs-api.open-meteo.com/v1/forecast (gfs_seamless)
     Pulls daily TMIN, TMAX, precipitation, and srad actually issued `k` days
     ahead, plus computes forecast-revision spread across consecutive runs as
     a proxy for uncertainty where multi-member archives are absent.

  2. NOAA GEFS v12 Reforecast on AWS Open Data (2000-2019, zero auth, 5 daily
     members c00 + p01..p04, 3-hourly out to 10-16 days):
       https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast/...
     Uses HTTP Range requests against each GRIB2 `.idx` inventory to fetch
     only the surface APCP / TMAX / TMIN / DSWRF records for the target
     members when `cfgrib` / `eccodes` is installed.

Cached per station under `<cache_dir>/reforecasts/<SITE_SAFE>.parquet` (or
`.csv.gz`), with schema:
    site_id, issue_date, lead_day, fc_TMIN, fc_TMAX, fc_precipitation,
    fc_srad, precip_ens_std, precip_ens_p90, temp_ens_std
Units match training: TMIN/TMAX & temp_ens_std in tenths of deg C, precip in
mm/day, srad in MJ/m^2/day.

CLI:
    python openFlowML/data/get_gefs_reforecast.py --cache-dir ~/.cache/openflow/dataset --start 2024-01-01
"""

import argparse
from datetime import datetime
import logging
import os
import sys
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(HERE)
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from data.utils import data_utils  # noqa: E402

logger = logging.getLogger(__name__)

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
GEFS_V12_BASE = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
GEFS_MEMBERS = ('c00', 'p01', 'p02', 'p03', 'p04')
OPEN_METEO_ARCHIVE_START = '2024-01-01'
OPEN_METEO_LEADS = tuple(range(1, 8))

REFORECAST_COLUMNS = [
    'site_id', 'issue_date', 'lead_day',
    'fc_TMIN', 'fc_TMAX', 'fc_precipitation', 'fc_srad',
    'precip_ens_std', 'precip_ens_p90', 'temp_ens_std',
]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=REFORECAST_COLUMNS)


def _site_slug(site_id: str) -> str:
    return site_id.replace(':', '_').replace('/', '_')


def _has_parquet() -> bool:
    for mod in ('pyarrow', 'fastparquet'):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def parse_open_meteo_previous_runs(site_id: str, payload: dict,
                                   leads: Sequence[int] = OPEN_METEO_LEADS) -> pd.DataFrame:
    """
    Parse one Open-Meteo Previous Runs JSON answer into REFORECAST_COLUMNS
    in training units (TMIN/TMAX in tenths deg C, precip mm, srad MJ/m^2/day).

    For each valid date, also computes cross-lead revision spread (std across
    available leads for that valid date) as a conservative ensemble-spread
    estimate when multi-member reforecast files aren't present.
    """
    hourly = (payload or {}).get('hourly') or {}
    times = hourly.get('time') or []
    if not times:
        return _empty()
    h = pd.DataFrame({'time': pd.to_datetime(times)})
    h['valid_date'] = h['time'].dt.normalize()
    for k in leads:
        for base in ('precipitation', 'temperature_2m', 'shortwave_radiation'):
            col = f'{base}_previous_day{k}'
            if col in hourly:
                h[col] = pd.to_numeric(hourly[col], errors='coerce')

    rows = []
    for k in leads:
        pcol = f'precipitation_previous_day{k}'
        tcol = f'temperature_2m_previous_day{k}'
        scol = f'shortwave_radiation_previous_day{k}'
        if pcol not in h or tcol not in h:
            continue
        agg_spec = {
            'precip': (pcol, lambda s: s.sum(min_count=20)),
            'tmin': (tcol, 'min'),
            'tmax': (tcol, 'max'),
            'n': (tcol, 'count'),
        }
        if scol in h:
            # Hourly W/m^2 summed over 24 h * 0.0036 -> MJ/m^2/day.
            agg_spec['srad'] = (scol, lambda s: s.sum(min_count=20) * 0.0036)
        g = h.groupby('valid_date').agg(**agg_spec).reset_index()
        g = g[g['n'] >= 20]
        for _, r in g.iterrows():
            if not (np.isfinite(r['tmin']) and np.isfinite(r['tmax'])):
                continue
            valid = pd.Timestamp(r['valid_date']).normalize()
            issue = valid - pd.Timedelta(days=int(k))
            p_val = float(r['precip']) if np.isfinite(r['precip']) else 0.0
            rows.append({
                'site_id': site_id,
                'valid_date': valid,
                'issue_date': issue,
                'lead_day': int(k),
                'fc_TMIN': float(r['tmin']) * 10.0,
                'fc_TMAX': float(r['tmax']) * 10.0,
                'fc_precipitation': max(0.0, p_val),
                'fc_srad': float(r['srad']) if ('srad' in r and np.isfinite(r['srad'])) else np.nan,
            })
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    # Estimate forecast uncertainty from run-to-run dispersion for each valid date.
    spread = df.groupby('valid_date').agg(
        precip_ens_std=('fc_precipitation', lambda s: float(np.nanstd(s)) if len(s) > 1 else 0.0),
        precip_ens_p90=('fc_precipitation', lambda s: float(np.nanpercentile(s, 90)) if len(s) else 0.0),
        temp_ens_std=('fc_TMAX', lambda s: float(np.nanstd(s)) if len(s) > 1 else 0.0),
    ).reset_index()
    df = df.merge(spread, on='valid_date', how='left')
    return df[REFORECAST_COLUMNS].sort_values(['issue_date', 'lead_day']).reset_index(drop=True)


def fetch_open_meteo_reforecast(site_id: str, lat: float, lon: float,
                                start_date, end_date,
                                tz: str = 'America/Denver') -> pd.DataFrame:
    """Fetch archived GFS forecasts from Open-Meteo Previous Runs (2024+)."""
    start = max(pd.Timestamp(start_date).normalize(), pd.Timestamp(OPEN_METEO_ARCHIVE_START))
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        return _empty()
    hourly_vars = (
        [f'precipitation_previous_day{k}' for k in OPEN_METEO_LEADS]
        + [f'temperature_2m_previous_day{k}' for k in OPEN_METEO_LEADS]
        + [f'shortwave_radiation_previous_day{k}' for k in OPEN_METEO_LEADS]
    )
    parts = []
    for cs, ce in data_utils.date_chunks(start.to_pydatetime(), end.to_pydatetime(), 180):
        params = {
            'latitude': round(float(lat), 4),
            'longitude': round(float(lon), 4),
            'start_date': cs.strftime('%Y-%m-%d'),
            'end_date': ce.strftime('%Y-%m-%d'),
            'hourly': ','.join(hourly_vars),
            'models': 'gfs_seamless',
            'timezone': tz,
        }
        resp = data_utils.request_with_retry(PREVIOUS_RUNS_URL, params=params, timeout=180)
        if resp is None:
            logger.warning("Open-Meteo Previous Runs failed for %s %s..%s",
                           site_id, params['start_date'], params['end_date'])
            continue
        try:
            payload = resp.json()
        except ValueError:
            continue
        if payload.get('error'):
            logger.warning("Open-Meteo Previous Runs error for %s: %s", site_id, payload.get('reason'))
            continue
        parsed = parse_open_meteo_previous_runs(site_id, payload)
        if not parsed.empty:
            parts.append(parsed)
    if not parts:
        return _empty()
    return (pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset=['site_id', 'issue_date', 'lead_day'], keep='last')
            .sort_values(['issue_date', 'lead_day']).reset_index(drop=True))


def gefs_v12_grib_url(issue_date, member: str = 'c00',
                      var_file: str = 'apcp_sfc',
                      window: str = 'Days:1-10') -> str:
    """Public AWS S3 URL for one NOAA GEFS v12 reforecast GRIB2 file (2000-2019)."""
    d = pd.Timestamp(issue_date)
    stamp = d.strftime('%Y%m%d00')
    return f"{GEFS_V12_BASE}/{d.year}/{stamp}/{member}/{window}/{var_file}_{stamp}_{member}.grib2"


def parse_grib_idx(idx_text: str) -> List[dict]:
    """Parse a NOAA GRIB2 `.idx` inventory into [{byte_start, byte_end, var, level, lead_str}]."""
    entries = []
    for line in idx_text.splitlines():
        parts = line.split(':')
        if len(parts) < 6:
            continue
        try:
            start = int(parts[1])
        except ValueError:
            continue
        entries.append({
            'byte_start': start,
            'byte_end': None,
            'var': parts[3],
            'level': parts[4],
            'lead_str': parts[5],
        })
    for i in range(len(entries) - 1):
        entries[i]['byte_end'] = entries[i + 1]['byte_start'] - 1
    return entries


def aggregate_gefs_members(site_id: str, issue_date,
                           member_daily: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine per-member daily DataFrames[lead_day, TMIN, TMAX, precipitation, srad]
    from GEFS v12 reforecasts into REFORECAST_COLUMNS (ensemble mean + std + p90).
    """
    if not member_daily:
        return _empty()
    issue = pd.Timestamp(issue_date).normalize()
    stacked = pd.concat(
        [df.assign(_m=m) for m, df in member_daily.items() if df is not None and not df.empty],
        ignore_index=True,
    )
    if stacked.empty:
        return _empty()
    rows = []
    for lead, g in stacked.groupby('lead_day'):
        p = pd.to_numeric(g['precipitation'], errors='coerce').dropna().to_numpy(dtype=float)
        tmin = pd.to_numeric(g['TMIN'], errors='coerce').dropna().to_numpy(dtype=float)
        tmax = pd.to_numeric(g['TMAX'], errors='coerce').dropna().to_numpy(dtype=float)
        srad = (pd.to_numeric(g['srad'], errors='coerce').dropna().to_numpy(dtype=float)
                if 'srad' in g else np.array([]))
        if not len(tmin) or not len(tmax):
            continue
        rows.append({
            'site_id': site_id,
            'issue_date': issue,
            'lead_day': int(lead),
            'fc_TMIN': float(np.mean(tmin)),
            'fc_TMAX': float(np.mean(tmax)),
            'fc_precipitation': float(np.mean(p)) if len(p) else 0.0,
            'fc_srad': float(np.mean(srad)) if len(srad) else np.nan,
            'precip_ens_std': float(np.std(p)) if len(p) > 1 else 0.0,
            'precip_ens_p90': float(np.percentile(p, 90)) if len(p) else 0.0,
            'temp_ens_std': float(np.std(tmax)) if len(tmax) > 1 else 0.0,
        })
    return pd.DataFrame(rows, columns=REFORECAST_COLUMNS) if rows else _empty()


def save_reforecast_table(df: pd.DataFrame, site_id: str, cache_dir: str) -> str:
    """Atomically write a station's archived forecast table into `<cache_dir>/reforecasts/`."""
    rdir = os.path.join(cache_dir, 'reforecasts')
    os.makedirs(rdir, exist_ok=True)
    slug = _site_slug(site_id)
    out = df.copy()
    if 'issue_date' in out.columns:
        out['issue_date'] = pd.to_datetime(out['issue_date']).dt.normalize()
    use_pq = _has_parquet()
    ext = '.parquet' if use_pq else '.csv.gz'
    target = os.path.join(rdir, f'{slug}{ext}')
    fd, tmp = tempfile.mkstemp(dir=rdir, suffix=f'.tmp{ext}')
    os.close(fd)
    try:
        if use_pq:
            out.to_parquet(tmp, index=False)
        else:
            out.to_csv(tmp, index=False, compression='gzip')
        os.replace(tmp, target)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return target


def load_reforecast_table(cache_dir: str, sites: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Load cached per-station reforecast tables from `<cache_dir>/reforecasts/`."""
    rdir = os.path.join(cache_dir, 'reforecasts')
    if not os.path.isdir(rdir):
        return _empty()
    parts = []
    if sites is not None:
        stems = [_site_slug(s) for s in sites]
    else:
        stems = sorted({
            fn[:-8] if fn.endswith('.parquet') else fn[:-7]
            for fn in os.listdir(rdir)
            if fn.endswith('.parquet') or fn.endswith('.csv.gz')
        })
    for stem in stems:
        for ext, reader in (('.parquet', pd.read_parquet),
                            ('.csv.gz', lambda p: pd.read_csv(p, compression='gzip'))):
            path = os.path.join(rdir, f'{stem}{ext}')
            if os.path.exists(path):
                try:
                    df = reader(path)
                    if not df.empty:
                        df['issue_date'] = pd.to_datetime(df['issue_date']).dt.normalize()
                        df['lead_day'] = df['lead_day'].astype(int)
                        parts.append(df)
                except Exception as e:
                    logger.warning("Failed reading reforecast cache %s: %s", path, e)
                break
    if not parts:
        return _empty()
    return (pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset=['site_id', 'issue_date', 'lead_day'], keep='last')
            .sort_values(['site_id', 'issue_date', 'lead_day']).reset_index(drop=True))


def sync_reforecasts(cache_dir: str, sites: Optional[Sequence[str]] = None,
                     start_date: str = OPEN_METEO_ARCHIVE_START,
                     end_date: Optional[str] = None) -> Dict[str, int]:
    """Incrementally sync Open-Meteo Previous Runs forecasts (2024+) into `cache_dir`."""
    import combine_data
    import ml_inference
    site_list = list(sites) if sites else combine_data.get_site_ids()
    end = pd.Timestamp(end_date or datetime.now()).normalize()
    counts: Dict[str, int] = {}
    for site_id in site_list:
        try:
            existing = load_reforecast_table(cache_dir, [site_id])
            fetch_start = pd.Timestamp(start_date).normalize()
            if not existing.empty:
                last_issue = existing['issue_date'].max()
                if last_issue >= end - pd.Timedelta(days=1):
                    counts[site_id] = len(existing)
                    continue
                fetch_start = max(fetch_start, last_issue - pd.Timedelta(days=7))
            coords = ml_inference._coords(site_id)
            fresh = fetch_open_meteo_reforecast(site_id, coords['lat'], coords['lon'], fetch_start, end)
            combined = (pd.concat([existing, fresh], ignore_index=True)
                        .drop_duplicates(subset=['site_id', 'issue_date', 'lead_day'], keep='last')
                        .sort_values(['issue_date', 'lead_day']).reset_index(drop=True))
            if not combined.empty:
                save_reforecast_table(combined, site_id, cache_dir)
            counts[site_id] = len(combined)
        except Exception as e:
            logger.warning("Reforecast sync failed for %s: %s", site_id, e)
            counts[site_id] = 0
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Sync archived weather forecasts into the feature store.')
    parser.add_argument('--cache-dir', default=os.path.expanduser('~/.cache/openflow/dataset'))
    parser.add_argument('--start', default=OPEN_METEO_ARCHIVE_START)
    parser.add_argument('--end', default=None)
    parser.add_argument('--site', action='append')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    counts = sync_reforecasts(args.cache_dir, sites=args.site, start_date=args.start, end_date=args.end)
    for sid, n in counts.items():
        print(f"  {sid:<16} {n} (issue_date, lead_day) forecast rows")
    return 0


if __name__ == '__main__':
    sys.exit(main())
