"""
Incremental per-station Feature Store cache for OpenFlow training & CV.

Fetching 15-25 years of USGS IV, CO DWR, gridMET, SNOTEL, USDM, and SMAP from
scratch on every CI run takes ~1.5-2 hours for 8 stations and blocks scaling
to more rivers or longer histories. This module caches each station's raw
merged daily DataFrame (the pre-normalization output of combine_data) and
refreshes it incrementally:

  - Cold cache / fingerprint change -> full fetch of [start_date, end_date].
  - Earlier start_date requested (e.g. extending 15 -> 25 years) -> backfill
    only the missing older span and prepend to the cached frame.
  - Newer end_date requested -> re-fetch only the trailing `refresh_days`
    window (default 14 days, picking up USGS provisional revisions, gridMET's
    ~2-day catch-up, and weekly USDM updates) and splice cleanly onto history.
  - Normalization (scalers.json) is intentionally NOT cached here so train.py
    and crossval.py always refit scalers strictly on each split's training
    dates with zero leakage.

Storage:
  <cache_dir>/
    manifest.json
    stations/<SITE_SAFE>.parquet   (automatic .csv.gz fallback without pyarrow)

Usage:
    python openFlowML/data/feature_store.py --cache-dir ~/.cache/openflow/dataset --years 25
    python openFlowML/data/feature_store.py --cache-dir ~/.cache/openflow/dataset --info
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
import sys
import tempfile
from typing import Callable, Dict, Optional, Sequence, Tuple

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(HERE)
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from data import basin_zones, get_upstream  # noqa: E402

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3
DEFAULT_CACHE_DIR = os.path.expanduser('~/.cache/openflow/dataset')
DEFAULT_REFRESH_DAYS = 14
# Boundary buffer on incremental splices: keep freshly fetched rows starting
# this many days after `tail_start` so short interior-gap interpolation at the
# fetch window's left edge never overwrites already-complete cached rows.
SPLICE_BUFFER_DAYS = 7

# Env vars that alter raw feature columns fetched by combine_data.
_FEATURE_TOGGLE_VARS = (
    'OPENFLOW_DISABLE_SMAP',
    'OPENFLOW_DISABLE_DROUGHT',
    'OPENFLOW_DISABLE_RESERVOIR',
    'OPENFLOW_DISABLE_UPSTREAM',
    'OPENFLOW_DISABLE_BASIN_PRECIP',
    'OPENFLOW_DISABLE_WEATHER_FILL',
    'OPENFLOW_DISABLE_ENERGY',
    'OPENFLOW_DISABLE_HYDRO_FORCING',
    'OPENFLOW_DISABLE_GRIDDED_SWE',
    'OPENFLOW_DISABLE_PRIOR',
)


def _has_parquet() -> bool:
    for mod in ('pyarrow', 'fastparquet'):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def _site_slug(site_id: str) -> str:
    return site_id.replace(':', '_').replace('/', '_')


def _norm_day(d) -> pd.Timestamp:
    ts = pd.Timestamp(d)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def station_fingerprint(site_id: str,
                        weather_source: Optional[str] = None,
                        registry: Optional[dict] = None,
                        zones: Optional[dict] = None,
                        hydro_source: Optional[str] = None) -> str:
    """
    Deterministic hash of every setting that affects a station's cached raw
    feature table. Changing a station's upstream gauges, basin zones, weather
    source, hydro forcing source, or disabled-source flags invalidates only
    the affected station(s).
    """
    if weather_source is None:
        import combine_data
        weather_source = combine_data.weather_source()
    if hydro_source is None:
        from data import get_hydro_forcing
        hydro_source = get_hydro_forcing.hydro_source()
    disabled = sorted(
        var for var in _FEATURE_TOGGLE_VARS
        if os.getenv(var, '').strip() in ('1', 'true', 'True')
    )
    up_cfg = get_upstream.upstream_config(site_id, registry)
    z_cfg = basin_zones.station_zones(site_id, zones)
    z_meta = None
    if z_cfg:
        z_meta = {
            'generated': z_cfg.get('generated'),
            'contributing_area_km2': z_cfg.get('contributing_area_km2'),
            'zone_km': z_cfg.get('zone_km'),
            'ungauged_area_km2': (z_cfg.get('ungauged') or {}).get('area_km2'),
            'ungauged_gauges': (z_cfg.get('ungauged') or {}).get('gauges'),
        }
    payload = {
        'schema_version': SCHEMA_VERSION,
        'site_id': site_id,
        'weather_source': weather_source,
        'hydro_source': hydro_source,
        'disabled': disabled,
        'upstream': up_cfg,
        'basin_zones': z_meta,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()[:16]


def _manifest_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, 'manifest.json')


def load_manifest(cache_dir: str) -> dict:
    path = _manifest_path(cache_dir)
    if not os.path.exists(path):
        return {'schema_version': SCHEMA_VERSION, 'stations': {}}
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {'schema_version': SCHEMA_VERSION, 'stations': {}}
        data.setdefault('stations', {})
        return data
    except (OSError, ValueError) as e:
        logger.warning("Corrupt cache manifest %s (%s); starting fresh", path, e)
        return {'schema_version': SCHEMA_VERSION, 'stations': {}}


def _atomic_write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _read_station_frame(cache_dir: str, site_id: str, fmt: Optional[str] = None) -> Optional[pd.DataFrame]:
    sdir = os.path.join(cache_dir, 'stations')
    slug = _site_slug(site_id)
    candidates = []
    if fmt == 'parquet':
        candidates = [(os.path.join(sdir, f'{slug}.parquet'), 'parquet'),
                      (os.path.join(sdir, f'{slug}.csv.gz'), 'csv.gz')]
    elif fmt == 'csv.gz':
        candidates = [(os.path.join(sdir, f'{slug}.csv.gz'), 'csv.gz'),
                      (os.path.join(sdir, f'{slug}.parquet'), 'parquet')]
    else:
        candidates = [(os.path.join(sdir, f'{slug}.parquet'), 'parquet'),
                      (os.path.join(sdir, f'{slug}.csv.gz'), 'csv.gz')]
    for path, kind in candidates:
        if not os.path.exists(path):
            continue
        try:
            if kind == 'parquet':
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, compression='gzip')
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
            if 'huc8' in df.columns:
                df['huc8'] = df['huc8'].fillna('').astype(str)
            return df
        except Exception as e:
            logger.warning("Failed reading cached %s (%s); will rebuild", path, e)
    return None


def _write_station_frame(cache_dir: str, site_id: str, df: pd.DataFrame) -> Tuple[str, str]:
    sdir = os.path.join(cache_dir, 'stations')
    os.makedirs(sdir, exist_ok=True)
    slug = _site_slug(site_id)
    out = df.copy()
    if 'Date' in out.columns:
        out['Date'] = pd.to_datetime(out['Date']).dt.normalize()
    if 'huc8' in out.columns:
        out['huc8'] = out['huc8'].fillna('').astype(str)

    use_parquet = _has_parquet()
    fmt = 'parquet' if use_parquet else 'csv.gz'
    ext = '.parquet' if use_parquet else '.csv.gz'
    target = os.path.join(sdir, f'{slug}{ext}')
    fd, tmp = tempfile.mkstemp(dir=sdir, suffix=f'.tmp{ext}')
    os.close(fd)
    try:
        if use_parquet:
            out.to_parquet(tmp, index=False)
        else:
            out.to_csv(tmp, index=False, compression='gzip')
        os.replace(tmp, target)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    # Clean up alternate format file if switching between parquet and csv.gz.
    other = os.path.join(sdir, f'{slug}.csv.gz' if use_parquet else f'{slug}.parquet')
    if os.path.exists(other):
        try:
            os.unlink(other)
        except OSError:
            pass
    return target, fmt


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out['Date'] = pd.to_datetime(out['Date']).dt.normalize()
    out = out.sort_values('Date').drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)
    return out


def _splice_tail(existing: pd.DataFrame, tail: pd.DataFrame, tail_start: pd.Timestamp,
                 refresh_days: int) -> pd.DataFrame:
    """Splice freshly fetched trailing rows onto existing cached history."""
    existing = _clean_frame(existing)
    tail = _clean_frame(tail)
    if tail.empty:
        return existing
    if existing.empty:
        return tail
    buf = min(SPLICE_BUFFER_DAYS, max(0, refresh_days // 2))
    cutoff = tail_start + pd.Timedelta(days=buf)
    # If existing history does not even reach cutoff, fall back to tail_start.
    if existing['Date'].max() < cutoff:
        cutoff = tail_start
    left = existing[existing['Date'] < cutoff]
    right = tail[tail['Date'] >= cutoff]
    if right.empty:
        right = tail
    return _clean_frame(pd.concat([left, right], ignore_index=True))


def _splice_backfill(backfill: pd.DataFrame, existing: pd.DataFrame,
                     cached_req_start: pd.Timestamp) -> pd.DataFrame:
    """Prepend older backfilled rows before existing cached history."""
    backfill = _clean_frame(backfill)
    existing = _clean_frame(existing)
    if backfill.empty:
        return existing
    if existing.empty:
        return backfill
    older = backfill[backfill['Date'] < cached_req_start]
    return _clean_frame(pd.concat([older, existing], ignore_index=True))


def load_or_update_station(
    site_id: str,
    start_date,
    end_date,
    fetch_fn: Callable[[str, datetime, datetime], pd.DataFrame],
    cache_dir: str,
    *,
    refresh_days: int = DEFAULT_REFRESH_DAYS,
    force_rebuild: bool = False,
    fingerprint: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Return (merged_df_for_window, status) for `site_id` over [start_date, end_date],
    updating the on-disk cache incrementally as needed.

    `fetch_fn(site_id, start_dt, end_dt)` must return the raw merged DataFrame
    for that date range (same schema as combine_data.merge_dataframes).

    status is one of: 'hit', 'incremental', 'backfill', 'backfill+incremental',
    'rebuilt', 'empty'.
    """
    req_start = _norm_day(start_date)
    req_end = _norm_day(end_date)
    fp = fingerprint if fingerprint is not None else station_fingerprint(site_id)

    manifest = load_manifest(cache_dir)
    entry = manifest.get('stations', {}).get(site_id)
    cached_df = None
    if (not force_rebuild and entry and entry.get('fingerprint') == fp
            and entry.get('schema_version') == SCHEMA_VERSION):
        cached_df = _read_station_frame(cache_dir, site_id, fmt=entry.get('format'))

    if cached_df is None or cached_df.empty:
        logger.info("Feature store [%s]: cold fetch %s..%s", site_id, req_start.date(), req_end.date())
        fresh = _clean_frame(fetch_fn(site_id, req_start.to_pydatetime(), req_end.to_pydatetime()))
        if fresh.empty:
            return fresh, 'empty'
        _, fmt = _write_station_frame(cache_dir, site_id, fresh)
        _update_manifest(cache_dir, manifest, site_id, fp, req_start, req_end, fresh, fmt)
        return _slice_window(fresh, req_start, req_end), 'rebuilt'

    cached_df = _clean_frame(cached_df)
    cached_req_start = _norm_day(entry.get('requested_start') or cached_df['Date'].min())
    cached_req_end = _norm_day(entry.get('requested_end') or cached_df['Date'].max())
    actions = []
    updated = cached_df
    new_req_start = cached_req_start
    new_req_end = cached_req_end

    # 1. Backfill older years if caller asks for an earlier start date.
    if req_start < cached_req_start - pd.Timedelta(days=1):
        bf_end = min(req_end, cached_req_start + pd.Timedelta(days=refresh_days))
        logger.info("Feature store [%s]: backfilling %s..%s", site_id, req_start.date(), bf_end.date())
        bf = _clean_frame(fetch_fn(site_id, req_start.to_pydatetime(), bf_end.to_pydatetime()))
        if not bf.empty:
            updated = _splice_backfill(bf, updated, cached_req_start)
            new_req_start = req_start
            actions.append('backfill')

    # 2. Refresh trailing window if caller asks for newer dates.
    if req_end > cached_req_end:
        anchor = min(updated['Date'].max(), cached_req_end)
        tail_start = max(req_start, anchor - pd.Timedelta(days=refresh_days))
        logger.info("Feature store [%s]: incremental refresh %s..%s",
                    site_id, tail_start.date(), req_end.date())
        tail = _clean_frame(fetch_fn(site_id, tail_start.to_pydatetime(), req_end.to_pydatetime()))
        if not tail.empty:
            updated = _splice_tail(updated, tail, tail_start, refresh_days)
            new_req_end = max(cached_req_end, req_end)
            actions.append('incremental')
        else:
            logger.warning("Feature store [%s]: incremental refresh returned empty; keeping cached rows",
                           site_id)

    if actions:
        _, fmt = _write_station_frame(cache_dir, site_id, updated)
        _update_manifest(cache_dir, manifest, site_id, fp, new_req_start, new_req_end, updated, fmt)
        status = '+'.join(actions)
    else:
        logger.info("Feature store [%s]: cache hit (%d rows, %s..%s)",
                    site_id, len(updated), updated['Date'].min().date(), updated['Date'].max().date())
        status = 'hit'

    return _slice_window(updated, req_start, req_end), status


def _slice_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (df['Date'] >= start) & (df['Date'] <= end)
    return df.loc[mask].reset_index(drop=True)


def _update_manifest(cache_dir: str, manifest: dict, site_id: str, fingerprint: str,
                     req_start: pd.Timestamp, req_end: pd.Timestamp,
                     df: pd.DataFrame, fmt: str) -> None:
    manifest['schema_version'] = SCHEMA_VERSION
    manifest['updated_utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    manifest.setdefault('stations', {})[site_id] = {
        'schema_version': SCHEMA_VERSION,
        'fingerprint': fingerprint,
        'format': fmt,
        'requested_start': str(req_start.date()),
        'requested_end': str(req_end.date()),
        'min_date': str(df['Date'].min().date()),
        'max_date': str(df['Date'].max().date()),
        'rows': int(len(df)),
        'updated_utc': manifest['updated_utc'],
    }
    _atomic_write_json(_manifest_path(cache_dir), manifest)


def sync_all(cache_dir: str = DEFAULT_CACHE_DIR,
             training_years: Optional[int] = None,
             sites: Optional[Sequence[str]] = None,
             refresh_days: int = DEFAULT_REFRESH_DAYS,
             force_rebuild: bool = False) -> Dict[str, str]:
    """Incrementally sync all configured training stations into `cache_dir`."""
    import combine_data
    years = int(training_years or os.getenv('OPENFLOW_TRAINING_YEARS', '') or combine_data.TRAINING_YEARS)
    site_list = list(sites) if sites else combine_data.get_site_ids()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    results: Dict[str, str] = {}
    for site_id in site_list:
        try:
            df, status = load_or_update_station(
                site_id, start_date, end_date,
                combine_data.fetch_station_merged,
                cache_dir,
                refresh_days=refresh_days,
                force_rebuild=force_rebuild,
            )
            results[site_id] = f"{status} ({len(df)} rows)"
        except Exception as e:
            logger.error("Feature store sync failed for %s: %s", site_id, e)
            results[site_id] = f"error: {e}"
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Sync or inspect the OpenFlow incremental feature store.')
    parser.add_argument('--cache-dir',
                        default=os.getenv('OPENFLOW_CACHE_DIR', '') or DEFAULT_CACHE_DIR,
                        help=f'Directory holding cached station tables (default {DEFAULT_CACHE_DIR})')
    parser.add_argument('--years', type=int, default=None,
                        help='Years of history to cover (default combine_data.TRAINING_YEARS)')
    parser.add_argument('--refresh-days', type=int, default=DEFAULT_REFRESH_DAYS,
                        help=f'Trailing days re-fetched on incremental update (default {DEFAULT_REFRESH_DAYS})')
    parser.add_argument('--site', action='append', help='Specific site_id(s) to sync; default all')
    parser.add_argument('--force', action='store_true', help='Force full rebuild of cached stations')
    parser.add_argument('--info', action='store_true', help='Print cache manifest without fetching')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if args.info:
        print(json.dumps(load_manifest(args.cache_dir), indent=2))
        return 0

    results = sync_all(
        cache_dir=args.cache_dir,
        training_years=args.years,
        sites=args.site,
        refresh_days=args.refresh_days,
        force_rebuild=args.force,
    )
    for site_id, summary in results.items():
        print(f"  {site_id:<16} {summary}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
