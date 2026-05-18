"""
SMAP L3 enhanced soil moisture (SPL3SMP_E v006) timeseries by HUC8.

Given a station's lat/lon and a date window, this module looks up the enclosing
HUC8 polygon, searches NASA Earthdata for every SMAP granule whose footprint
intersects that polygon over the window, downloads them (with a granule cache
shared across calls so neighboring stations don't redownload the same global
daily file), filters each granule to recommended-quality pixels inside the
polygon's bbox, and returns the daily polygon mean.

Public API:
    main(lat, lon, start_date, end_date) -> DataFrame[Date, soil_moisture]

A failure at any step (HUC8 lookup, Earthdata auth, search, download,
extraction) degrades to an empty DataFrame -- combine_data treats missing
soil moisture as "no SMAP today" (forward-filled, then site-median fallback,
plus an sm_observed indicator) rather than dropping the row.

Performance: SPL3SMP_E granules are global daily files (~30-100 MB each), so
two stations in different HUC8s on the same day pull the same granule. The
module-level _GRANULE_PATH_CACHE deduplicates within a single process; set
OPENFLOW_SMAP_CACHE_DIR to a persistent path to keep granules across runs.
"""

import argparse
import logging
import os
import tempfile
from datetime import date, datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# NSIDC SMAP L3 enhanced (9 km) passive radiometer soil moisture.
SMAP_SHORT_NAME = "SPL3SMP_E"
SMAP_VERSION = "006"
# HDF5 fill value for missing pixels in the SMAP product.
FILL_VALUE = -9999.0
# Valid SMAP retrieval range for volumetric soil moisture (m^3/m^3).
VALID_MIN = 0.0
VALID_MAX = 1.0
# retrieval_qual_flag bit 0: 0 = retrieval is recommended quality, 1 = not.
# Conservative filter: drop any pixel where bit 0 is set.
QUAL_RECOMMENDED_BIT = 0

# Process-level cache so the per-station SMAP loop doesn't re-download the
# same global daily granule for every basin. Keyed by granule filename.
_GRANULE_PATH_CACHE: dict = {}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=['Date', 'soil_moisture'])


def _to_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()


def _login_earthdata():
    """
    Authenticate with NASA Earthdata via earthaccess. Returns the Auth on
    success, None on any failure -- the caller short-circuits to an empty
    series so a missing credential doesn't bring down combine_data.
    """
    try:
        import earthaccess
    except ImportError as e:
        logger.warning("earthaccess not installed: %s", e)
        return None
    try:
        auth = earthaccess.login(strategy="environment")
        if auth is not None and getattr(auth, 'authenticated', False):
            return auth
        logger.warning("Earthdata authentication did not succeed")
        return None
    except Exception as e:
        logger.warning("Earthdata authentication error: %s", e)
        return None


def _get_huc8_polygon(lat: float, lon: float):
    """
    HUC8 polygon (list of (lon, lat) tuples) enclosing the point, simplified.
    Returns None when the WBD lookup fails or the simplification dependencies
    aren't installed.
    """
    try:
        from data.utils.get_poly import get_huc_polygon, simplify_polygon
    except ImportError as e:
        logger.warning("Polygon utilities unavailable: %s", e)
        return None
    result = get_huc_polygon(lat, lon, huc_level=8)
    if not result:
        return None
    polygon, _huc_id, _attributes = result
    if not polygon:
        return None
    return simplify_polygon(polygon)


def _search_granules(polygon, start_date, end_date):
    """SMAP granules intersecting the polygon's bbox over the date window."""
    try:
        import earthaccess
    except ImportError as e:
        logger.warning("earthaccess not installed: %s", e)
        return []
    lons, lats = zip(*polygon)
    bbox = (min(lons), min(lats), max(lons), max(lats))
    try:
        granules = earthaccess.search_data(
            short_name=SMAP_SHORT_NAME,
            version=SMAP_VERSION,
            temporal=(_to_date(start_date), _to_date(end_date)),
            bounding_box=bbox,
        )
    except Exception as e:
        logger.warning("SMAP granule search failed: %s", e)
        return []
    return list(granules) if granules else []


def _granule_date(granule) -> Optional[date]:
    """
    Observation date for a granule. SMAP L3 filenames embed YYYYMMDD
    (e.g. SMAP_L3_SM_P_E_20240115_R18290_001.h5), so we lift it from there
    and fall back to the granule's temporal metadata if the filename pattern
    doesn't match.
    """
    try:
        links = granule.data_links() if hasattr(granule, 'data_links') else []
        for link in links or []:
            name = link.rsplit('/', 1)[-1]
            for token in name.split('_'):
                if len(token) == 8 and token.isdigit():
                    try:
                        return datetime.strptime(token, "%Y%m%d").date()
                    except ValueError:
                        continue
    except Exception:
        pass
    try:
        umm = granule.get("umm", {}) if hasattr(granule, 'get') else {}
        beg = (umm.get("TemporalExtent", {})
                  .get("RangeDateTime", {})
                  .get("BeginningDateTime"))
        if beg:
            return datetime.strptime(beg[:10], "%Y-%m-%d").date()
    except Exception:
        pass
    return None


def _extract_polygon_mean(hdf_path: str, polygon,
                          use_quality_flag: bool = True) -> Optional[float]:
    """
    Average volumetric soil moisture across all valid SMAP pixels inside the
    polygon's bounding box, across both AM and PM passes. Returns None when
    no valid pixel intersects (cloudy / RFI day, frozen ground, or polygon
    entirely off the EASE-grid for that pass).

    Pixels are filtered against the explicit fill value, the [0, 1] valid
    range, and -- when the dataset is present -- the SMAP retrieval_qual_flag
    bit 0 ("recommended quality"). Without the quality filter, winter Colorado
    retrievals include frozen-ground pixels that look numerically plausible
    but the SMAP team flags as not recommended.

    Bbox is sufficient at SMAP's 9 km grid spacing -- the HUC8 footprint is
    rarely much larger than a handful of cells and the bbox vs true polygon
    distinction is below the retrieval noise floor.
    """
    try:
        import h5py
        import numpy as np
    except ImportError as e:
        logger.warning("h5py/numpy unavailable: %s", e)
        return None

    lons, lats = zip(*polygon)
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    # SMAP L3 v006 splits AM and PM passes into separate top-level groups.
    pass_groups = [
        'Soil_Moisture_Retrieval_Data_AM',
        'Soil_Moisture_Retrieval_Data_PM',
        'Soil_Moisture_Retrieval_Data',
    ]

    collected = []
    try:
        with h5py.File(hdf_path, 'r') as f:
            for grp_name in pass_groups:
                if grp_name not in f:
                    continue
                grp = f[grp_name]
                if not all(k in grp for k in ('soil_moisture', 'latitude', 'longitude')):
                    continue
                sm = grp['soil_moisture'][:]
                lat = grp['latitude'][:]
                lon = grp['longitude'][:]
                in_bbox = ((lon >= min_lon) & (lon <= max_lon) &
                           (lat >= min_lat) & (lat <= max_lat))
                valid = ((sm != FILL_VALUE) & (sm >= VALID_MIN) & (sm <= VALID_MAX) &
                         (lon != FILL_VALUE) & (lat != FILL_VALUE))
                # Quality flag: drop pixels where the recommended-quality bit
                # is set. Older granules occasionally omit the dataset; in that
                # case we proceed without the filter rather than dropping all.
                if use_quality_flag and 'retrieval_qual_flag' in grp:
                    qflag = grp['retrieval_qual_flag'][:].astype('int32')
                    recommended = ((qflag >> QUAL_RECOMMENDED_BIT) & 1) == 0
                    valid = valid & recommended
                mask = in_bbox & valid
                if mask.any():
                    collected.extend(sm[mask].astype(float).tolist())
    except Exception as e:
        logger.warning("Failed to read %s: %s", hdf_path, e)
        return None

    if not collected:
        return None
    return float(sum(collected) / len(collected))


def _granule_cache_key(granule) -> Optional[str]:
    """
    Stable id for a granule, used as the cache key. Falls back through
    `.data_links()` (the most reliable per-granule identifier) before giving
    up; granules without a usable id are simply not cached.
    """
    try:
        links = granule.data_links() if hasattr(granule, 'data_links') else []
        for link in links or []:
            name = link.rsplit('/', 1)[-1]
            if name:
                return name
    except Exception:
        pass
    return None


def _get_cache_dir() -> str:
    """
    Persistent SMAP granule cache directory. Defaults to a subdirectory of the
    system tempdir; override with OPENFLOW_SMAP_CACHE_DIR to share the cache
    across runs (e.g. a CI cache action).
    """
    base = (os.environ.get('OPENFLOW_SMAP_CACHE_DIR') or
            os.path.join(tempfile.gettempdir(), 'openflow_smap_cache'))
    os.makedirs(base, exist_ok=True)
    return base


def _download_granule(granule, cache_dir):
    """
    Download `granule` into `cache_dir`, or return its cached path. Returns
    None on failure (and does NOT delete the cached file -- the cache is
    intentionally process-lifetime+).
    """
    key = _granule_cache_key(granule)
    if key:
        cached = _GRANULE_PATH_CACHE.get(key)
        if cached and os.path.exists(cached):
            logger.debug("Granule cache hit: %s", key)
            return cached
        # Also check disk in case a previous process populated the dir.
        on_disk = os.path.join(cache_dir, key)
        if os.path.exists(on_disk):
            _GRANULE_PATH_CACHE[key] = on_disk
            return on_disk
    try:
        import earthaccess
    except ImportError:
        return None
    try:
        files = earthaccess.download([granule], local_path=cache_dir)
    except Exception as e:
        logger.warning("Granule download failed: %s", e)
        return None
    if not files:
        return None
    path = files[0]
    if key:
        _GRANULE_PATH_CACHE[key] = path
    return path


def main(lat: float, lon: float, start_date, end_date) -> pd.DataFrame:
    """
    Fetch the SMAP L3 enhanced soil-moisture timeseries for the HUC8 enclosing
    (lat, lon) over [start_date, end_date].

    Returns a DataFrame with columns ['Date', 'soil_moisture'] where
    soil_moisture is volumetric m^3/m^3 in [0, 1]. Multiple AM/PM passes on
    the same day are averaged. Empty DataFrame on any failure -- the
    combine_data spine treats missing soil moisture the same way it treats
    missing SWE (defaults to 0, never drops the row).
    """
    try:
        polygon = _get_huc8_polygon(lat, lon)
    except Exception as e:
        logger.warning("HUC8 polygon lookup failed: %s", e)
        return _empty()
    if not polygon:
        logger.warning("No HUC8 polygon found for (%s, %s)", lat, lon)
        return _empty()

    if _login_earthdata() is None:
        return _empty()

    granules = _search_granules(polygon, start_date, end_date)
    if not granules:
        logger.warning("No SMAP granules found in [%s, %s] for the polygon",
                       start_date, end_date)
        return _empty()
    logger.info("Found %d SMAP granules", len(granules))

    cache_dir = _get_cache_dir()
    rows = []
    for granule in granules:
        obs_date = _granule_date(granule)
        if obs_date is None:
            continue
        path = _download_granule(granule, cache_dir)
        if not path:
            continue
        # Cached files are intentionally NOT removed -- the next station's
        # call to main() in this process will hit the cache for the same
        # global daily granule.
        value = _extract_polygon_mean(path, polygon)
        if value is None:
            continue
        rows.append((obs_date.strftime('%Y-%m-%d'), value))

    if not rows:
        return _empty()

    df = pd.DataFrame(rows, columns=['Date', 'soil_moisture'])
    df['soil_moisture'] = pd.to_numeric(df['soil_moisture'], errors='coerce')
    # AM + PM passes on the same date collapse to a single daily mean.
    daily = df.groupby('Date', as_index=False)['soil_moisture'].mean()
    return daily.sort_values('Date').reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fetch SMAP L3 enhanced soil moisture timeseries by HUC8.')
    parser.add_argument('--lat', type=float, required=True)
    parser.add_argument('--lon', type=float, required=True)
    parser.add_argument('--start-date', type=str, required=True, help='YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, required=True, help='YYYY-MM-DD')
    args = parser.parse_args()
    df = main(args.lat, args.lon, args.start_date, args.end_date)
    if df.empty:
        print("No data")
    else:
        print(df.to_string(index=False))
