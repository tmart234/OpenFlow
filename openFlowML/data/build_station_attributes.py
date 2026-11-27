"""
Static physical catchment attributes from USGS NLDI + EPA StreamCat (Phase 5).

Queries EPA's zero-auth StreamCat REST API (`https://api.epa.gov/StreamCat/streams/metrics`)
using each station's NHDPlus COMID / NWM reachId (resolved via `get_nwm_prior.resolve_reach_id`)
and persists a compact, checked-in `station_attributes.json` lookup table.

Shipping `station_attributes.json` inside model releases means:
  - ZERO runtime API calls during training or mobile inference
  - Unseen rivers (e.g. Clear Creek or newly added WY/NM/UT/ID/MT gauges) receive
    rich physical conditioning (watershed area, elevation, relief, climate normals,
    aridity/water balance, base-flow index, soil permeability/texture, bedrock depth,
    forest/shrub cover, wetness index, dam storage regulation, ungauged fraction)
    instead of an empty `station_idx = 0` vector
  - Exact watershed area (`area_km2`) supports optional specific discharge (`mm/day`)
    normalization across rivers spanning 3+ orders of magnitude in drainage area.

CLI usage:
    python openFlowML/data/build_station_attributes.py
    python openFlowML/data/build_station_attributes.py --site-ids .github/site_ids_regional.txt
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# Support both `from data import build_station_attributes` and direct script execution.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from data import basin_zones, get_nwm_prior, get_upstream
from data.utils import data_utils

logger = logging.getLogger(__name__)

ATTRIBUTES_PATH = os.path.join(_HERE, 'station_attributes.json')
STREAMCAT_METRICS_URL = "https://api.epa.gov/StreamCat/streams/metrics"
STREAMCAT_BATCH_SIZE = 40

# StreamCat metric names requested from the API (`aoi=ws`, `showareasqkm=true`).
STREAMCAT_METRIC_NAMES = (
    'elev',
    'precip9120',
    'precip8110',
    'tmean9120',
    'tmean8110',
    'precip_minus_evt',
    'runoff',
    'bfi',
    'perm',
    'clay',
    'sand',
    'rckdep',
    'pctconif2019',
    'pctdecid2019',
    'pctmxfst2019',
    'pctshrb2019',
    'wetindex',
    'damdens',
    'damnidstor',
)

# Fixed-order 18-feature static attribute vector fed to `static_input` in Phase 5 models.
# Every entry is numeric, finite, and physically meaningful across Western US basins.
STATIC_FEATURE_NAMES: List[str] = [
    'log_area_km2',
    'elev_mean_m',
    'elev_cat_m',
    'elev_relief_m',
    'precip_mean_mm',
    'tmean_c',
    'water_balance',
    'runoff_mean_mm',
    'bfi_pct',
    'soil_perm',
    'clay_pct',
    'sand_pct',
    'rock_depth_cm',
    'forest_pct',
    'shrub_pct',
    'wetness_index',
    'log_dam_storage',
    'ungauged_frac',
]


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else float(default)
    except (TypeError, ValueError):
        return float(default)


def fetch_streamcat_metrics(comids: Sequence[int], timeout: int = 45) -> Dict[int, dict]:
    """
    Batch-query EPA StreamCat watershed metrics for a sequence of NHDPlus COMIDs.
    Returns {comid: raw_item_dict}. Zero auth required.
    """
    unique = sorted({int(c) for c in comids if c is not None and int(c) > 0})
    out: Dict[int, dict] = {}
    if not unique:
        return out
    metric_param = ','.join(STREAMCAT_METRIC_NAMES)
    for i in range(0, len(unique), STREAMCAT_BATCH_SIZE):
        chunk = unique[i:i + STREAMCAT_BATCH_SIZE]
        params = {
            'comid': ','.join(str(c) for c in chunk),
            'aoi': 'cat,ws',
            'showareasqkm': 'true',
            'name': metric_param,
        }
        resp = data_utils.request_with_retry(STREAMCAT_METRICS_URL, params=params, timeout=timeout)
        if resp is None:
            logger.warning("EPA StreamCat request failed for batch of %d COMIDs", len(chunk))
            continue
        try:
            payload = resp.json()
        except ValueError:
            logger.warning("EPA StreamCat returned invalid JSON for batch of %d COMIDs", len(chunk))
            continue
        items = payload.get('items') if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or item.get('comid') is None:
                continue
            try:
                cid = int(item['comid'])
            except (TypeError, ValueError):
                continue
            # Normalize keys to lowercase for resilient parsing across API revisions.
            out[cid] = {str(k).lower(): v for k, v in item.items()}
    return out


def compute_ungauged_frac(site_id: str, zones: Optional[dict] = None,
                          registry: Optional[dict] = None) -> float:
    """
    Fraction of contributing area NOT upstream of registered upstream gauges.
    Reads `basin_zones.json` when available; defaults to 1.0 when a station has
    no registered upstream gauges, or 0.5 when upstream gauges exist without a
    delineated ungauged polygon.
    """
    z_entry = (zones or {}).get(site_id)
    if isinstance(z_entry, dict):
        total = _safe_float(z_entry.get('contributing_area_km2'), 0.0)
        ungauged = z_entry.get('ungauged')
        if total > 0.0 and isinstance(ungauged, dict) and ungauged.get('area_km2') is not None:
            return float(np.clip(_safe_float(ungauged.get('area_km2'), total) / total, 0.0, 1.0))
    reg_entry = (registry or {}).get(site_id)
    if isinstance(reg_entry, dict) and reg_entry.get('upstream'):
        return 0.5
    return 1.0


def extract_station_record(site_id: str, comid: int, sc_row: dict, *,
                           ungauged_frac: float = 1.0,
                           area_override_km2: Optional[float] = None,
                           huc8: Optional[str] = None,
                           name: Optional[str] = None) -> dict:
    """Parse one StreamCat row into OpenFlow's canonical station attribute record."""
    row = {str(k).lower(): v for k, v in (sc_row or {}).items()}
    area_km2 = _safe_float(row.get('wsareasqkm'), 0.0)
    if area_km2 <= 0.0 and area_override_km2:
        area_km2 = max(float(area_override_km2), 0.0)
    elev_ws = _safe_float(row.get('elevws'), 0.0)
    elev_cat = _safe_float(row.get('elevcat'), elev_ws)
    elev_relief = max(elev_ws - elev_cat, 0.0)
    precip = _safe_float(row.get('precip9120ws'), np.nan)
    if not np.isfinite(precip) or precip <= 0.0:
        precip = _safe_float(row.get('precip8110ws'), 0.0)
    tmean = _safe_float(row.get('tmean9120ws'), np.nan)
    if not np.isfinite(tmean):
        tmean = _safe_float(row.get('tmean8110ws'), 0.0)
    forest = (
        _safe_float(row.get('pctconif2019ws'), 0.0)
        + _safe_float(row.get('pctdecid2019ws'), 0.0)
        + _safe_float(row.get('pctmxfst2019ws'), 0.0)
    )
    dam_stor = max(_safe_float(row.get('damnidstorws'), 0.0), 0.0)
    dam_dens = max(_safe_float(row.get('damdensws'), 0.0), 0.0)

    features = {
        'log_area_km2': round(float(np.log1p(max(area_km2, 0.0))), 6),
        'elev_mean_m': round(elev_ws, 3),
        'elev_cat_m': round(elev_cat, 3),
        'elev_relief_m': round(elev_relief, 3),
        'precip_mean_mm': round(precip, 3),
        'tmean_c': round(tmean, 4),
        'water_balance': round(_safe_float(row.get('precip_minus_evtws'), 0.0), 4),
        'runoff_mean_mm': round(_safe_float(row.get('runoffws'), 0.0), 3),
        'bfi_pct': round(_safe_float(row.get('bfiws'), 0.0), 3),
        'soil_perm': round(_safe_float(row.get('permws'), 0.0), 4),
        'clay_pct': round(_safe_float(row.get('clayws'), 0.0), 3),
        'sand_pct': round(_safe_float(row.get('sandws'), 0.0), 3),
        'rock_depth_cm': round(_safe_float(row.get('rckdepws'), 0.0), 3),
        'forest_pct': round(float(np.clip(forest, 0.0, 100.0)), 3),
        'shrub_pct': round(float(np.clip(_safe_float(row.get('pctshrb2019ws'), 0.0), 0.0, 100.0)), 3),
        'wetness_index': round(_safe_float(row.get('wetindexws'), 0.0), 3),
        'log_dam_storage': round(float(np.log1p(dam_stor)), 6),
        'ungauged_frac': round(float(np.clip(ungauged_frac, 0.0, 1.0)), 4),
    }
    rec = {
        'site_id': site_id,
        'comid': int(comid),
        'area_km2': round(area_km2, 3),
        'dam_density': round(dam_dens, 6),
        'dam_storage_m3_per_km2': round(dam_stor, 3),
        'regulated': bool(dam_stor >= 10000.0),
        'features': features,
        'source': 'EPA StreamCat (NHDPlusV2)',
        'generated': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
    }
    if huc8:
        rec['huc8'] = str(huc8)
    if name:
        rec['name'] = str(name)
    return rec


def build_for_sites(site_ids: Iterable[str], *,
                    registry: Optional[dict] = None,
                    zones: Optional[dict] = None,
                    existing: Optional[dict] = None,
                    force: bool = False,
                    comid_overrides: Optional[Dict[str, int]] = None,
                    area_overrides: Optional[Dict[str, float]] = None,
                    huc8_map: Optional[Dict[str, str]] = None,
                    name_map: Optional[Dict[str, str]] = None) -> dict:
    """
    Build or update the station attributes catalog for `site_ids`.
    Preserves existing entries unless `force=True` or a site is missing.
    """
    registry = registry if registry is not None else get_upstream.load_registry()
    if zones is None and os.path.exists(basin_zones.ZONES_PATH):
        with open(basin_zones.ZONES_PATH) as f:
            zones = json.load(f)
    zones = zones or {}
    comid_overrides = comid_overrides or {}
    area_overrides = area_overrides or {}
    huc8_map = huc8_map or {}
    name_map = name_map or {}

    catalog = {
        '_doc': [
            "Static physical catchment attributes built by openFlowML/data/build_station_attributes.py.",
            "Source: USGS NLDI (NHDPlusV2 COMID) + EPA StreamCat watershed metrics (zero runtime auth).",
            "Consumed by Phase 5 static_input MLP, attribute-gated upstream skip, and specific discharge (mm/day).",
        ],
        '_feature_names': list(STATIC_FEATURE_NAMES),
    }
    if existing:
        for k, v in existing.items():
            if not k.startswith('_') and isinstance(v, dict):
                catalog[k] = v

    to_fetch: Dict[str, int] = {}
    for sid in site_ids:
        sid = sid.strip()
        if not sid or sid.startswith('#'):
            continue
        if not force and sid in catalog and isinstance(catalog[sid].get('features'), dict):
            if all(f in catalog[sid]['features'] for f in STATIC_FEATURE_NAMES):
                continue
        cid = comid_overrides.get(sid)
        if cid is None:
            reg_entry = registry.get(sid) if isinstance(registry, dict) else None
            cid = get_nwm_prior.resolve_reach_id(
                sid,
                nws_lid=(reg_entry or {}).get('nws_lid'),
                usgs_id=(reg_entry or {}).get('usgs_id'),
            )
        if cid is None:
            logger.warning("Could not resolve COMID for %s; skipping static attributes", sid)
            continue
        to_fetch[sid] = int(cid)

    if to_fetch:
        sc_rows = fetch_streamcat_metrics(list(to_fetch.values()))
        for sid, cid in to_fetch.items():
            row = sc_rows.get(cid)
            if not row:
                logger.warning("No EPA StreamCat row returned for %s (comid=%d)", sid, cid)
                continue
            reg_entry = registry.get(sid) if isinstance(registry, dict) else None
            uf = compute_ungauged_frac(sid, zones=zones, registry=registry)
            area_fb = area_overrides.get(sid)
            if area_fb is None and isinstance(zones.get(sid), dict):
                area_fb = zones[sid].get('contributing_area_km2')
            catalog[sid] = extract_station_record(
                sid, cid, row,
                ungauged_frac=uf,
                area_override_km2=area_fb,
                huc8=huc8_map.get(sid),
                name=name_map.get(sid) or (reg_entry or {}).get('short_name') or (reg_entry or {}).get('name'),
            )
            logger.info("Built attributes for %s (comid=%d, area=%.1f km^2, elev=%.0f m)",
                        sid, cid, catalog[sid]['area_km2'], catalog[sid]['features']['elev_mean_m'])

    return catalog


def load_station_attributes(path: Optional[str] = None) -> dict:
    """Load `station_attributes.json` (returning `{}` if absent)."""
    target = path or ATTRIBUTES_PATH
    if not os.path.exists(target):
        return {}
    with open(target) as f:
        return json.load(f)


def station_area_km2(site_id: str, attrs: Optional[dict]) -> Optional[float]:
    """Watershed drainage area in km^2 for `site_id`, or None if unavailable."""
    entry = (attrs or {}).get(site_id)
    if not isinstance(entry, dict):
        return None
    area = _safe_float(entry.get('area_km2'), 0.0)
    return area if area > 0.0 else None


def station_areas_map(site_ids: Iterable[str], attrs: Optional[dict]) -> Dict[str, float]:
    """Map of {site_id: area_km2} for every site with positive known drainage area."""
    out: Dict[str, float] = {}
    for sid in site_ids:
        area = station_area_km2(sid, attrs)
        if area is not None:
            out[sid] = area
    return out


def fit_static_scalers(attrs: dict, site_ids: Sequence[str],
                       feature_names: Sequence[str] = STATIC_FEATURE_NAMES) -> Dict[str, dict]:
    """
    Fit z-score scalers ({feature: {'mean': float, 'scale': float}}) strictly on
    `site_ids` present in `attrs`. Constant features keep `scale = 1.0`.
    """
    rows = []
    for sid in sorted(set(site_ids)):
        entry = (attrs or {}).get(sid)
        if not isinstance(entry, dict) or not isinstance(entry.get('features'), dict):
            continue
        feats = entry['features']
        rows.append([_safe_float(feats.get(f), np.nan) for f in feature_names])
    if not rows:
        return {f: {'mean': 0.0, 'scale': 1.0} for f in feature_names}
    mat = np.asarray(rows, dtype='float64')
    scalers: Dict[str, dict] = {}
    for j, name in enumerate(feature_names):
        col = mat[:, j]
        valid = col[np.isfinite(col)]
        if len(valid) == 0:
            scalers[name] = {'mean': 0.0, 'scale': 1.0}
        else:
            mu = float(np.mean(valid))
            sd = float(np.std(valid))
            scalers[name] = {'mean': mu, 'scale': sd if sd > 1e-8 else 1.0}
    return scalers


def station_static_vector(site_id: str, attrs: Optional[dict],
                          static_scalers: Optional[dict],
                          feature_names: Sequence[str] = STATIC_FEATURE_NAMES) -> np.ndarray:
    """
    Build the scaled float32 `(S,)` static attribute vector for `site_id`.
    Sites present in `attrs` get their exact scaled physical vector even when
    unseen in training (`station_idx = 0`). Unknown sites fall back cleanly to
    zeros (the training-set mean in scaled space).
    """
    out = np.zeros((len(feature_names),), dtype='float32')
    entry = (attrs or {}).get(site_id)
    if not isinstance(entry, dict) or not isinstance(entry.get('features'), dict):
        return out
    feats = entry['features']
    sc = static_scalers or {}
    for j, name in enumerate(feature_names):
        raw = _safe_float(feats.get(name), np.nan)
        if not np.isfinite(raw):
            out[j] = 0.0
            continue
        p = sc.get(name) or {'mean': 0.0, 'scale': 1.0}
        scale = float(p.get('scale', 1.0))
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        mean = float(p.get('mean', 0.0))
        out[j] = np.float32((raw - mean) / scale)
    return out


def build_static_vectors(site_ids: Iterable[str], attrs: Optional[dict],
                         static_scalers: Optional[dict],
                         feature_names: Sequence[str] = STATIC_FEATURE_NAMES) -> Dict[str, np.ndarray]:
    """Return `{site_id: float32 (S,)}` for every site in `site_ids`."""
    return {
        sid: station_static_vector(sid, attrs, static_scalers, feature_names)
        for sid in sorted(set(site_ids))
    }


def _read_site_list(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--site-ids', default=None,
                        help='Path to site_ids file (defaults to .github/site_ids.txt)')
    parser.add_argument('--out', default=ATTRIBUTES_PATH,
                        help='Output JSON path (default: openFlowML/data/station_attributes.json)')
    parser.add_argument('--force', action='store_true',
                        help='Re-fetch StreamCat metrics even for sites already cached')
    args = parser.parse_args(argv)

    repo_root = os.path.dirname(_PKG_ROOT)
    site_file = args.site_ids or os.path.join(repo_root, '.github', 'site_ids.txt')
    sites = _read_site_list(site_file)
    existing = load_station_attributes(args.out) if os.path.exists(args.out) else None
    catalog = build_for_sites(sites, existing=existing, force=args.force)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(catalog, f, indent=2, sort_keys=False)
    n_sites = sum(1 for k in catalog if not k.startswith('_'))
    logger.info("Wrote %d station attribute records -> %s", n_sites, args.out)
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    raise SystemExit(main())
