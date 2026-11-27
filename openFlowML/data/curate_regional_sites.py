"""
Curated Mountain West hydrological gauge discovery & vetting (Phase 5).

Queries USGS NWIS Site Service across Mountain West states (CO, WY, NM, UT, ID, MT)
and applies deterministic hydrological quality filters:
  1. Active stream gauge (`siteType=ST`) reporting daily mean discharge (`00060`, `stat_cd=00003`)
  2. Continuous record length >= `min_years` (default 15 years) and recent reporting
     (within `max_stale_days`, default 60 days)
  3. Record completeness >= `min_completeness` (default 0.90, i.e. <10% missing days)
  4. Drainage area within `[min_area_km2, max_area_km2]` (default 25..50,000 km^2)
  5. Spatial / hydrological diversity cap (`max_per_huc8`, default 2, and per-state
     balance) so the training pool spans snowmelt, monsoon, and mixed regimes instead
     of stacking gauges on one reach.

Always preserves the core production gauges from `.github/site_ids.txt` at the top,
emits `.github/site_ids_regional.txt`, and optionally enriches `station_attributes.json`
via EPA StreamCat so static attributes & specific discharge work out of the box.

CLI usage:
    python openFlowML/data/curate_regional_sites.py
    python openFlowML/data/curate_regional_sites.py --states CO,WY,NM,UT,ID,MT --max-total 80 --with-attributes
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from data import build_station_attributes as bsa
from data.utils import data_utils

logger = logging.getLogger(__name__)

NWIS_SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
SQMI_TO_KM2 = 2.58998811
DEFAULT_STATES = ('CO', 'WY', 'NM', 'UT', 'ID', 'MT')


def parse_rdb(text: str) -> List[Dict[str, str]]:
    """Parse USGS NWIS tab-delimited RDB text into a list of row dicts."""
    lines = [ln for ln in (text or '').splitlines() if ln and not ln.startswith('#')]
    if len(lines) < 3:
        return []
    header = lines[0].split('\t')
    # lines[1] is the column width/type spec row (e.g. '5s\t15s\t...').
    rows: List[Dict[str, str]] = []
    for ln in lines[2:]:
        parts = ln.split('\t')
        if len(parts) < len(header):
            parts.extend([''] * (len(header) - len(parts)))
        rows.append(dict(zip(header, parts)))
    return rows


def _to_float(val) -> Optional[float]:
    try:
        v = float(val)
        return v if pd.notna(v) else None
    except (TypeError, ValueError):
        return None


def _to_int(val) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return 0


def fetch_state_candidates(state: str, timeout: int = 60) -> List[dict]:
    """
    Fetch active USGS stream gauges in `state` with both expanded drainage-area
    metadata and daily discharge (00060 / stat 00003) series catalog counts.
    """
    st = state.strip().upper()
    exp_resp = data_utils.request_with_retry(NWIS_SITE_URL, params={
        'format': 'rdb',
        'stateCd': st,
        'siteType': 'ST',
        'parameterCd': '00060',
        'siteStatus': 'active',
        'hasDataTypeCd': 'dv',
        'siteOutput': 'expanded',
    }, timeout=timeout)
    cat_resp = data_utils.request_with_retry(NWIS_SITE_URL, params={
        'format': 'rdb',
        'stateCd': st,
        'siteType': 'ST',
        'parameterCd': '00060',
        'siteStatus': 'active',
        'seriesCatalogOutput': 'true',
        'outputDataTypeCd': 'dv',
    }, timeout=timeout)
    if exp_resp is None or cat_resp is None:
        logger.warning("USGS NWIS site query failed for state %s", st)
        return []

    expanded: Dict[str, dict] = {}
    for row in parse_rdb(exp_resp.text):
        sno = (row.get('site_no') or '').strip()
        if sno:
            expanded[sno] = row

    # Keep the longest daily-mean discharge (00060, stat_cd 00003) series per site.
    best_series: Dict[str, dict] = {}
    for row in parse_rdb(cat_resp.text):
        if (row.get('parm_cd') or '').strip() != '00060':
            continue
        stat = (row.get('stat_cd') or '').strip()
        if stat and stat != '00003':
            continue
        sno = (row.get('site_no') or '').strip()
        if not sno:
            continue
        cnt = _to_int(row.get('count_nu'))
        prev = best_series.get(sno)
        if prev is None or cnt > _to_int(prev.get('count_nu')):
            best_series[sno] = row

    out: List[dict] = []
    for sno, srow in best_series.items():
        erow = expanded.get(sno, {})
        da_sqmi = _to_float(erow.get('contrib_drain_area_va')) or _to_float(erow.get('drain_area_va'))
        area_km2 = round(da_sqmi * SQMI_TO_KM2, 2) if da_sqmi and da_sqmi > 0 else None
        huc_raw = (erow.get('huc_cd') or srow.get('huc_cd') or '').strip()
        huc8 = huc_raw[:8] if len(huc_raw) >= 8 else huc_raw
        out.append({
            'site_id': f'USGS:{sno}',
            'site_no': sno,
            'state': st,
            'name': (erow.get('station_nm') or srow.get('station_nm') or '').strip(),
            'huc8': huc8,
            'area_km2': area_km2,
            'begin_date': (srow.get('begin_date') or '').strip(),
            'end_date': (srow.get('end_date') or '').strip(),
            'count_nu': _to_int(srow.get('count_nu')),
        })
    logger.info("State %s: fetched %d active USGS daily discharge candidates", st, len(out))
    return out


def filter_candidates(candidates: Sequence[dict], *,
                      min_years: float = 15.0,
                      min_completeness: float = 0.90,
                      min_area_km2: float = 25.0,
                      max_area_km2: float = 50000.0,
                      max_stale_days: int = 60,
                      max_per_huc8: int = 2,
                      max_per_state: Optional[int] = 15,
                      max_total: Optional[int] = 80,
                      exclude_site_ids: Optional[Iterable[str]] = None,
                      as_of: Optional[pd.Timestamp] = None) -> List[dict]:
    """
    Deterministically vet and diversify candidate gauges. Pure helper (no network).

    Scores each passing gauge by `(record_years, completeness)` so longer, cleaner
    records win within each HUC8 and state quota.
    """
    ref_day = (pd.Timestamp(as_of).normalize()
               if as_of is not None
               else pd.Timestamp(datetime.now(timezone.utc).date()))
    excluded = set(exclude_site_ids or ())
    vetted: List[dict] = []
    seen: set = set()

    for c in candidates:
        sid = c.get('site_id')
        if not sid or sid in excluded or sid in seen:
            continue
        seen.add(sid)

        area = c.get('area_km2')
        if area is None or area < min_area_km2 or area > max_area_km2:
            continue

        try:
            bdt = pd.Timestamp(c['begin_date']).normalize()
            edt = pd.Timestamp(c['end_date']).normalize()
        except Exception:
            continue
        if pd.isna(bdt) or pd.isna(edt) or edt <= bdt:
            continue
        if (ref_day - edt).days > max_stale_days:
            continue

        span_days = int((edt - bdt).days) + 1
        years = span_days / 365.25
        if years < min_years:
            continue
        count = int(c.get('count_nu') or 0)
        completeness = min(count / float(max(span_days, 1)), 1.0)
        if completeness < min_completeness:
            continue
        # Skip obvious canal/ditch/diversion names even when typed ST in NWIS.
        nm_lower = str(c.get('name') or '').lower()
        if any(bad in nm_lower for bad in ('canal', 'ditch', 'diversion', 'wasteway', 'drain ', 'tunnel')):
            continue

        enriched = dict(c)
        enriched['record_years'] = round(years, 1)
        enriched['completeness'] = round(completeness, 4)
        vetted.append(enriched)

    # Prefer long, complete records; break ties deterministically by site_id.
    vetted.sort(key=lambda r: (-r['record_years'], -r['completeness'], r['site_id']))

    # Apply HUC8 cap and state quota using round-robin across states for balanced
    # geographic representation when `max_total` is set.
    by_state: Dict[str, List[dict]] = {}
    for row in vetted:
        by_state.setdefault(row.get('state') or 'UNK', []).append(row)

    selected: List[dict] = []
    huc_counts: Dict[str, int] = {}
    state_counts: Dict[str, int] = {}
    states_order = sorted(by_state.keys())
    cursors = {st: 0 for st in states_order}

    while True:
        progressed = False
        for st in states_order:
            if max_total is not None and len(selected) >= max_total:
                return selected
            if max_per_state is not None and state_counts.get(st, 0) >= max_per_state:
                continue
            pool = by_state[st]
            while cursors[st] < len(pool):
                cand = pool[cursors[st]]
                cursors[st] += 1
                huc = cand.get('huc8') or ''
                if huc and max_per_huc8 > 0 and huc_counts.get(huc, 0) >= max_per_huc8:
                    continue
                selected.append(cand)
                if huc:
                    huc_counts[huc] = huc_counts.get(huc, 0) + 1
                state_counts[st] = state_counts.get(st, 0) + 1
                progressed = True
                break
        if not progressed:
            break
    return selected


def write_site_list(core_sites: Sequence[str], regional_sites: Sequence[dict],
                    out_path: str) -> None:
    """Write `.github/site_ids_regional.txt` with core gauges preserved first."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    lines = [
        "# Curated Mountain West regional training site list (Phase 5).",
        f"# Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d')} by openFlowML/data/curate_regional_sites.py.",
        "# Core production stations from .github/site_ids.txt are listed first,",
        "# followed by vetted CO/WY/NM/UT/ID/MT gauges (>=15 yrs, >=90% complete, 25-50,000 km^2).",
        "# Select via OPENFLOW_SITE_IDS_FILE=.github/site_ids_regional.txt or workflow dispatch.",
        "",
        "# --- Core production stations ---",
    ]
    seen = set()
    for sid in core_sites:
        sid = sid.strip()
        if sid and not sid.startswith('#') and sid not in seen:
            lines.append(sid)
            seen.add(sid)

    by_state: Dict[str, List[dict]] = {}
    for row in regional_sites:
        sid = row['site_id']
        if sid in seen:
            continue
        seen.add(sid)
        by_state.setdefault(row.get('state') or 'OTHER', []).append(row)

    for st in sorted(by_state.keys()):
        lines.append("")
        lines.append(f"# --- {st} ({len(by_state[st])} vetted gauges) ---")
        for row in by_state[st]:
            lines.append(row['site_id'])
    lines.append("")
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--states', default=','.join(DEFAULT_STATES),
                        help='Comma-separated state codes (default: CO,WY,NM,UT,ID,MT)')
    parser.add_argument('--min-years', type=float, default=15.0)
    parser.add_argument('--min-completeness', type=float, default=0.90)
    parser.add_argument('--min-area-km2', type=float, default=25.0)
    parser.add_argument('--max-area-km2', type=float, default=50000.0)
    parser.add_argument('--max-per-huc8', type=int, default=2)
    parser.add_argument('--max-per-state', type=int, default=12)
    parser.add_argument('--max-total', type=int, default=72,
                        help='Max additional regional gauges beyond core .github/site_ids.txt')
    parser.add_argument('--out', default=None,
                        help='Output site list path (default: .github/site_ids_regional.txt)')
    parser.add_argument('--with-attributes', action='store_true',
                        help='Also fetch EPA StreamCat static attributes into station_attributes.json')
    args = parser.parse_args(argv)

    repo_root = os.path.dirname(_PKG_ROOT)
    core_path = os.path.join(repo_root, '.github', 'site_ids.txt')
    out_path = args.out or os.path.join(repo_root, '.github', 'site_ids_regional.txt')
    core_sites = bsa._read_site_list(core_path) if os.path.exists(core_path) else []

    states = [s.strip().upper() for s in args.states.split(',') if s.strip()]
    all_candidates: List[dict] = []
    for st in states:
        all_candidates.extend(fetch_state_candidates(st))

    selected = filter_candidates(
        all_candidates,
        min_years=args.min_years,
        min_completeness=args.min_completeness,
        min_area_km2=args.min_area_km2,
        max_area_km2=args.max_area_km2,
        max_per_huc8=args.max_per_huc8,
        max_per_state=args.max_per_state,
        max_total=args.max_total,
        exclude_site_ids=core_sites,
    )
    write_site_list(core_sites, selected, out_path)
    logger.info("Wrote %d core + %d curated regional gauges (%d total) -> %s",
                len(core_sites), len(selected), len(core_sites) + len(selected), out_path)

    if args.with_attributes:
        all_sites = list(core_sites) + [r['site_id'] for r in selected]
        area_map = {r['site_id']: r['area_km2'] for r in selected if r.get('area_km2')}
        huc_map = {r['site_id']: r['huc8'] for r in selected if r.get('huc8')}
        name_map = {r['site_id']: r['name'] for r in selected if r.get('name')}
        existing = bsa.load_station_attributes()
        catalog = bsa.build_for_sites(
            all_sites, existing=existing,
            area_overrides=area_map, huc8_map=huc_map, name_map=name_map)
        with open(bsa.ATTRIBUTES_PATH, 'w') as f:
            json.dump(catalog, f, indent=2)
        n_attrs = sum(1 for k in catalog if not k.startswith('_'))
        logger.info("Updated %s with %d total station attribute records", bsa.ATTRIBUTES_PATH, n_attrs)
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    raise SystemExit(main())
