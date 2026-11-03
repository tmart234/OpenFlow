"""
Build basin_zones.json: travel-distance precipitation zones per station.

    python openFlowML/data/build_basin_zones.py            # every station in .github/site_ids.txt
    python openFlowML/data/build_basin_zones.py --site USGS:08276500

Per station:
  1. USGS NLDI network navigation (upstream with tributaries) from the gauge
     -> every upstream HUC12 pour point. Closed basins that do not drain to
     the river are not on the network, so they are excluded automatically.
  2. The same navigation limited to zone_km[0] / zone_km[1] river-km splits
     those HUC12s into local / mid / far zones.
  3. For each upstream gauge in upstream_gauges.json (its `usgs_id`), the
     HUC12s upstream of THAT gauge are "gauged"; the rest are "ungauged".
  4. HUC12 polygons from the USGS WBD MapServer are unioned per zone and for
     the ungauged area, cleaned and simplified to ~1 km.
  5. Forecast sample points: a regular grid (~--points per station) inside
     the zones, each weighted by the area it represents, plus the NWS
     gridpoint (WFO/x,y) it falls in, so forecasts need no lookup at run time.

Needs shapely + requests + network. Training and inference only read the
committed JSON (see basin_zones.py), so they don't need shapely.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import date

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from data import basin_zones, get_upstream  # noqa: E402

logger = logging.getLogger(__name__)

NLDI = "https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{usgs}/navigation/UT/huc12pp"
WBD_HUC12 = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/query"
NWS_POINTS = "https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
DWR_STATION = "https://dwr.state.co.us/Rest/GET/api/v2/surfacewater/surfacewaterstations/"
USER_AGENT = "OpenFlow basin zones (github.com/tmart234/OpenFlow)"
ALL_KM = 5000
SIMPLIFY_DEG = 0.01
CLEAN_DEG = 0.004
DEFAULT_POINTS = 36


def _get(url, params=None, retries=4, **kw):
    headers = {'User-Agent': USER_AGENT, **kw.pop('headers', {})}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=180, **kw)
            if r.status_code == 200:
                return r
            logger.warning("HTTP %s from %s", r.status_code, url)
        except requests.RequestException as e:
            logger.warning("%s: %s", url, e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on {url}")


def usgs_id_for(site_id: str) -> str:
    prefix, sid = site_id.split(':', 1)
    if prefix == 'USGS':
        return sid
    if prefix == 'DWR':
        rows = _get(DWR_STATION, {'format': 'json', 'abbrev': sid}).json().get('ResultList', [])
        if rows and rows[0].get('usgsSiteId'):
            return rows[0]['usgsSiteId']
    raise ValueError(f"No USGS id for {site_id}; NLDI needs one")


def upstream_huc12(usgs: str, distance_km: float = ALL_KM) -> set:
    r = _get(NLDI.format(usgs=usgs), {'distance': distance_km})
    return {f['properties']['identifier'] for f in r.json().get('features', [])}


def huc12_polygons(ids, batch=60):
    """{huc12: (shapely geometry, area_km2)} from the WBD MapServer."""
    from shapely.geometry import shape
    out = {}
    ids = sorted(ids)
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        data = {'where': "huc12 IN (%s)" % ",".join(f"'{h}'" for h in chunk),
                'outFields': 'huc12,areasqkm', 'returnGeometry': 'true', 'outSR': 4326,
                'maxAllowableOffset': 0.002, 'geometryPrecision': 5, 'f': 'geojson'}
        for attempt in range(4):
            try:
                r = requests.post(WBD_HUC12, data=data, timeout=300,
                                  headers={'User-Agent': USER_AGENT})
                feats = r.json().get('features')
                if r.status_code == 200 and feats is not None:
                    break
            except (requests.RequestException, ValueError) as e:
                logger.warning("WBD batch %d: %s", i, e)
            time.sleep(2 ** attempt)
        else:
            raise RuntimeError("WBD HUC12 query failed")
        for f in feats:
            props = f['properties']
            out[props['huc12']] = (shape(f['geometry']).buffer(0), float(props['areasqkm']))
        logger.info("WBD: %d/%d HUC12 polygons", len(out), len(ids))
    return out


def _clean(geom):
    """Close slivers between generalized neighbours, then simplify to ~1 km."""
    g = geom.buffer(CLEAN_DEG).buffer(-CLEAN_DEG)
    return g.simplify(SIMPLIFY_DEG, preserve_topology=True)


def rings_of(geom):
    polys = getattr(geom, 'geoms', [geom])
    rings = []
    for p in polys:
        if p.is_empty:
            continue
        for ring in [p.exterior, *p.interiors]:
            rings.append([[round(x, 4), round(y, 4)] for x, y in ring.coords])
    return rings


def assign_zones(all_ids, local_ids, mid_ids):
    """{huc12: zone} from nested distance-limited navigations."""
    return {h: ('local' if h in local_ids else 'mid' if h in mid_ids else 'far') for h in all_ids}


def sample_points(zone_geoms, ungauged_geom, total_area_km2, n_points=DEFAULT_POINTS):
    """
    Regular lon/lat grid inside the zones, ~n_points in total. Each point
    carries its zone, whether it is in the ungauged area, and the area (km2)
    it represents. Every non-empty zone (and the ungauged area) gets at
    least one point at its representative point.
    """
    from shapely.geometry import Point
    from shapely.prepared import prep
    union = None
    for g in zone_geoms.values():
        union = g if union is None else union.union(g)
    if union is None or union.is_empty:
        return []
    minx, miny, maxx, maxy = union.bounds
    lat0 = (miny + maxy) / 2
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat0))
    step_km = max(math.sqrt(total_area_km2 / max(n_points, 1)), 3.0)
    dy, dx = step_km / 110.57, step_km / km_per_deg_lon
    prepared = {z: prep(g) for z, g in zone_geoms.items() if not g.is_empty}
    ung = prep(ungauged_geom) if ungauged_geom is not None and not ungauged_geom.is_empty else None
    pts = []
    y = miny + dy / 2
    while y < maxy:
        x = minx + dx / 2
        while x < maxx:
            p = Point(x, y)
            for z, pg in prepared.items():
                if pg.contains(p):
                    pts.append({'lat': round(y, 4), 'lon': round(x, 4), 'zone': z,
                                'ungauged': bool(ung and ung.contains(p)),
                                'w': round(step_km * step_km, 1)})
                    break
            x += dx
        y += dy
    for z, g in zone_geoms.items():
        if not g.is_empty and not any(p['zone'] == z for p in pts):
            rp = g.representative_point()
            pts.append({'lat': round(rp.y, 4), 'lon': round(rp.x, 4), 'zone': z,
                        'ungauged': bool(ung and ung.contains(rp)), 'w': round(g.area * 111 * 111, 1)})
    if ung is not None and not any(p['ungauged'] for p in pts):
        rp = ungauged_geom.representative_point()
        zone = next((z for z, pg in prepared.items() if pg.contains(rp)), 'local')
        pts.append({'lat': round(rp.y, 4), 'lon': round(rp.x, 4), 'zone': zone, 'ungauged': True,
                    'w': round(ungauged_geom.area * 111 * 111, 1)})
    return pts


def nws_gridpoint(lat, lon):
    try:
        r = _get(NWS_POINTS.format(lat=lat, lon=lon), retries=3,
                 headers={'Accept': 'application/geo+json'})
        p = r.json()['properties']
        return f"{p['gridId']}/{p['gridX']},{p['gridY']}"
    except Exception as e:
        logger.warning("No NWS gridpoint for %.4f,%.4f: %s", lat, lon, e)
        return None


def build_station(site_id, zone_km=basin_zones.DEFAULT_ZONE_KM, n_points=DEFAULT_POINTS,
                  registry=None):
    from shapely.ops import unary_union
    usgs = usgs_id_for(site_id)
    logger.info("%s (USGS %s): navigating NLDI", site_id, usgs)
    all_ids = upstream_huc12(usgs)
    if not all_ids:
        raise RuntimeError(f"NLDI found no upstream HUC12s for {site_id}")
    local_ids = upstream_huc12(usgs, zone_km[0])
    mid_ids = upstream_huc12(usgs, zone_km[1])
    zone_of = assign_zones(all_ids, local_ids, mid_ids)

    cfg = get_upstream.upstream_config(site_id, registry)
    gauges = [g['usgs_id'] for g in (cfg or {}).get('upstream', []) if g.get('usgs_id')]
    gauged = set()
    for g in gauges:
        gauged |= upstream_huc12(g)
    polys = huc12_polygons(all_ids)

    zones, zone_geoms = {}, {}
    for z in basin_zones.ZONES:
        members = [h for h in all_ids if zone_of[h] == z and h in polys]
        geom = _clean(unary_union([polys[h][0] for h in members])) if members else None
        zone_geoms[z] = geom if geom is not None else unary_union([])
        zones[z] = {'area_km2': round(sum(polys[h][1] for h in members), 1),
                    'huc12_count': len(members),
                    'huc8': sorted({h[:8] for h in members}),
                    'rings': rings_of(geom) if geom is not None else []}
    ung_members = [h for h in all_ids if h not in gauged and h in polys]
    ung_geom = _clean(unary_union([polys[h][0] for h in ung_members])) if ung_members else None
    total = sum(z['area_km2'] for z in zones.values())
    points = sample_points(zone_geoms, ung_geom, total, n_points)
    for p in points:
        p['nws'] = nws_gridpoint(p['lat'], p['lon'])
    return {
        'usgs_id': usgs,
        'generated': date.today().isoformat(),
        'source': 'USGS NLDI (UT huc12pp) + WBD HUC12 polygons',
        'zone_km': list(zone_km),
        'contributing_area_km2': round(total, 1),
        'zones': zones,
        'ungauged': {'area_km2': round(sum(polys[h][1] for h in ung_members), 1),
                     'gauges': gauges,
                     'rings': rings_of(ung_geom) if ung_geom is not None else []},
        'points': points,
    }


def main(argv=None):
    import combine_data
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--site', action='append', help='Site id(s); default: all training sites')
    parser.add_argument('--points', type=int, default=DEFAULT_POINTS)
    parser.add_argument('--zone-km', type=float, nargs=2, default=basin_zones.DEFAULT_ZONE_KM)
    parser.add_argument('--out', default=basin_zones.ZONES_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    sites = args.site or combine_data.get_site_ids()
    existing = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
    existing.setdefault('_doc', [
        "Generated by openFlowML/data/build_basin_zones.py -- do not hand-edit polygons/points.",
        "Per station: travel-distance precipitation zones (local/mid/far by river-km, zone_km) over the",
        "NLDI contributing area, the area NOT upstream of the station's registered upstream gauges",
        "('ungauged'), and area-weighted forecast sample points with their NWS gridpoint (WFO/x,y).",
    ])
    for site in sites:
        try:
            existing[site] = build_station(site, tuple(args.zone_km), args.points)
            z = existing[site]
            logger.info("%s: %.0f km2 (local %.0f / mid %.0f / far %.0f; ungauged %.0f), %d points",
                        site, z['contributing_area_km2'], z['zones']['local']['area_km2'],
                        z['zones']['mid']['area_km2'], z['zones']['far']['area_km2'],
                        z['ungauged']['area_km2'], len(z['points']))
        except Exception as e:
            logger.error("Failed to build zones for %s: %s", site, e)
    with open(args.out, 'w') as f:
        json.dump(existing, f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
