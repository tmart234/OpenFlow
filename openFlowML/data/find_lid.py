"""
Helper: find candidate NWS / AHPS LIDs near a given latitude/longitude.

The CBRFC baseline (data/get_cbrfc.py) needs a mapping site_id -> NWS LID.
Mappings are curated by hand in data/cbrfc_lid_map.json because there's no
algorithmic equivalence between a USGS site number and an AHPS LID. This
script queries the NWPS gauge index for gauges within a radius and prints
their LID + name + distance so a maintainer can pick the right one.

Usage:
    python -m data.find_lid --lat 38.52 --lon -106.96
    python -m data.find_lid --site-id USGS:09163500  # auto-resolves coords

Output is a ranked list; the closest gauge by haversine is usually the right
match for a flow-monitoring gauge near the LID's location. Copy the chosen
mapping into data/cbrfc_lid_map.json and re-run train.py to enable CBRFC
backtesting for that site.
"""

import argparse
import logging
import math
import sys
from typing import List, Optional

from data.utils import data_utils

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

# NWPS publishes a gauges index; lat-bbox filter keeps the response small.
NWPS_GAUGES_URL = "https://api.water.noaa.gov/nwps/v1/gauges"


def _haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_nearest_lids(lat: float, lon: float, *,
                      bbox_deg: float = 0.5, top_n: int = 5) -> List[dict]:
    """
    Return up to `top_n` NWS gauges nearest to (lat, lon) within a
    ±bbox_deg lat/lon window. Each entry has {lid, name, lat, lon,
    distance_km, rfc}.
    """
    params = {
        # NWPS supports bbox filtering as a comma-joined "minLon,minLat,maxLon,maxLat".
        'bbox': f"{lon - bbox_deg},{lat - bbox_deg},{lon + bbox_deg},{lat + bbox_deg}",
    }
    response = data_utils.request_with_retry(NWPS_GAUGES_URL, params=params)
    if response is None:
        logger.error("NWPS gauge search returned no response")
        return []
    try:
        payload = response.json()
    except ValueError:
        logger.error("NWPS returned non-JSON")
        return []
    gauges = payload.get('gauges') or payload.get('data') or []
    enriched = []
    for g in gauges:
        try:
            g_lat = float(g.get('latitude') or g.get('lat'))
            g_lon = float(g.get('longitude') or g.get('lon'))
        except (TypeError, ValueError):
            continue
        enriched.append({
            'lid': g.get('lid') or g.get('id'),
            'name': g.get('name') or g.get('siteName', ''),
            'lat': g_lat,
            'lon': g_lon,
            'distance_km': _haversine(lat, lon, g_lat, g_lon),
            'rfc': g.get('rfc') or g.get('forecastCenter', ''),
        })
    enriched.sort(key=lambda x: x['distance_km'])
    return enriched[:top_n]


def _coords_for_site_id(site_id: str) -> Optional[tuple]:
    """Resolve an OpenFlow site_id (USGS:XXX or DWR:XXX) to (lat, lon) via the existing coordinate utilities."""
    from data.utils.get_coordinates import get_usgs_coordinates, get_dwr_coordinates
    prefix, _, sid = site_id.partition(':')
    if prefix == 'USGS':
        coords = get_usgs_coordinates(sid)
    elif prefix == 'DWR':
        coords = get_dwr_coordinates(sid)
    else:
        logger.error("Unsupported prefix in %s", site_id)
        return None
    if not coords:
        return None
    return float(coords['latitude']), float(coords['longitude'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--site-id', help='OpenFlow site_id like USGS:09163500')
    group.add_argument('--latlon', nargs=2, type=float, metavar=('LAT', 'LON'),
                       help='Direct lat lon')
    parser.add_argument('--top-n', type=int, default=5)
    parser.add_argument('--bbox-deg', type=float, default=0.5,
                        help='Search half-window in degrees (default 0.5 ~= 55 km)')
    args = parser.parse_args()

    if args.site_id:
        coords = _coords_for_site_id(args.site_id)
        if not coords:
            logger.error("Could not resolve coordinates for %s", args.site_id)
            return 1
        lat, lon = coords
        print(f"# Site {args.site_id} -> ({lat}, {lon})")
    else:
        lat, lon = args.latlon

    hits = find_nearest_lids(lat, lon, bbox_deg=args.bbox_deg, top_n=args.top_n)
    if not hits:
        print("# No NWS gauges found within the bbox; widen --bbox-deg.")
        return 1
    print(f"# Top {len(hits)} NWS gauges near ({lat:.4f}, {lon:.4f}):")
    print("# LID    distance_km  name                                      RFC")
    for h in hits:
        print(f"  {h['lid']:6s}  {h['distance_km']:>10.2f}  "
              f"{(h['name'] or '')[:40]:40s}  {h['rfc']}")
    if args.site_id:
        top = hits[0]
        print()
        print(f"# To wire the closest match, append to data/cbrfc_lid_map.json:")
        print(f'  "{args.site_id}": "{top["lid"]}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
