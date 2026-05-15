import argparse
import logging
from datetime import datetime, timedelta

import pandas as pd

from data.utils import data_utils

"""
Snow Water Equivalent (SWE) timeseries by HUC, sourced from the USDA NRCS AWDB
REST API.

Given a station's lat/lon we look up the enclosing HUC, find every active
SNOTEL site in that HUC that reports daily WTEQ, fetch their WTEQ timeseries,
and return a single daily SWE series that's the cross-station mean within the
HUC.

This replaces the older hand-curated swe_dicts approach -- AWDB-by-HUC works
for any HUC in the country instead of only a handful of Colorado basins.
"""

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

WBD_QUERY_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/{layer}/query"
AWDB_STATIONS_URL = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/stations"
AWDB_DATA_URL = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"

# WBD MapServer layer ids per HUC level (matches data/utils/get_poly.py).
_HUC_LAYER = {2: 0, 4: 1, 6: 2, 8: 4, 10: 5, 12: 6}


def get_huc_id(lat, lon, level=8):
    """
    Resolve the HUC code containing a point via the WBD ArcGIS MapServer.

    Lightweight (no shapely) so combine_data can call it from the core env.
    """
    if level not in _HUC_LAYER:
        raise ValueError(f"Unsupported HUC level: {level}")
    params = {
        "f": "json",
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": f"huc{level}",
        "inSR": 4326,
        "returnGeometry": "false",
    }
    response = data_utils.request_with_retry(WBD_QUERY_URL.format(layer=_HUC_LAYER[level]), params=params)
    if response is None:
        return None
    try:
        features = response.json().get("features", [])
    except ValueError:
        return None
    if not features:
        return None
    return features[0].get("attributes", {}).get(f"huc{level}")


def get_snotel_station_triplets(huc_id, active_only=True):
    """Return SNOTEL station triplets reporting WTEQ in the given HUC."""
    if not huc_id:
        return []
    params = {"hucs": huc_id, "elements": "WTEQ"}
    if active_only:
        params["activeOnly"] = "true"
    response = data_utils.request_with_retry(AWDB_STATIONS_URL, params=params)
    if response is None:
        return []
    try:
        stations = response.json()
    except ValueError:
        return []
    return [s["stationTriplet"] for s in stations if s.get("stationTriplet")]


def get_swe_timeseries(station_triplets, start_date, end_date):
    """
    Fetch daily WTEQ for each triplet and return one DataFrame [Date, SWE]
    where SWE is the cross-station mean for each day.
    """
    if not station_triplets:
        return pd.DataFrame(columns=['Date', 'SWE'])
    params = {
        "stationTriplets": ",".join(station_triplets),
        "elements": "WTEQ",
        "duration": "DAILY",
        "beginDate": start_date.strftime('%Y-%m-%d'),
        "endDate": end_date.strftime('%Y-%m-%d'),
    }
    response = data_utils.request_with_retry(AWDB_DATA_URL, params=params)
    if response is None:
        return pd.DataFrame(columns=['Date', 'SWE'])
    try:
        payload = response.json()
    except ValueError:
        return pd.DataFrame(columns=['Date', 'SWE'])

    rows = []
    for station in payload or []:
        for element in station.get('data', []) or []:
            for point in element.get('values', []) or []:
                date = point.get('date')
                value = point.get('value')
                if date is None or value is None:
                    continue
                try:
                    rows.append((str(date)[:10], float(value)))
                except (ValueError, TypeError):
                    continue

    if not rows:
        return pd.DataFrame(columns=['Date', 'SWE'])

    df = pd.DataFrame(rows, columns=['Date', 'SWE'])
    daily = df.groupby('Date')['SWE'].mean().reset_index()
    return daily.sort_values('Date').reset_index(drop=True)


def get_swe(lat, lon, start_date, end_date, huc_level=8):
    """
    End-to-end SWE lookup for a point: lat/lon -> HUC -> SNOTEL triplets -> WTEQ.

    Returns a DataFrame with columns [Date, SWE]. Empty if no HUC is found, no
    SNOTEL stations report WTEQ in that HUC, or AWDB returns nothing.
    """
    huc_id = get_huc_id(lat, lon, huc_level)
    if not huc_id:
        logger.warning("Could not resolve HUC%d for (%s, %s)", huc_level, lat, lon)
        return pd.DataFrame(columns=['Date', 'SWE'])
    triplets = get_snotel_station_triplets(huc_id)
    if not triplets:
        logger.warning("No SNOTEL WTEQ stations found in HUC %s", huc_id)
        return pd.DataFrame(columns=['Date', 'SWE'])
    logger.info("HUC %s: averaging WTEQ across %d SNOTEL stations", huc_id, len(triplets))
    return get_swe_timeseries(triplets, start_date, end_date)


def main(lat, lon, start_date=None, end_date=None, huc_level=8):
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    if end_date is None:
        end_date = datetime.now()
    df = get_swe(lat, lon, start_date, end_date, huc_level)
    data_utils.preview_data(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch HUC-aggregated SWE timeseries from NRCS AWDB.')
    parser.add_argument('--lat', type=float, required=True)
    parser.add_argument('--lon', type=float, required=True)
    parser.add_argument('--start_date', type=str, default=None, help='YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default=None, help='YYYY-MM-DD')
    parser.add_argument('--huc_level', type=int, default=8, choices=[2, 4, 6, 8, 10, 12])
    args = parser.parse_args()
    start = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
    main(args.lat, args.lon, start, end, args.huc_level)
