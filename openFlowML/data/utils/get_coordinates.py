import requests
import argparse
import sys
import logging

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def get_usgs_coordinates(site_number):
    """
    Look up the decimal lat/lon for a USGS site via the NWIS Site Service.

    The legacy waterdata.usgs.gov/nwis/inventory endpoint was decommissioned;
    waterservices.usgs.gov/nwis/site/ is the current RDB endpoint. The RDB
    header row is parsed to locate columns by name rather than fixed index.
    """
    url = "https://waterservices.usgs.gov/nwis/site/"
    params = {"format": "rdb", "sites": site_number}
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return None
        # Drop comment (#) and blank lines: what remains is the header row,
        # the RDB format-spec row, then one or more data rows.
        lines = [ln for ln in response.text.splitlines() if ln and not ln.startswith('#')]
        if len(lines) < 3:
            return None
        header = lines[0].split('\t')
        try:
            lat_idx = header.index('dec_lat_va')
            lon_idx = header.index('dec_long_va')
            site_idx = header.index('site_no')
        except ValueError:
            logger.error("Unexpected RDB header from NWIS Site Service")
            return None
        for row in lines[2:]:
            fields = row.split('\t')
            if len(fields) <= max(lat_idx, lon_idx, site_idx):
                continue
            if fields[site_idx] == site_number and fields[lat_idx] and fields[lon_idx]:
                return {'latitude': fields[lat_idx], 'longitude': fields[lon_idx]}
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching USGS coordinates for {site_number}: {e}")
        return None

def get_dwr_coordinates(abbrev):
    """
    Look up the lat/lon for a Colorado DWR surface water station by abbreviation.

    Uses the DWR REST API's JSON response (the previous implementation requested
    JSON but parsed it as tab-delimited text, so it always returned None).
    """
    url = "https://dwr.state.co.us/Rest/GET/api/v2/surfacewater/surfacewaterstations/"
    params = {"format": "json", "abbrev": abbrev, "fields": "abbrev,latitude,longitude"}
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return None
        results = response.json().get('ResultList', [])
        if not results:
            return None
        station = results[0]
        latitude = station.get('latitude')
        longitude = station.get('longitude')
        if latitude is None or longitude is None:
            return None
        return {'latitude': latitude, 'longitude': longitude}
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.error(f"Error fetching DWR coordinates for {abbrev}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Fetch latitude and longitude for a given USGS site number.')
    parser.add_argument('site_type', type=str, help='site type (ex: usgs, dwr)')
    parser.add_argument('site_number', type=str, help='site number or abbreviation')
    args = parser.parse_args()
    if args.site_type.lower() == 'usgs':
        result = get_usgs_coordinates(args.site_number)
    elif args.site_type.lower() == 'dwr':
        result = get_dwr_coordinates(args.site_number)
    else:
        print("Invalid source. Use 'usgs' or 'dwr'.")
        sys.exit(1)
    print(result)

if __name__ == '__main__':
    main()
