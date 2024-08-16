import requests
import argparse
import sys
import logging
'''
TODO: remove? may not need this with get_all_stations.py
'''

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def get_usgs_coordinates(site_number):
    base_url = "https://waterdata.usgs.gov/nwis/inventory"
    params = {
        'search_site_no': site_number,
        'search_site_no_match_type': 'exact',
        'group_key': 'NONE',
        'format': 'sitefile_output',
        'sitefile_output_format': 'rdb',
        'column_name': ['site_no', 'station_nm', 'dec_lat_va', 'dec_long_va'],
        'list_of_search_criteria': 'search_site_no'
    }
    
    response = requests.get(base_url, params=params)
    lines = response.text.splitlines()
    
    try:
         # Filter out comment lines and retrieve relevant data
        data_lines = [line for line in lines if not line.startswith('#')]
        if len(data_lines) < 3:
            return None
        data = next((line for line in data_lines[2:] if line.split('\t')[0] == site_number), None)
        if data is None:
            return None
        fields = data.split('\t')
        return {
            'latitude': fields[2],
            'longitude': fields[3]
        }
    except Exception:
        return None

def get_dwr_coordinates(abbrev):
    base_url = "https://dwr.state.co.us/Rest/GET/api/v2/surfacewater/surfacewaterstations"
    params = {
        "format": "json",
        "dateFormat": "dateOnly",
        "fields": "abbrev,longitude,latitude",
        "encoding": "deflate",
        "abbrev": abbrev,
    }

    response = requests.get(base_url, params=params)
    lines = response.text.splitlines()
    
    try:
        # Filter out comment lines and retrieve relevant data
        data_lines = [line for line in lines if not line.startswith('#')]
        if len(data_lines) < 3:
            return None
        data = data_lines[2]
        fields = data.split('\t')
        return {
            'latitude': fields[1],
            'longitude': fields[2]
        }
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description='Fetch latitude and longitude for a given USGS site number.')
    parser.add_argument('site_type', type=str, help='site type (ex: usgs, dwr)')
    parser.add_argument('site', type=str, help='site number or abbreviation')
    args = parser.parse_args()
    if args.site_type.lower() == 'usgs':
        result = get_usgs_coordinates(args.site)
    elif args.site_type.lower() == 'dwr':
        result = get_dwr_coordinates(args.site)
    else:
        print("Invalid source. Use 'usgs' or 'dwr'.")
        sys.exit(1)
    print(result)

if __name__ == '__main__':
    main()
