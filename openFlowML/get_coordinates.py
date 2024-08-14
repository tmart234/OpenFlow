import requests
import argparse

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
    
    # Filter out comment lines and retrieve relevant data
    data_lines = [line for line in lines if not line.startswith('#')]
    data = data_lines[2]  # Since we're skipping the first two lines after comments
    
    fields = data.split('\t')
    site_no, station_nm, dec_lat_va, dec_long_va = fields[0], fields[1], fields[2], fields[3]
    
    return {
        'latitude': dec_lat_va,
        'longitude': dec_long_va
    }

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
    
    # Filter out comment lines and retrieve relevant data
    data_lines = [line for line in lines if not line.startswith('#')]
    data = data_lines[2]  # Since we're skipping the first two lines after comments
    
    fields = data.split('\t')
    station_nm, dec_lat_va, dec_long_va = fields[0], fields[1], fields[2]
    
    return {
        'latitude': dec_lat_va,
        'longitude': dec_long_va
    }

def main():
    parser = argparse.ArgumentParser(description='Fetch latitude and longitude for a given USGS site number.')
    parser.add_argument('site_type', type=str, help='site type (ex: usgs, dwr)')
    parser.add_argument('site_number', type=str, help='site number')
    args = parser.parse_args()
    if args.site_type.lowercase == 'usgs':
        result = get_usgs_coordinates(args.site_type, args.site_number)
    elif args.site_type.lowercase == 'dwr':
        result = get_dwr_coordinates(args.site_type, args.site_number)
    print(result)

if __name__ == '__main__':
    main()
