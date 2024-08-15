import json
import requests
import logging

# Check if the root logger already has handlers (configured in another module)
if not logging.getLogger().hasHandlers():
    # If not, set up basic logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get logger for this module
logger = logging.getLogger(__name__)

def fetch_and_parse_station_data(url, data_source):
    # Fetch the JSON data from the URL
    response = requests.get(url)
    json_data = response.text
    
    # Parse the JSON data
    data = json.loads(json_data)
    
    # Create a list to store the parsed station data
    parsed_stations = []
    
    if data_source == 'DWR':
        stations = data['ResultList']
        for station in stations:
            parsed_station = {
                'id': station.get('stationNum'),
                'abbreviation': station.get('abbrev'),
                'name': station.get('stationName'),
                'latitude': station.get('latitude'),
                'longitude': station.get('longitude'),
                'usgs_site_id': station.get('usgsSiteId'),
                'county': station.get('county'),
                'state': station.get('state'),
                'division': station.get('division'),
                'water_district': station.get('waterDistrict'),
                'data_source': station.get('dataSource'),
                'start_date': station.get('startDate'),
                'end_date': station.get('endDate'),
                'measurement_unit': station.get('measUnit'),
                'last_reading': None,
                'last_reading_date': None
            }
            parsed_stations.append(parsed_station)
    
    elif data_source == 'USGS':
        time_series_list = data['value']['timeSeries']
        for time_series in time_series_list:
            source_info = time_series['sourceInfo']
            variable = time_series['variable']
            values = time_series['values'][0]
            
            site_code = next((code['value'] for code in source_info['siteCode'] if code['agencyCode'] == 'USGS'), None)
            
            parsed_station = {
                'id': site_code,
                'abbreviation': None,
                'name': source_info.get('siteName'),
                'latitude': source_info['geoLocation']['geogLocation'].get('latitude'),
                'longitude': source_info['geoLocation']['geogLocation'].get('longitude'),
                'usgs_site_id': site_code,
                'county': next((prop['value'] for prop in source_info['siteProperty'] if prop['name'] == 'countyCd'), None),
                'state': next((prop['value'] for prop in source_info['siteProperty'] if prop['name'] == 'stateCd'), None),
                'division': None,
                'water_district': None,
                'data_source': 'USGS',
                'start_date': None,
                'end_date': None,
                'measurement_unit': variable['unit']['unitCode'],
                'last_reading': values['value'][0]['value'] if values['value'] else None,
                'last_reading_date': values['value'][0]['dateTime'] if values['value'] else None
            }
            parsed_stations.append(parsed_station)
    
    else:
        logger.error(f"Unsupported data source: {data_source}. Use 'DWR' or 'USGS'.")
        raise ValueError("Unsupported data source. Use 'DWR' or 'USGS'.")
    
    logger.info(f"Parsed {len(parsed_stations)} stations from {data_source}")
    return parsed_stations

def main():
    dwr_url = "https://dwr.state.co.us/Rest/GET/api/v2/surfacewater/surfacewaterstations/?format=json"
    usgs_url = "https://waterservices.usgs.gov/nwis/iv/?format=json&stateCd=CO&parameterCd=00060&siteStatus=active"

    logger.info("Fetching DWR station data...")
    dwr_stations = fetch_and_parse_station_data(dwr_url, 'DWR')
    logger.info("Fetching USGS station data...")
    usgs_stations = fetch_and_parse_station_data(usgs_url, 'USGS')

    # Combine the lists
    all_stations = dwr_stations + usgs_stations

    # Log examples and total count
    logger.info("Example DWR station:")
    logger.info(json.dumps(dwr_stations[0], indent=2))
    logger.info("Example USGS station:")
    logger.info(json.dumps(usgs_stations[0], indent=2))
    logger.info(f"Total number of stations: {len(all_stations)}")

    # You can now work with the all_stations list
    # For example, you can iterate through all stations:
    for station in all_stations:
        logger.debug(f"Station: {station['name']}, ID: {station['id']}, Lat: {station['latitude']}, Long: {station['longitude']}, Source: {station['data_source']}")

    return all_stations

if __name__ == "__main__":
    main()