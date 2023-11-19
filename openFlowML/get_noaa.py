import requests
import pandas as pd
import io
from datetime import datetime, timedelta
import time
import csv
import math
import re
import json
import numpy as np
import sys
import logging

# given a coordinate, find closest NOAA station
# tests a single NOAA station to get historical temperature data
# handle errors accordingly if data is not available
# NOAA station may not have current daily data so script wiil find one with recent data
# TODO: need better ways to check completness of numeric temperature data before deciding it's good data

Country = 'US'
noaa_api_token = "ensQWPauKcbtSOmsAvlwRVfWyQjJpbHa" # not sensitive or currently used
headers = {"token": noaa_api_token}
fileds = ["TMIN","TMAX"]
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_station_with_recent_data(sorted_stations, startStr, fields, endStr):
    # Convert startStr and endStr to datetime objects
    start_date_obj = datetime.strptime(startStr, "%Y-%m-%d")
    three_days_ago = start_date_obj - timedelta(days=3)
    end_date_obj = datetime.strptime(endStr, "%Y-%m-%d")
    
    for station_id in sorted_stations:
        metadata = get_station_metadata(station_id)
        if metadata:
            # check for high data coverage
            if metadata.get("datacoverage") > 0.87:
                maxdate_str = metadata.get("maxdate")
                mindate_str = metadata.get("mindate")
                maxdate = datetime.strptime(maxdate_str, "%Y-%m-%d")
                mindate = datetime.strptime(mindate_str, "%Y-%m-%d")
                logging.debug(f"Station ID: {station_id} has an end of: {maxdate} and start of {mindate}")
                
                # Now, you're comparing datetime objects with datetime objects
                if (maxdate >= end_date_obj or maxdate >= three_days_ago) and mindate <= start_date_obj:
                    bool_value = check_fields(fields, station_id[0], startStr, endStr)
                    if bool_value:
                        return station_id
    return None

def check_fields(fields, station_id, start_str, end_str):
    url = "https://www.ncei.noaa.gov/access/services/search/v1/data"
    ncei_search_params = {
        "dataset": "daily-summaries",
        "startDate": start_str + "T00:00:00",
        "endDate": end_str + "T00:00:00", 
        "dataTypes": ",".join(fields),
        "stations": station_id,
    }

    # Encode the parameters without encoding the colons in the datetime strings
    encoded_params = [
        f"{k}={','.join(v) if isinstance(v, list) else v}" for k, v in ncei_search_params.items()
    ]

    # Join the encoded parameters with '&' and add them to the URL
    request_url = url + "?" + "&".join(encoded_params)
    logging.debug(f"checking fields for: %s", request_url)

    search_response = get_data(request_url)
    # Assuming search_response is a JSON string
    search_response_json = json.loads(search_response)
    data_types = search_response_json.get("dataTypes", {}).get("buckets", [])
    # Check if the desired fields are in the response
    response_fields = {data_type["key"] for data_type in data_types}
    if all(field in response_fields for field in fields):
        return True
    logging.info("bad fields... checking next ID")
    return False

def get_data(url, headers=None, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                if response.text:
                    return response.text
                else:
                    logging.info("Response status code is 200, but no data received.")
                    return None
            elif response.status_code == 503:  # Retry on 503 errors
                retries += 1
                logging.info(f"Received a 503 error. Retrying... ({retries}/{max_retries})")
                time.sleep(0.3)  # Sleep for 0.3 seconds before retrying
            else:
                logging.error(f"Request failed with status code {response.status_code}.")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"An error occurred: {e}")
            return None
    logging.error(f"Exceeded maximum retries ({max_retries}) for URL {url}.")
    return None

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat, dlon = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def find_closest_ghcnd_station(latitude, longitude, fields, startStr, endStr):
    # store all US stations in us_stations
    stations_url = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
    response_text = get_data(stations_url)
    if response_text is not None:
        us_stations = []
        for line in response_text.splitlines():
            if line.startswith(Country):
                us_stations.append(line)
    else:
        logging.error("Failed to fetch GHCND stations.")
        us_stations = None

    if us_stations is not None:
        logging.debug("US stations found!:")
        logging.debug(us_stations)
    else:
        logging.error("No US stations found. Exiting....")
        exit()

    closest_station = None

     # regex pattern for stations
    pattern = re.compile(r"US[a-zA-Z0-9_]{6}\d+")

    stations_with_distances = []

    for line in us_stations:
        station_id, lat, lon, *_ = line.split()

        if not pattern.match(station_id):
            logging.debug(f"bad match for: {station_id}")
            continue

        lat, lon = float(lat), float(lon)
        distance = haversine_distance(latitude, longitude, lat, lon)
        stations_with_distances.append((station_id, distance))

    # Sort stations by distance and limit the list to 50 items
    # increase size if this part is failing
    sorted_stations = sorted(stations_with_distances, key=lambda x: x[1])[:50]
    logging.info(f"Close station list: {sorted_stations}")

    if not sorted_stations:
        logging.error("No stations found within the distance limit. Exiting...")
        return None

    closest_station = None
    for station, distance in sorted_stations:
        station_with_data = find_station_with_recent_data([(station, distance)], startStr, fields, endStr)
        if station_with_data:
            closest_station = station_with_data
            logging.info(f"The closest station with recent data and valid fields is {closest_station[0]} and it is {closest_station[1]} distance")
            break

    if not closest_station:
        logging.error("No station found with recent data and valid fields.")
    return closest_station

def get_station_metadata(noaa_station_id):
    # Check if noaa_station_id is a tuple, and if so, take the first element
    if isinstance(noaa_station_id, tuple):
        noaa_station_id = noaa_station_id[0]
    # Ensure noaa_station_id is a string
    noaa_station_id = str(noaa_station_id)
    if not noaa_station_id.startswith("GHCND:"):
        noaa_station_id = "GHCND:" + noaa_station_id
    cdo_api_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/stations/"
    time.sleep(0.3)  # Sleep for 0.3 seconds to avoid hitting the rate limit

    metadata_url = f"{cdo_api_url}{noaa_station_id}"
    metadata_response = get_data(metadata_url, headers=headers)

    if metadata_response:
        metadata = json.loads(metadata_response)
        if not metadata:
            logging.info(f"No metadata found for URL: {metadata}")
            return None
        logging.debug(f"{noaa_station_id} metadata: {metadata}")
        return metadata
    else:
        logging.error("No metadata received.")
        return None

def fetch_temperature_data(nearest_station_id, start_str, end_str):
    temperature_data = {}
    metadata = get_station_metadata(nearest_station_id)
    logging.debug(metadata)
    
    if not metadata:
        logging.error("Failed to fetch station metadata.")
        return temperature_data

    ncei_search_url = "https://www.ncei.noaa.gov/access/services/data/v1"
    ncei_search_params = {
        "dataset": "daily-summaries",
        "startDate": start_str + "T00:00:00",
        "endDate": end_str + "T00:00:00",
        "dataTypes": "TMIN,TMAX",
        "stations": nearest_station_id,
    }

    # Construct the request URL
    encoded_params = "&".join([f"{k}={','.join(v) if isinstance(v, list) else v}" for k, v in ncei_search_params.items()])
    request_url = ncei_search_url + "?" + encoded_params
    logging.info("Temperature data URL: %s", request_url)


    response_text = get_data(request_url, headers=headers)
    if response_text:
        response_text = "\n".join([line.strip().replace('"', '') for line in response_text.splitlines()])
        reader = csv.DictReader(io.StringIO(response_text))
        
        # Convert reader output to a list of dictionaries and then to a DataFrame
        temperature_data_list = list(reader)
        temperature_df = pd.DataFrame(temperature_data_list)

        # Replace empty strings with NaN and convert to numeric for TMAX and TMIN
        temperature_df['TMAX'].replace('', np.nan, inplace=True)
        temperature_df['TMIN'].replace('', np.nan, inplace=True)
        temperature_df['TMAX'] = pd.to_numeric(temperature_df['TMAX'], errors='coerce')
        temperature_df['TMIN'] = pd.to_numeric(temperature_df['TMIN'], errors='coerce')

        # Convert the 'DATE' column to datetime but do not set it as index
        temperature_df['Date'] = pd.to_datetime(temperature_df['DATE'], format="%Y-%m-%d")

        # Drop the old 'DATE' column
        temperature_df.drop(columns=['DATE'], inplace=True)

        return temperature_df
    else:
        logging.error("Could not get temperature data!")
        return pd.DataFrame()  # Return an empty DataFrame on error
    
def main(latitude=38.52, longitude=-106.96, startStr=None, endStr=None):
    if startStr is None:
        startStr = (datetime.now() - timedelta(days=7*365 + 7)).strftime('%Y-%m-%d')
    if endStr is None:
        endStr = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    nearest_station_id = find_closest_ghcnd_station(float(latitude), float(longitude), fileds, startStr, endStr)

    if nearest_station_id:
        logging.info(f"Nearest station ID with good data: {nearest_station_id[0]}")
        temperature_data = fetch_temperature_data(nearest_station_id[0], startStr, endStr)
        logging.info(temperature_data)
        return nearest_station_id[0], temperature_data
    else:
        logging.error("No station found near the specified location!")
        return sys.exit(1)  # Exit with a status of 1, indicating failure

if __name__ == "__main__":
    if len(sys.argv) > 4:  # Check if enough arguments are passed
        latitude = float(sys.argv[1])
        longitude = float(sys.argv[2])
        start_date = sys.argv[3]
        end_date = sys.argv[4]
    else:
        # Defaults will be used if not enough arguments are passed
        latitude, longitude, start_date, end_date = 38.52, -106.96, None, None

    main(latitude, longitude, start_date, end_date)