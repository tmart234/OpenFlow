import requests
import pandas as pd
import io
from datetime import datetime, timedelta
import time
import csv
import math
import re
import json
import os
import sys

# given a coordinate, find closest NOAA station
# tests a single NOAA station to get historical temperature data
# handle errors accordingly if data is not available
# NOAA station may not have current daily data so script wiil find one with recent data
# TODO: need better ways to check completness of numeric temperature data before deciding it's good data

Country = 'US'
noaa_api_token = "ensQWPauKcbtSOmsAvlwRVfWyQjJpbHa" # not sensitive or currently used
headers = {"token": noaa_api_token}
fileds = ["TMIN","TMAX"]

def find_station_with_recent_data(sorted_stations, startStr, fields, endStr):
    # Convert startStr and endStr to datetime objects
    start_date_obj = datetime.strptime(startStr, "%Y-%m-%d")
    end_date_obj = datetime.strptime(endStr, "%Y-%m-%d")
    
    for station_id in sorted_stations:
        metadata = get_station_metadata(station_id)
        if metadata:
            # check for high data coverage
            if metadata.get("datacoverage") > 0.95:
                maxdate_str = metadata.get("maxdate")
                mindate_str = metadata.get("mindate")
                maxdate = datetime.strptime(maxdate_str, "%Y-%m-%d")
                mindate = datetime.strptime(mindate_str, "%Y-%m-%d")
                print(f"Station ID: {station_id} has an end of: {maxdate} and start of {mindate}")
                
                # Now, you're comparing datetime objects with datetime objects
                if maxdate >= end_date_obj and mindate <= start_date_obj:
                    bool_value = check_fields(fields, station_id[0], startStr, endStr)
                    if bool_value:
                        return station_id
    return None

def check_fields(fields, id, start_str, end_str):
    url = "https://www.ncei.noaa.gov/access/services/search/v1/data"
    ncei_search_params = {
        "dataset": "daily-summaries",
        "startDate": start_str.strftime("%Y-%m-%d") + "T00:00:00",
        "endDate": end_str.strftime("%Y-%m-%d") + "T00:00:00",
        "dataTypes": ",".join(fields),
        "stations": id,
    }

    # Encode the parameters without encoding the colons in the datetime strings
    encoded_params = [
        f"{k}={','.join(v) if isinstance(v, list) else v}" for k, v in ncei_search_params.items()
    ]

    # Join the encoded parameters with '&' and add them to the URL
    request_url = url + "?" + "&".join(encoded_params)
    print(f"checking fields for: {request_url}")

    search_response = get_data(request_url)
    # Assuming search_response is a JSON string
    search_response_json = json.loads(search_response)
    data_types = search_response_json.get("dataTypes", {}).get("buckets", [])
    # Check if the desired fields are in the response
    response_fields = {data_type["key"] for data_type in data_types}
    if all(field in response_fields for field in fields):
        return True
    print("bad fields... checking next ID")
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
                    print("Response status code is 200, but no data received.")
                    return None
            elif response.status_code == 503:  # Retry on 503 errors
                retries += 1
                print(f"Received a 503 error. Retrying... ({retries}/{max_retries})")
                time.sleep(0.3)  # Sleep for 0.3 seconds before retrying
            else:
                print(f"Request failed with status code {response.status_code}.")
                return None
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None
    print(f"Exceeded maximum retries ({max_retries}) for URL {url}.")
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
        print("Failed to fetch GHCND stations.")
        us_stations = None

    if us_stations is not None:
        print("US stations found!!")
        #print(us_stations)
    else:
        print("No US stations found. Exiting....")
        exit()

    closest_station = None

     # regex pattern for stations
    pattern = re.compile(r"US[a-zA-Z0-9_]{6}\d+")

    stations_with_distances = []

    for line in us_stations:
        station_id, lat, lon, *_ = line.split()

        if not pattern.match(station_id):
            #print(f"bad match for: {station_id}")
            continue

        lat, lon = float(lat), float(lon)
        distance = haversine_distance(latitude, longitude, lat, lon)
        stations_with_distances.append((station_id, distance))

    # Sort stations by distance and limit the list to 50 items
    # increase size if this part is failing
    sorted_stations = sorted(stations_with_distances, key=lambda x: x[1])[:50]
    print(f"Close station list: {sorted_stations}")

    if not sorted_stations:
        print("No stations found within the distance limit. Exiting...")
        return None

    closest_station = None
    closest_station = find_station_with_recent_data(sorted_stations, startStr, fields, endStr)
    if closest_station:
        print(f"The closest station with recent data and valid fields is {closest_station[0]} and it is {closest_station[1]} distance")
    else:
        print("No station found with recent data and valid fields.")
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
            print(f"No metadata found for URL: {metadata}")
            return None
        #print(f"{noaa_station_id} metadata: {metadata}")
        return metadata
    else:
        print("No metadata received.")
        return None


def fetch_temperature_data(nearest_station_id, startStr, endStr):
    temperature_data = {}
    # Convert datetime objects to strings with the desired format
    start_str = startStr.strftime("%Y-%m-%d")
    end_str = endStr.strftime("%Y-%m-%d")
    # Get station metadata
    metadata = get_station_metadata(nearest_station_id)
    print(metadata)
    
    if not metadata:
        print("Failed to fetch station metadata.")
        return temperature_data

    ncei_search_url = "https://www.ncei.noaa.gov/access/services/data/v1"
    ncei_search_params = {
        "dataset": "daily-summaries",
        "startDate": start_str + "T00:00:00",
        "endDate": end_str + "T00:00:00",
        "dataTypes": "TMIN,TMAX",
        "stations": nearest_station_id,
    }

    # Encode the parameters without encoding the colons in the datetime strings
    encoded_params = [
        f"{k}={','.join(v) if isinstance(v, list) else v}" for k, v in ncei_search_params.items()
    ]
    # Join the encoded parameters with '&' and add them to the URL
    request_url = ncei_search_url + "?" + "&".join(encoded_params)
    print("Temperature data URL:", request_url)

    response_text = get_data(request_url, headers=headers)
    if response_text:
        # Preprocess the response text to remove extra spaces
        response_text = "\n".join([line.strip().replace('"', '') for line in response_text.splitlines()])
        # Use csv.DictReader to read the response data
        reader = csv.DictReader(io.StringIO(response_text))
        for row in reader:
            date_str = row["DATE"]
            try:
                # temperature values are given in tenths of degrees Celsius.
                min_temp = (float(int(row["TMIN"].strip())) / 10) * (9 / 5) + 32  # Convert from Celsius to Fahrenheit
                max_temp = (float(int(row["TMAX"].strip())) / 10) * (9 / 5) + 32  # Convert from Celsius to Fahrenheit
                temperature_data[date_str] = {"TMIN": min_temp, "TMAX": max_temp}
            except ValueError:
                print(f"Skipping row with non-numeric temperature data: {row}")
    else:
        print("Could not get temperature data!!")

    # Convert temperature_data to a pandas DataFrame
    temperature_df = pd.DataFrame.from_dict(temperature_data, orient="index", columns=["TMIN", "TMAX"])
    temperature_df.index = pd.to_datetime(temperature_df.index, format="%Y-%m-%d")
    return temperature_df

def main(latitude, longitude, startStr, endStr, statiom):
    nearest_station_id = find_closest_ghcnd_station(latitude, longitude, fileds, startStr, endStr)

    if nearest_station_id:
        print(f"Nearest station ID with good data: {nearest_station_id[0]}")
        temperature_data = fetch_temperature_data(nearest_station_id[0], startStr, endStr)
        
        # Save the temperature data to a CSV file by converting PD DataFrame to a CSV file
        script_directory = os.path.dirname(os.path.abspath(__file__))
        
        # Change the naming format here
        csv_file_path = os.path.join(script_directory, f"{station}_noaa_data.csv")
        temperature_data.to_csv(csv_file_path)
        
        # Set the path as an environment variable file
        env_file = os.getenv('GITHUB_ENV')
        with open(env_file, "a") as myfile:
            myfile.write(f"CSV_FILE_PATH={csv_file_path}")    
        
        return nearest_station_id[0], temperature_data
    else:
        print("No station found near the specified location.")
        return None

if __name__ == "__main__":
    latitude = float(sys.argv[1])
    longitude = float(sys.argv[2])
    start_date = sys.argv[3]
    end_date = sys.argv[4]
    usgs_station = sys.argv[5]
    main(latitude, longitude, start_date, end_date, usgs_station)
