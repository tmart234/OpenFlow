import requests
import pandas as pd
import io
from io import StringIO
from datetime import datetime, timedelta
import time
import csv
import math

# given a coordinate, find closest NOAA station
# tests a single NOAA station to get 1 year of historical temperature data
# handle errors accordingly if data is not available
# NOAA station may not have current daily data so script wiil find one with recent data

Country = 'US'

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat, dlon = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def find_closest_ghcnd_station(latitude, longitude, noaa_api_token):
    stations_url = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
    response = requests.get(stations_url)
    if response.status_code != 200:
        print("Failed to fetch GHCND stations.")
        us_stations = None
    else:
        us_stations = []
        # Process the response content line by line
        for line in response.text.splitlines():
            if line.startswith(Country):
                us_stations.append(line)
    #print(us_stations)

    closest_station = None
    min_distance = float("inf")
    current_year = datetime.now().year
    one_month_ago = datetime(current_year, datetime.now().month, 1) - timedelta(days=1)

    for line in us_stations:
        station_id, lat, lon, *_ = line.split()
        lat, lon = float(lat), float(lon)
        distance = haversine_distance(latitude, longitude, lat, lon)
        if distance < min_distance:
            metadata_str = get_station_metadata(station_id, noaa_api_token)
            print(metadata_str)
            if metadata_str:
                maxdate_str = metadata_str.get("maxdate")
                maxdate = datetime.strptime(maxdate_str, "%Y-%m-%d")
                print(metadata_str)
                print(maxdate_str)
                # check that the station has valid data for the last year and last month
                if maxdate.year == current_year and maxdate > one_month_ago:
                    min_distance = distance
                    closest_station = station_id
    return closest_station


# convert cordinate to boundary box
def lat_lon_to_bbox(lat, lon, miles):
    miles_to_degrees = 1 / 69
    delta = miles * miles_to_degrees
    return lat - delta, lon - delta, lat + delta, lon + delta

def get_station_metadata(noaa_station_id, noaa_api_token):
    if not noaa_station_id.startswith("GHCND:"):
        noaa_station_id = "GHCND:" + noaa_station_id
    #print(noaa_station_id)
    cdo_api_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/stations/"
    headers = {"token": noaa_api_token}
    # Sleep for 0.3 second before making the request to avoid hitting the rate limit (5 per second or 1000 per day)
    time.sleep(0.3)
    metadata_response = requests.get(f"{cdo_api_url}{noaa_station_id}", headers=headers)
    print(f"Request URL: {metadata_response.url}")
    print(f"Response status code: {metadata_response.status_code}")

    metadata = None
    if metadata_response.status_code == 200:
        metadata = metadata_response.json()
    print(metadata)
    return metadata



def fetch_temperature_data(nearest_station_id, noaa_api_token):
    temperature_data = {}
    
    # Get station metadata
    metadata = get_station_metadata(nearest_station_id, noaa_api_token)
    print(metadata)
    
    if not metadata:
        print("Failed to fetch station metadata.")
        return temperature_data

    # Use the end date on record if available
    end_date_str = metadata.get("maxdate") if metadata.get("maxdate") else datetime.today().strftime("%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    one_year_ago = end_date - timedelta(days=365)
    one_year_ago_str = one_year_ago.strftime("%Y-%m-%dT%H:%M:%SZ")

    ncei_search_url = "https://www.ncei.noaa.gov/access/services/search/v1/data"
    bbox = lat_lon_to_bbox(latitude, longitude, 5)
    ncei_search_params = {
        "dataset": "daily-summaries",
        "startDate": one_year_ago_str,
        "endDate": end_date_str,
        "boundingBox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "dataTypes": "TMIN,TMAX",
        "stations": nearest_station_id,
        "format": "csv",
    }
    search_response = requests.get(ncei_search_url, params=ncei_search_params)
    print("Temperature data URL:", search_response.url)
    if search_response.status_code == 200 and search_response.text.strip():
        reader = csv.DictReader(io.StringIO(search_response.text))
        for row in reader:
            date_str = row["DATE"]
            min_temp = float(row["TMIN"]) * (9 / 5) + 32  # Convert from Celsius to Fahrenheit
            max_temp = float(row["TMAX"]) * (9 / 5) + 32  # Convert from Celsius to Fahrenheit
            temperature_data[date_str] = (min_temp, max_temp)

    # Save temperature data to a CSV file
    with open("temperature_data.csv", "w", newline="") as csvfile:
        fieldnames = ["DATE", "TMIN", "TMAX"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for date_str, (min_temp, max_temp) in temperature_data.items():
            writer.writerow({"DATE": date_str, "TMIN": min_temp, "TMAX": max_temp})

    return temperature_data


noaa_api_token = "ensQWPauKcbtSOmsAvlwRVfWyQjJpbHa"
noaa_station_id = "GHCND:USR0000CRED"
latitude = 39.045002
longitude = -106.257903


if __name__ == "__main__":
    nearest_station_id = find_closest_ghcnd_station(latitude, longitude, noaa_api_token)
    print(nearest_station_id)
    if nearest_station_id:
        print("Nearest station ID:", nearest_station_id) 
        fetch_temperature_data(nearest_station_id, noaa_api_token) 
    else: 
        print("No station found near the specified location.")