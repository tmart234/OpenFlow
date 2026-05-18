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

# Given a coordinate, find the closest NOAA GHCND station that actually has
# usable daily TMIN/TMAX/PRCP data over the requested window.
#
# We gate stations in two passes:
#   1. NCEI metadata pass: `datacoverage > 0.87` and a date range that brackets
#      our window. This is fast (one metadata call per candidate) but is a
#      coverage promise across the station's entire history -- a station with
#      88% lifetime coverage can still be missing the last 6 months.
#   2. Realized-data pass: after `fetch_temperature_data` returns, we measure
#      the actual non-NaN fraction of TMIN/TMAX over the requested window
#      and reject the station if it drops below MIN_OBSERVED_FRACTION. This is
#      what the old "TODO: better completeness checks" comment was about --
#      `find_station_with_recent_data` could happily return a station whose
#      daily data was 60% NaN once you actually downloaded it.
#
# PRCP is reported by NCEI in tenths of mm; we divide by 10 to land in mm,
# the same unit Open-Meteo serves for precipitation_sum.

Country = 'US'
noaa_api_token = "ensQWPauKcbtSOmsAvlwRVfWyQjJpbHa" # not sensitive or currently used
headers = {"token": noaa_api_token}
# TMIN / TMAX are required (data_utils.CORE_REQUIRED). PRCP joins as an
# optional column -- if the closest station's PRCP is too gappy we still
# accept the station for temperature and combine_data will fill PRCP with 0.
fileds = ["TMIN", "TMAX", "PRCP"]
REQUIRED_FIELDS = ("TMIN", "TMAX")
# Stations with realized coverage below this on any required field are
# rejected so we walk down to the next candidate.
MIN_OBSERVED_FRACTION = 0.7
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
        "dataTypes": "TMIN,TMAX,PRCP",
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

        # Rename 'STATION' column to 'NOAA_station'
        temperature_df.rename(columns={'STATION': 'NOAA_station'}, inplace=True)

        # Coerce TMIN/TMAX (already in degrees in NCEI daily-summaries) and
        # PRCP (tenths of mm in GHCND; we convert to mm to match Open-Meteo
        # precipitation_sum). Missing values come through as empty strings.
        for col in ('TMAX', 'TMIN'):
            if col in temperature_df.columns:
                temperature_df[col].replace('', np.nan, inplace=True)
                temperature_df[col] = pd.to_numeric(temperature_df[col], errors='coerce')
        if 'PRCP' in temperature_df.columns:
            temperature_df['PRCP'].replace('', np.nan, inplace=True)
            temperature_df['PRCP'] = pd.to_numeric(temperature_df['PRCP'], errors='coerce') / 10.0
            # Surface PRCP under combine_data's `precipitation` column name so
            # downstream code doesn't need to know about the GHCND code.
            temperature_df.rename(columns={'PRCP': 'precipitation'}, inplace=True)

        # Convert the 'DATE' column to datetime but do not set it as index
        temperature_df['Date'] = pd.to_datetime(temperature_df['DATE'], format="%Y-%m-%d")

        # Drop the old 'DATE' column
        temperature_df.drop(columns=['DATE'], inplace=True)

        return temperature_df
    else:
        logging.error("Could not get temperature data!")
        return pd.DataFrame()  # Return an empty DataFrame on error


def _realized_coverage_ok(df, fields, min_fraction):
    """
    Validate that a station's actual returned data isn't mostly NaN on the
    fields we require. Returns True iff every field in `fields` is non-NaN
    on at least `min_fraction` of the returned rows.

    Why this exists: NCEI metadata `datacoverage` is a lifetime average; a
    station with 88% lifetime coverage can still be missing the last 6 months
    of TMIN. Without this gate the model trains on a column that's mostly NaN
    and gap-filled to 0, which silently destroys temperature signal.
    """
    if df is None or df.empty:
        return False
    n = len(df)
    for field in fields:
        if field not in df.columns:
            return False
        present = df[field].notna().sum()
        if present / n < min_fraction:
            logging.info("Realized coverage for %s = %.2f < %.2f, rejecting",
                         field, present / n, min_fraction)
            return False
    return True


def main(latitude=38.52, longitude=-106.96, startStr=None, endStr=None):
    if startStr is None:
        startStr = (datetime.now() - timedelta(days=7*365 + 7)).strftime('%Y-%m-%d')
    if endStr is None:
        endStr = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    # Build a ranked list of candidate stations once; the metadata pass inside
    # find_closest_ghcnd_station gates on lifetime datacoverage + field
    # availability. We then walk the candidates ourselves and apply a second
    # realized-data gate (MIN_OBSERVED_FRACTION on TMIN/TMAX over the actual
    # returned window) so a station with stale-but-present metadata can't slip
    # past with mostly-NaN data.
    candidate = find_closest_ghcnd_station(
        float(latitude), float(longitude), fileds, startStr, endStr)
    tried = []
    while candidate is not None:
        station_id = candidate[0]
        if station_id in tried:  # paranoia: never loop forever
            break
        tried.append(station_id)
        logging.info("Trying station %s (distance %.2f km)", station_id, candidate[1])
        temperature_data = fetch_temperature_data(station_id, startStr, endStr)
        if _realized_coverage_ok(temperature_data, REQUIRED_FIELDS, MIN_OBSERVED_FRACTION):
            logging.info("Accepted station %s (realized coverage passes %.0f%% gate)",
                         station_id, MIN_OBSERVED_FRACTION * 100)
            return station_id, temperature_data
        # Walk down the candidate list -- ask find_closest_ghcnd_station for
        # the next-best station EXCLUDING the ones we've already tried.
        logging.info("Station %s failed realized-coverage gate; trying next", station_id)
        candidate = _next_candidate(latitude, longitude, fileds,
                                    startStr, endStr, exclude=tried)
    logging.error("No station found near the specified location with good realized coverage!")
    # Return empty results rather than exiting: a single station with no
    # nearby NOAA data must not abort the whole training run.
    return None, pd.DataFrame()


def _next_candidate(latitude, longitude, fields, startStr, endStr, exclude):
    """
    Re-run the closest-station search excluding stations we already tried.

    Cheap to re-issue: the GHCND station list is cached at the HTTP layer and
    the metadata calls are per-station with 0.3s sleeps. We accept the extra
    latency in exchange for not having to refactor find_closest_ghcnd_station
    into a generator.
    """
    return _find_closest_ghcnd_station_excluding(latitude, longitude, fields,
                                                  startStr, endStr, exclude)


def _find_closest_ghcnd_station_excluding(latitude, longitude, fields,
                                          startStr, endStr, exclude):
    """Same as find_closest_ghcnd_station but skips stations in `exclude`."""
    stations_url = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
    response_text = get_data(stations_url)
    if response_text is None:
        return None
    us_stations = [line for line in response_text.splitlines() if line.startswith(Country)]
    pattern = re.compile(r"US[a-zA-Z0-9_]{6}\d+")
    stations_with_distances = []
    for line in us_stations:
        station_id, lat, lon, *_ = line.split()
        if not pattern.match(station_id) or station_id in exclude:
            continue
        lat, lon = float(lat), float(lon)
        distance = haversine_distance(latitude, longitude, lat, lon)
        stations_with_distances.append((station_id, distance))
    sorted_stations = sorted(stations_with_distances, key=lambda x: x[1])[:50]
    for station, distance in sorted_stations:
        result = find_station_with_recent_data([(station, distance)], startStr, fields, endStr)
        if result:
            return result
    return None

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