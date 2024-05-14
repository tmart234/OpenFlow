import requests
import json

def get_stations_by_huc(huc_code):
    url = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/stations"
    params = {
        "hucs": huc_code,
        "elements": "WTEQ"
    }
    response = requests.get(url, params=params)
    return response.json() if response.status_code == 200 else None

def get_swe_data(station_triplets, begin_date, end_date):
    url = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"
    params = {
        "stationTriplets": ",".join(station_triplets),
        "elements": "WTEQ",
        "beginDate": begin_date,
        "endDate": end_date,
        "duration": "DAILY"  # Change this to test different durations
    }
    response = requests.get(url, params=params)
    return response.json() if response.status_code == 200 else None

# Example HUC code for Animas, please replace with the actual HUC if needed
huc_code = "14080104"
begin_date = "2019-10-01"
end_date = "2021-09-30"

# Fetch station data
stations = get_stations_by_huc(huc_code)
if stations:
    station_triplets = [station["stationTriplet"] for station in stations if "stationTriplet" in station]
    print(station_triplets)

    # Fetch SWE data
    swe_data = get_swe_data(station_triplets, begin_date, end_date)
    print(json.dumps(swe_data, indent=2))
else:
    print("No stations found or failed to retrieve stations.")
