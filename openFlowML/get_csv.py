import requests
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import time
import json

# USER PARAMS
# Parameters to control
flow_site_id = "09058000"
# Replace with your own NOAA API key
noaa_api_key = "your_noaa_api_key_here"
# Replace with desired station ID
noaa_station_id = "USW00012839"
# GPS cords
latitude = "39.987749"
longitude = "-106.510539"

# Calculate today and one year before today
today = datetime.today()
one_year_ago = today - timedelta(days=365)
today_str = today.strftime("%Y-%m-%d")
one_year_ago_str = one_year_ago.strftime("%Y-%m-%d")

# Step 1: Fetch CSV data from the USGS endpoint
url = f"https://nwis.waterservices.usgs.gov/nwis/iv/?sites={flow_site_id}&parameterCd=00060&startDT={one_year_ago_str}T14:32:51.443-06:00&endDT={today_str}T14:32:51.443-06:00&siteStatus=all&format=rdb"
response = requests.get(url)
content = response.text.splitlines()

data = []

for row in csv.reader(content, delimiter='\t'):
    if row[0].startswith("USGS"):
        data.append(row)

# Step 2: Calculate the daily average of the values in the CSV data
daily_values = defaultdict(list)

for row in data:
    date_str = row[2].split(" ")[0]
    value = row[4]
    if value.replace(".", "", 1).isdigit():  # Check if the value is a float
        daily_values[date_str].append(float(value))

daily_averages = {date: sum(values) / len(values) for date, values in daily_values.items()}


# Step 3: Fetch high and low temperature data from NCEI Data Service API
temperature_data = {}

ncei_base_url = "https://www.ncei.noaa.gov/access/services/data/v1"
ncei_params = {
    "dataset": "daily-summaries",
    "stations": noaa_station_id,
    "startDate": one_year_ago_str,
    "endDate": today_str,
    "dataTypes": "TMIN,TMAX",
    "format": "csv",
}

response = requests.get(ncei_base_url, params=ncei_params)

if response.status_code != 200:
    print(f"Error: Received a {response.status_code} status code from the NCEI server.")
    exit()

content = response.text.splitlines()
for row in content[:10]:
    print(row)
reader = csv.DictReader(content)
for row in reader:
    date_str = row["DATE"]
    min_temp = float(row["TMIN"]) * (9 / 5) + 32  # Convert from Celsius to Fahrenheit
    max_temp = float(row["TMAX"]) * (9 / 5) + 32  # Convert from Celsius to Fahrenheit
    temperature_data[date_str] = (min_temp, max_temp)


# Step 4: Combine the data from steps 2 and 3
combined_data = []

for date, average in daily_averages.items():
    min_temp, max_temp = temperature_data.get(date, (None, None))
    combined_data.append([date, average, min_temp, max_temp])

# Step 5: Save the combined data to a new CSV file
with open("combined_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Date", "Average", "Min_Temp", "Max_Temp"])
    writer.writerows(combined_data)