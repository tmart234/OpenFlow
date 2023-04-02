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

# Step 3: Save the daily averages to a CSV file
with open("daily_averages.csv", mode="w", newline="") as csv_file:
    fieldnames = ["Date", "Average"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for date, average in daily_averages.items():
        writer.writerow({"Date": date, "Average": average})

