import requests
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import time

# Parameters to control
site_id = "09058000"

# Calculate today and one year before today
today = datetime.today()
one_year_ago = today - timedelta(days=365)
today_str = today.strftime("%Y-%m-%d")
one_year_ago_str = one_year_ago.strftime("%Y-%m-%d")

# Step 1: Fetch CSV data from the USGS endpoint
url = f"https://nwis.waterservices.usgs.gov/nwis/iv/?sites={site_id}&parameterCd=00060&startDT={one_year_ago_str}T14:32:51.443-06:00&endDT={today_str}T14:32:51.443-06:00&siteStatus=all&format=rdb"
response = requests.get(url)
content = response.text.splitlines()

data = []

for row in csv.reader(content):
    if row[0].startswith("USGS"):
        data.append(row)

# Step 2: Calculate the daily average of the values in the CSV data
daily_values = defaultdict(list)

for row in data:
    date_str = row[2].split(" ")[0]
    value = float(row[4])
    daily_values[date_str].append(value)

daily_averages = {date: sum(values) / len(values) for date, values in daily_values.items()}

# Step 3: Fetch high and low temperature data from OpenWeatherMap API
# Replace with your own API key
openweathermap_api_key = "your_openweathermap_api_key"
latitude = "40.0385"
longitude = "-105.5085"

temperature_data = {}

for date in daily_averages.keys():
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    timestamp = int(date_obj.timestamp())
    weather_url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={latitude}&lon={longitude}&dt={timestamp}&units=metric&appid={openweathermap_api_key}"
    weather_response = requests.get(weather_url)
    weather_data = weather_response.json()

    min_temp = weather_data["current"]["temp_min"]
    max_temp = weather_data["current"]["temp_max"]
    temperature_data[date] = (min_temp, max_temp)
    time.sleep(1)  # To avoid hitting rate limits

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
