import requests
from datetime import datetime, timedelta

# USER PARAMS
flow_site_id = "09058000"

today = datetime.today()
one_year_ago = today - timedelta(days=365)
today_str = today.strftime("%Y-%m-%d")
one_year_ago_str = one_year_ago.strftime("%Y-%m-%d")

url = f"https://nwis.waterservices.usgs.gov/nwis/iv/?sites={flow_site_id}&parameterCd=00060&startDT={one_year_ago_str}&endDT={today_str}&siteStatus=all&format=rdb"
response = requests.get(url)
content = response.text.splitlines()

# purpose is to store the daily average
# ex: {"2022-04-02":234,"2022-04-03":456,"2022-04-0":567}
avg_dict = {}

# Initialize avg_dict with keys for dates from a year ago to today
date_range = [one_year_ago + timedelta(days=i) for i in range(366)]
for date in date_range:
    date_str = date.strftime("%Y-%m-%d")
    avg_dict[date_str] = {
        "values": [],
        "count": 0,
        "sum": 0
    }

# loop through all dates
for line in content:
    if line.startswith("USGS"):
        data = line.split("\t")
        date_time_str = data[2]
        target_date = date_time_str.split("T")[0].split()[0]  # Get date without the time component
        discharge_value = data[4]

        try:
            discharge_value = float(discharge_value)
            avg_dict[target_date]["values"].append(discharge_value)
            avg_dict[target_date]["count"] += 1
            avg_dict[target_date]["sum"] += discharge_value
        except ValueError:
            pass

# Calculate daily average
for date, data in avg_dict.items():
    avg_dict[date] = data["sum"] / data["count"] if data["count"] > 0 else None

print(avg_dict)