import requests
from datetime import datetime, timedelta
import csv


# Adds flow to data. Given a csv of temps & dates, find the date range and add a column for flow in cfs
# USER PARAMS
flow_site_id = "09058000"
input_csv_file = "Kremmling_Climate_daily.csv"  # Replace with the name of your input CSV file
output_csv_file = "Kremmling_Climate_daily_output.csv"  

# Determine the minimum and maximum dates from the CSV file
min_date = None
max_date = None

with open(input_csv_file, "r") as input_file:
    csv_reader = csv.reader(input_file)
    header = next(csv_reader)

    for row in csv_reader:
        date_str = row[0]
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")

        if min_date is None or date_obj < min_date:
            min_date = date_obj
        if max_date is None or date_obj > max_date:
            max_date = date_obj

min_date_str = min_date.strftime("%Y-%m-%d")
max_date_str = max_date.strftime("%Y-%m-%d")

# Update the URL with the new date range
url = f"https://nwis.waterservices.usgs.gov/nwis/iv/?sites={flow_site_id}&parameterCd=00060&startDT={min_date_str}&endDT={max_date_str}&siteStatus=all&format=rdb"
response = requests.get(url)
content = response.text.splitlines()

# purpose is to store the daily average
# ex: {"2022-04-02":234,"2022-04-03":456,"2022-04-0":567}
avg_dict = {}

# Calculate the number of days between min_date and max_date
num_days = (max_date - min_date).days

# Initialize avg_dict with keys for dates from min_date to max_date
date_range = [min_date + timedelta(days=i) for i in range(num_days + 1)]
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

# Read the input CSV file
with open(input_csv_file, "r") as input_file:
    csv_reader = csv.reader(input_file)
    header = next(csv_reader)

    # Add the new column header
    header.append("avg flow")

    # Read the rows and append the "avg flow" column values
    rows = []
    for row in csv_reader:
        date_str = row[0]
        avg_flow = avg_dict.get(date_str)
        if avg_flow is not None:
            row.append(avg_flow)
        rows.append(row)

# Write the output CSV file with the new column
with open(output_csv_file, "w", newline="") as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(header)
    csv_writer.writerows(rows)