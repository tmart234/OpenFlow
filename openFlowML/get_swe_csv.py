import requests
from datetime import datetime

# script to get historical SWE for a location and date range

# Set your variables here
start_date = "2020-01-01"
end_date = "2021-01-01"
station_id = "360"
state = "MT"

url_template = "https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport/daily/start_of_period/{station_id}:{state}:SNTL%7Cid=%22%22%7Cname/{start_date},{end_date}/WTEQ::value"

url = url_template.format(
    start_date=start_date,
    end_date=end_date,
    station_id=station_id,
    state=state,
)

response = requests.get(url)
content = response.text.splitlines()

data_dict = {}

for line in content:
    if line.startswith("Date") or ',' not in line:
        continue  # skip the header line or lines without a comma

    date_str, value = line.split(',')
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    data_dict[date_obj] = float(value)

print(data_dict)
