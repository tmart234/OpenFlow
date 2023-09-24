import requests
from datetime import datetime

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
#print(url)
response = requests.get(url)
content = response.text.splitlines()

data_dict = {}

for line in content:
    if ',' not in line and not line.strip().startswith('#'):
        continue  # skip lines without a comma
    try:
        date_str, value = line.split(',')
    except Exception as e:
        print(f"Error processing line: {line}")
        print(f"Exception: {e}")
        continue

    # Check if the second element is a number
    if not value.replace('.', '', 1).isdigit():
        continue

    data_dict[date_str] = float(value)

print(data_dict)
