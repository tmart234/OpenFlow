import requests

# example data as defaults
def fetch_snow_data(start_date="2020-01-01", end_date="2021-01-01", station_id="360", state="MT"):
    url_template = "https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport/daily/start_of_period/{station_id}:{state}:SNTL%7Cid=%22%22%7Cname/{start_date},{end_date}/WTEQ::value"
    url = url_template.format(
        start_date=start_date,
        end_date=end_date,
        station_id=station_id,
        state=state,
    )

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad responses
        content = response.text.splitlines()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

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

    return data_dict

if __name__ == "__main__":
    # Example usage
    result = fetch_snow_data()
    print(result)
