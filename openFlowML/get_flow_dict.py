import requests
import argparse
from datetime import datetime

def get_daily_flow_data(flow_site_id, start_date, end_date):
    url = f"https://nwis.waterservices.usgs.gov/nwis/iv/?sites={flow_site_id}&parameterCd=00060&startDT={start_date}&endDT={end_date}&siteStatus=all&format=rdb"
    print("Fetching flow data from: ")
    print(url)
    response = requests.get(url)
    content = response.text.splitlines()

    # Initialize a dictionary to store daily min and max flow values
    daily_flow_data = {}

    # Find the start line that contains actual data headers (e.g., "agency_cd site_no ...")
    for index, line in enumerate(content):
        if line.startswith('agency_cd'):
            start_line = index + 1
            break

    # Extract the actual data lines
    data_lines = content[start_line:]

    for line in data_lines:
        columns = line.split('\t')
        if len(columns) >= 5:
            datetime_str = f"{columns[2]} {columns[3]}"
            try:
                dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
                formatted_datetime_str = dt.strftime("%Y-%m-%d %H:%M")
                
                flow = columns[4]
                date_str = formatted_datetime_str.split()[0]
                flow = float(flow)
                
                if date_str in daily_flow_data:
                    if "min" not in daily_flow_data[date_str] or flow < daily_flow_data[date_str]["min"]:
                        daily_flow_data[date_str]["min"] = flow
                    if "max" not in daily_flow_data[date_str] or flow > daily_flow_data[date_str]["max"]:
                        daily_flow_data[date_str]["max"] = flow
                else:
                    daily_flow_data[date_str] = {"min": flow, "max": flow}
            except ValueError:
                print(f"Skipping line due to unexpected datetime format: {line}")

    return daily_flow_data

def main(args):
    daily_flow_data = get_daily_flow_data(args.flow_site_id, args.start_date, args.end_date)

    # Print the daily flow data
    for date, data in daily_flow_data.items():
        min_flow = data["min"]
        max_flow = data["max"]
        print(f"Date: {date}, Min Flow: {min_flow}, Max Flow: {max_flow}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch daily flow data for a given USGS site.')
    parser.add_argument('flow_site_id', type=str, help='USGS flow site ID')
    parser.add_argument('start_date', type=str, help='Start date in the format YYYY-MM-DD')
    parser.add_argument('end_date', type=str, help='End date in the format YYYY-MM-DD')
    args = parser.parse_args()
    main(args)
