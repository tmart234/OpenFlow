import requests
import argparse
import pandas as pd
from datetime import datetime

def get_daily_flow_data(flow_site_id, start_date, end_date):
    url = f"https://nwis.waterservices.usgs.gov/nwis/iv/?sites={flow_site_id}&parameterCd=00060&startDT={start_date}&endDT={end_date}&siteStatus=all&format=rdb"
    print("Fetching flow data from:")
    print(url)
    response = requests.get(url)
    content = response.text.splitlines()

    # Initialize lists to store data for creating DataFrame
    dates, min_flows, max_flows = [], [], []

    for index, line in enumerate(content):
        if line.startswith('USGS'):
            start_line = index
            break

    data_lines = content[start_line:]

    for line in data_lines:
        columns = line.split('\t')
        if len(columns) >= 5:
            datetime_str = columns[2][:16]  # This grabs only the "YYYY-MM-DD HH:MM" part
            try:
                dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
                formatted_datetime_str = dt.strftime("%Y-%m-%d %H:%M")
                
                flow = columns[4]
                date_str = formatted_datetime_str.split()[0]
                flow = float(flow)
                
                if date_str in dates:
                    index = dates.index(date_str)
                    min_flows[index] = min(min_flows[index], flow)
                    max_flows[index] = max(max_flows[index], flow)
                else:
                    dates.append(date_str)
                    min_flows.append(flow)
                    max_flows.append(flow)
            except ValueError:
                print(f"Skipping line due to unexpected datetime format: {line}")

    df = pd.DataFrame({
        'Date': dates,
        'Min Flow': min_flows,
        'Max Flow': max_flows
    })

    return df

def main(args):
    df = get_daily_flow_data(args.flow_site_id, args.start_date, args.end_date)

    # Print the daily flow data
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch daily flow data for a given USGS site.')
    parser.add_argument('flow_site_id', type=str, help='USGS flow site ID')
    parser.add_argument('start_date', type=str, help='Start date in the format YYYY-MM-DD')
    parser.add_argument('end_date', type=str, help='End date in the format YYYY-MM-DD')
    args = parser.parse_args()
    main(args)
