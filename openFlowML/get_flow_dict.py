import requests
from datetime import datetime, timedelta

def get_daily_flow_data(flow_site_id, start_date, end_date):
    # Update the URL with the provided date range
    url = f"https://nwis.waterservices.usgs.gov/nwis/iv/?sites={flow_site_id}&parameterCd=00060&startDT={start_date}&endDT={end_date}&siteStatus=all&format=rdb"
    response = requests.get(url)
    content = response.text.splitlines()

    # Initialize a dictionary to store daily min and max flow values
    daily_flow_data = {}

    for line in content:
        # Skip lines that start with "#"
        if line.startswith("#"):
            continue
        # Split each line by spaces
        columns = line.split()
        # Check if the line has at least 5 columns
        if len(columns) >= 5:
            # Combine date, time, and timezone strings into a single string
            datetime_str = f"{columns[2]} {columns[3]} {columns[4]}"
            
            try:
                # Create a datetime object from the combined string
                dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M %Z")
                # Convert the datetime object back to a formatted string
                formatted_datetime_str = dt.strftime("%Y-%m-%d %H:%M %Z")
                
                flow = columns[5]
                # Extract the date from the timestamp (e.g., "2023-09-16 17:30" -> "2023-09-16")
                date_str = formatted_datetime_str.split()[0]
                
                # Convert the flow value to a float
                flow = float(flow)
                
                # Check if the date is already in the dictionary
                if date_str in daily_flow_data:
                    # Update the minimum and maximum flow values if needed
                    min_flow = min(daily_flow_data[date_str]["min"], flow)
                    max_flow = max(daily_flow_data[date_str]["max"], flow)
                    daily_flow_data[date_str]["min"] = min_flow
                    daily_flow_data[date_str]["max"] = max_flow
                else:
                    # If it's a new date, initialize the dictionary entry
                    daily_flow_data[date_str] = {"min": flow, "max": flow}
            except ValueError:
                # Handle the case where the datetime format is not as expected
                print(f"Skipping line due to unexpected datetime format: {line}")

    return daily_flow_data

def main():
    # USER PARAMS
    flow_site_id = "09058000"
    # Specify your start and end dates as needed
    start_date = "2018-09-17"
    end_date = "2023-09-16"

    daily_flow_data = get_daily_flow_data(flow_site_id, start_date, end_date)

    # Print the daily flow data
    for date, data in daily_flow_data.items():
        min_flow = data["min"]
        max_flow = data["max"]
        print(f"Date: {date}, Min Flow: {min_flow}, Max Flow: {max_flow}")

if __name__ == "__main__":
    main()
