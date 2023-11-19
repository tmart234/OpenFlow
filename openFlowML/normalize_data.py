from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

'''
Normalize and interpolate data
'''

def get_mean_temperature(station_id, date):
    # TODO: Construct URL for fetching mean temperature
    url = f"http://example.com/mean_temperature?station_id={station_id}&date={date}"

    # Parse the response to extract the mean temperature
    # Return the mean temperature for that date
    mean_temperature = -999

    return mean_temperature

def fill_with_rolling_mean(data, temperature_column, window_size=7):
    # Fills missing temperature values with a rolling mean.
    # First replace NaN values with a rolling mean
    data[temperature_column] = data[temperature_column].rolling(window=window_size, min_periods=1, center=True).mean()

    # If there are still NaN values at the start or end of the series, fill them with the first or last valid value
    data[temperature_column] = data[temperature_column].bfill()  # Backward fill
    data[temperature_column] = data[temperature_column].ffill()  # Forward fill

    return data


def interpolate_temperatures(data):
    # Applies rolling mean interpolation to temperature columns
    data = fill_with_rolling_mean(data, 'TMIN')
    data = fill_with_rolling_mean(data, 'TMAX')
    return data

def normalize_date_to_year_fraction(date_series):
    date_series = pd.to_datetime(date_series)
    day_of_year = date_series.dt.dayofyear
    year_fraction = (day_of_year - 1) / (365 + date_series.dt.is_leap_year.astype(int))
    return year_fraction

def normalize_data(file_path, data):
    try:
        data = pd.read_csv(file_path)

        # Interpolate temperatures first
        data = interpolate_temperatures(data)

        numeric_columns = ['TMIN', 'TMAX', 'Min Flow', 'Max Flow']

        # Check if required columns exist
        missing_columns = [col for col in numeric_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the data: {missing_columns}")

        # Replace empty strings with NaN
        data.replace({'': np.nan}, inplace=True)

        # One-hot encoding for stationID
        if 'stationID' in data.columns:
            data = pd.get_dummies(data, columns=['stationID'], prefix=['station'])

        # Convert columns to numeric values
        for column in numeric_columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')

        # Fill remaining NaN values with column mean
        for column in numeric_columns:
            if data[column].isnull().any():
                data[column].fillna(data[column].mean(), inplace=True)

        # Normalize numeric columns
        scalers = {}
        for column in numeric_columns:
            scaler = StandardScaler()
            data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))
            scalers[column] = scaler

        # Normalize the date column
        data['date_normalized'] = normalize_date_to_year_fraction(data['Date'])

        # Drop the original date column
        data.drop(columns=['Date'], inplace=True)

        return data
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

if __name__ == "__main__":
    file_path = 'combined_data_all_sites.csv'
    normalized_data = normalize_data(file_path)