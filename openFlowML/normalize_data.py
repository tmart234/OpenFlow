from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def normalize_date_to_year_fraction(date_series):
    date_series = pd.to_datetime(date_series)
    day_of_year = date_series.dt.dayofyear
    year_fraction = (day_of_year - 1) / (365 + date_series.dt.is_leap_year.astype(int))
    return year_fraction

def normalize_data(file_path, data):
    output_file = file_path.replace('.csv', '_normalized.csv')
    numeric_columns = ['TMIN', 'TMAX', 'Min Flow', 'Max Flow']

    # Replace empty strings with NaN
    data.replace({'': np.nan}, inplace=True)

    # Convert columns to numeric values, converting any errors to NaN
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    # Use 7-day rolling average to fill NaN values for specific columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].rolling(7, min_periods=1).mean())

    # If after rolling average fill, there are still NaN values, fill with column mean
    for column in numeric_columns:
        if data[column].isnull().any():
            data[column].fillna(data[column].mean(), inplace=True)

    # Normalize numeric columns
    scalers = {}
    for column in numeric_columns:
        scaler = StandardScaler()
        data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))
        scalers[column] = scaler

    # Normalize the date column to a fraction of the year
    data['date_normalized'] = normalize_date_to_year_fraction(data['Date'])
    
    # Drop the original date column
    data = data.drop(columns=['Date'])

    return data

if __name__ == "__main__":
    file_path = 'combined_data_all_sites.csv'
    data = pd.read_csv(file_path)
    normalize_data(file_path, data)
