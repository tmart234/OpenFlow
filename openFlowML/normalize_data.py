from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def save_normalized_data(file_path, data):
    output_file = file_path.replace('.csv', '_normalized.csv')
    data.to_csv(output_file, index=False)

def normalize_data(file_path, data):

    output_file = file_path.replace('.csv', '_normalized.csv')
    numeric_columns = ['TMIN', 'TMAX', 'Min Flow', 'Max Flow']

    # Convert empty strings to NaN and ensure columns are of type float
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    # Use 7-day rolling average to fill NaN values for specific columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].rolling(7, min_periods=1).mean())

    # Create a scaler for each column
    scalers = {}
    for column in numeric_columns:
        if data[column].isnull().any():  # If after forward fill, still have NaN values
            data[column].fillna(data[column].mean(), inplace=True)  # fill with column mean
        
        scaler = StandardScaler()
        data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))
        scalers[column] = scaler

    # Save the normalized data
    return data

if __name__ == "__main__":
    file_path = 'combined_data_all_sites.csv'
    data = pd.read_csv(file_path)
    normalize_data(file_path, data)
