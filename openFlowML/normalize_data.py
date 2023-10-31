import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_data(input_file_path, output_file_path):
    # Read the CSV
    data = pd.read_csv(input_file_path)

    # Columns to normalize - temperature and flow data
    columns_to_normalize = ['TMIN', 'TMAX', 'FlowMin', 'FlowMax']

    # Create a scaler for each column
    scalers = {}
    for column in columns_to_normalize:
        scaler = StandardScaler()
        data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))
        scalers[column] = scaler

    # Save the normalized data
    data.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    INPUT_PATH = 'combined_data.csv'
    OUTPUT_PATH = INPUT_PATH.replace('.csv', '_normalized.csv')
    normalize_data(INPUT_PATH, OUTPUT_PATH)
