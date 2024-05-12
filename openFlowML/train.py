import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Input, Concatenate, LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split
import combine_data
import numpy as np
import pandas as pd
import logging
#import tf2onnx

"""
TODO:
1) add embedding layers to transform each station ID into a dense vector
    - dense vectors can be input into further layers to process the temporal and other contextual features like SWE and temperature
2) Zero-shot Learning for generalization of station IDs
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_as_h5(model, output_filename="lstm_model.h5"):
    # Save the Keras model to HDF5 format
    model.save(output_filename)

def reshape_data_for_lstm(data, historical_flow_timesteps=60, forecast_temperature_timesteps=14, forecast_flow_timesteps=14):
    X, Y = [], []
    required_columns = ["Min Flow", "Max Flow", "TMIN", "TMAX"]

    # Ensure required columns are present
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Data missing required columns. Available columns: {data.columns}")

    # Debugging: Print the DataFrame columns before one-hot encoding
    logging.info("DataFrame columns before one-hot encoding: %s", data.columns)

    # Assuming station columns are already one-hot encoded
    station_columns = [col for col in data.columns if col.startswith('station_')]
    feature_columns = required_columns + station_columns + ['date_normalized']

    total_required_days = historical_flow_timesteps + forecast_temperature_timesteps

    # Loop through the data to create input and output sets
    for i in range(len(data) - total_required_days - forecast_flow_timesteps + 1):
        input_features = data.iloc[i:i+total_required_days][feature_columns].values
        X.append(input_features)

        future_flow = data.iloc[i+total_required_days:i+total_required_days+forecast_flow_timesteps][["Min Flow", "Max Flow"]].values
        # Reshape future_flow to match the output shape of the model
        future_flow = future_flow.reshape(-1, forecast_flow_timesteps, 2)
        Y.append(future_flow[0])  # Assuming future_flow is not empty

    return np.array(X), np.array(Y)

# Build LSTM model
def build_lstm_model(input_shape, forecast_horizon=14):
    # Define the model
    model = Sequential()

    # Add LSTM layers
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))

    # Add Dense output layer
    model.add(Dense(forecast_horizon * 2, activation="linear"))
    model.add(tf.keras.layers.Reshape((forecast_horizon, 2)))

    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    return model

# Main function to execute the training process
def main():
    data = combine_data.main()
    if data is None or data.empty:
        raise ValueError("Error: Data is not available or not in expected format.")

    # Extract one-hot encoded station columns
    station_columns = [col for col in data.columns if col.startswith('station_')]

    # Ensure that site_id_data is no longer needed
    X, Y = reshape_data_for_lstm(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    # Adjust the input_shape to match the LSTM input requirements
    # The number of site IDs is determined by the count of one-hot encoded station columns
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # Modify the fit call (note that site_id is no longer used)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    save_as_h5(model)

if __name__ == '__main__':
    main()