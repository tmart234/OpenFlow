import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Input, Concatenate, LSTM, Dropout, Dense
from darts.models import BlockRNNModel, AutoARIMA
from sklearn.model_selection import train_test_split
import combine_data
import numpy as np
import pandas as pd
import logging

"""
TODO:
1) remove station ID one-hot encoding and add embedding layers to transform each station ID into a dense vector
    - dense vectors can be input into further layers to process the temporal and other contextual features like SWE and temperature
2) Zero-shot Learning for generalization of station IDs
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_as_h5(model, output_filename="lstm_model.h5"):
    # Save the Keras model to HDF5 format
    model.save(output_filename)

# make mix/max flow prediction for next 14 days, 
# take in a window of 14 day future temps, 
# and take in window of 60 days of past flow data
# with station id
# output a flattened prediction of shape [?, forecast_horizon * 2]
# then immediately reshaped it to the desired [?, forecast_horizon, 2].
# Reshape the data to be suitable for LSTM training

def train_autoarima(train_series):
    model = AutoARIMA(seasonal=True, suppress_warnings=True)
    model.fit(train_series)
    return model

def train_lstm(input_shape):
    model = BlockRNNModel(
        model='LSTM',
        input_chunk_length=input_shape[0],
        output_chunk_length=input_shape[1],
        n_rnn_layers=2,
        hidden_dim=50,
        dropout=0.2,
        n_epochs=10,
        optimizer_kwargs={'lr': 1e-3},
        random_state=42
    )
    return model

def preprocess_data(data, target_columns, past_covariate_columns):
    scaler = Scaler()
    series_target = TimeSeries.from_dataframe(data[target_columns])
    series_past_covariates = TimeSeries.from_dataframe(data[past_covariate_columns])
    series_target = scaler.fit_transform(series_target)
    series_past_covariates = scaler.transform(series_past_covariates)
    return series_target, series_past_covariates

def main():
    data = pd.read_csv("your_data.csv")  # Example: load your data here
    # Split data into features and targets
    features = ["Min Flow", "Max Flow", "TMIN", "TMAX", "station_id"]
    target = ["Min Flow", "Max Flow"]
    data_features, data_target = data[features], data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, test_size=0.2, random_state=42)
    
    # Prepare the data for Darts models
    train_series_target, train_series_covariates = preprocess_data(X_train, target, features)
    test_series_target, test_series_covariates = preprocess_data(X_test, target, features)
    
    # Train models
    arima_model = train_autoarima(train_series_target)
    lstm_input_shape = (len(train_series_covariates), len(target))  # Hypothetical shape
    lstm_model = train_lstm(lstm_input_shape)
    
    # Fit LSTM model (note: we fit directly on TimeSeries objects)
    lstm_model.fit(series=train_series_target, past_covariates=train_series_covariates, verbose=True)
    
    # Predictions
    arima_pred = arima_model.predict(len(test_series_target))
    lstm_pred = lstm_model.predict(n=len(test_series_target), series=train_series_target, past_covariates=test_series_covariates)
    
    # Evaluate models (just a print for now, adjust with actual evaluation metrics)
    print("ARIMA predictions:", arima_pred.values())
    print("LSTM predictions:", lstm_pred.values())

if __name__ == '__main__':
    main()