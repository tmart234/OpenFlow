from darts.models import BlockRNNModel, AutoARIMA
from sklearn.model_selection import train_test_split
import combine_data
import numpy as np
import pandas as pd
import logging
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

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
        input_chunk_length=input_shape[0],  # Define appropriate chunk length
        output_chunk_length=input_shape[1],  # Define appropriate output length
        n_rnn_layers=2,
        hidden_dim=50,
        dropout=0.2,
        n_epochs=10,
        optimizer_kwargs={'lr': 1e-3},
        random_state=42
    )
    return model

def prepare_data(data):
    scaler = Scaler()
    series = TimeSeries.from_dataframe(data)
    series = scaler.fit_transform(series)
    return series, scaler

def main():
    data = combine_data.main(training_num_years=5)  # Assuming combine_data.main() returns a DataFrame

    if data is None or data.empty:
        raise ValueError("Error: Data is not available or not in expected format.")

    # Assuming 'flow' and 'temperature' are columns in your data
    target_columns = ['flow']
    covariate_columns = ['temperature', 'SWE']  # Example covariate columns

    series_target, scaler = prepare_data(data[target_columns])
    series_covariates, _ = prepare_data(data[covariate_columns])

    train_target, val_target = series_target.split_before(0.8)
    train_covariates, val_covariates = series_covariates.split_before(0.8)

    # Train AutoARIMA
    arima_model = train_autoarima(train_target)

    # Train LSTM
    input_shape = (60, 1)  # Example input shape, adjust based on actual data
    lstm_model = train_lstm(input_shape)
    lstm_model.fit(train_target, past_covariates=train_covariates, verbose=True)

    # Prediction
    arima_forecast = arima_model.predict(len(val_target))
    lstm_forecast = lstm_model.predict(n=len(val_target), series=train_target, past_covariates=val_covariates)

    # Evaluate (example using MAPE, you might choose other metrics such as MSE or MAE)
    from darts.metrics import mape
    arima_mape = mape(val_target, arima_forecast)
    lstm_mape = mape(val_target, lstm_forecast)

    print(f"ARIMA MAPE: {arima_mape}")
    print(f"LSTM MAPE: {lstm_mape}")

    # Optionally: Save models
    # arima_model.save_model('arima_model.pkl')
    # lstm_model.save_model('lstm_model.pkl')

if __name__ == '__main__':
    main()