import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import combine_data
import tf2onnx

def save_as_onnx(model):
    # Convert Keras model to ONNX format
    onnx_model, _ = tf2onnx.convert.from_keras(model)

    # Save the ONNX model to a file
    onnx_model_path = 'lstm_model.onnx'
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

# use the past 60 days to make prediction for next 14 days
def reshape_data_for_lstm(data, timesteps=60, forecast_horizon=14):
    X, y_min, y_max = [], [], []
    
    for i in range(len(data) - timesteps - forecast_horizon + 1):
        X.append(data.iloc[i:i+timesteps].values)  # This assumes you're using all columns as features
        
        y_min_flow = data.iloc[i+timesteps:i+timesteps+forecast_horizon]["Min Flow"].values
        y_max_flow = data.iloc[i+timesteps:i+timesteps+forecast_horizon]["Max Flow"].values
        
        y_min.append(y_min_flow)
        y_max.append(y_max_flow)

    # Stacking y_min and y_max to have a shape: [samples, forecast_horizon, 2]
    y = np.stack([y_min, y_max], axis=-1)
    
    return np.array(X), y


def build_lstm_model(input_shape, output_shape):
    model = Sequential()

    # Add the first LSTM layer
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Add a second LSTM layer
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))

    # Add Dense layer for prediction
    model.add(Dense(output_shape))

    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_model(data, epochs=10, batch_size=32, validation_split=0.2):
    X, Y = reshape_data_for_lstm(data)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=validation_split)

    # Reshape data for LSTM
    # LSTM expects input shape as (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

    return model, history

if __name__ == '__main__':
    data = combine_data.main()
    if isinstance(data, str):
        print("Error: Data is recognized as string. Expected pandas DataFrame.")
        exit(1)  # Exit the script
    model, history = train_lstm_model(data)
    save_as_onnx(model)