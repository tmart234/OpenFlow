import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_dataset(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Load data from the CSV file
data = pd.read_csv("combined_data.csv")

# Feature columns
X = data[["Min_Temp", "Max_Temp", "Average", "SWE"]].values

# Target column
y = data["Average"].shift(-1).dropna().values

# Normalize the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Drop the last row from the feature matrix since there's no target for it
X = X[:-1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create time series datasets
time_steps = 3
X_train, y_train = create_dataset(X_train, y_train, time_steps)
X_test, y_test = create_dataset(X_test, y_test, time_steps)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, activation="relu", input_shape=(time_steps, X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

# Train the LSTM model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Evaluate the model using the testing set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error:", mse)
print("R2 score:", r2)

# Predict the flow value for a certain date using the model
# Replace these values with the predicted date's predicted temperature and the current date's flows and SWE value
min_temp = 15.0
max_temp = 25.0
current_average = 750.0
swe = 150.0
input_data = scaler.transform([[min_temp, max_temp, current_average, swe]])
input_data_series = np.array([input_data[-time_steps:]])
flow_prediction = model.predict(input_data_series)
print("Predicted flow value for the given date:", flow_prediction[0][0])
