import yfinance as yf
import numpy as np
import pandas as pd
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load Stock Data
def get_stock_data(stock_symbol, start_date="2010-01-01", end_date="2024-01-01"):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# Add Technical Indicators
def add_indicators(data):
    data = data.copy()  # Ensure the original dataframe is not modified

    # Ensure 'Close' column is treated as a Series
    close_prices = data['Close'].squeeze()  # Convert 2D array to 1D if needed

    # Calculate indicators correctly
    data['SMA'] = SMAIndicator(close_prices, window=14).sma_indicator().astype(float)
    data['RSI'] = RSIIndicator(close_prices, window=14).rsi().astype(float)
    data['MACD'] = MACD(close_prices).macd().astype(float)

    return data.dropna()  # Drop NaN values to avoid errors

# Prepare Data for LSTM
def prepare_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close', 'SMA', 'RSI', 'MACD']])
    return scaled_data, scaler

# Create LSTM Dataset
def create_lstm_dataset(data, timesteps=10):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(data[i+timesteps, 0])
    return np.array(X), np.array(y)

# Train LSTM Model
def train_model():
    sp500 = get_stock_data("^GSPC")
    sp500 = add_indicators(sp500)
    sp500_scaled, scaler = prepare_data(sp500)
    X_train, y_train = create_lstm_dataset(sp500_scaled)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    model.save("stock_model.h5")
    np.save("scaler.npy", scaler)

# Train and Save Model
if __name__ == "__main__":
    train_model()