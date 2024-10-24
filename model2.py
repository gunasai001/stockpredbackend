import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import load_model
from data import NSE

# Load the trained model and scaler
model = load_model('stock_price_prediction.keras')
scaler = MinMaxScaler(feature_range=(0, 1))
encoder = OneHotEncoder(sparse_output=False)

# Fit the encoder and scaler once
df = pd.read_csv('all_stocks_historical_data.csv')
df.columns = df.columns.str.strip()  # Strip any whitespace from column names

# Normalize features
features = ['open', 'high', 'low', 'close', 'VOLUME']
df_features = df[features].values
scaler.fit(df_features)

# One-hot encode the stock symbols
stock_symbols_encoded = encoder.fit_transform(df[['symbol']].values)

# Function to get historical data for a symbol
def get_historical_data(symbol, time_step=100):
    data = NSE()
    stock_data = data.get_past_7_days_data(symbol)
    features = ['open', 'high', 'low', 'close', 'VOLUME']
    
    if len(stock_data) < time_step:
        raise ValueError(f"Not enough historical data for {symbol}. Need at least {time_step} days, but only found {len(stock_data)}.")
    
    last_data = stock_data[features].values[-time_step:]
    last_data_scaled = scaler.transform(last_data)
    
    return last_data_scaled

# Function to make predictions for a given stock symbol
def predict_stock_prices(symbol, model, encoder, scaler, time_step=100, future_days=7):
    try:
        historical_data = get_historical_data(symbol, time_step)
        price_placeholder = np.reshape(historical_data, (1, time_step, len(features)))
        
        # One-hot encode the stock symbol for prediction
        symbol_placeholder = encoder.transform(np.array([[symbol]]))
        
        # Make predictions
        predicted_prices = model.predict([price_placeholder, symbol_placeholder])
        
        # Create a dummy array with the correct shape for inverse transformation
        dummy_array = np.zeros((future_days, len(features)))
        dummy_array[:, 3] = predicted_prices[0]  # Assuming 'close' is at index 3
        
        # Inverse transform the dummy array
        predicted_prices_transformed = scaler.inverse_transform(dummy_array)
        
        # Extract only the 'close' prices
        predicted_close_prices = predicted_prices_transformed[:, 3]
        
        return predicted_close_prices
    except Exception as e:
        print(f"Error predicting for {symbol}: {str(e)}")
        return None
