import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from data import NSE
# Load your CSV file
df = pd.read_csv('all_stocks_historical_data.csv')

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Check unique stock symbols
print("Unique Stock Symbols:", df['symbol'].unique())

# Select relevant columns
features = ['open', 'high', 'low', 'close', 'VOLUME']
df_features = df[features].values

# Normalize the features (between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df_features)

# One-hot encode the stock symbols
encoder = OneHotEncoder(sparse_output=False)
stock_symbols_encoded = encoder.fit_transform(df[['symbol']].values)

# Prepare time-series data for training
def create_dataset(data, stock_symbols, time_step=100, future_days=7):
    X_data, y_data, symbol_data = [], [], []
    for i in range(time_step, len(data) - future_days):
        X_data.append(data[i-time_step:i])
        y_data.append(data[i:i+future_days, 3])  # Index 3 refers to 'close' prices
        symbol_data.append(stock_symbols[i])
    return np.array(X_data), np.array(y_data), np.array(symbol_data)

# Parameters
# time_step = 100  # Use the past 60 days to predict
future_days = 7  # Predict next 7 days
# X_train, y_train, stock_symbols_train = create_dataset(df_scaled, stock_symbols_encoded, time_step, future_days)

# # Reshape input for LSTM (samples, timesteps, features)
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(features)))

# # Define the model architecture
# def build_model(time_steps, num_features, num_symbols, future_days):
#     # LSTM input for price history
#     price_input = Input(shape=(time_steps, num_features))

#     # LSTM layers to process historical stock prices
#     x = LSTM(units=50, activation='relu', return_sequences=True)(price_input)
#     x = Dropout(0.2)(x)

#     x = LSTM(units=60, activation='relu', return_sequences=True)(x)
#     x = Dropout(0.3)(x)

#     x = LSTM(units=80, activation='relu', return_sequences=False)(x)
#     x = Dropout(0.4)(x)

#     # Input for stock identifier (one-hot encoded)
#     stock_id_input = Input(shape=(num_symbols,))

#     # Combine stock identifier with LSTM output
#     combined = Concatenate()([x, stock_id_input])

#     # Dense layers for future price prediction
#     y = Dense(units=512, activation='relu')(combined)
#     y = Dense(units=256, activation='relu')(y)

#     # Output layer predicting for the next 'future_days' days
#     output = Dense(units=future_days)(y)

#     # Build and compile the model
#     model = Model(inputs=[price_input, stock_id_input], outputs=output)
#     model.compile(optimizer='adam', loss='mean_squared_error')

#     return model

# # Get the number of features and symbols for model input
# num_features = X_train.shape[2]  # Number of stock-related features (Open, High, Low, etc.)
# num_symbols = stock_symbols_train.shape[1]  # Number of unique stock symbols (encoded as one-hot vectors)

# # Build the model
# model = build_model(time_steps=time_step, num_features=num_features, num_symbols=num_symbols, future_days=future_days)

# # Print the model summary
# model.summary()

# # Define early stopping callback
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # Train the model
# history = model.fit(
#     [X_train, stock_symbols_train], 
#     y_train, 
#     epochs=50, 
#     batch_size=32, # Increase epochs, early stopping will prevent overfitting
#     validation_split=0.2,  # Use 20% of data for validation
#     callbacks=[early_stopping],
#     verbose=1
# )

# # Save the trained model using the new Keras format
# model.save('stock_price_prediction.keras')
model = load_model('stock_price_prediction.keras')
# Function to get historical data for predictions
def get_historical_data(symbol, time_step=100):
    data = NSE()
    # Filter the DataFrame for the specific stock symbol
    # df = data.get_past_7_days_data(symbol)
    stock_data = data.get_past_7_days_data(symbol)
    features=['open', 'high', 'low', 'close','VOLUME ']
    # Check if we have enough data for this stock
    if len(stock_data) < time_step:
        raise ValueError(f"Not enough historical data for {symbol}. Need at least {time_step} days, but only found {len(stock_data)}.")
    
    # Select the last 'time_step' rows and return the relevant features
    last_data = stock_data[features].values[-time_step:]
    
    # Scale the data using the same scaler fitted on training data
    last_data_scaled = scaler.transform(last_data)
    
    return last_data_scaled

# Updated function to make predictions
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

# Updated example usage
symbol_to_predict = ['RELIANCE', 'TCS', 'HDFCBANK', 'BRITANNIA', 'ICICIBANK', 'SBIN', 'INFY',
                     'HINDUNILVR', 'ITC', 'LT', 'BAJFINANCE', 'ADANIENT', 'MARUTI', 'NTPC',
                     'AXISBANK', 'HCLTECH', 'TATAMOTORS', 'M&M', 'ULTRACEMCO', 'TITAN', 'ASIANPAINT',
                     'BAJAJ-AUTO', 'WIPRO', 'JSWSTEEL', 'NESTLEIND']

for symbol in symbol_to_predict:
    predicted_prices = predict_stock_prices(symbol, model, encoder, scaler)
    if predicted_prices is not None:
        print(f"\nPredicted Close Prices for {symbol} for the next {future_days} days:")
        for day, price in enumerate(predicted_prices, 1):
            print(f"Day {day}: {price:.2f}")
    else:
        print(f"\nUnable to predict prices for {symbol}")

# Plot training history
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 6))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Training History')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()