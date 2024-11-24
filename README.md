# 📊 **Stock Market Analysis 📈 & Prediction using LSTM**

This project focuses on analyzing and predicting stock market trends using **Long Short-Term Memory (LSTM)** networks, a type of **Recurrent Neural Network (RNN)** that is well-suited for time-series forecasting. We will be leveraging data from **Yahoo Finance** and analyzing key metrics such as stock correlation and investment risk.

---

## 📥 **Getting the Data**

To start with, we need to load historical stock market data, and for that, we'll use the **`yfinance`** library. Yahoo Finance provides comprehensive financial data, and `yfinance` allows us to easily access this data directly into Python. 

### Install yfinance:
```bash
pip install yfinance
```

### Load Stock Market Data:
```python
import yfinance as yf

# Specify the stock ticker symbol
ticker_symbol = 'AAPL'  # Example: Apple stock

# Get the stock data from Yahoo Finance
stock_data = yf.download(ticker_symbol, start='2010-01-01', end='2023-01-01')

# Display the first few rows of the dataset
stock_data.head()
```

This will give us data like `Open`, `High`, `Low`, `Close`, `Volume`, and `Adj Close` over the specified time period.

---

## 🔍 **Exploring and Visualizing the Data**

Once we have the data, it’s important to explore and visualize it to understand the trends. For this, we'll use **Pandas**, **Matplotlib**, and **Seaborn**.

```

---

## 📊 **Measure the Correlation Between Stocks**

If you want to measure how different stocks move in relation to each other (e.g., Apple vs. Google), you can compute the **correlation** between stock prices.

### Load Data for Multiple Stocks:
```python
tickers = ['AAPL', 'GOOGL', 'MSFT']
stock_data_multiple = yf.download(tickers, start='2010-01-01', end='2023-01-01')['Close']

# Calculate the correlation matrix
correlation_matrix = stock_data_multiple.corr()

# Plot the correlation heatmap
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Stock Prices')
plt.show()
```

---

## ⚖️ **Measure the Risk of Investment**

The **risk** of investing in a stock can be measured using its **volatility**, typically represented as the **standard deviation** of stock returns. High volatility means higher risk.

### Calculate Volatility:
```python
# Calculate daily returns
stock_data['Daily Return'] = stock_data['Close'].pct_change()

# Calculate volatility (standard deviation of daily returns)
volatility = stock_data['Daily Return'].std() * (252 ** 0.5)  # Annualized volatility
print(f'Annualized Volatility of {ticker_symbol}: {volatility:.4f}')
```

---

## 📈 **Stock Price Prediction using LSTM**

LSTM models are particularly effective for time-series forecasting because they can capture long-term dependencies in sequential data. In this step, we'll build an LSTM model to predict the future stock prices.

### Prepare the Data for LSTM:
We will prepare the data by normalizing it and splitting it into training and testing sets.

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Use only the 'Close' prices for prediction
data = stock_data[['Close']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Prepare data for LSTM
X = []
y = []

# Create sequences of data to feed to the model
look_back = 60  # Number of previous days to predict the next day
for i in range(look_back, len(data_scaled)):
    X.append(data_scaled[i-look_back:i, 0])
    y.append(data_scaled[i, 0])

X = np.array(X)
y = np.array(y)

# Reshape data to match LSTM input
X = X.reshape(X.shape[0], X.shape[1], 1)
```

---

### Build the LSTM Model:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build the LSTM model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
```

### Train the Model:
```python
model.fit(X, y, epochs=10, batch_size=32)
```

---

### Model Evaluation and Prediction:
After training, you can evaluate the model's performance and use it to predict future stock prices.

```python
# Predict stock prices
predicted_price = model.predict(X)

# Reverse the normalization
predicted_price = scaler.inverse_transform(predicted_price)
```

---

## 📊 **Visualize Predictions vs Actual Prices**

```python
# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'][-len(predicted_price):].values, color='blue', label='Actual Price')
plt.plot(predicted_price, color='red', label='Predicted Price')
plt.title(f'{ticker_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
```

---

## 🎯 **Key Takeaways**

- **yfinance** allows easy access to historical stock market data for analysis and prediction.
- **LSTM** models are effective for time-series forecasting, particularly in predicting stock prices.
- **Exploratory Data Analysis (EDA)** is crucial for understanding stock trends, correlations, and risk levels.
- The **LSTM model** can predict future stock prices with reasonable accuracy based on historical data.
