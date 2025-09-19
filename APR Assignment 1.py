# APR(CS6103) Assignment-1: Stock Price Prediction using Linear Regression
# Author: Abdul Khazmuddin
# Roll: 2511CS06
# Email: abdul_2511cs06@iitp.ac.in
# Phone: 7003532322

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

stock_name = "AAPL"
# stock data (example: Apple Inc. - AAPL) from yfinance
data = yf.download(stock_name, start="2020-01-01", end="2023-01-01")

# Feature engineering
# This will predict NEXT day's closing price using today's Open, High, Low, Volume
data['Target'] = data['Close'].shift(-1)  # next day's close
data = data.dropna()

X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Target']

# Spliting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Training Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluating model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R² Score:", r2)

# coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print("\nModel Coefficients:")
print(coef_df.head())
