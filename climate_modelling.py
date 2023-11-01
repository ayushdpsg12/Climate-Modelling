import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Load real climate data (replace with your dataset)
data_url = 'path_to_real_climate_data.csv'
df = pd.read_csv(data_url)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Visualize the time series data
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Temperature'], label='Observed Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Observed Temperature Time Series')
plt.legend()
plt.show()

# Split data into training and testing sets
train_size = int(0.8 * len(df))
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Hyperparameters for SARIMA (these need further tuning)
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)

# Fit a SARIMA model with additional features
exog_train = train_data[['Greenhouse_Gases', 'Ocean_Surface_Temperature']]
exog_test = test_data[['Greenhouse_Gases', 'Ocean_Surface_Temperature']]
model = SARIMAX(
    train_data['Temperature'], exog=exog_train,
    order=order, seasonal_order=seasonal_order,
    enforce_stationarity=False, enforce_invertibility=False
)
fitted_model = model.fit(disp=False)

# Predict using the fitted model
predictions = fitted_model.get_forecast(steps=len(test_data), exog=exog_test)
predicted_mean = predictions.predicted_mean
conf_int = predictions.conf_int()

# Calculate Mean Squared Error (MSE) for evaluation
mse = mean_squared_error(test_data['Temperature'], predicted_mean)
print(f"Mean Squared Error: {mse:.2f}")

# Visualize the predictions and confidence interval
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data['Temperature'], label='Train Temperature')
plt.plot(test_data.index, test_data['Temperature'], label='Test Temperature')
plt.plot(test_data.index, predicted_mean, label='Predictions', linestyle='dashed')
plt.fill_between(test_data.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='gray', alpha=0.2)
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Time Series Prediction with SARIMA and Additional Features')
plt.legend()
plt.show()

# Plot ACF and PACF to help with parameter selection
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(train_data['Temperature'], lags=40, ax=axes[0])
plot_pacf(train_data['Temperature'], lags=40, ax=axes[1])
plt.show()

# Print model summary and residuals analysis
print(fitted_model.summary())
plt.figure(figsize=(10, 6))
plt.plot(fitted_model.resid)
plt.title('Model Residuals')
plt.show()
