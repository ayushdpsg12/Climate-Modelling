#Climate Change Prediction with SARIMA Model

This project aims to predict climate change trends using the SARIMA (Seasonal Autoregressive Integrated Moving Average) model. 
The code leverages real-world data on temperature, greenhouse gases, and ocean surface temperature to make predictions about future temperature changes.

## Getting Started

To run the code, you'll need Python and the necessary libraries installed. You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn

## Project Overview

- **Data**: The project uses real climate data with temperature, greenhouse gases, and ocean surface temperature as key variables. The data is structured as a time series, with each data point associated with a specific date.

- **Data Visualization**: The project begins with data visualization to better understand how temperature, greenhouse gases, and ocean surface temperature change over time. This helps identify trends and patterns in the dataset.

- **SARIMA Model**: SARIMA, a sophisticated time series analysis technique, is the core of this project. It takes historical temperature data and exogenous variables into account to predict future temperature changes. The SARIMA model captures patterns, trends, and seasonality to make accurate predictions.

- **Model Parameters**: Selecting the right model parameters is crucial. The code uses autocorrelation and partial autocorrelation analysis to determine the best settings for the SARIMA model, such as autoregressive, differencing, and moving average orders.

- **Exogenous Features**: Exogenous features, including greenhouse gases and ocean surface temperature, are incorporated to enhance the model's predictive capabilities. These features capture the potential impact of external factors on temperature changes.

- **Model Fitting and Forecasting**: The SARIMA model is trained using historical data, and then it is used to make forecasts for future temperature values based on testing data.

- **Model Evaluation**: The Mean Squared Error (MSE) is calculated to assess the model's predictive accuracy. A lower MSE indicates a better fit between predicted and actual temperature values.

- **Visualization of Predictions**: The code generates graphs to visualize observed and predicted temperature values, along with confidence intervals. Confidence intervals provide a range of possible values for the predictions.
