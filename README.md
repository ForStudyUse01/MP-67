# MP-67
# Stock Market Prediction App

A simple Streamlit application that fetches historical stock data and predicts future stock prices.

## Features

- View historical stock prices with interactive charts
- Predict future stock prices using linear regression
- Select different time periods for analysis
- Choose number of days for prediction
- Real-time stock data using Yahoo Finance

## Requirements

The application requires the following Python libraries:
- streamlit
- pandas
- numpy
- yfinance
- scikit-learn
- plotly

## How to Run

1. Make sure you have all the required libraries installed
2. Open a terminal in the project directory
3. Run the following command:

```
streamlit run main.py
```

4. The application will open in your default web browser

## How to Use

1. Enter a stock symbol in the sidebar (e.g., AAPL for Apple, MSFT for Microsoft)
2. Select the time period for historical data
3. Adjust the number of days for prediction
4. View historical data and prediction charts in the respective tabs

## Note

This is a simple prediction model for demonstration purposes only. Real stock market prediction requires more complex models and analysis. The predictions should not be used for actual investment decisions. 
