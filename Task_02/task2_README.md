# Task 2: Short-Term Stock Price Prediction

## Objective
This task involved implementing a machine learning model to predict the next day's closing stock price using historical data from Yahoo Finance. The goal was to use features like Open, High, Low, and Volume to predict the next day's Close price, demonstrating proficiency in time series forecasting and regression techniques.

## Dataset
The dataset was obtained from Yahoo Finance using the yfinance library, specifically for the AAPL (Apple Inc.) stock ticker. The model used 3 years of daily stock data, incorporating features including opening price, highest price, lowest price, and trading volume to predict the subsequent day's closing price.

## Key Findings
The model implementation compared both Linear Regression and Random Forest algorithms to determine the optimal approach for stock price prediction. Performance was evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) metrics. The analysis revealed insights about which features (Open, High, Low, Volume) had the greatest influence on predicting closing prices, and demonstrated the challenges inherent in financial market prediction.

## Files Produced
- `Task2_Stock_Price_Prediction.ipynb`: Complete Jupyter Notebook with all analysis, model training, and visualizations
- `./figures/`: Directory containing generated plots including actual vs predicted price comparisons