# Task 2: Short-Term Stock Price Prediction

## Objective
This task involved implementing a machine learning model to predict the next day's closing stock price using historical data from Yahoo Finance. The goal was to use features like Open, High, Low, and Volume to predict the next day's Close price, demonstrating proficiency in time series forecasting and regression techniques.

## Dataset
The dataset was obtained from Yahoo Finance using the yfinance library, specifically for the AAPL (Apple Inc.) stock ticker. The model used 3 years of daily stock data, incorporating features including opening price, highest price, lowest price, and trading volume to predict the subsequent day's closing price.

## Key Findings
Linear Regression outperformed Random Forest on this task (MAE $4.14 vs $15.53) — with a chronological
train/test split and only Open/High/Low/Volume as features, the linear model generalizes better than
the more complex Random Forest, which overfits the training period. `High` was the most influential
feature for predicting next-day Close, by coefficient magnitude.

## Performance Metrics
(from a live run against real AAPL data pulled via yfinance)

| Model | MAE | RMSE |
| --- | --- | --- |
| Linear Regression (selected) | $4.14 | $5.68 |
| Random Forest | $15.53 | $21.59 |

MAE as a percentage of the average test-set closing price ($285.34): **1.45%**.

## Files Produced
- `Task2_Stock_Price_Prediction.ipynb`: Complete Jupyter Notebook with all analysis, model training, and visualizations
- `./figures/`: Directory containing generated plots (actual vs. predicted prices, model coefficients)