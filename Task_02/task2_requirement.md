IMPLEMENT TASK 2: Short-Term Stock Price Prediction

Phase: Implementation (Specification complete)

Goal:
Create a complete Jupyter Notebook named `Task2_Stock_Price_Prediction.ipynb` that predicts the next day’s closing stock price using historical data from Yahoo Finance.

Requirements (strict):

1. Libraries:
   - pandas, numpy, matplotlib, seaborn
   - scikit-learn
   - yfinance

2. Stock Selection:
   - Use one stock ticker (AAPL or TSLA).
   - Fetch at least 2–5 years of daily data.

3. Data Processing:
   - Display `.head()`, `.info()`, `.describe()`.
   - Create target column by shifting Close price by −1 day.
   - Drop rows with NaN target.

4. Features & Target:
   - Features: Open, High, Low, Volume
   - Target: Next day Close price

5. Train-Test Split:
   - Use chronological split (no shuffle).
   - 80% training, 20% testing.

6. Model Training:
   - Train Linear Regression or Random Forest Regressor.
   - Explain model choice in Markdown.

7. Evaluation:
   - Compute MAE and RMSE.
   - Print results clearly.

8. Visualization:
   - Plot Actual vs Predicted Close prices.
   - Show plot inline and save to `./figures/`.

9. Code Quality:
   - Modular code
   - Clear comments
   - Markdown explanations before major sections

10. Deliverables:
   - Notebook content
   - `task2_README.md`
   - Confirmation of saved figure

Produce the full notebook content and README text. End with a brief summary of results and observations.
