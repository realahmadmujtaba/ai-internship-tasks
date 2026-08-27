# AI/ML Internship Tasks

Three machine learning tasks completed during the DevelopersHub AI/ML engineering internship, each as a self-contained Jupyter notebook.

## Tasks

| Task | Folder | Focus |
| --- | --- | --- |
| 01 | `Task_01/` | Exploratory data analysis and visualization of the Iris dataset |
| 02 | `Task_02/` | Next-day stock closing price prediction (AAPL, live data via yfinance) |
| 03 | `Task_03/` | Heart disease risk classification (UCI dataset) |
| Phase 2 | `Phase_2/` | BERT classifier, ML pipeline, and multimodal ML (DevelopersHub Phase 2) |

## Built with

Python, Jupyter, scikit-learn, pandas

## Run the notebooks

```bash
pip install -r requirements.txt
jupyter lab
```

## Results

All notebooks were re-run end-to-end to verify these numbers (see each task's own README for details).

| Task | Metric | Result |
| --- | --- | --- |
| 01 — Iris EDA | — | No missing values in 150 samples; petal length/width give the clearest species separation |
| 02 — Stock prediction | MAE / RMSE (Linear Regression) | $4.14 / $5.68 (1.45% of average closing price) |
| 03 — Heart disease | Accuracy / ROC AUC (Logistic Regression) | 86.89% / 0.9513 |

## What I learned

Re-running Task 3's notebook surfaced a real bug: it filled missing values with
`df[column].fillna(..., inplace=True)` on a chained column selection, which silently does nothing
under pandas' copy-on-write behavior — the "cleaned" dataframe still had NaNs, and model training
crashed downstream. Fixed by reassigning (`df[column] = df[column].fillna(...)`) instead of relying
on `inplace=True` through chained indexing — a good reminder to always verify a "fix" actually
changed the data rather than trusting that the code ran without an error.
