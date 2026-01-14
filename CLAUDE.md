## DevelopersHub AI/ML Engineering Internship

This document defines the permanent operating rules for Code while working on the DevelopersHub AI/ML Engineering Internship tasks.Complete all tasks step by step without skipping any required part.

---

## 1. Core Operating Principles

- Follow the internship PDF as the single source of truth.
- Do not invent or assume requirements outside the PDF.
- Work on **one task at a time**.
- Every task must be completed in **three phases**:
  1. Specification
  2. Implementation
  3. Explanation & Results
- Do not rush or compress steps.
- Assume all work will be reviewed by internship evaluators.

---

## 2. Mandatory Workflow (Step-by-Step)

For every task, must follow this order strictly:

### Phase 1: Specification

- Clearly restate the task objective.
- Define dataset(s) to be used.
- List tools, libraries, and models.
- Define evaluation metrics.
- List required outputs (plots, metrics, explanations).

### Phase 2: Implementation

- Write clean, runnable Python code.
- Use modular sections or functions.
- Comment each major step.
- Follow standard ML best practices.

### Phase 3: Explanation & Insights

- Explain what was done.
- Explain why each step was required.
- Interpret results in simple technical language.
- Mention limitations or improvements if applicable.

must **not** move to implementation without completing Phase 1.

---

## 3. Code Quality Rules

- Code must be:
  - Clean
  - Modular
  - Well-commented
  - Easy to read
- Use meaningful variable and function names.
- Avoid unnecessary complexity.
- Code must run in a Jupyter Notebook without modification.

---

## 4. Notebook & Markdown Structure

Each task notebook must include:

1. Title & Objective
2. Dataset Description
3. Data Loading
4. Data Exploration / Preprocessing
5. Visualization
6. Model Training
7. Model Evaluation
8. Results & Final Insights

Use Markdown headings before each section.

Use fenced code blocks:

``python
import pandas as pd

---

## 5. Library Usage Policy

Libraries must be chosen strictly based on task requirements:

1. pandas, numpy → data handling

2. matplotlib, seaborn → visualization

3. scikit-learn → machine learning

4. yfinance → stock market data

5. transformers → LLMs & fine-tuning

6. torch → deep learning backend

7. streamlit → chatbot interfaces

Do not introduce unnecessary libraries.

---

## 6. Model Training & Evaluation Rules

Always split data into training and testing sets.

Use correct metrics:

1. Classification → Accuracy, Confusion Matrix, ROC-AUC

2. Regression → MAE, RMSE

3. Plot predicted vs actual values when required.

4. Label all plots clearly.

Explain metrics in simple language.

---

7. Task-Specific Rules

- Task 1: Exploring & Visualizing a Dataset

  - Dataset: Iris

  - Mandatory checks:

  - .shape, .columns, .head()

  - .info(), .describe()

  - Mandatory plots:

  - Scatter plots

  - Histograms

  - Box plots

  - Libraries: pandas, seaborn, matplotlib only

- Task 2: Stock Price Prediction

- Data source: Yahoo Finance (yfinance)

- Features: Open, High, Low, Volume

- Target: Next-day Close price

- Models: Linear Regression or Random Forest

- Plot actual vs predicted Close prices

- Task 3: Heart Disease Prediction

- Dataset: UCI Heart Disease

- Handle missing values properly.

- Perform EDA.

- Models: Logistic Regression or Decision Tree

- Metrics:

- Accuracy

- Confusion Matrix

- ROC Curve

Identify important features.

- Task 4: General Health Query Chatbot

- Use prompt engineering.

- Act as a helpful medical assistant.

- Avoid diagnosis or prescriptions.

- Add safety disclaimers when needed.

- Task 5: Mental Health Support Chatbot

- Base model: DistilGPT2 / GPT-Neo / Mistral

- Dataset: EmpatheticDialogues

- Fine-tuning via Hugging Face Trainer API

- Responses must be empathetic and safe.

- Interface: CLI or Streamlit

- Task 6: House Price Prediction

- Preprocess numerical and categorical features.

- Models: Linear Regression or Gradient Boosting

- Metrics:

- MAE

- RMSE

Visualize predicted vs actual prices.

---

8. Explanation Rules

Always explain:

1. What was done

2. Why it was done

3. What the results mean

4. Keep explanations clear and technical.

5. Assume the reviewer is an internship evaluator.

---

9. Submission Readiness

1. Code must be GitHub-ready.

1. Notebook must be submission-ready.

1. Follow the internship checklist strictly.

1. No missing sections allowed.

---

10. Safety & Failure Prevention

1. Ask before implementing if something is unclear.

1. Do not skip plots, metrics, or explanations.

1. Do not hallucinate results.

1. Do not exceed task scope unnecessarily.
