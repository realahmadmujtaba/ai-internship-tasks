# Task 2: End-to-End ML Pipeline for Customer Churn Prediction

## Objective

Build a **production-ready machine learning pipeline** to predict customer churn (whether a customer will leave a service). The pipeline combines data preprocessing and model training into a single, reusable object.

## Dataset

**Telco Customer Churn Dataset** (IBM Dataset)
- **Total samples:** ~7,000 customers
- **Features:** 19 customer attributes
- Demographic: Age, Gender, Partner, Dependents
- Services: Phone, Internet, Online Security, Tech Support, etc.
- Account: Tenure, Contract Type, Billing Method
- Financial: Monthly Charges, Total Charges
- **Target:** Churn (Yes/No)
- **Churn rate:** ~26.5% (imbalanced dataset)

## Methodology & Approach

### 1. **Data Exploration (EDA)**
- Analyze distribution of features
- Identify relationships with churn
- Check for missing values
- Visualize churn patterns by contract type, service type, etc.

### 2. **Data Preprocessing**
- **Binary encoding:** Convert Yes/No to 0/1
- **Label encoding:** Binary categorical variables
- **One-hot encoding:** Multiclass categorical variables
- **Scaling:** Standardize numeric features to 0-mean, unit-variance

### 3. **Pipeline Architecture**
The ML pipeline consists of:
```
Input Data
↓
[Preprocessing Step]
StandardScaler (numeric features)
LabelEncoder (binary categorical)
OneHotEncoder (multiclass categorical)
↓
[Model Training Step]
LogisticRegression
RandomForestClassifier
↓
Output Prediction
```

#

## 4. **Hyperparameter Tuning**
- **GridSearchCV:** Exhaustive search over parameter grid
- **Cross-validation:** 5-fold to prevent overfitting
- **Scoring metric:** F1-score (better for imbalanced data)
- **Parallelization:** n_jobs=-1 for faster computation

### 5. **Model Comparison**
- **Logistic Regression (baseline):** Simple, interpretable
- **Logistic Regression (tuned):** Optimized regularization
- **Random Forest (baseline):** Ensemble, handles non-linearity
- **Random Forest (tuned):** Optimized depth and splitting

### 6. **Evaluation Metrics**
- **Accuracy:** Overall correctness (misleading for imbalanced data)
- **Precision:** True positives / All predicted positives
- **Recall:** True positives / All actual positives
- **F1-Score:** Harmonic mean of Precision and Recall
- **ROC-AUC:** Area under ROC curve (threshold-independent)

## Key Results

| Model | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|-------|----------|----------|-----------|--------|---------|
| LR (Baseline) | 80.2% | 0.642 | 0.62 | 0.67 | 0.84 |
| LR (Tuned) | 80.5% | 0.651 | 0.64 | 0.67 | 0.85 |
| RF (Baseline) | 81.1% | 0.660 | 0.65 | 0.67 | 0.85 |
| **RF (Tuned)** | **81.8%** | **0.672** | **0.66** | **0.69** | **0.86** |

*Note: Results vary based on data sample and random seed*

## Key Learnings

1. **Pipelines prevent data leakage:** Preprocessing is fitted only on training data
2. **Imbalanced data:** Standard accuracy is misleading; use F1-score or ROC-AUC
3. **Hyperparameter tuning:** Small improvements from tuning (1-2%)
4. **Feature scaling importance:** Tree models don't need scaling, but linear models do
5. **Production readiness:** Model must be saved along with preprocessing objects

## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

#

## Run the Script
```bash
python task2_ml_pipeline.py
```

#

## Steps Executed
1. Load and explore Telco Churn dataset
2. Preprocess and encode features
3. Split data (80% train, 20% test)
4. Build and train baseline models
5. Evaluate baseline performance
6. Perform hyperparameter tuning with GridSearchCV
7. Evaluate tuned models
8. Compare and select best model
9. Analyze feature importance
10. Save pipeline to production
11. Create inference function

### Outputs
- `churn_prediction_pipeline.pkl` - Trained pipeline (ready for production)
- `preprocessing_info.pkl` - Preprocessing metadata
- `task2_eda.png` - Exploratory data analysis visualizations
- `task2_model_comparison.png` - Model performance comparison
- `task2_feature_importance.png` - Feature importance ranking

## Visualizations Generated

1. **EDA Visualizations:**
- Churn distribution
- Churn by contract type
- Churn by internet service
- Monthly charges comparison

2. **Model Comparison:**
- Accuracy comparison across models
- F1-score comparison
- Confusion matrix for best model
- ROC curve with AUC score

3. **Feature Analysis:**
- Top 10 most important features
- Importance magnitude

## Production Deployment

### Load and Use Saved Pipeline
```python
import joblib# Load pipeline
pipeline = joblib.load('churn_prediction_pipeline.pkl')
preprocessing_info = joblib.load('preprocessing_info.pkl')# New customer data
new_customer = {
'SeniorCitizen': 0,
'Tenure': 24,
'MonthlyCharges': 65.5,
'TotalCharges': 1570.0,
'PhoneService': 'Yes',
'InternetService': 'Fiber optic',
'Contract': 'Month-to-month',# ... other features
}# Preprocess and predict
customer_df = pd.DataFrame([new_customer])# ... apply preprocessing ...
prediction = pipeline.predict(customer_df)[0]
probability = pipeline.predict_proba(customer_df)[0][1]

print(f"Churn Prediction: {prediction}")# 0 or 1
print(f"Churn Probability: {probability:.2%}")
```

#

## REST API Example
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
pipeline = joblib.load('churn_prediction_pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
data = request.json
customer_df = pd.DataFrame([data])
prediction = pipeline.predict(customer_df)[0]
probability = pipeline.predict_proba(customer_df)[0][1]

return jsonify({
'prediction': int(prediction),
'probability': float(probability)
})

if __name__ == '__main__':
app.run(debug=True)
```

## Skills Gained

- Data preprocessing and feature engineering
- Machine learning pipeline construction
- Hyperparameter tuning with GridSearchCV
- Handling imbalanced datasets
- Model evaluation and comparison
- Feature importance analysis
- Production-ready model export

## Code Structure

```
task2_ml_pipeline.py
Step 1: Import Libraries
Step 2: Load and Explore Dataset
Step 3: Data Exploration & Visualization
Step 4: Data Preprocessing
Step 5: Train/Test Split
Step 6: Create ML Pipeline
Step 7: Train Baseline Models
Step 8: Evaluate Baseline
Step 9: Hyperparameter Tuning
Step 10: Compare Models
Step 11: Visualizations
Step 12: Feature Importance
Step 13: Save Pipeline
Step 14: Inference Function
Summary
```

## Pipeline Configuration Details

### Logistic Regression Grid
```python
'classifier__C': [0.001, 0.01, 0.1, 1, 10]
'classifier__penalty': ['l2']
'classifier__solver': ['lbfgs']
```

#

## Random Forest Grid
```python
'classifier__n_estimators': [50, 100, 200]
'classifier__max_depth': [5, 10, 15, 20]
'classifier__min_samples_split': [5, 10]
'classifier__min_samples_leaf': [2, 4]
```

## Key Concepts

### What is a Pipeline?
A pipeline chains multiple steps (preprocessing, modeling) into a single object:
- **Prevents data leakage:** Preprocessing fit only on training data
- **Reusable:** Same pipeline for training and prediction
- **Production-ready:** Single object to deploy

### Why GridSearchCV?
- **Systematic search:** Tests all parameter combinations
- **Cross-validation:** Estimates performance on unseen data
- **Parallel processing:** Uses multiple CPU cores
- **Best parameters:** Automatically selects optimal configuration

### Why F1-Score for Imbalanced Data?
- **Accuracy is misleading:** With 26% churn, predicting "no churn" for all gives 74% accuracy
- **F1-Score:** Combines precision and recall, penalizes both false positives and false negatives
- **Business relevance:** Captures both types of errors

## Troubleshooting

**Issue:** GridSearchCV taking too long
- Solution: Reduce parameter grid or use RandomizedSearchCV

**Issue:** Memory error
- Solution: Use smaller dataset or n_jobs=1

**Issue:** Model performs worse on new data
- Solution: Check for data drift or different distribution

## References

- [Scikit-learn Pipeline Documentation](https://scikit-learn.org/stable/modules/compose.html)
- [GridSearchCV Guide](https://scikit-learn.org/stable/modules/grid_search.html)
- [Telco Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- [ML Imbalanced Data Handling](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-machine-learning/)

## Author
AI/ML Internship - DevelopersHub Corporation

## License
Open source - Feel free to modify and use
