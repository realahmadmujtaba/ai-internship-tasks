# Task 3: Heart Disease Prediction

## Overview
This project implements a machine learning model to predict heart disease risk using the UCI Heart Disease dataset. The notebook performs comprehensive data analysis, preprocessing, model training, and evaluation.

## Dataset
- **Source**: UCI Machine Learning Repository - Heart Disease Dataset
- **Size**: 303 samples with 14 features
- **Target**: Binary classification (0 = No disease, 1 = Disease)

## Features
The dataset includes the following clinical attributes:
- age: Patient age
- sex: Gender (0 = female, 1 = male)
- cp: Chest pain type
- trestbps: Resting blood pressure
- chol: Cholesterol measurement
- fbs: Fasting blood sugar
- restecg: Resting electrocardiographic results
- thalach: Maximum heart rate achieved
- exang: Exercise-induced angina
- oldpeak: ST depression induced by exercise
- slope: Slope of the peak exercise ST segment
- ca: Number of major vessels colored by fluoroscopy
- thal: Thalassemia (blood disorder)

## Methodology
1. **Data Loading**: Load and examine the dataset structure
2. **Data Cleaning**: Handle missing values and prepare data
3. **Exploratory Data Analysis**: Visualize distributions and correlations
4. **Feature Engineering**: Prepare features for modeling
5. **Model Training**: Train Logistic Regression and Decision Tree models
6. **Evaluation**: Assess performance using accuracy, confusion matrix, and ROC-AUC
7. **Feature Importance**: Identify key predictors of heart disease

## Models Used
- **Logistic Regression**: Interpretable model for binary classification
- **Decision Tree**: Non-linear model for comparison

## Key Findings
- Top predictors by Logistic Regression coefficient: `ca` (number of major vessels), `thal`
  (thalassemia), `sex`, `cp` (chest pain type), `exang` (exercise-induced angina).
- Top predictors by Decision Tree importance: `thal`, `cp`, `ca`, `age`, `chol`.

## Files Included
- `Task3_Heart_Disease_Prediction.ipynb`: Main Jupyter notebook with complete implementation
- `run_heart_disease_prediction.py`: Standalone script version (same pipeline, runnable end-to-end)
- `heart_disease.csv`: Dataset used for training
- `figures/`: Directory containing generated visualizations

## Performance Metrics
(from `run_heart_disease_prediction.py`, 242/61 train/test split, verified by running the script)

| Model | Accuracy | ROC AUC |
| --- | --- | --- |
| Logistic Regression | 86.89% | 0.9513 |
| Decision Tree | 78.69% | 0.8047 |

## Important Disclaimer
This model is designed for educational purposes only and should not be used for actual medical diagnosis. Always consult healthcare professionals for proper medical assessment and diagnosis.