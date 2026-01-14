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
- The most important features for predicting heart disease were typically:
  - Exercise-induced angina (exang)
  - Number of major vessels colored (ca)
  - Maximum heart rate achieved (thalach)
  - ST depression induced by exercise (oldpeak)
  - Chest pain type (cp)

## Files Included
- `Task3_Heart_Disease_Prediction.ipynb`: Main Jupyter notebook with complete implementation
- `heart_disease.csv`: Dataset used for training
- `figures/`: Directory containing generated visualizations

## Performance Metrics
- **Accuracy**: Typically 80-85% depending on the model
- **ROC AUC**: Usually above 0.85 indicating good discrimination ability

## Important Disclaimer
This model is designed for educational purposes only and should not be used for actual medical diagnosis. Always consult healthcare professionals for proper medical assessment and diagnosis.