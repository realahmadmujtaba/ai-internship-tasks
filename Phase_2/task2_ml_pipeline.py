"""
========================================
TASK 2: END-TO-END ML PIPELINE WITH SCIKIT-LEARN
========================================

OBJECTIVE:
Build a reusable and production-ready machine learning pipeline 
for predicting customer churn (will customer leave or stay?).

WHAT IS A PIPELINE?
A pipeline combines data preprocessing and model training into a single, 
reusable object. This ensures consistent preprocessing for both training 
and prediction, and prevents data leakage.

SKILLS GAINED:
- ML pipeline construction
- Hyperparameter tuning with GridSearchCV
- Model export and reusability
- Production-readiness practices

STEP 1: IMPORT LIBRARIES
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, confusion_matrix, roc_auc_score, roc_curve)
import joblib

print("✓ All libraries imported successfully!")


"""
STEP 2: LOAD AND EXPLORE TELCO CHURN DATASET
The Telco Churn dataset contains customer information and whether they churned
"""

print("\n" + "="*50)
print("LOADING TELCO CHURN DATASET")
print("="*50)

# Load dataset from pandas or download it
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print(f"\nDataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Display first few rows
print("\n--- First 5 Rows ---")
print(df.head())

# Data info
print("\n--- Data Types & Missing Values ---")
print(df.info())

# Statistical summary
print("\n--- Statistical Summary ---")
print(df.describe())

# Check for missing values
print("\n--- Missing Values ---")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("No missing values found!")
else:
    print(missing[missing > 0])

# Target variable distribution
print("\n--- Target Variable Distribution (Churn) ---")
churn_counts = df['Churn'].value_counts()
print(churn_counts)
print(f"\nChurn Rate: {(churn_counts['Yes'] / len(df)) * 100:.2f}%")


"""
STEP 3: DATA EXPLORATION & VISUALIZATION
Understand relationships between features and target
"""

print("\n" + "="*50)
print("DATA EXPLORATION")
print("="*50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Churn distribution
churn_counts.plot(kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Churn Distribution')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_xticklabels(['No Churn', 'Churned'], rotation=0)

# 2. Churn by Contract Type
contract_churn = pd.crosstab(df['Contract'], df['Churn'])
contract_churn.plot(kind='bar', ax=axes[0, 1], color=['#3498db', '#e74c3c'])
axes[0, 1].set_title('Churn by Contract Type')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)

# 3. Churn by Internet Service
internet_churn = pd.crosstab(df['InternetService'], df['Churn'])
internet_churn.plot(kind='bar', ax=axes[1, 0], color=['#3498db', '#e74c3c'])
axes[1, 0].set_title('Churn by Internet Service')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)

# 4. Churn by Monthly Charges
df.boxplot(column='MonthlyCharges', by='Churn', ax=axes[1, 1])
axes[1, 1].set_title('Monthly Charges by Churn')
axes[1, 1].set_xlabel('Churn')
axes[1, 1].set_ylabel('Monthly Charges')
plt.suptitle('')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/task2_eda.png', dpi=300, bbox_inches='tight')
print("✓ EDA visualizations saved as 'task2_eda.png'")
plt.show()


"""
STEP 4: DATA PREPROCESSING
Prepare data for machine learning
"""

print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

# Separate features and target
X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

# Convert target to binary (Yes -> 1, No -> 0)
y = (y == 'Yes').astype(int)

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric features ({len(numeric_features)}): {numeric_features[:5]}...")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# Handle categorical features with binary values
binary_features = [col for col in categorical_features 
                   if X[col].nunique() == 2]
multiclass_features = [col for col in categorical_features 
                       if X[col].nunique() > 2]

print(f"\nBinary categorical features: {binary_features}")
print(f"Multiclass categorical features: {multiclass_features}")

# Convert binary categorical features
le_dict = {}
for col in binary_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# One-hot encode multiclass categorical features
X = pd.get_dummies(X, columns=multiclass_features, drop_first=True)

print(f"\nAfter preprocessing:")
print(f"Features shape: {X.shape}")
print(f"Numeric features: {numeric_features}")
print(f"Categorical features (after encoding): {X.columns.difference(numeric_features).tolist()[:5]}...")


"""
STEP 5: SPLIT DATA
Divide into training and testing sets
"""

print("\n" + "="*50)
print("SPLITTING DATA")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} ({(X_train.shape[0]/len(X))*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({(X_test.shape[0]/len(X))*100:.1f}%)")

print(f"\nTraining set churn rate: {y_train.mean()*100:.2f}%")
print(f"Test set churn rate: {y_test.mean()*100:.2f}%")


"""
STEP 6: CREATE ML PIPELINE
Build a reusable pipeline with preprocessing and model
"""

print("\n" + "="*50)
print("BUILDING ML PIPELINE")
print("="*50)

# Define preprocessing steps
print("\n1. Creating ColumnTransformer for data preprocessing...")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),  # Scale numeric features
        # Categorical features are already encoded
    ],
    remainder='passthrough'  # Keep other columns as is
)

print("✓ Preprocessor created (StandardScaler for numeric features)")

# Create pipeline with Logistic Regression
print("\n2. Creating Pipeline with Logistic Regression...")

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

print("✓ Logistic Regression Pipeline created")

# Create pipeline with Random Forest
print("\n3. Creating Pipeline with Random Forest...")

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

print("✓ Random Forest Pipeline created")


"""
STEP 7: TRAIN BASELINE MODELS
Train both models without hyperparameter tuning
"""

print("\n" + "="*50)
print("TRAINING BASELINE MODELS")
print("="*50)

print("\n1. Training Logistic Regression Pipeline...")
lr_pipeline.fit(X_train, y_train)
print("✓ Logistic Regression trained")

print("\n2. Training Random Forest Pipeline...")
rf_pipeline.fit(X_train, y_train)
print("✓ Random Forest trained")


"""
STEP 8: EVALUATE BASELINE MODELS
"""

print("\n" + "="*50)
print("BASELINE MODEL EVALUATION")
print("="*50)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Evaluate model on training and test sets
    """
    print(f"\n--- {model_name} ---")
    
    # Train predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    # Test predictions
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model': model_name,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
        'predictions': y_test_pred,
        'probabilities': y_test_proba
    }
    
    print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"  Train F1-Score: {metrics['train_f1']:.4f}")
    print(f"  Test F1-Score:  {metrics['test_f1']:.4f}")
    print(f"  Precision:      {metrics['test_precision']:.4f}")
    print(f"  Recall:         {metrics['test_recall']:.4f}")
    print(f"  ROC-AUC:        {metrics['test_auc']:.4f}")
    
    return metrics

# Evaluate baseline models
lr_metrics = evaluate_model(lr_pipeline, X_train, X_test, y_train, y_test, 
                            'Logistic Regression (Baseline)')
rf_metrics = evaluate_model(rf_pipeline, X_train, X_test, y_train, y_test,
                            'Random Forest (Baseline)')


"""
STEP 9: HYPERPARAMETER TUNING WITH GRIDSEARCHCV
Find best hyperparameters for each model
"""

print("\n" + "="*50)
print("HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("="*50)

# Define hyperparameter grids
print("\n1. Tuning Logistic Regression...")

param_grid_lr = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10],  # Inverse regularization strength
    'classifier__penalty': ['l2'],
    'classifier__solver': ['lbfgs']
}

# GridSearchCV for Logistic Regression
grid_search_lr = GridSearchCV(
    lr_pipeline, 
    param_grid_lr, 
    cv=5,                    # 5-fold cross-validation
    scoring='f1',            # Optimize for F1-score
    n_jobs=-1,               # Use all available cores
    verbose=1
)

grid_search_lr.fit(X_train, y_train)

print(f"\n✓ Logistic Regression tuning complete!")
print(f"  Best parameters: {grid_search_lr.best_params_}")
print(f"  Best CV F1-Score: {grid_search_lr.best_score_:.4f}")

# Evaluate tuned LR model
lr_tuned_metrics = evaluate_model(grid_search_lr.best_estimator_, X_train, X_test, 
                                  y_train, y_test, 'Logistic Regression (Tuned)')

# Define hyperparameter grid for Random Forest
print("\n2. Tuning Random Forest...")

param_grid_rf = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, 15, 20],
    'classifier__min_samples_split': [5, 10],
    'classifier__min_samples_leaf': [2, 4]
}

# GridSearchCV for Random Forest
grid_search_rf = GridSearchCV(
    rf_pipeline, 
    param_grid_rf, 
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search_rf.fit(X_train, y_train)

print(f"\n✓ Random Forest tuning complete!")
print(f"  Best parameters: {grid_search_rf.best_params_}")
print(f"  Best CV F1-Score: {grid_search_rf.best_score_:.4f}")

# Evaluate tuned RF model
rf_tuned_metrics = evaluate_model(grid_search_rf.best_estimator_, X_train, X_test,
                                  y_train, y_test, 'Random Forest (Tuned)')


"""
STEP 10: COMPARE MODELS
"""

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

comparison_df = pd.DataFrame([
    {
        'Model': 'Logistic Regression (Baseline)',
        'Test Accuracy': lr_metrics['test_accuracy'],
        'Test F1-Score': lr_metrics['test_f1'],
        'ROC-AUC': lr_metrics['test_auc']
    },
    {
        'Model': 'Random Forest (Baseline)',
        'Test Accuracy': rf_metrics['test_accuracy'],
        'Test F1-Score': rf_metrics['test_f1'],
        'ROC-AUC': rf_metrics['test_auc']
    },
    {
        'Model': 'Logistic Regression (Tuned)',
        'Test Accuracy': lr_tuned_metrics['test_accuracy'],
        'Test F1-Score': lr_tuned_metrics['test_f1'],
        'ROC-AUC': lr_tuned_metrics['test_auc']
    },
    {
        'Model': 'Random Forest (Tuned)',
        'Test Accuracy': rf_tuned_metrics['test_accuracy'],
        'Test F1-Score': rf_tuned_metrics['test_f1'],
        'ROC-AUC': rf_tuned_metrics['test_auc']
    }
])

print("\n" + comparison_df.to_string(index=False))

# Select best model
best_model_idx = comparison_df['F1-Score'].idxmax()
best_model = comparison_df.loc[best_model_idx, 'Model']
best_f1 = comparison_df.loc[best_model_idx, 'Test F1-Score']

print(f"\n{'='*50}")
print(f"✓ BEST MODEL: {best_model}")
print(f"  F1-Score: {best_f1:.4f}")
print(f"{'='*50}")


"""
STEP 11: VISUALIZATIONS
"""

print("\n" + "="*50)
print("GENERATING VISUALIZATIONS")
print("="*50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Model Comparison - Accuracy
models = comparison_df['Model'].tolist()
accuracies = comparison_df['Test Accuracy'].tolist()
colors = ['#3498db', '#3498db', '#2ecc71', '#2ecc71']
axes[0, 0].barh(models, accuracies, color=colors)
axes[0, 0].set_title('Test Accuracy Comparison')
axes[0, 0].set_xlim([0.75, 0.85])
for i, v in enumerate(accuracies):
    axes[0, 0].text(v - 0.005, i, f'{v:.4f}', va='center', ha='right', fontweight='bold')

# 2. Model Comparison - F1-Score
f1_scores = comparison_df['Test F1-Score'].tolist()
axes[0, 1].barh(models, f1_scores, color=colors)
axes[0, 1].set_title('Test F1-Score Comparison')
axes[0, 1].set_xlim([0.5, 0.7])
for i, v in enumerate(f1_scores):
    axes[0, 1].text(v - 0.01, i, f'{v:.4f}', va='center', ha='right', fontweight='bold')

# 3. Confusion Matrix for Best Model
best_predictions = rf_tuned_metrics['predictions']
cm = confusion_matrix(y_test, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'],
            ax=axes[1, 0], cbar=False)
axes[1, 0].set_title(f'Confusion Matrix - {best_model}')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')

# 4. ROC Curve for Best Model
fpr, tpr, _ = roc_curve(y_test, rf_tuned_metrics['probabilities'])
auc = rf_tuned_metrics['test_auc']
axes[1, 1].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
axes[1, 1].set_title(f'ROC Curve - {best_model}')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/task2_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Model comparison visualizations saved as 'task2_model_comparison.png'")
plt.show()


"""
STEP 12: FEATURE IMPORTANCE
Identify which features are most important for predictions
"""

print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Get feature importance from best model (Random Forest)
best_rf_model = grid_search_rf.best_estimator_
feature_importance = best_rf_model.named_steps['classifier'].feature_importances_

# Get feature names
feature_names = X.columns.tolist()

# Create dataframe
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10).to_string(index=False))

# Plot top 10 features
plt.figure(figsize=(10, 6))
top_n = 10
top_features = importance_df.head(top_n)
plt.barh(range(len(top_features)), top_features['Importance'], color='#3498db')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance')
plt.title(f'Top {top_n} Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/task2_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Feature importance plot saved as 'task2_feature_importance.png'")
plt.show()


"""
STEP 13: SAVE THE PIPELINE
Export the best model for production use
"""

print("\n" + "="*50)
print("SAVING PIPELINE FOR PRODUCTION")
print("="*50)

# Save the best pipeline
best_pipeline = grid_search_rf.best_estimator_
model_path = '/mnt/user-data/outputs/churn_prediction_pipeline.pkl'
joblib.dump(best_pipeline, model_path)

print(f"✓ Best pipeline saved to '{model_path}'")

# Save preprocessing information
preprocessing_info = {
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'binary_features': binary_features,
    'multiclass_features': multiclass_features,
    'feature_names': feature_names,
    'label_encoders': le_dict,
    'best_params': grid_search_rf.best_params_,
    'test_metrics': {
        'accuracy': rf_tuned_metrics['test_accuracy'],
        'f1_score': rf_tuned_metrics['test_f1'],
        'precision': rf_tuned_metrics['test_precision'],
        'recall': rf_tuned_metrics['test_recall'],
        'auc': rf_tuned_metrics['test_auc']
    }
}

joblib.dump(preprocessing_info, '/mnt/user-data/outputs/preprocessing_info.pkl')
print("✓ Preprocessing info saved")


"""
STEP 14: INFERENCE FUNCTION
Make predictions on new data
"""

print("\n" + "="*50)
print("INFERENCE FUNCTION")
print("="*50)

def predict_churn(customer_data_dict, pipeline, preprocessing_info):
    """
    Predict churn for a new customer
    
    Parameters:
    - customer_data_dict: Dictionary with customer features
    - pipeline: Trained pipeline object
    - preprocessing_info: Preprocessing information
    
    Returns:
    - Prediction (0/1) and probability
    """
    
    # Create DataFrame from dictionary
    customer_df = pd.DataFrame([customer_data_dict])
    
    # Encode binary categorical features
    le_dict = preprocessing_info['label_encoders']
    for col in preprocessing_info['binary_features']:
        if col in customer_df.columns:
            customer_df[col] = le_dict[col].transform(customer_df[col])
    
    # One-hot encode multiclass features
    customer_df = pd.get_dummies(
        customer_df, 
        columns=preprocessing_info['multiclass_features'],
        drop_first=True
    )
    
    # Ensure all features are present
    for feature in preprocessing_info['feature_names']:
        if feature not in customer_df.columns:
            customer_df[feature] = 0
    
    # Reorder columns to match training data
    customer_df = customer_df[preprocessing_info['feature_names']]
    
    # Make prediction
    prediction = pipeline.predict(customer_df)[0]
    probability = pipeline.predict_proba(customer_df)[0][1]
    
    return prediction, probability


# Test prediction
print("\nExample Prediction:")
sample_customer = {
    'SeniorCitizen': 0,
    'Tenure': 12,
    'MonthlyCharges': 65.5,
    'TotalCharges': 786.0,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'Gender': 'Male'
}

churn_pred, churn_prob = predict_churn(sample_customer, best_pipeline, preprocessing_info)
churn_status = 'Will Churn' if churn_pred == 1 else 'Will Not Churn'

print(f"\nCustomer Profile: {sample_customer}")
print(f"\nPrediction: {churn_status}")
print(f"Churn Probability: {churn_prob:.2%}")


print("\n" + "="*50)
print("✓ TASK 2 COMPLETE!")
print("="*50)
print("\nKey Achievements:")
print("✓ Loaded and explored Telco Churn dataset")
print("✓ Preprocessed and prepared data")
print("✓ Built ML pipelines (Logistic Regression & Random Forest)")
print("✓ Performed hyperparameter tuning with GridSearchCV")
print("✓ Compared and evaluated models")
print("✓ Identified feature importance")
print("✓ Saved production-ready pipeline")
print("✓ Created inference function for new predictions")
print("\nProduction-Ready Files:")
print("  - churn_prediction_pipeline.pkl (main model)")
print("  - preprocessing_info.pkl (preprocessing metadata)")
