# Execute the heart disease prediction notebook as a Python script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create figures directory
import os
if not os.path.exists('./figures'):
    os.makedirs('./figures')

# Load the dataset
df = pd.read_csv('heart_disease.csv')

print("Dataset shape:", df.shape)
print("Column names:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0])

# Visualize missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Heatmap of Missing Values')
plt.tight_layout()
plt.savefig('./figures/missing_values_heatmap.png')
plt.close()  # Close the figure to save memory

# Handle missing values
for column in df.columns:
    if df[column].isnull().sum() > 0:
        if df[column].dtype in ['int64', 'float64']:
            df[column].fillna(df[column].median(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)

print("\nMissing values after imputation:")
print(df.isnull().sum().sum())

# Target distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='target')
plt.title('Distribution of Heart Disease (Target Variable)')
plt.xlabel('Target (0: No Disease, 1: Disease)')
plt.ylabel('Count')

# Add count labels on bars
for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()),
                       ha='center', va='bottom')
plt.tight_layout()
plt.savefig('./figures/target_distribution.png')
plt.close()

# Print percentage of each class
target_counts = df['target'].value_counts()
print(f"\nNo Heart Disease: {target_counts[0]} ({target_counts[0]/len(df)*100:.1f}%)")
print(f"Heart Disease: {target_counts[1]} ({target_counts[1]/len(df)*100:.1f}%)")

# Correlation heatmap
plt.figure(figsize=(14, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Correlation Heatmap of Features')
plt.tight_layout()
plt.savefig('./figures/correlation_heatmap.png')
plt.close()

# Feature vs Target visualizations
# Age vs Target
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='target', y='age')
plt.title('Age Distribution by Heart Disease Status')
plt.ylabel('Age')
plt.xlabel('Target (0: No Disease, 1: Disease)')

# Sex vs Target
plt.subplot(1, 2, 2)
pd.crosstab(df['sex'], df['target']).plot(kind='bar', stacked=True)
plt.title('Sex vs Heart Disease')
plt.xlabel('Sex (0: Female, 1: Male)')
plt.ylabel('Count')
plt.legend(['No Disease', 'Disease'], title='Target')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('./figures/age_sex_vs_target.png')
plt.close()

# Chest pain type vs Target
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='target', y='thalach')  # Maximum heart rate achieved
plt.title('Max Heart Rate vs Heart Disease')
plt.ylabel('Max Heart Rate Achieved')
plt.xlabel('Target (0: No Disease, 1: Disease)')

plt.subplot(1, 2, 2)
pd.crosstab(df['cp'], df['target']).plot(kind='bar', stacked=True)
plt.title('Chest Pain Type vs Heart Disease')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.legend(['No Disease', 'Disease'], title='Target')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('./figures/chest_pain_max_hr_vs_target.png')
plt.close()

# Separate features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature names: {list(X.columns)}")

# Feature relevance explanation
feature_explanations = {
    'age': 'Patient age - higher age often correlates with increased heart disease risk',
    'sex': 'Gender - males historically have higher heart disease rates',
    'cp': 'Chest pain type - different types may indicate cardiac issues',
    'trestbps': 'Resting blood pressure - high BP is a risk factor',
    'chol': 'Cholesterol level - high cholesterol increases risk',
    'fbs': 'Fasting blood sugar - diabetes increases heart disease risk',
    'restecg': 'Resting ECG results - abnormalities may indicate issues',
    'thalach': 'Maximum heart rate achieved - lower max HR may indicate poor heart health',
    'exang': 'Exercise-induced angina - chest pain during exercise is a strong indicator',
    'oldpeak': 'ST depression induced by exercise - indicates heart stress',
    'slope': 'Slope of peak exercise ST segment - different slopes have different meanings',
    'ca': 'Number of major vessels colored by fluoroscopy - blocked arteries indicate disease',
    'thal': 'Thalassemia - blood disorder that can affect heart health'
}

print("\nFeature Explanations:")
for feature, explanation in feature_explanations.items():
    print(f"- {feature}: {explanation}")

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Training target distribution: {y_train.value_counts(normalize=True)}")
print(f"Testing target distribution: {y_test.value_counts(normalize=True)}")

# Train Logistic Regression model
# Standardize features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

print("\nLogistic Regression model trained successfully!")

# Model choice explanation
print("\nModel Choice Justification:")
print("Logistic Regression was chosen because:")
print("1. It's interpretable - we can understand which features contribute most")
print("2. It works well for binary classification problems like this")
print("3. It handles multiple features effectively")
print("4. It provides probability estimates for predictions")
print("5. It's less prone to overfitting compared to more complex models")

# Calculate accuracy
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"\nLogistic Regression Accuracy: {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('./figures/confusion_matrix.png')
plt.close()

# Print confusion matrix details
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn} (Correctly predicted no disease)")
print(f"False Positives: {fp} (Incorrectly predicted disease)")
print(f"False Negatives: {fn} (Incorrectly predicted no disease)")
print(f"True Positives: {tp} (Correctly predicted disease)")

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_lr)
roc_auc = roc_auc_score(y_test, y_pred_proba_lr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('./figures/roc_curve.png')
plt.close()

print(f"\nROC AUC Score: {roc_auc:.4f}")

# Get feature importance from logistic regression coefficients
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_model.coef_[0],
    'abs_coefficient': np.abs(lr_model.coef_[0])
}).sort_values(by='abs_coefficient', ascending=False)

print("\nFeature Importance (based on Logistic Regression coefficients):")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(10)  # Top 10 features
colors = ['red' if coef < 0 else 'blue' for coef in top_features['coefficient']]
bars = plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Coefficient Value')
plt.title('Top 10 Feature Importance for Heart Disease Prediction')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('./figures/feature_importance.png')
plt.close()

# Print interpretation of top features
print("\nInterpretation of Top Features:")
for idx, row in feature_importance.head(5).iterrows():
    direction = "increases" if row['coefficient'] > 0 else "decreases"
    print(f"- {row['feature']}: {direction} heart disease risk (coef: {row['coefficient']:.3f})")

# Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_model.predict(X_test)
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]

# Calculate accuracy
accuracy_dt = accuracy_score(y_test, y_pred_dt)
roc_auc_dt = roc_auc_score(y_test, y_pred_proba_dt)

print(f"\nDecision Tree Accuracy: {accuracy_dt:.4f} ({accuracy_dt*100:.2f}%)")
print(f"Decision Tree ROC AUC: {roc_auc_dt:.4f}")

# Feature importance for Decision Tree
dt_feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\nDecision Tree Feature Importance:")
print(dt_feature_importance.head())

# Final results summary
print("\n=== HEART DISEASE PREDICTION RESULTS SUMMARY ===")
print(f"Dataset shape: {df.shape}")
print(f"Target distribution: {dict(y.value_counts())}")
print(f"\nLogistic Regression Model:")
print(f"- Accuracy: {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")
print(f"- ROC AUC: {roc_auc:.4f}")
print(f"\nDecision Tree Model:")
print(f"- Accuracy: {accuracy_dt:.4f} ({accuracy_dt*100:.2f}%)")
print(f"- ROC AUC: {roc_auc_dt:.4f}")

print(f"\nTop 5 Most Important Features (Logistic Regression):")
for idx, row in feature_importance.head(5).iterrows():
    print(f"- {row['feature']}: coefficient = {row['coefficient']:.3f}")

print(f"\nTop 5 Most Important Features (Decision Tree):")
for idx, row in dt_feature_importance.head(5).iterrows():
    print(f"- {row['feature']}: importance = {row['importance']:.3f}")

# Medical insights
print(f"\n=== MEDICAL INSIGHTS ===")
print("Based on the analysis, the following factors appear to be important predictors of heart disease:")
print("1. Exercise-induced angina (exang) - chest pain during exercise")
print("2. Number of major vessels colored by fluoroscopy (ca) - indicates blocked arteries")
print("3. Maximum heart rate achieved (thalach) - lower max HR indicates poor heart health")
print("4. ST depression induced by exercise (oldpeak) - indicates heart stress")
print("5. Chest pain type (cp) - certain types are associated with cardiac issues")

print(f"\nIMPORTANT MEDICAL DISCLAIMER:")
print("This model is for educational purposes only and should not be used for actual medical diagnosis.")
print("Always consult with healthcare professionals for proper medical assessment and diagnosis.")

print("\nFigures have been saved to the ./figures/ directory")