"""
========================================
TASK 3: MULTIMODAL MACHINE LEARNING
Housing Price Prediction Using Images + Tabular Data
========================================

OBJECTIVE:
Predict housing prices using both house images (CNN features) 
and structured data (tabular features).

WHAT IS MULTIMODAL LEARNING?
Multimodal learning combines different data types (images, text, numbers, etc.)
to make better predictions. In this task, we use:
1. House images → CNN extracts visual features
2. Tabular data → Numeric features (size, bedrooms, etc.)
3. Combined features → Final price prediction

SKILLS GAINED:
- Multimodal machine learning
- Convolutional Neural Networks (CNNs)
- Feature fusion (image + tabular)
- Regression modeling and evaluation

STEP 1: IMPORT LIBRARIES
"""


import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import os
os.makedirs('./outputs', exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

import os
from pathlib import Path

print("✓ All libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")


"""
STEP 2: CREATE SYNTHETIC HOUSING DATASET WITH IMAGES
For demonstration, we'll create synthetic house images and tabular data
In production, you would use real house images and data
"""

print("\n" + "="*50)
print("CREATING SYNTHETIC HOUSING DATASET")
print("="*50)

# Create directories for synthetic images
os.makedirs('./house_images', exist_ok=True)

# Generate synthetic house images and data
np.random.seed(42)

def create_synthetic_house_image(house_id, price):
    """
    Create a synthetic house image representation
    (In real scenario, you'd have actual house photos)
    """
    # Create image based on price (higher price = more colorful/detailed)
    img = np.random.rand(224, 224, 3) * 0.5 + 0.3  # Base noise
    
    # Add price-dependent features
    price_normalized = min(price / 1000000, 1.0)
    
    # Add colored regions based on price
    img[50:100, 50:100, :] = price_normalized  # Color intensity based on price
    img[150:200, 150:200, :] = 0.3 + (price_normalized * 0.7)
    
    # Convert to uint8
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Save as image
    from PIL import Image
    Image.fromarray(img_uint8).save(f'./house_images/house_{house_id}.png')
    
    return f'house_{house_id}.png'

# Generate dataset
print("\nGenerating synthetic housing data...")
n_samples = 200

data = {
    'house_id': range(n_samples),
    'square_feet': np.random.randint(800, 5000, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.uniform(1, 4, n_samples).round(1),
    'age_years': np.random.randint(0, 100, n_samples),
    'garage_spaces': np.random.randint(0, 4, n_samples),
    'pool': np.random.choice([0, 1], n_samples),
    'basement': np.random.choice([0, 1], n_samples),
}

# Generate prices with relationship to features
prices = (
    data['square_feet'] * 150 +
    data['bedrooms'] * 30000 +
    data['bathrooms'] * 25000 +
    data['garage_spaces'] * 20000 +
    data['pool'] * 50000 +
    data['basement'] * 40000 +
    np.random.normal(0, 50000, n_samples)  # Add noise
)

# Ensure positive prices
prices = np.maximum(prices, 100000)
data['price'] = prices.astype(int)

# Create image filenames
print("Creating synthetic house images...")
image_files = []
for idx, (house_id, price) in enumerate(zip(data['house_id'], data['price'])):
    if idx % 50 == 0:
        print(f"  Created {idx}/{n_samples} images")
    img_file = create_synthetic_house_image(house_id, price)
    image_files.append(img_file)

data['image_file'] = image_files

# Create DataFrame
df = pd.DataFrame(data)

print(f"\n✓ Dataset created!")
print(f"Total samples: {len(df)}")
print(f"\n--- Dataset Info ---")
print(df.head())
print(f"\n--- Statistics ---")
print(df.describe())

# Verify images exist
image_dir = './house_images'
n_images = len(os.listdir(image_dir))
print(f"\n✓ {n_images} synthetic house images created in {image_dir}")


"""
STEP 3: SPLIT DATA INTO TRAIN/TEST/VAL
"""

print("\n" + "="*50)
print("SPLITTING DATA")
print("="*50)

# First split: 80% train+val, 20% test
df_trainval, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Second split: 80% train, 20% val (of the train+val data)
df_train, df_val = train_test_split(df_trainval, test_size=0.2, random_state=42)

print(f"Training set: {len(df_train)} samples ({len(df_train)/len(df)*100:.1f}%)")
print(f"Validation set: {len(df_val)} samples ({len(df_val)/len(df)*100:.1f}%)")
print(f"Test set: {len(df_test)} samples ({len(df_test)/len(df)*100:.1f}%)")

# Price ranges
print(f"\n--- Price Statistics ---")
print(f"Train - Min: ${df_train['price'].min():,}, Max: ${df_train['price'].max():,}, Mean: ${df_train['price'].mean():,.0f}")
print(f"Val   - Min: ${df_val['price'].min():,}, Max: ${df_val['price'].max():,}, Mean: ${df_val['price'].mean():,.0f}")
print(f"Test  - Min: ${df_test['price'].min():,}, Max: ${df_test['price'].max():,}, Mean: ${df_test['price'].mean():,.0f}")


"""
STEP 4: FEATURE ENGINEERING FOR TABULAR DATA
"""

print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

def prepare_tabular_features(df):
    """Prepare and normalize tabular features"""
    features = df[['square_feet', 'bedrooms', 'bathrooms', 'age_years', 
                   'garage_spaces', 'pool', 'basement']].copy()
    
    # Normalize features to 0-1 range
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    return features_normalized, scaler, features.columns.tolist()

# Prepare features
X_train_tabular, scaler_train, feature_names = prepare_tabular_features(df_train)
X_val_tabular, _, _ = prepare_tabular_features(df_val)
X_test_tabular, _, _ = prepare_tabular_features(df_test)

# For consistency, use training scaler for val and test
X_val_tabular = scaler_train.transform(df_val[feature_names])
X_test_tabular = scaler_train.transform(df_test[feature_names])

# Target variable
y_train = df_train['price'].values
y_val = df_val['price'].values
y_test = df_test['price'].values

# Normalize prices for training (helps with neural network training)
price_scaler = StandardScaler()
y_train_normalized = price_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_normalized = price_scaler.transform(y_val.reshape(-1, 1)).flatten()

print(f"Tabular features shape: {X_train_tabular.shape}")
print(f"Feature names: {feature_names}")
print(f"✓ Features normalized and scaled")


"""
STEP 5: PREPARE IMAGE DATA
Load and preprocess house images
"""

print("\n" + "="*50)
print("PREPARING IMAGE DATA")
print("="*50)

def load_and_preprocess_images(df, image_dir, img_size=(224, 224)):
    """
    Load images and preprocess them
    """
    images = []
    valid_indices = []
    
    print(f"\nLoading {len(df)} images...")
    
    for idx, (index, row) in enumerate(df.iterrows()):
        if idx % 50 == 0:
            print(f"  Loaded {idx}/{len(df)} images")
        
        try:
            img_path = os.path.join(image_dir, row['image_file'])
            
            # Load and resize image
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img)
            
            # Preprocess using ResNet50 preprocessing
            img_array = resnet_preprocess(img_array)
            
            images.append(img_array)
            valid_indices.append(index)
        
        except Exception as e:
            print(f"  Warning: Could not load image {row['image_file']}: {e}")
            continue
    
    return np.array(images), valid_indices

# Load images
print("Loading training images...")
X_train_images, train_indices = load_and_preprocess_images(df_train, './house_images')

print("Loading validation images...")
X_val_images, val_indices = load_and_preprocess_images(df_val, './house_images')

print("Loading test images...")
X_test_images, test_indices = load_and_preprocess_images(df_test, './house_images')

# Filter other data to match loaded images
df_train = df_train.iloc[train_indices].reset_index(drop=True)
df_val = df_val.iloc[val_indices].reset_index(drop=True)
df_test = df_test.iloc[test_indices].reset_index(drop=True)

X_train_tabular = X_train_tabular[train_indices]
X_val_tabular = X_val_tabular[val_indices]
X_test_tabular = X_test_tabular[test_indices]

y_train = df_train['price'].values
y_val = df_val['price'].values
y_test = df_test['price'].values
y_train_normalized = price_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_normalized = price_scaler.transform(y_val.reshape(-1, 1)).flatten()

print(f"\n✓ Images loaded successfully!")
print(f"Train images shape: {X_train_images.shape}")
print(f"Val images shape: {X_val_images.shape}")
print(f"Test images shape: {X_test_images.shape}")


"""
STEP 6: BUILD CNN FOR IMAGE FEATURE EXTRACTION
"""

print("\n" + "="*50)
print("BUILDING CNN IMAGE FEATURE EXTRACTOR")
print("="*50)

# Load pre-trained ResNet50 (trained on ImageNet)
print("\nLoading pre-trained ResNet50...")
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model weights (transfer learning)
base_model.trainable = False

print("✓ ResNet50 loaded")
print(f"Base model parameters: {base_model.count_params():,}")

# Build feature extraction model
print("\nBuilding feature extraction model...")
feature_extraction_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu')
])

print("✓ Feature extraction model built")

# Extract features (one-time, we don't train this)
print("\nExtracting image features (this may take a moment)...")

# Extract from training images in batches
batch_size = 32
image_features_train = []
for i in range(0, len(X_train_images), batch_size):
    batch = X_train_images[i:i+batch_size]
    features = feature_extraction_model.predict(batch, verbose=0)
    image_features_train.extend(features)

X_train_image_features = np.array(image_features_train)

# Extract from validation images
image_features_val = []
for i in range(0, len(X_val_images), batch_size):
    batch = X_val_images[i:i+batch_size]
    features = feature_extraction_model.predict(batch, verbose=0)
    image_features_val.extend(features)

X_val_image_features = np.array(image_features_val)

# Extract from test images
image_features_test = []
for i in range(0, len(X_test_images), batch_size):
    batch = X_test_images[i:i+batch_size]
    features = feature_extraction_model.predict(batch, verbose=0)
    image_features_test.extend(features)

X_test_image_features = np.array(image_features_test)

print(f"✓ Image features extracted!")
print(f"Train image features shape: {X_train_image_features.shape}")
print(f"Val image features shape: {X_val_image_features.shape}")
print(f"Test image features shape: {X_test_image_features.shape}")


"""
STEP 7: FUSE IMAGE AND TABULAR FEATURES
Combine features from both modalities
"""

print("\n" + "="*50)
print("FEATURE FUSION")
print("="*50)

# Combine image and tabular features
X_train_fused = np.concatenate([X_train_image_features, X_train_tabular], axis=1)
X_val_fused = np.concatenate([X_val_image_features, X_val_tabular], axis=1)
X_test_fused = np.concatenate([X_test_image_features, X_test_tabular], axis=1)

print(f"Fused feature dimensions:")
print(f"  Image features: {X_train_image_features.shape[1]}")
print(f"  Tabular features: {X_train_tabular.shape[1]}")
print(f"  Total fused features: {X_train_fused.shape[1]}")

print(f"\nTrain fused features shape: {X_train_fused.shape}")
print(f"Val fused features shape: {X_val_fused.shape}")
print(f"Test fused features shape: {X_test_fused.shape}")


"""
STEP 8: BUILD AND TRAIN REGRESSION MODEL
Train on fused features to predict house prices
"""

print("\n" + "="*50)
print("TRAINING REGRESSION MODELS")
print("="*50)

# Model 1: Random Forest
print("\n1. Training Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_fused, y_train)
print("✓ Random Forest trained")

# Model 2: Gradient Boosting
print("\n2. Training Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train_fused, y_train)
print("✓ Gradient Boosting trained")

# Model 3: Neural Network (using Keras)
print("\n3. Building and Training Neural Network...")

nn_model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train_fused.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

# Compile model
nn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("Model architecture:")
nn_model.summary()

# Train model
print("\nTraining neural network...")
history = nn_model.fit(
    X_train_fused, y_train_normalized,
    validation_data=(X_val_fused, y_val_normalized),
    epochs=50,
    batch_size=16,
    verbose=1
)

print("✓ Neural Network trained")


"""
STEP 9: EVALUATE ALL MODELS
"""

print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

def evaluate_model(model, X_test, y_test, model_name, is_neural_net=False):
    """Evaluate regression model"""
    print(f"\n--- {model_name} ---")
    
    # Make predictions
    if is_neural_net:
        y_pred_normalized = model.predict(X_test, verbose=0)
        y_pred = price_scaler.inverse_transform(y_pred_normalized)
    else:
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"  MAE (Mean Absolute Error):   ${mae:,.0f}")
    print(f"  RMSE (Root Mean Squared Error): ${rmse:,.0f}")
    print(f"  R² Score:                   {r2:.4f}")
    print(f"  MAPE (Mean Absolute %):     {mape:.2f}%")
    
    return {
        'model': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'predictions': y_pred
    }

# Evaluate all models
rf_results = evaluate_model(rf_model, X_test_fused, y_test, 'Random Forest')
gb_results = evaluate_model(gb_model, X_test_fused, y_test, 'Gradient Boosting')
nn_results = evaluate_model(nn_model, X_test_fused, y_test, 'Neural Network', is_neural_net=True)


"""
STEP 10: VISUALIZE TRAINING HISTORY
"""

print("\n" + "="*50)
print("VISUALIZING RESULTS")
print("="*50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Neural Network Training History - Loss
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].set_title('Neural Network - Training History')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Model Comparison - MAE
models_names = [rf_results['model'], gb_results['model'], nn_results['model']]
mae_values = [rf_results['mae'], gb_results['mae'], nn_results['mae']]
colors = ['#3498db', '#2ecc71', '#e74c3c']
axes[0, 1].bar(models_names, mae_values, color=colors)
axes[0, 1].set_title('Model Comparison - MAE')
axes[0, 1].set_ylabel('MAE ($)')
for i, v in enumerate(mae_values):
    axes[0, 1].text(i, v + 5000, f'${v:,.0f}', ha='center', va='bottom')

# 3. Model Comparison - R² Score
r2_values = [rf_results['r2'], gb_results['r2'], nn_results['r2']]
axes[1, 0].bar(models_names, r2_values, color=colors)
axes[1, 0].set_title('Model Comparison - R² Score')
axes[1, 0].set_ylabel('R² Score')
axes[1, 0].set_ylim([0, 1])
for i, v in enumerate(r2_values):
    axes[1, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')

# 4. Actual vs Predicted (Best Model)
best_results = max([rf_results, gb_results, nn_results], key=lambda x: x['r2'])
axes[1, 1].scatter(y_test, best_results['predictions'], alpha=0.6, s=50)
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[1, 1].set_xlabel('Actual Price ($)')
axes[1, 1].set_ylabel('Predicted Price ($)')
axes[1, 1].set_title(f'{best_results["model"]} - Actual vs Predicted')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./outputs/task3_multimodal_results.png', dpi=300, bbox_inches='tight')
print("✓ Results visualization saved as 'task3_multimodal_results.png'")
plt.show()


"""
STEP 11: RESIDUAL ANALYSIS
"""

print("\n" + "="*50)
print("RESIDUAL ANALYSIS")
print("="*50)

residuals = y_test - best_results['predictions']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residual distribution
axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[0].axvline(residuals.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: ${residuals.mean():,.0f}')
axes[0].axvline(0, color='g', linestyle='--', linewidth=2, label='Zero')
axes[0].set_xlabel('Residuals ($)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Residual Distribution')
axes[0].legend()

# Residuals vs Predicted
axes[1].scatter(best_results['predictions'], residuals, alpha=0.6, s=50)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Price ($)')
axes[1].set_ylabel('Residuals ($)')
axes[1].set_title('Residuals vs Predicted Values')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./outputs/task3_residual_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Residual analysis saved as 'task3_residual_analysis.png'")
plt.show()


"""
STEP 12: FEATURE IMPORTANCE
"""

print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Get feature importance from best model
if best_results['model'] == 'Random Forest':
    importances = rf_model.feature_importances_
elif best_results['model'] == 'Gradient Boosting':
    importances = gb_model.feature_importances_

# Create feature names
feature_names_full = ['Image_Feature_' + str(i) for i in range(128)] + feature_names

# Get top features
feature_importance_df = pd.DataFrame({
    'Feature': feature_names_full,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10).to_string(index=False))

# Plot top features
fig, ax = plt.subplots(figsize=(10, 6))
top_n = 10
top_features = feature_importance_df.head(top_n)
ax.barh(range(len(top_features)), top_features['Importance'], color='#3498db')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'])
ax.set_xlabel('Importance')
ax.set_title(f'Top {top_n} Features - {best_results["model"]}')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('./outputs/task3_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Feature importance plot saved as 'task3_feature_importance.png'")
plt.show()


"""
STEP 13: SAVE MODELS
"""

print("\n" + "="*50)
print("SAVING MODELS")
print("="*50)

# Save models
joblib.dump(rf_model, './outputs/multimodal_rf_model.pkl')
print("✓ Random Forest model saved")

joblib.dump(gb_model, './outputs/multimodal_gb_model.pkl')
print("✓ Gradient Boosting model saved")

nn_model.save('./outputs/multimodal_nn_model.h5')
print("✓ Neural Network model saved")

# Save preprocessing objects
preprocessing_data = {
    'feature_extraction_model': feature_extraction_model,
    'price_scaler': price_scaler,
    'feature_scaler': scaler_train,
    'feature_names': feature_names
}

joblib.dump(preprocessing_data, './outputs/multimodal_preprocessing.pkl')
print("✓ Preprocessing objects saved")


"""
STEP 14: INFERENCE FUNCTION
Make predictions on new house data
"""

print("\n" + "="*50)
print("INFERENCE FUNCTION")
print("="*50)

def predict_house_price(image_path, tabular_features, loaded_objects, model_type='rf'):
    """
    Predict house price from image and tabular features
    
    Parameters:
    - image_path: Path to house image
    - tabular_features: dict with {feature_name: value}
    - loaded_objects: dict with preprocessing objects
    - model_type: 'rf', 'gb', or 'nn'
    
    Returns:
    - Predicted price
    """
    
    feature_extraction_model = loaded_objects['feature_extraction_model']
    price_scaler = loaded_objects['price_scaler']
    feature_scaler = loaded_objects['feature_scaler']
    feature_names = loaded_objects['feature_names']
    
    # Load and preprocess image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = resnet_preprocess(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Extract image features
    image_features = feature_extraction_model.predict(img_array, verbose=0)
    
    # Prepare tabular features
    tabular_array = np.array([tabular_features[f] for f in feature_names])
    tabular_array = feature_scaler.transform(tabular_array.reshape(1, -1))
    
    # Fuse features
    fused_features = np.concatenate([image_features, tabular_array], axis=1)
    
    # Predict
    if model_type == 'rf':
        y_pred = rf_model.predict(fused_features)[0]
    elif model_type == 'gb':
        y_pred = gb_model.predict(fused_features)[0]
    elif model_type == 'nn':
        y_pred_normalized = nn_model.predict(fused_features, verbose=0)
        y_pred = price_scaler.inverse_transform(y_pred_normalized)[0][0]
    
    return y_pred

# Test on sample
print("\nExample Prediction:")
sample_tabular = {
    'square_feet': 2500,
    'bedrooms': 4,
    'bathrooms': 2.5,
    'age_years': 10,
    'garage_spaces': 2,
    'pool': 1,
    'basement': 1
}

sample_image_path = './house_images/house_0.png'

predicted_price = predict_house_price(
    sample_image_path, 
    sample_tabular,
    preprocessing_data,
    model_type='rf'
)

actual_price = test_results['predictions'][0] if len(test_results['predictions']) > 0 else None

print(f"\nSample House Features: {sample_tabular}")
print(f"Predicted Price: ${predicted_price:,.0f}")


"""
STEP 15: SUMMARY
"""

print("\n" + "="*50)
print("✓ TASK 3 COMPLETE!")
print("="*50)
print("\nKey Achievements:")
print("✓ Created multimodal dataset (images + tabular)")
print("✓ Built CNN for image feature extraction")
print("✓ Fused image and tabular features")
print("✓ Trained 3 regression models:")
print("  - Random Forest")
print("  - Gradient Boosting")
print("  - Neural Network")
print("✓ Evaluated with MAE and RMSE")
print("✓ Analyzed residuals and feature importance")
print("✓ Saved all models and preprocessing objects")
print("✓ Created inference function for new predictions")

print("\n--- Final Results ---")
print(f"Best Model: {best_results['model']}")
print(f"  MAE: ${best_results['mae']:,.0f}")
print(f"  RMSE: ${best_results['rmse']:,.0f}")
print(f"  R² Score: {best_results['r2']:.4f}")
print(f"  MAPE: {best_results['mape']:.2f}%")

print("\nFiles saved:")
print("  - multimodal_rf_model.pkl")
print("  - multimodal_gb_model.pkl")
print("  - multimodal_nn_model.h5")
print("  - multimodal_preprocessing.pkl")
print("  - task3_multimodal_results.png")
print("  - task3_residual_analysis.png")
print("  - task3_feature_importance.png")
