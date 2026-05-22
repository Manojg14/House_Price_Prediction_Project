"""
House Price Prediction - Model Training Script
This script trains the model and saves all necessary files for the Flask application

Steps:
1. Load and preprocess data
2. Train the model
3. Save model, scaler, and encoders
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("🤖 HOUSE PRICE PREDICTION MODEL TRAINING")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n📂 Loading dataset...")
df = pd.read_csv('house_price_prediction_dataset.csv')
print(f"✓ Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")
print(f"   Price range: ₹{df['Price'].min():,.0f} - ₹{df['Price'].max():,.0f}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n🔧 Preprocessing data...")

# Define categorical columns
categorical_cols = ['City', 'Furnishing', 'Main Road', 'Guest Room', 
                   'Basement', 'Water Supply', 'Air Conditioning', 'Preferred Tenant']

# Initialize label encoders dictionary
le_dict = {}

# Create a copy for processing
df_processed = df.copy()

# Encode categorical variables
print("   • Encoding categorical variables...")
for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df[col])
    le_dict[col] = le
    print(f"     ✓ {col}: {len(le.classes_)} unique values")

# Prepare features and target
X = df_processed.drop('Price', axis=1)
y = df_processed['Price']

print(f"\n   • Features shape: {X.shape}")
print(f"   • Target shape: {y.shape}")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("\n✂️  Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   ✓ Training set: {X_train.shape[0]} samples")
print(f"   ✓ Testing set: {X_test.shape[0]} samples")

# ============================================================================
# 4. FEATURE SCALING
# ============================================================================
print("\n📊 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✓ Features scaled successfully")

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================
print("\n🧠 Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("   ✓ Model training completed")

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================
print("\n📈 Evaluating model...")

# Make predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Calculate metrics
mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred_test)

mae_train = mean_absolute_error(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_pred_train)

print("\n   TRAINING SET METRICS:")
print(f"   • MAE (Mean Absolute Error):  ₹{mae_train:,.2f}")
print(f"   • RMSE (Root Mean Sq Error):  ₹{rmse_train:,.2f}")
print(f"   • R² Score:                   {r2_train:.4f}")

print("\n   TESTING SET METRICS:")
print(f"   • MAE (Mean Absolute Error):  ₹{mae_test:,.2f}")
print(f"   • RMSE (Root Mean Sq Error):  ₹{rmse_test:,.2f}")
print(f"   • R² Score:                   {r2_test:.4f}")

# Feature importance
print("\n   TOP 5 IMPORTANT FEATURES:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).abs().sort_values('Coefficient', ascending=False)

for idx, row in feature_importance.head(5).iterrows():
    print(f"   • {row['Feature']}: {row['Coefficient']:.2f}")

# ============================================================================
# 7. SAVE MODEL AND ARTIFACTS
# ============================================================================
print("\n💾 Saving model artifacts...")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✓ Model saved to 'model.pkl'")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✓ Scaler saved to 'scaler.pkl'")

# Save encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump(le_dict, f)
print("   ✓ Encoders saved to 'encoders.pkl'")

# Save feature names
feature_names = X.columns.tolist()
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("   ✓ Feature names saved to 'feature_names.pkl'")

# ============================================================================
# 8. SAVE SAMPLE PREDICTIONS
# ============================================================================
print("\n📋 Creating sample predictions...")

sample_indices = np.random.choice(X_test.index, 10, replace=False)
sample_df = df.loc[sample_indices, ['Area', 'Bedrooms', 'Bathrooms', 'City', 'Price']].copy()
sample_df['Predicted_Price'] = model.predict(X_test_scaled[X_test.index.isin(sample_indices)])
sample_df['Error'] = sample_df['Price'] - sample_df['Predicted_Price']
sample_df['Error_Percentage'] = (sample_df['Error'] / sample_df['Price'] * 100).round(2)

print("\n   SAMPLE PREDICTIONS:")
print(sample_df.to_string(index=False))

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 80)

print("\n📦 FILES SAVED:")
print("   • model.pkl              - Trained Linear Regression model")
print("   • scaler.pkl             - StandardScaler for feature scaling")
print("   • encoders.pkl           - Label encoders for categorical variables")
print("   • feature_names.pkl      - Feature column names")

print("\n📊 MODEL PERFORMANCE:")
print(f"   • Test R² Score: {r2_test:.4f} (Accuracy)")
print(f"   • Test RMSE: ₹{rmse_test:,.2f}")
print(f"   • Test MAE: ₹{mae_test:,.2f}")

print("\n🚀 NEXT STEPS:")
print("   1. Ensure all .pkl files are in the same directory as app.py")
print("   2. Ensure 'house_price_prediction_dataset.csv' is in the same directory")
print("   3. Run: pip install -r requirements.txt")
print("   4. Run: python app.py")
print("   5. Open: http://localhost:5000 in your browser")

print("\n" + "=" * 80)
print("If you see this message, your model is ready to use!")
print("=" * 80 + "\n")