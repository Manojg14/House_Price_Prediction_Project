from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json

app = Flask(__name__)

# Global model and preprocessing objects
model = None
label_encoders = {}
feature_columns = None
scaler_params = {}

def load_and_prepare_data():
    """Load and preprocess the dataset"""
    global model, label_encoders, feature_columns, scaler_params
    
    # Load dataset
    df = pd.read_csv('Jupyter Notebook/house_price_prediction_dataset.csv')
    
    # Create a copy for processing
    data = df.copy()
    
    # Identify categorical columns
    categorical_cols = ['City', 'Furnishing', 'Main Road', 'Guest Room', 'Basement', 
                       'Water Supply', 'Air Conditioning', 'Preferred Tenant']
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    # Separate features and target
    X = data.drop('Price', axis=1)
    y = data['Price']
    
    feature_columns = X.columns.tolist()
    
    # Store scaling parameters for predictions
    for col in X.columns:
        scaler_params[col] = {
            'mean': X[col].mean(),
            'std': X[col].std(),
            'min': X[col].min(),
            'max': X[col].max()
        }
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'train_r2': float(r2_score(y_train, y_pred_train)),
        'test_r2': float(r2_score(y_test, y_pred_test)),
        'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
        'test_mae': float(mean_absolute_error(y_test, y_pred_test)),
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        'dataset_size': len(df),
        'training_size': len(X_train),
        'testing_size': len(X_test)
    }
    
    return metrics

def predict_price(features_dict):
    """Make prediction for a single house"""
    global model, label_encoders, feature_columns
    
    # Create feature array
    features = []
    categorical_cols = ['City', 'Furnishing', 'Main Road', 'Guest Room', 'Basement', 
                       'Water Supply', 'Air Conditioning', 'Preferred Tenant']
    
    for col in feature_columns:
        if col in categorical_cols:
            # Encode categorical value
            le = label_encoders[col]
            value = le.transform([features_dict[col]])[0]
            features.append(value)
        else:
            # Use numeric value as is
            features.append(float(features_dict[col]))
    
    # Predict
    features_array = np.array([features])
    prediction = model.predict(features_array)[0]
    
    return max(0, prediction)  # Ensure non-negative prediction

# Load and train model on startup
print("Loading and training model...")
metrics = load_and_prepare_data()
print("Model trained successfully!")
print(f"Test R² Score: {metrics['test_r2']:.4f}")

@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['Area', 'Bedrooms', 'Bathrooms', 'Stories', 'Parking', 'Age',
                          'City', 'Furnishing', 'Main Road', 'Guest Room', 'Basement',
                          'Water Supply', 'Air Conditioning', 'Preferred Tenant', 'Locality Rating']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Make prediction
        prediction = predict_price(data)
        
        return jsonify({
            'success': True,
            'predicted_price': round(prediction, 2),
            'formatted_price': f"₹{prediction:,.2f}"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Return model information"""
    return jsonify({
        'metrics': metrics,
        'features': feature_columns
    })

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
