import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("="*60)
print("MODEL EVALUATION REPORT")
print("="*60)

# Load models
print("\n📦 Loading models...")
rf = joblib.load('models/best_route_model.pkl')
scaler_rf = joblib.load('models/scaler_rf.pkl')
xgb_ridership = joblib.load('models/xgb_ridership_model.pkl')
xgb_scaler = joblib.load('models/xgb_feature_scaler.pkl')
print("✅ Models loaded successfully")

# Load data
print("\n📊 Loading training data...")
df = pd.read_csv('training_data_converted_strategy_output.csv')
print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ====================================================================
# EVALUATE MODEL 1: RIDERSHIP FORECASTING (XGBoost)
# ====================================================================
print("\n" + "="*60)
print("MODEL 1: RIDERSHIP FORECASTING (XGBoost Regressor)")
print("="*60)

# Prepare features for ridership model
ridership_features = [
    'route_no_enc',
    'day_of_week',
    'month',
    'depot_enc',
    'is_holiday',
    'hours_left'
]

# Check if ridership column exists
ridership_target_col = None
for col in ['ridership', 'total_ridership', 'passenger_count', 'ridership_count']:
    if col in df.columns:
        ridership_target_col = col
        print(f"✅ Found target column: {col}")
        break

if ridership_target_col is None:
    print("⚠️ Warning: Ridership target column not found in data")
    print("Available columns:", df.columns.tolist())
    print("\nSkipping ridership model evaluation...")
    ridership_metrics = None
else:
    # Check if all features exist
    missing_features = [f for f in ridership_features if f not in df.columns]
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
        ridership_metrics = None
    else:
        # Prepare data
        X_ridership = df[ridership_features].dropna()
        y_ridership = df.loc[X_ridership.index, ridership_target_col]
        
        # Split data
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_ridership, y_ridership, test_size=0.2, random_state=42
        )
        
        print(f"\n📊 Dataset Split:")
        print(f"   Training set: {len(X_train_r)} samples")
        print(f"   Test set: {len(X_test_r)} samples")
        
        # Scale and predict
        X_test_scaled = xgb_scaler.transform(X_test_r)
        y_pred_r = xgb_ridership.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_r, y_pred_r)
        rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
        r2 = r2_score(y_test_r, y_pred_r)
        mape = np.mean(np.abs((y_test_r - y_pred_r) / (y_test_r + 1))) * 100
        
        print(f"\n📈 Performance Metrics:")
        print(f"   MAE (Mean Absolute Error):  {mae:.2f} passengers")
        print(f"   RMSE (Root Mean Squared):   {rmse:.2f} passengers")
        print(f"   R² Score:                   {r2:.4f}")
        print(f"   MAPE (Mean Abs % Error):    {mape:.2f}%")
        
        # Calculate accuracy as percentage
        accuracy_ridership = (1 - mape/100) * 100
        print(f"\n🎯 Model Accuracy: {accuracy_ridership:.2f}%")
        
        ridership_metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'accuracy': accuracy_ridership
        }

# ====================================================================
# EVALUATE MODEL 2: ROUTE SELECTION (Random Forest)
# ====================================================================
print("\n" + "="*60)
print("MODEL 2: ROUTE SELECTION (Random Forest Classifier)")
print("="*60)

# Prepare features for route selection model
route_features = [
    'passenger_on_bus',
    'disruption_weather_impact',
    'is_weekend',
    'is_peak_hour_encoded',
    'bus_replacement_priority',
    'bus_replacement_route_type_encoded',
    'deadmileage_to_disruption',
    'geo_distance_to_disruption',
    'travel_time_min_from_hub'
]

# Check if target column exists
route_target_col = None
for col in ['is_best_route', 'best_route', 'is_selected', 'selected_route']:
    if col in df.columns:
        route_target_col = col
        print(f"✅ Found target column: {col}")
        break

if route_target_col is None:
    print("⚠️ Warning: Route selection target column not found")
    print("Available columns:", df.columns.tolist())
    print("\nSkipping route selection model evaluation...")
    route_metrics = None
else:
    # Check if all features exist
    missing_features = [f for f in route_features if f not in df.columns]
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
        route_metrics = None
    else:
        # Prepare data
        X_route = df[route_features].dropna()
        y_route = df.loc[X_route.index, route_target_col]
        
        # Split data
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
            X_route, y_route, test_size=0.2, random_state=42, stratify=y_route
        )
        
        print(f"\n📊 Dataset Split:")
        print(f"   Training set: {len(X_train_rf)} samples")
        print(f"   Test set: {len(X_test_rf)} samples")
        print(f"   Class distribution: {dict(y_test_rf.value_counts())}")
        
        # Scale and predict
        X_test_scaled_rf = scaler_rf.transform(X_test_rf)
        y_pred_rf = rf.predict(X_test_scaled_rf)
        y_pred_proba = rf.predict_proba(X_test_scaled_rf)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_rf, y_pred_rf)
        precision = precision_score(y_test_rf, y_pred_rf, zero_division=0)
        recall = recall_score(y_test_rf, y_pred_rf, zero_division=0)
        f1 = f1_score(y_test_rf, y_pred_rf, zero_division=0)
        
        print(f"\n📈 Performance Metrics:")
        print(f"   Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision:  {precision:.4f} ({precision*100:.2f}%)")
        print(f"   Recall:     {recall:.4f} ({recall*100:.2f}%)")
        print(f"   F1-Score:   {f1:.4f} ({f1*100:.2f}%)")
        
        print(f"\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test_rf, y_pred_rf)
        print(cm)
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test_rf, y_pred_rf))
        
        route_metrics = {
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }

# ====================================================================
# SUMMARY
# ====================================================================
print("\n" + "="*60)
print("📊 SUMMARY")
print("="*60)

if ridership_metrics:
    print(f"\n✅ Ridership Forecasting Model:")
    print(f"   Accuracy: {ridership_metrics['accuracy']:.2f}%")
    print(f"   R² Score: {ridership_metrics['r2']:.4f}")
    print(f"   MAE: {ridership_metrics['mae']:.2f} passengers")
else:
    print("\n❌ Ridership model could not be evaluated")

if route_metrics:
    print(f"\n✅ Route Selection Model:")
    print(f"   Accuracy: {route_metrics['accuracy']:.2f}%")
    print(f"   Precision: {route_metrics['precision']:.2f}%")
    print(f"   F1-Score: {route_metrics['f1']:.2f}%")
else:
    print("\n❌ Route selection model could not be evaluated")

print("\n" + "="*60)

# Save metrics to file
metrics_dict = {
    'ridership_model': ridership_metrics,
    'route_selection_model': route_metrics
}

import json
with open('model_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=4)

print("✅ Metrics saved to model_metrics.json")
print("="*60)
