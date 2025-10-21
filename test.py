import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

print("🔄 Retraining XGBoost Ridership Model from Scratch")
print("="*60)

# Load data
df = pd.read_csv('training_data_converted_strategy_output.csv')

# Route and depot mappings
route_no_mapping = {
    "PJ01": 0, "100": 1, "200": 2, "201": 3, "300": 4, "302": 5, "303": 6,
    "400": 7, "401": 8, "500": 9, "600": 10, "601": 11, "T504": 12, "T505": 13,
    "T506": 14, "T507": 15, "T508": 16, "506": 72, "420": 141, "771": 48,
    "602": 83, "751": 44, "822": 120
}

depot_mapping = {
    "29": 0, "7": 1, "10": 2, "22": 3, "27": 4,
    "2": 5, "5": 6, "37": 7, "4": 8, "38": 9
}

# Map route and depot
df['route_no_enc'] = df['disruption_route_name'].map(route_no_mapping)
df['depot_enc'] = df['disruption_depot_id'].astype(str).map(depot_mapping)

# Create time features
df['day_of_week'] = df.get('disruption_day_of_week', 3)
df['month'] = 6
df['is_holiday'] = df['is_weekend']
df['hours_left'] = 12

# Use passenger_on_bus as target (since no ridership column)
features = ['route_no_enc', 'day_of_week', 'month', 'depot_enc', 'is_holiday', 'hours_left']
target = 'passenger_on_bus'

# Clean data
df_clean = df.dropna(subset=features + [target])
X = df_clean[features]
y = df_clean[target]

print(f"📊 Training data: {len(X)} samples")
print(f"📊 Target range: {y.min():.0f} - {y.max():.0f} passengers")
print(f"📊 Average: {y.mean():.1f} passengers\n")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features - IMPORTANT: Fit on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model with proper parameters
print("🏋️ Training XGBoost model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)

xgb_model.fit(X_train_scaled, y_train)

# Evaluate
y_pred_train = xgb_model.predict(X_train_scaled)
y_pred_test = xgb_model.predict(X_test_scaled)

# Training metrics
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = r2_score(y_train, y_pred_train)

# Test metrics
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)
mape_test = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1))) * 100

print("\n📈 Model Performance:")
print("="*60)
print(f"Training Set:")
print(f"  MAE:  {mae_train:.2f} passengers")
print(f"  RMSE: {rmse_train:.2f} passengers")
print(f"  R²:   {r2_train:.4f}")

print(f"\nTest Set:")
print(f"  MAE:  {mae_test:.2f} passengers")
print(f"  RMSE: {rmse_test:.2f} passengers")
print(f"  R²:   {r2_test:.4f}")
print(f"  MAPE: {mape_test:.2f}%")

accuracy = max(0, 100 - mape_test)
print(f"\n🎯 Estimated Accuracy: {accuracy:.2f}%")
print("="*60)

# Show sample predictions
print("\n📋 Sample Predictions:")
comparison = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred_test[:10],
    'Error': np.abs(y_test.values[:10] - y_pred_test[:10])
})
print(comparison.to_string(index=False))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Feature Importance:")
print(feature_importance.to_string(index=False))

# Save models
print("\n💾 Saving models...")
joblib.dump(xgb_model, 'models/xgb_ridership_model_retrained.pkl')
joblib.dump(scaler, 'models/xgb_feature_scaler_retrained.pkl')

print("✅ New models saved!")
print("\nTo use the new models, replace:")
print("  models/xgb_ridership_model.pkl → models/xgb_ridership_model_retrained.pkl")
print("  models/xgb_feature_scaler.pkl → models/xgb_feature_scaler_retrained.pkl")
