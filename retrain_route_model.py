import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

print("🔄 Retraining Route Selection Model with Improvements...")

# Load data
df = pd.read_csv('training_data_converted_strategy_output.csv')

# Features
features = [
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

# Add engineered features
df['distance_efficiency'] = df['geo_distance_to_disruption'] / (df['deadmileage_to_disruption'] + 0.001)
df['time_per_km'] = df['travel_time_min_from_hub'] / (df['geo_distance_to_disruption'] + 0.001)
df['priority_passenger_product'] = df['bus_replacement_priority'] * df['passenger_on_bus']
df['weather_peak_interaction'] = df['disruption_weather_impact'] * df['is_peak_hour_encoded']

enhanced_features = features + [
    'distance_efficiency',
    'time_per_km',
    'priority_passenger_product',
    'weather_peak_interaction'
]

X = df[enhanced_features].dropna()
y = df.loc[X.index, 'best_route']

print(f"📊 Original class distribution: {dict(y.value_counts())}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE to balance classes
smote = SMOTE(sampling_strategy=0.8, random_state=42)  # Bring minority to 80% of majority
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"📊 After SMOTE: {dict(pd.Series(y_train_balanced).value_counts())}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Train improved model
rf_improved = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',  # Additional balancing
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

print("🏋️ Training improved model...")
rf_improved.fit(X_train_scaled, y_train_balanced)

# Evaluate
y_pred = rf_improved.predict(X_test_scaled)
y_pred_proba = rf_improved.predict_proba(X_test_scaled)

print("\n📊 Improved Model Performance:")
print(classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save improved model
joblib.dump(rf_improved, 'models/best_route_model_improved_test.pkl')
joblib.dump(scaler, 'models/scaler_rf_improved_test.pkl')

print("\n✅ Improved model saved!")
print("Replace the old models with these improved versions.")
