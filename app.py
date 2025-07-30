from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

knn = joblib.load('/Users/fahmi.taib/Desktop/Deployment Code Test/Prototype/new arishem design/knn_model.pkl')
scaler_knn = joblib.load('/Users/fahmi.taib/Desktop/Deployment Code Test/Prototype/new arishem design/scaler_knn.pkl')
rf = joblib.load('/Users/fahmi.taib/Desktop/Deployment Code Test/Prototype/new arishem design/best_route_model.pkl')
scaler_rf = joblib.load('/Users/fahmi.taib/Desktop/Deployment Code Test/Prototype/new arishem design/scaler_rf.pkl')
xgb_ridership = joblib.load('/Users/fahmi.taib/Desktop/Deployment Code Test/Prototype/new arishem design/xgb_ridership_model.pkl')
xgb_scaler = joblib.load('/Users/fahmi.taib/Desktop/Deployment Code Test/Prototype/new arishem design/xgb_feature_scaler.pkl')

df = pd.read_csv('/Users/fahmi.taib/Desktop/Deployment Code Test/training_data_converted_strategy_output.csv')

features_rf = [
    'passenger_on_bus', 'disruption_weather_impact', 'is_weekend', 'is_peak_hour_encoded',
    'bus_replacement_priority', 'bus_replacement_route_type_encoded',
    'deadmileage_to_disruption', 'geo_distance_to_disruption', 'travel_time_min_from_hub'
]

class DisruptionInput(BaseModel):
    lat_disruption: float
    lng_disruption: float
    passenger_on_bus: int
    disruption_weather_impact: int
    is_weekend: int
    is_peak_hour_encoded: int
    bus_replacement_priority: int
    bus_replacement_route_type_encoded: int
    deadmileage_to_disruption: float
    geo_distance_to_disruption: float
    travel_time_min_from_hub: float

@app.post("/predict_best_route")
def predict_best_route(data: DisruptionInput):
    coords_scaled = scaler_knn.transform([[data.lat_disruption, data.lng_disruption]])
    distances, indices = knn.kneighbors(coords_scaled)

    candidate_rows = df.iloc[indices[0]][features_rf + ['bus_replacement_route_no']].to_dict(orient='records')

    rf_features_list = []
    for candidate in candidate_rows:
        features = {
            'passenger_on_bus': data.passenger_on_bus,
            'disruption_weather_impact': data.disruption_weather_impact,
            'is_weekend': data.is_weekend,
            'is_peak_hour_encoded': data.is_peak_hour_encoded,
            'bus_replacement_priority': data.bus_replacement_priority,
            'bus_replacement_route_type_encoded': data.bus_replacement_route_type_encoded,
            'deadmileage_to_disruption': data.deadmileage_to_disruption,
            'geo_distance_to_disruption': data.geo_distance_to_disruption,
            'travel_time_min_from_hub': data.travel_time_min_from_hub
        }
        rf_features_list.append(features)

    X_rf = pd.DataFrame(rf_features_list)
    X_rf_scaled = scaler_rf.transform(X_rf)
    preds = rf.predict(X_rf_scaled)
    probs = rf.predict_proba(X_rf_scaled)[:, 1]

    best_route = None
    best_prob = None
    for i, pred in enumerate(preds):
        if pred == 1:
            best_route = candidate_rows[i]['bus_replacement_route_no']
            best_prob = probs[i]
            break
    else:
        best_idx = probs.argmax()
        best_route = candidate_rows[best_idx]['bus_replacement_route_no']
        best_prob = probs[best_idx]

    return {
        "best_route": best_route,
        "best_route_probability": float(best_prob),
        "candidate_routes": [c['bus_replacement_route_no'] for c in candidate_rows]
    }

class RidershipForecastInput(BaseModel):
    route_no_enc: int
    day_of_week: int
    month: int
    depot_enc: int
    is_holiday: int
    hours_left: float

@app.post("/forecast_ridership")
def forecast_ridership(data: RidershipForecastInput):
    X = [[
        data.route_no_enc,
        data.day_of_week,
        data.month,
        data.depot_enc,
        data.is_holiday,
        data.hours_left
    ]]
    X_scaled = xgb_scaler.transform(X)
    confidence_score = 0.9093

    prediction = xgb_ridership.predict(X_scaled)
    return {"forecasted_ridership": float(prediction[0]),
            "confidence_score": confidence_score
    }
