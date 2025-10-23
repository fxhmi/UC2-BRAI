from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SwiftRoute API v2.1", version="2.1.0")

# CORS - Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
knn = None
scaler_knn = None
rf = None
scaler_rf = None
xgb_ridership = None
xgb_scaler = None
df = None
ridership_predictor = None

# Load models at startup
@app.on_event("startup")
async def load_models():
    global knn, scaler_knn, rf, scaler_rf, xgb_ridership, xgb_scaler, df, ridership_predictor
    
    logger.info("🔄 Loading models...")
    
    try:
        # Load route prediction models
        knn = joblib.load('models/knn_model.pkl')
        scaler_knn = joblib.load('models/scaler_knn.pkl')
        rf = joblib.load('models/best_route_model_improved.pkl')
        scaler_rf = joblib.load('models/scaler_rf_improved.pkl')
        
        # Load ridership models
        xgb_ridership = joblib.load('models/xgb_ridership_model.pkl')
        xgb_scaler = joblib.load('models/xgb_feature_scaler.pkl')
        
        # Load data
        df = pd.read_csv('training_data_converted_strategy_output.csv')
        
        # Try to load hourly predictor
        try:
            from predict_disruption_ridership import DisruptionRidershipPredictor
            ridership_predictor = DisruptionRidershipPredictor()
            logger.info("✅ Hourly ridership predictor loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load hourly predictor: {e}")
            ridership_predictor = None
        
        logger.info("✅ All core models loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise

# Features list
features_rf = [
    'passenger_on_bus', 'disruption_weather_impact', 'is_weekend', 'is_peak_hour_encoded',
    'bus_replacement_priority', 'bus_replacement_route_type_encoded',
    'deadmileage_to_disruption', 'geo_distance_to_disruption', 'travel_time_min_from_hub'
]

enhanced_features = features_rf + [
    'distance_efficiency', 'time_per_km', 'priority_passenger_product', 'weather_peak_interaction'
]

# Pydantic models
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

class RidershipForecastInput(BaseModel):
    route_no_enc: int
    day_of_week: int
    month: int
    depot_enc: int
    is_holiday: int
    hours_left: float

class RidershipRequest(BaseModel):
    route_no: str
    disruption_datetime: str
    depot: Optional[int] = None

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "SwiftRoute API v2.1 - Running on Azure",
        "status": "operational",
        "version": "2.1.0",
        "endpoints": [
            "/health",
            "/predict_best_route",
            "/forecast_ridership",
            "/predict_disruption_ridership"
        ]
    }

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": knn is not None,
        "ridership_predictor_available": ridership_predictor is not None,
        "data_shape": df.shape if df is not None else None
    }

# Predict best route
@app.post("/predict_best_route")
async def predict_best_route(data: DisruptionInput):
    try:
        if knn is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # KNN
        coords_scaled = scaler_knn.transform([[data.lat_disruption, data.lng_disruption]])
        n_neighbors = min(15, len(df))
        distances, indices = knn.kneighbors(coords_scaled, n_neighbors=n_neighbors)
        
        # Get candidates
        candidate_rows = df.iloc[indices[0]][features_rf + ['bus_replacement_route_no']].to_dict(orient='records')
        
        # Engineer features
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
            
            features['distance_efficiency'] = features['geo_distance_to_disruption'] / (features['deadmileage_to_disruption'] + 0.001)
            features['time_per_km'] = features['travel_time_min_from_hub'] / (features['geo_distance_to_disruption'] + 0.001)
            features['priority_passenger_product'] = features['bus_replacement_priority'] * features['passenger_on_bus']
            features['weather_peak_interaction'] = features['disruption_weather_impact'] * features['is_peak_hour_encoded']
            
            rf_features_list.append(features)
        
        # Predict
        X_rf = pd.DataFrame(rf_features_list)[enhanced_features]
        X_rf_scaled = scaler_rf.transform(X_rf)
        preds = rf.predict(X_rf_scaled)
        probs = rf.predict_proba(X_rf_scaled)[:, 1]
        
        # Find best route
        best_route = None
        best_prob = None
        
        for i, pred in enumerate(preds):
            if pred == 1:
                best_route = candidate_rows[i]['bus_replacement_route_no']
                best_prob = probs[i]
                break
        
        if best_route is None:
            best_idx = probs.argmax()
            best_route = candidate_rows[best_idx]['bus_replacement_route_no']
            best_prob = probs[best_idx]
        
        candidate_routes = list(dict.fromkeys([c['bus_replacement_route_no'] for c in candidate_rows]))
        
        # Calculate confidence
        prob_std = np.std(probs)
        base_accuracy = 86.0
        confidence_adjustment = (best_prob - 0.5) * 20
        spread_bonus = min(10, prob_std * 20)
        final_confidence = max(70, min(95, base_accuracy + confidence_adjustment + spread_bonus))
        
        return {
            "best_route": best_route,
            "best_route_probability": float(best_prob),
            "confidence_score": round(final_confidence / 100, 4),
            "model_accuracy": 86.0,
            "model_precision": 76.0,
            "model_recall": 85.0,
            "candidate_routes": candidate_routes,
            "n_candidates_evaluated": len(candidate_rows),
            "model_version": "v2.1_azure"
        }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Forecast ridership (old method)
@app.post("/forecast_ridership")
async def forecast_ridership(data: RidershipForecastInput):
    try:
        if xgb_ridership is None:
            raise HTTPException(status_code=503, detail="Ridership model not loaded")
        
        X = [[data.route_no_enc, data.day_of_week, data.month, data.depot_enc, data.is_holiday, data.hours_left]]
        X_scaled = xgb_scaler.transform(X)
        prediction = xgb_ridership.predict(X_scaled)
        
        base_confidence = 0.9093
        confidence_adjustments = 0
        if data.hours_left < 2:
            confidence_adjustments -= 0.05
        if data.is_holiday == 1:
            confidence_adjustments -= 0.03
        
        final_confidence = max(0.70, min(0.95, base_confidence + confidence_adjustments))
        prediction_value = max(0, min(60, float(prediction[0])))
        
        return {
            "forecasted_ridership": round(prediction_value, 1),
            "confidence_score": round(final_confidence, 4),
            "model_version": "xgboost_v2.0",
            "prediction_range": {
                "lower_bound": round(prediction_value * 0.85, 1),
                "upper_bound": round(prediction_value * 1.15, 1)
            }
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Predict disruption ridership (new method)
@app.post("/predict_disruption_ridership")
async def predict_disruption_ridership(request: RidershipRequest):
    if ridership_predictor is None:
        raise HTTPException(status_code=503, detail="Hourly ridership predictor not available")
    
    try:
        result = ridership_predictor.predict_at_disruption(
            route_no=request.route_no,
            disruption_datetime=request.disruption_datetime,
            depot=request.depot
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
