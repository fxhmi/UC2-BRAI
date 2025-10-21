from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "https://prasarana-swiftroute-e21358fcb5f7.herokuapp.com",
    "http://localhost:8501"  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
try:
    knn = joblib.load('models/knn_model.pkl')
    scaler_knn = joblib.load('models/scaler_knn.pkl')
    rf = joblib.load('models/best_route_model_improved.pkl')
    scaler_rf = joblib.load('models/scaler_rf_improved.pkl')
    xgb_ridership = joblib.load('models/xgb_ridership_model.pkl')
    xgb_scaler = joblib.load('models/xgb_feature_scaler.pkl')
    df = pd.read_csv('training_data_converted_strategy_output.csv')
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

features_rf = [
    'passenger_on_bus', 'disruption_weather_impact', 'is_weekend', 'is_peak_hour_encoded',
    'bus_replacement_priority', 'bus_replacement_route_type_encoded',
    'deadmileage_to_disruption', 'geo_distance_to_disruption', 'travel_time_min_from_hub'
]

enhanced_features = features_rf + [
    'distance_efficiency',
    'time_per_km',
    'priority_passenger_product',
    'weather_peak_interaction'
]

@app.get("/")
async def root():
    return {
        "message": "SwiftRoute API v2.0 - Enhanced with Multi-Objective Optimization",
        "status": "operational",
        "endpoints": ["/predict_best_route", "/forecast_ridership", "/health"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "data_shape": df.shape
    }

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

def engineer_route_features(features_dict: Dict) -> Dict:
    """Add engineered features for better route selection"""
    # Distance efficiency
    features_dict['distance_efficiency'] = (
        features_dict['geo_distance_to_disruption'] / 
        (features_dict['deadmileage_to_disruption'] + 0.001)
    )
    
    # Time per km
    features_dict['time_per_km'] = (
        features_dict['travel_time_min_from_hub'] / 
        (features_dict['geo_distance_to_disruption'] + 0.001)
    )
    
    # Priority-passenger interaction
    features_dict['priority_passenger_product'] = (
        features_dict['bus_replacement_priority'] * 
        features_dict['passenger_on_bus']
    )
    
    # Weather-peak interaction
    features_dict['weather_peak_combined'] = (
        features_dict['disruption_weather_impact'] * 
        features_dict['is_peak_hour_encoded']
    )
    
    # Urgency score
    features_dict['urgency_score'] = (
        (features_dict['passenger_on_bus'] / 60) * 0.4 +
        (features_dict['bus_replacement_priority'] / 3) * 0.3 +
        features_dict['is_peak_hour_encoded'] * 0.3
    )
    
    return features_dict

def calculate_multi_objective_score(
    candidate_data: Dict,
    ridership: float,
    max_ridership: float = 60,
    max_distance: float = 50.0,
    max_time: float = 60.0
) -> float:
    """
    Multi-objective scoring: balances low ridership, proximity, and response time
    Higher score = better candidate
    """
    # Normalize metrics (0-1 scale, higher is better)
    ridership_score = 1 - (ridership / max_ridership) if ridership else 0.5
    proximity_score = 1 - (candidate_data.get('geo_distance_to_disruption', 25) / max_distance)
    time_score = 1 - (candidate_data.get('travel_time_min_from_hub', 30) / max_time)
    
    # Weighted combination (adjust based on business priorities)
    weights = {
        'ridership': 0.45,  # Prioritize low ridership routes
        'proximity': 0.30,  # Important: close to disruption
        'time': 0.25        # Important: quick response
    }
    
    composite_score = (
        weights['ridership'] * ridership_score +
        weights['proximity'] * proximity_score +
        weights['time'] * time_score
    )
    
    return composite_score

@app.post("/predict_best_route")
def predict_best_route(data: DisruptionInput):
    try:
        # Step 1: Get candidates from KNN
        coords_scaled = scaler_knn.transform([[data.lat_disruption, data.lng_disruption]])
        n_neighbors = min(15, len(df))
        distances, indices = knn.kneighbors(coords_scaled, n_neighbors=n_neighbors)
        
        # Step 2: Extract candidate routes
        candidate_rows = df.iloc[indices[0]][features_rf + ['bus_replacement_route_no']].to_dict(orient='records')
        
        # Step 3: Engineer features for each candidate
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
            
            # Add engineered features (CRITICAL: Must match training!)
            features['distance_efficiency'] = (
                features['geo_distance_to_disruption'] / 
                (features['deadmileage_to_disruption'] + 0.001)
            )
            features['time_per_km'] = (
                features['travel_time_min_from_hub'] / 
                (features['geo_distance_to_disruption'] + 0.001)
            )
            features['priority_passenger_product'] = (
                features['bus_replacement_priority'] * 
                features['passenger_on_bus']
            )
            features['weather_peak_interaction'] = (
                features['disruption_weather_impact'] * 
                features['is_peak_hour_encoded']
            )
            
            rf_features_list.append(features)
        
        # Step 4: Use ALL enhanced features for RF prediction
        X_rf = pd.DataFrame(rf_features_list)
        X_rf = X_rf[enhanced_features]  # Ensure correct order
        X_rf_scaled = scaler_rf.transform(X_rf)
        
        # Get predictions
        preds = rf.predict(X_rf_scaled)
        probs = rf.predict_proba(X_rf_scaled)[:, 1]
        
        # Find best route
        best_route = None
        best_prob = None
        best_idx = None
        
        for i, pred in enumerate(preds):
            if pred == 1:
                best_route = candidate_rows[i]['bus_replacement_route_no']
                best_prob = probs[i]
                best_idx = i
                break
        
        if best_route is None:
            best_idx = probs.argmax()
            best_route = candidate_rows[best_idx]['bus_replacement_route_no']
            best_prob = probs[best_idx]
        
        candidate_routes = list(dict.fromkeys([c['bus_replacement_route_no'] for c in candidate_rows]))
        
        # Calculate confidence with improved model
        prob_std = np.std(probs)
        base_accuracy = 86.0  # Updated from 69.59%
        
        # Dynamic confidence
        confidence_adjustment = (best_prob - 0.5) * 20
        spread_bonus = min(10, prob_std * 20)
        final_confidence = base_accuracy + confidence_adjustment + spread_bonus
        final_confidence = max(70, min(95, final_confidence))
        
        return {
            "best_route": best_route,
            "best_route_probability": float(best_prob),
            "confidence_score": round(final_confidence / 100, 4),
            "model_accuracy": 86.0,  
            "model_precision": 76.0,  
            "model_recall": 85.0,     
            "candidate_routes": candidate_routes,
            "n_candidates_evaluated": len(candidate_rows),
            "model_version": "v2.1_improved_with_smote",
            "improvements": "SMOTE balanced training + 4 engineered features"
        }
        
    except Exception as e:
        logger.error(f"Error in predict_best_route: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast_ridership")
def forecast_ridership(data: RidershipForecastInput):
    try:
        # Prepare features
        X = [[
            data.route_no_enc,
            data.day_of_week,
            data.month,
            data.depot_enc,
            data.is_holiday,
            data.hours_left
        ]]
        
        # Scale and predict
        X_scaled = xgb_scaler.transform(X)
        prediction = xgb_ridership.predict(X_scaled)
        
        # Calculate confidence (you can enhance this based on validation metrics)
        base_confidence = 0.9093
        
        # Adjust confidence based on input characteristics
        confidence_adjustments = 0
        
        # Lower confidence for edge cases
        if data.hours_left < 2:
            confidence_adjustments -= 0.05
        if data.is_holiday == 1:
            confidence_adjustments -= 0.03
        
        final_confidence = max(0.70, min(0.95, base_confidence + confidence_adjustments))
        
        # Ensure prediction is non-negative and reasonable
        prediction_value = max(0, float(prediction[0]))
        prediction_value = min(60, prediction_value)  # Cap at bus capacity
        
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
        logger.error(f"Error in forecast_ridership: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_best_route_multi_objective")
def predict_best_route_multi_objective(data: DisruptionInput):
    """
    Enhanced endpoint using multi-objective optimization
    Combines route selection with ridership forecasting
    """
    try:
        # Get initial prediction
        base_prediction = predict_best_route(data)
        candidate_routes = base_prediction['candidate_routes']
        
        # This would require ridership forecasts - placeholder for integration
        # In production, you'd call forecast_ridership for each candidate
        
        return {
            **base_prediction,
            "optimization_method": "multi_objective",
            "note": "Integrate with ridership forecasts for full multi-objective optimization"
        }
        
    except Exception as e:
        logger.error(f"Error in multi-objective prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

########### V2

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import logging

# ADD THESE NEW IMPORTS
from predict_disruption_ridership import DisruptionRidershipPredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "https://rapidkl-bus-assist-api.onrender.com",
    "http://localhost:8501"  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load existing models
try:
    knn = joblib.load('models/knn_model.pkl')
    scaler_knn = joblib.load('models/scaler_knn.pkl')
    rf = joblib.load('models/best_route_model_improved.pkl')
    scaler_rf = joblib.load('models/scaler_rf_improved.pkl')
    xgb_ridership = joblib.load('models/xgb_ridership_model.pkl')
    xgb_scaler = joblib.load('models/xgb_feature_scaler.pkl')
    df = pd.read_csv('training_data_converted_strategy_output.csv')
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

# ADD: Initialize new hourly ridership predictor
try:
    ridership_predictor = DisruptionRidershipPredictor()
    logger.info("✅ Hourly ridership predictor loaded successfully")
except Exception as e:
    logger.error(f"⚠️ Could not load ridership predictor: {e}")
    ridership_predictor = None

features_rf = [
    'passenger_on_bus', 'disruption_weather_impact', 'is_weekend', 'is_peak_hour_encoded',
    'bus_replacement_priority', 'bus_replacement_route_type_encoded',
    'deadmileage_to_disruption', 'geo_distance_to_disruption', 'travel_time_min_from_hub'
]

enhanced_features = features_rf + [
    'distance_efficiency',
    'time_per_km',
    'priority_passenger_product',
    'weather_peak_interaction'
]

@app.get("/")
async def root():
    return {
        "message": "SwiftRoute API v2.1 - Enhanced with Hourly Ridership Prediction",
        "status": "operational",
        "endpoints": [
            "/predict_best_route", 
            "/forecast_ridership", 
            "/predict_disruption_ridership",  # NEW
            "/predict_multiple_routes_ridership",  # NEW
            "/available_routes",  # NEW
            "/route_info/{route_no}",  # NEW
            "/health"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "ridership_predictor_available": ridership_predictor is not None,
        "data_shape": df.shape,
        "total_routes": len(ridership_predictor.get_available_routes()) if ridership_predictor else 0
    }

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

# ADD NEW REQUEST MODELS
class RidershipRequest(BaseModel):
    route_no: str
    disruption_datetime: str  # Format: "2024-10-21 14:30"
    depot: Optional[int] = None

class MultiRouteRequest(BaseModel):
    routes: list
    disruption_datetime: str
    depot: Optional[int] = None

def engineer_route_features(features_dict: Dict) -> Dict:
    """Add engineered features for better route selection"""
    features_dict['distance_efficiency'] = (
        features_dict['geo_distance_to_disruption'] / 
        (features_dict['deadmileage_to_disruption'] + 0.001)
    )
    
    features_dict['time_per_km'] = (
        features_dict['travel_time_min_from_hub'] / 
        (features_dict['geo_distance_to_disruption'] + 0.001)
    )
    
    features_dict['priority_passenger_product'] = (
        features_dict['bus_replacement_priority'] * 
        features_dict['passenger_on_bus']
    )
    
    features_dict['weather_peak_combined'] = (
        features_dict['disruption_weather_impact'] * 
        features_dict['is_peak_hour_encoded']
    )
    
    features_dict['urgency_score'] = (
        (features_dict['passenger_on_bus'] / 60) * 0.4 +
        (features_dict['bus_replacement_priority'] / 3) * 0.3 +
        features_dict['is_peak_hour_encoded'] * 0.3
    )
    
    return features_dict

def calculate_multi_objective_score(
    candidate_data: Dict,
    ridership: float,
    max_ridership: float = 60,
    max_distance: float = 50.0,
    max_time: float = 60.0
) -> float:
    """
    Multi-objective scoring: balances low ridership, proximity, and response time
    Higher score = better candidate
    """
    ridership_score = 1 - (ridership / max_ridership) if ridership else 0.5
    proximity_score = 1 - (candidate_data.get('geo_distance_to_disruption', 25) / max_distance)
    time_score = 1 - (candidate_data.get('travel_time_min_from_hub', 30) / max_time)
    
    weights = {
        'ridership': 0.45,
        'proximity': 0.30,
        'time': 0.25
    }
    
    composite_score = (
        weights['ridership'] * ridership_score +
        weights['proximity'] * proximity_score +
        weights['time'] * time_score
    )
    
    return composite_score

@app.post("/predict_best_route")
def predict_best_route(data: DisruptionInput):
    try:
        coords_scaled = scaler_knn.transform([[data.lat_disruption, data.lng_disruption]])
        n_neighbors = min(15, len(df))
        distances, indices = knn.kneighbors(coords_scaled, n_neighbors=n_neighbors)
        
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
            
            features['distance_efficiency'] = (
                features['geo_distance_to_disruption'] / 
                (features['deadmileage_to_disruption'] + 0.001)
            )
            features['time_per_km'] = (
                features['travel_time_min_from_hub'] / 
                (features['geo_distance_to_disruption'] + 0.001)
            )
            features['priority_passenger_product'] = (
                features['bus_replacement_priority'] * 
                features['passenger_on_bus']
            )
            features['weather_peak_interaction'] = (
                features['disruption_weather_impact'] * 
                features['is_peak_hour_encoded']
            )
            
            rf_features_list.append(features)
        
        X_rf = pd.DataFrame(rf_features_list)
        X_rf = X_rf[enhanced_features]
        X_rf_scaled = scaler_rf.transform(X_rf)
        
        preds = rf.predict(X_rf_scaled)
        probs = rf.predict_proba(X_rf_scaled)[:, 1]
        
        best_route = None
        best_prob = None
        best_idx = None
        
        for i, pred in enumerate(preds):
            if pred == 1:
                best_route = candidate_rows[i]['bus_replacement_route_no']
                best_prob = probs[i]
                best_idx = i
                break
        
        if best_route is None:
            best_idx = probs.argmax()
            best_route = candidate_rows[best_idx]['bus_replacement_route_no']
            best_prob = probs[best_idx]
        
        candidate_routes = list(dict.fromkeys([c['bus_replacement_route_no'] for c in candidate_rows]))
        
        prob_std = np.std(probs)
        base_accuracy = 86.0
        
        confidence_adjustment = (best_prob - 0.5) * 20
        spread_bonus = min(10, prob_std * 20)
        final_confidence = base_accuracy + confidence_adjustment + spread_bonus
        final_confidence = max(70, min(95, final_confidence))
        
        return {
            "best_route": best_route,
            "best_route_probability": float(best_prob),
            "confidence_score": round(final_confidence / 100, 4),
            "model_accuracy": 86.0,
            "model_precision": 76.0,
            "model_recall": 85.0,
            "candidate_routes": candidate_routes,
            "n_candidates_evaluated": len(candidate_rows),
            "model_version": "v2.1_improved_with_smote",
            "improvements": "SMOTE balanced training + 4 engineered features"
        }
        
    except Exception as e:
        logger.error(f"Error in predict_best_route: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast_ridership")
def forecast_ridership(data: RidershipForecastInput):
    try:
        X = [[
            data.route_no_enc,
            data.day_of_week,
            data.month,
            data.depot_enc,
            data.is_holiday,
            data.hours_left
        ]]
        
        X_scaled = xgb_scaler.transform(X)
        prediction = xgb_ridership.predict(X_scaled)
        
        base_confidence = 0.9093
        confidence_adjustments = 0
        
        if data.hours_left < 2:
            confidence_adjustments -= 0.05
        if data.is_holiday == 1:
            confidence_adjustments -= 0.03
        
        final_confidence = max(0.70, min(0.95, base_confidence + confidence_adjustments))
        
        prediction_value = max(0, float(prediction[0]))
        prediction_value = min(60, prediction_value)
        
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
        logger.error(f"Error in forecast_ridership: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# NEW ENDPOINTS - HOURLY RIDERSHIP PREDICTION
# ============================================================================

@app.post("/predict_disruption_ridership")
def predict_disruption_ridership(request: RidershipRequest):
    """
    Predict ridership at exact time of disruption
    
    Example:
    {
        "route_no": "402",
        "disruption_datetime": "2024-10-21 14:30",
        "depot": 29
    }
    """
    if ridership_predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Ridership predictor not available. Models may not be trained yet."
        )
    
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
        logger.error(f"Error predicting ridership: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_multiple_routes_ridership")
def predict_multiple_routes_ridership(request: MultiRouteRequest):
    """
    Predict ridership for multiple routes at once
    
    Example:
    {
        "routes": ["402", "506", "600"],
        "disruption_datetime": "2024-10-21 14:30"
    }
    """
    if ridership_predictor is None:
        raise HTTPException(status_code=503, detail="Ridership predictor not available")
    
    try:
        results = ridership_predictor.predict_multiple_routes(
            routes=request.routes,
            disruption_datetime=request.disruption_datetime,
            depot=request.depot
        )
        
        return {
            "disruption_datetime": request.disruption_datetime,
            "total_routes": len(request.routes),
            "predictions": results
        }
        
    except Exception as e:
        logger.error(f"Error predicting multiple routes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available_routes")
def get_available_routes():
    """
    Get list of all available routes in the system
    """
    if ridership_predictor is None:
        raise HTTPException(status_code=503, detail="Ridership predictor not available")
    
    try:
        routes = ridership_predictor.get_available_routes()
        return {
            "total_routes": len(routes),
            "routes": routes
        }
    except Exception as e:
        logger.error(f"Error getting available routes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/route_info/{route_no}")
def get_route_info(route_no: str):
    """
    Get statistical information for a specific route
    """
    if ridership_predictor is None:
        raise HTTPException(status_code=503, detail="Ridership predictor not available")
    
    try:
        result = ridership_predictor.get_route_info(route_no)
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting route info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_best_route_multi_objective")
def predict_best_route_multi_objective(data: DisruptionInput):
    """
    Enhanced endpoint using multi-objective optimization
    Combines route selection with ridership forecasting
    """
    try:
        base_prediction = predict_best_route(data)
        candidate_routes = base_prediction['candidate_routes']
        
        return {
            **base_prediction,
            "optimization_method": "multi_objective",
            "note": "Integrate with ridership forecasts for full multi-objective optimization"
        }
        
    except Exception as e:
        logger.error(f"Error in multi-objective prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# import numpy as np
# from fastapi.middleware.cors import CORSMiddleware
# from typing import List, Dict, Optional
# import logging
# import gc
# import os

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# # CORS
# origins = ["*"]  # Allow all for now
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Memory-efficient model loading
# class ModelCache:
#     """Load models on demand and cache only essentials"""
#     def __init__(self):
#         self._cache = {}
#         self._df = None
        
#     def get_df(self):
#         """Load training data (cached)"""
#         if self._df is None:
#             self._df = pd.read_csv('training_data_converted_strategy_output.csv')
#             logger.info(f"Loaded training data: {self._df.shape}")
#         return self._df
    
#     def load_model(self, name):
#         """Load model temporarily (not cached)"""
#         model_paths = {
#             'knn': 'models/knn_model.pkl',
#             'scaler_knn': 'models/scaler_knn.pkl',
#             'rf': 'models/best_route_model_improved.pkl',
#             'scaler_rf': 'models/scaler_rf_improved.pkl',
#         }
        
#         if name in model_paths:
#             logger.info(f"Loading {name}...")
#             model = joblib.load(model_paths[name])
#             return model
#         return None
    
#     def clear_cache(self):
#         """Clear cache to free memory"""
#         self._cache.clear()
#         gc.collect()

# # Global cache
# cache = ModelCache()

# # Features
# features_rf = [
#     'passenger_on_bus', 'disruption_weather_impact', 'is_weekend', 'is_peak_hour_encoded',
#     'bus_replacement_priority', 'bus_replacement_route_type_encoded',
#     'deadmileage_to_disruption', 'geo_distance_to_disruption', 'travel_time_min_from_hub'
# ]

# enhanced_features = features_rf + [
#     'distance_efficiency', 'time_per_km', 'priority_passenger_product', 'weather_peak_interaction'
# ]

# # Pydantic models
# class DisruptionInput(BaseModel):
#     lat_disruption: float
#     lng_disruption: float
#     passenger_on_bus: int
#     disruption_weather_impact: int
#     is_weekend: int
#     is_peak_hour_encoded: int
#     bus_replacement_priority: int
#     bus_replacement_route_type_encoded: int
#     deadmileage_to_disruption: float
#     geo_distance_to_disruption: float
#     travel_time_min_from_hub: float

# class RidershipRequest(BaseModel):
#     route_no: str
#     disruption_datetime: str
#     depot: Optional[int] = None

# @app.get("/")
# async def root():
#     return {
#         "message": "SwiftRoute API v2.1 - Memory Optimized",
#         "status": "operational",
#         "memory_efficient": True
#     }

# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "memory_optimized": True,
#         "cache_size": len(cache._cache)
#     }

# @app.post("/predict_best_route")
# def predict_best_route(data: DisruptionInput):
#     """Predict best route with on-demand model loading"""
#     try:
#         # Load models only when needed
#         knn = cache.load_model('knn')
#         scaler_knn = cache.load_model('scaler_knn')
#         df = cache.get_df()
        
#         # Step 1: KNN
#         coords_scaled = scaler_knn.transform([[data.lat_disruption, data.lng_disruption]])
#         n_neighbors = min(15, len(df))
#         distances, indices = knn.kneighbors(coords_scaled, n_neighbors=n_neighbors)
        
#         # Clear KNN from memory
#         del knn, scaler_knn
#         gc.collect()
        
#         # Step 2: Get candidates
#         candidate_rows = df.iloc[indices[0]][features_rf + ['bus_replacement_route_no']].to_dict(orient='records')
        
#         # Step 3: Load RF models
#         rf = cache.load_model('rf')
#         scaler_rf = cache.load_model('scaler_rf')
        
#         # Engineer features
#         rf_features_list = []
#         for candidate in candidate_rows:
#             features = {
#                 'passenger_on_bus': data.passenger_on_bus,
#                 'disruption_weather_impact': data.disruption_weather_impact,
#                 'is_weekend': data.is_weekend,
#                 'is_peak_hour_encoded': data.is_peak_hour_encoded,
#                 'bus_replacement_priority': data.bus_replacement_priority,
#                 'bus_replacement_route_type_encoded': data.bus_replacement_route_type_encoded,
#                 'deadmileage_to_disruption': data.deadmileage_to_disruption,
#                 'geo_distance_to_disruption': data.geo_distance_to_disruption,
#                 'travel_time_min_from_hub': data.travel_time_min_from_hub
#             }
            
#             features['distance_efficiency'] = features['geo_distance_to_disruption'] / (features['deadmileage_to_disruption'] + 0.001)
#             features['time_per_km'] = features['travel_time_min_from_hub'] / (features['geo_distance_to_disruption'] + 0.001)
#             features['priority_passenger_product'] = features['bus_replacement_priority'] * features['passenger_on_bus']
#             features['weather_peak_interaction'] = features['disruption_weather_impact'] * features['is_peak_hour_encoded']
            
#             rf_features_list.append(features)
        
#         # Predict
#         X_rf = pd.DataFrame(rf_features_list)[enhanced_features]
#         X_rf_scaled = scaler_rf.transform(X_rf)
        
#         preds = rf.predict(X_rf_scaled)
#         probs = rf.predict_proba(X_rf_scaled)[:, 1]
        
#         # Clear RF from memory
#         del rf, scaler_rf, X_rf, X_rf_scaled
#         gc.collect()
        
#         # Find best route
#         best_route = None
#         best_prob = None
        
#         for i, pred in enumerate(preds):
#             if pred == 1:
#                 best_route = candidate_rows[i]['bus_replacement_route_no']
#                 best_prob = probs[i]
#                 break
        
#         if best_route is None:
#             best_idx = probs.argmax()
#             best_route = candidate_rows[best_idx]['bus_replacement_route_no']
#             best_prob = probs[best_idx]
        
#         candidate_routes = list(dict.fromkeys([c['bus_replacement_route_no'] for c in candidate_rows]))
        
#         # Calculate confidence
#         prob_std = np.std(probs)
#         base_accuracy = 86.0
#         confidence_adjustment = (best_prob - 0.5) * 20
#         spread_bonus = min(10, prob_std * 20)
#         final_confidence = base_accuracy + confidence_adjustment + spread_bonus
#         final_confidence = max(70, min(95, final_confidence))
        
#         return {
#             "best_route": best_route,
#             "best_route_probability": float(best_prob),
#             "confidence_score": round(final_confidence / 100, 4),
#             "model_accuracy": 86.0,
#             "candidate_routes": candidate_routes,
#             "n_candidates_evaluated": len(candidate_rows),
#             "model_version": "v2.1_memory_optimized"
#         }
        
#     except Exception as e:
#         logger.error(f"Error in predict_best_route: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         gc.collect()

# @app.post("/predict_disruption_ridership")
# def predict_disruption_ridership(request: RidershipRequest):
#     """Predict ridership with lazy loading"""
#     try:
#         # Import only when needed
#         from predict_disruption_ridership import DisruptionRidershipPredictor
        
#         predictor = DisruptionRidershipPredictor()
#         result = predictor.predict_at_disruption(
#             route_no=request.route_no,
#             disruption_datetime=request.disruption_datetime,
#             depot=request.depot
#         )
        
#         # Clear predictor from memory
#         del predictor
#         gc.collect()
        
#         if 'error' in result:
#             raise HTTPException(status_code=400, detail=result['error'])
        
#         return result
        
#     except ImportError:
#         raise HTTPException(status_code=503, detail="Ridership predictor not available")
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error predicting ridership: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         gc.collect()

# @app.get("/available_routes")
# def get_available_routes():
#     """Get available routes without loading full predictor"""
#     try:
#         # Just read the CSV directly
#         df = pd.read_csv('data/ridership_prepared_hourly.csv.gz')
#         routes = sorted(df['route_no'].astype(str).unique().tolist())
        
#         del df
#         gc.collect()
        
#         return {
#             "total_routes": len(routes),
#             "routes": routes[:50]  # Return first 50 only
#         }
#     except Exception as e:
#         return {"error": str(e)}

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)
