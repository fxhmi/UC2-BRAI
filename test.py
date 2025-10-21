import requests
import json

API_URL = "https://rapidkl-bus-assist-api.onrender.com"

print("🧪 Testing Render API...")

# Test 1: Health check
print("\n1️⃣ Health check...")
response = requests.get(f"{API_URL}/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# Test 2: Predict best route
print("\n2️⃣ Testing route prediction...")
payload = {
    "lat_disruption": 3.0943772,
    "lng_disruption": 101.6009691,
    "passenger_on_bus": 30,
    "disruption_weather_impact": 2,
    "is_weekend": 0,
    "is_peak_hour_encoded": 1,
    "bus_replacement_priority": 1,
    "bus_replacement_route_type_encoded": 1,
    "deadmileage_to_disruption": 1.0,
    "geo_distance_to_disruption": 1.0,
    "travel_time_min_from_hub": 15.0
}
response = requests.post(f"{API_URL}/predict_best_route", json=payload)
print(f"Status: {response.status_code}")
print(f"Best route: {response.json().get('best_route')}")

# Test 3: Predict ridership
print("\n3️⃣ Testing ridership prediction...")
payload = {
    "route_no": "506",
    "disruption_datetime": "2024-10-21 14:30"
}
response = requests.post(f"{API_URL}/predict_disruption_ridership", json=payload)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"✅ Predicted ridership: {data['predicted_ridership']} passengers")
    print(f"   Confidence: {data['confidence_score']:.0%}")
    print(f"   Range: {data['prediction_range']['lower_bound']:.0f}-{data['prediction_range']['upper_bound']:.0f}")
else:
    print(f"❌ Error: {response.text}")

print("\n✅ Tests complete!")
