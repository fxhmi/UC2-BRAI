import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
import re
warnings.filterwarnings('ignore')

class DisruptionRidershipPredictor:
    """
    Predict ridership at exact time and location of disruption
    Production-ready with adaptive confidence and risk assessment
    """
    
    def __init__(self, model_dir='models', data_dir='data'):
        """Initialize predictor with trained models - memory optimized"""
        self.model_dir = model_dir
        self.data_dir = data_dir
        
        # Load models
        try:
            self.model = joblib.load(os.path.join(model_dir, 'xgb_hourly_ridership_model.pkl'))
            self.scaler = joblib.load(os.path.join(model_dir, 'xgb_hourly_ridership_scaler.pkl'))
            self.route_encoder = joblib.load(os.path.join(model_dir, 'encoders/route_encoder.pkl'))
            self.time_cat_encoder = joblib.load(os.path.join(model_dir, 'encoders/time_category_encoder.pkl'))
        except Exception as e:
            raise Exception(f"Error loading models: {e}")
        
        # Load feature list
        try:
            with open(os.path.join(data_dir, 'feature_columns.txt'), 'r') as f:
                self.feature_columns = [line.strip() for line in f.readlines()]
        except Exception as e:
            raise Exception(f"Error loading feature list: {e}")
        
        # Load route statistics
        try:
            csv_path = os.path.join(data_dir, 'ridership_prepared_hourly.csv')
            if not os.path.exists(csv_path):
                csv_path = csv_path + '.gz'
            
            df_stats = pd.read_csv(csv_path)
            df_stats['route_no'] = df_stats['route_no'].astype(str)
            
            self.route_stats = df_stats.groupby('route_no').agg({
                'ridership_total': ['mean', 'std', 'max', 'min'],
                'depot': 'first'
            }).reset_index()
            self.route_stats.columns = ['route_no', 'route_avg_ridership', 
                                         'route_std_ridership', 'route_max_ridership',
                                         'route_min_ridership', 'depot']
            
            self.hourly_stats = df_stats.groupby(['route_no', 'hour'])['ridership_total'].mean().reset_index()
            self.hourly_stats.columns = ['route_no', 'hour', 'avg_ridership']
            
            self.dow_stats = df_stats.groupby(['route_no', 'day_of_week'])['ridership_total'].mean().reset_index()
            self.dow_stats.columns = ['route_no', 'day_of_week', 'avg_ridership']
            
        except Exception as e:
            raise Exception(f"Error loading route statistics: {e}")
    
    def get_available_routes(self):
        """Get list of all available routes"""
        routes = self.route_stats['route_no'].unique().tolist()
        
        def natural_sort_key(route):
            parts = re.split(r'(\d+)', str(route))
            return [int(part) if part.isdigit() else part for part in parts]
        
        try:
            return sorted(routes, key=natural_sort_key)
        except:
            return sorted([str(r) for r in routes])
    
    def predict_at_disruption(self, route_no, disruption_datetime, depot=None):
        """
        Predict ridership at exact time of disruption
        
        Parameters:
        - route_no: str (e.g., '506', '600')
        - disruption_datetime: datetime object or string ('2024-10-21 14:30')
        - depot: int (optional, will lookup if not provided)
        
        Returns:
        - dict with prediction, confidence, and details
        """
        
        route_no = str(route_no).strip()
        
        if isinstance(disruption_datetime, str):
            disruption_datetime = pd.to_datetime(disruption_datetime)
        
        # Extract temporal features
        hour = disruption_datetime.hour
        day_of_week = disruption_datetime.dayofweek
        month = disruption_datetime.month
        week_of_year = disruption_datetime.isocalendar().week
        day_of_year = disruption_datetime.dayofyear
        
        is_weekend = int(day_of_week >= 5)
        is_morning_peak = int(7 <= hour <= 9)
        is_evening_peak = int(17 <= hour <= 19)
        is_peak_hour = int(is_morning_peak or is_evening_peak)
        
        # Time category
        if 0 <= hour < 6:
            time_category = 'early_morning'
        elif 6 <= hour < 10:
            time_category = 'morning_peak'
        elif 10 <= hour < 16:
            time_category = 'midday'
        elif 16 <= hour < 20:
            time_category = 'evening_peak'
        else:
            time_category = 'night'
        
        # Malaysian holidays 2024-2025
        holidays = pd.to_datetime([
            '2024-01-01', '2024-01-25', '2024-02-01', '2024-02-10', '2024-02-11',
            '2024-03-28', '2024-04-10', '2024-04-11', '2024-05-01', '2024-05-22',
            '2024-06-03', '2024-06-17', '2024-07-07', '2024-08-31', '2024-09-16',
            '2024-10-24', '2024-11-01', '2024-12-25',
            '2025-01-01', '2025-01-29', '2025-02-01', '2025-03-31', '2025-04-01',
            '2025-05-01', '2025-05-12', '2025-06-06', '2025-08-31', '2025-09-16',
            '2025-10-24', '2025-10-31', '2025-12-25'
        ])
        is_public_holiday = int(disruption_datetime.date() in [h.date() for h in holidays])
        
        # School holidays
        school_periods = [
            (datetime(2024, 3, 16), datetime(2024, 3, 24)),
            (datetime(2024, 5, 25), datetime(2024, 6, 9)),
            (datetime(2024, 8, 24), datetime(2024, 9, 1)),
            (datetime(2024, 11, 16), datetime(2024, 12, 31)),
            (datetime(2025, 3, 15), datetime(2025, 3, 23)),
            (datetime(2025, 5, 24), datetime(2025, 6, 8)),
            (datetime(2025, 8, 23), datetime(2025, 8, 31)),
            (datetime(2025, 11, 15), datetime(2025, 12, 31)),
        ]
        is_school_holiday = int(any(start <= disruption_datetime <= end for start, end in school_periods))
        
        # Get route statistics
        route_data = self.route_stats[self.route_stats['route_no'] == route_no]
        if len(route_data) == 0:
            available = self.get_available_routes()
            return {
                'error': f'Route {route_no} not found in training data',
                'available_routes': available[:20],
                'total_available': len(available)
            }
        
        route_avg = route_data['route_avg_ridership'].values[0]
        route_std = route_data['route_std_ridership'].values[0]
        route_max = route_data['route_max_ridership'].values[0]
        route_min = route_data['route_min_ridership'].values[0]
        depot_val = depot if depot is not None else int(route_data['depot'].values[0])
        
        # Get hourly average
        hourly_data = self.hourly_stats[(self.hourly_stats['route_no'] == route_no) & 
                                         (self.hourly_stats['hour'] == hour)]
        route_hour_avg = hourly_data['avg_ridership'].values[0] if len(hourly_data) > 0 else route_avg
        
        # Get day-of-week average
        dow_data = self.dow_stats[(self.dow_stats['route_no'] == route_no) & 
                                   (self.dow_stats['day_of_week'] == day_of_week)]
        route_dow_avg = dow_data['avg_ridership'].values[0] if len(dow_data) > 0 else route_avg
        
        # Encode categorical
        try:
            route_no_encoded = self.route_encoder.transform([route_no])[0]
            time_cat_encoded = self.time_cat_encoder.transform([time_category])[0]
        except Exception as e:
            return {
                'error': f'Unable to encode route {route_no}',
                'details': str(e)
            }
        
        # Build features
        features_dict = {
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'week_of_year': week_of_year,
            'day_of_year': day_of_year,
            'is_weekend': is_weekend,
            'is_morning_peak': is_morning_peak,
            'is_evening_peak': is_evening_peak,
            'is_peak_hour': is_peak_hour,
            'is_public_holiday': is_public_holiday,
            'is_school_holiday': is_school_holiday,
            'route_no_encoded': route_no_encoded,
            'depot': depot_val,
            'time_category_encoded': time_cat_encoded,
            'route_avg_ridership': route_avg,
            'route_std_ridership': route_std,
            'route_max_ridership': route_max,
            'route_hour_avg_ridership': route_hour_avg,
            'route_dow_avg_ridership': route_dow_avg,
            'ridership_lag_1d': route_hour_avg,
            'ridership_lag_7d': route_hour_avg,
            'ridership_rolling_3h': route_avg,
            'ridership_rolling_7d': route_avg,
            'peak_route_interaction': is_peak_hour * route_avg,
            'weekend_hour_interaction': is_weekend * hour,
            'holiday_weekend_interaction': is_public_holiday * is_weekend
        }
        
        # Predict
        X = np.array([[features_dict[col] for col in self.feature_columns]])
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        prediction = max(0, min(235, prediction))
        
        # Calculate confidence
        if prediction > 20:
            confidence = 0.90
        elif prediction > 10:
            confidence = 0.85
        elif prediction > 5:
            confidence = 0.75
        else:
            confidence = 0.65
        
        if is_peak_hour:
            confidence = min(0.95, confidence + 0.05)
        if is_public_holiday:
            confidence -= 0.08
        if is_school_holiday:
            confidence -= 0.05
        if hour < 6 or hour > 22:
            confidence -= 0.10
        
        confidence = max(0.50, min(0.95, confidence))
        
        # Prediction range
        mae = 4.24
        if prediction > 20:
            range_multiplier = 1.0
        elif prediction > 10:
            range_multiplier = 1.2
        elif prediction > 5:
            range_multiplier = 1.5
        else:
            range_multiplier = 2.0
        
        if is_public_holiday or is_school_holiday:
            range_multiplier *= 1.2
        if hour < 6 or hour > 22:
            range_multiplier *= 1.3
        
        lower_bound = max(0, prediction - mae * range_multiplier)
        upper_bound = min(235, prediction + mae * range_multiplier)
        
        # Risk assessment
        if prediction > 40:
            risk_level = "High"
            risk_note = "High demand - prioritize this route"
        elif prediction > 25:
            risk_level = "Medium"
            risk_note = "Moderate demand expected"
        elif prediction > 10:
            risk_level = "Low"
            risk_note = "Normal demand levels"
        else:
            risk_level = "Very Low"
            risk_note = "Low demand expected"
        
        if confidence >= 0.85:
            reliability = "High"
        elif confidence >= 0.75:
            reliability = "Good"
        elif confidence >= 0.65:
            reliability = "Moderate"
        else:
            reliability = "Low"
        
        return {
            'route_no': route_no,
            'disruption_datetime': disruption_datetime.strftime('%Y-%m-%d %H:%M'),
            'predicted_ridership': round(prediction, 1),
            'confidence_score': round(confidence, 2),
            'reliability': reliability,
            'prediction_range': {
                'lower_bound': round(lower_bound, 1),
                'upper_bound': round(upper_bound, 1)
            },
            'risk_assessment': {
                'level': risk_level,
                'note': risk_note
            },
            'conditions': {
                'hour': hour,
                'is_peak_hour': bool(is_peak_hour),
                'is_weekend': bool(is_weekend),
                'time_category': time_category
            }
        }
    
    def predict_multiple_routes(self, routes, disruption_datetime, depot=None):
        """Predict for multiple routes"""
        return [self.predict_at_disruption(r, disruption_datetime, depot) for r in routes]
    
    def get_route_info(self, route_no):
        """Get route statistics"""
        route_no = str(route_no).strip()
        route_data = self.route_stats[self.route_stats['route_no'] == route_no]
        
        if len(route_data) == 0:
            return {'error': f'Route {route_no} not found'}
        
        return {
            'route_no': route_no,
            'average_ridership': round(route_data['route_avg_ridership'].values[0], 1),
            'max_ridership': round(route_data['route_max_ridership'].values[0], 1),
            'depot': int(route_data['depot'].values[0])
        }
