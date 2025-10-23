# import streamlit as st
# import requests
# import datetime
# import folium
# from streamlit_folium import folium_static
# from PIL import Image
# import plotly.express as px
# import plotly.graph_objects as go
# import pandas as pd
# from streamlit_javascript import st_javascript
# import re
# import base64
# import numpy as np

# API_URL = "https://prasarana-swiftroute-e21358fcb5f7.herokuapp.com"

# # JavaScript for Malaysia timezone
# js_code = """
# (() => {
#     const dtf = new Intl.DateTimeFormat('en-CA', {
#       timeZone: 'Asia/Kuala_Lumpur',
#       year: 'numeric',
#       month: '2-digit',
#       day: '2-digit',
#       hour: '2-digit',
#       minute: '2-digit',
#       second: '2-digit',
#       hour12: false
#     });
#     const [
#       { value: year },,
#       { value: month },,
#       { value: day },,
#       { value: hour },,
#       { value: minute },,
#       { value: second }
#     ] = dtf.formatToParts(new Date());
#     return `${year}-${month}-${day}T${hour}:${minute}:${second}`;
# })()
# """

# st.set_page_config(page_title="SwiftRoute AI v2.0", layout="wide")

# # [Keep your existing CSS styles - they're good]
# st.markdown(
#     """
#     <style>
#     body, .css-1d391kg, .css-1v3fvcr {
#         color: #c4c0c0 !important;  
#         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#     }
#     div.block-container {
#         padding-top: 0.25rem !important;
#         padding-bottom: 1rem !important;
#         padding-left: 1rem !important;
#         padding-right: 1rem !important;
#     }
#     h1, .css-1v3fvcr h1 {
#         margin-top: 0.1rem !important;
#         margin-bottom: 0.3rem !important;
#     }
#     .stButton button {
#         margin-top: 0.15rem !important;
#         padding-top: 0.5rem !important;
#         padding-bottom: 0.5rem !important;
#         background-color: #D50000 !important;
#         color: white !important;
#         font-weight: bold;
#         border-radius: 8px;
#     }
#     .stButton button:hover {
#         background-color: #FF3333 !important;
#         color: white !important;
#     }
#     [data-testid="stSidebar"] {
#         position: fixed !important;
#         top: 0 !important;
#         right: 0 !important;
#         width: 480px !important;
#         height: 100vh !important;
#         overflow-y: auto !important;
#         background-color: #002060 !important;
#         border-left: 3px solid #11134b !important;
#         box-shadow: -2px 0 8px rgba(0,0,0,0.09);
#         z-index: 9999 !important;
#     }
#     main > div.block-container {
#         max-width: 1300px !important;
#         margin-left: auto !important;
#         margin-right: 480px !important;
#         padding-left: 1rem !important;
#         padding-right: 1rem !important;
#     }
#     [data-testid="stSidebar"] label,
#     [data-testid="stSidebar"] div[data-baseweb="input"] > input,
#     [data-testid="stSidebar"] div[data-baseweb="select"] > div > div,
#     [data-testid="stSidebar"] div[data-baseweb="select"] span {
#         color: #fcfafa !important;
#     }
#     [data-testid="stSidebar"] div[data-baseweb="select"] > div > div,
#     [data-testid="stSidebar"] div[data-baseweb="select"]:focus > div,
#     [data-testid="stSidebar"] div[data-baseweb="input"] > input:focus {
#         color: black !important;
#         background-color: #ffffff !important;
#     }
#     [data-testid="stSidebar"] input::placeholder {
#         color: #666666 !important;
#     }
#     [data-testid="stSidebar"] div[data-baseweb="select"] {
#         background-color: #fff !important;
#     }
#     .stApp {
#         background-color: #f9f6ef;  
#     }
#     [data-testid="stSidebar"] span,
#     [data-testid="stSidebar"] label,
#     [data-testid="stSidebar"] p {
#         color: white !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # [Keep your header with logo - it's good]
# with open("assets/Prasarana_vertical-01-scaled-removebg-preview.jpg", "rb") as img_file:
#     logo_base64 = base64.b64encode(img_file.read()).decode()

# st.markdown(f"""
#     <div style='position: relative; top: 0px; left: 0; right: 0;
#                 width: 100%; max-width: 100%; margin: 0 auto;
#                 display: flex; align-self: normal; justify-content: space-between;
#                 padding: 14px 30px 14px 30px;
#                 background-color: #ffffffee;
#                 border-bottom: 1px solid #ccc;
#                 box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.07);
#                 border-radius: 16px 16px 0 0;
#                 z-index: 1000;'>
#         <div style='display: flex; align-items: center; gap: 24px;'>
#             <img src='data:image/png;base64,{logo_base64}' width='120' style='border-radius: 12px; box-shadow: 0 0 8px #eee;' />
#             <div>
#                 <h2 style='margin: 0; color: #1f4e79;'>AI Replacement Bus Launch Point Optimisation v2.0</h2>
#                 <p style='margin: 0; font-size: 19px; color: #444;'>Enhanced with Multi-Objective Optimization & Advanced Ridership Forecasting</p>
#             </div>
#         </div>
#     </div>
# """, unsafe_allow_html=True)

# # Initialize session state
# if "best_route_result" not in st.session_state:
#     st.session_state["best_route_result"] = None
# if "candidate_routes_result" not in st.session_state:
#     st.session_state["candidate_routes_result"] = None
# if "ridership_forecasts" not in st.session_state:
#     st.session_state["ridership_forecasts"] = {}
# if "disrupted_route_forecast" not in st.session_state:
#     st.session_state["disrupted_route_forecast"] = None
# if "guide_step" not in st.session_state:
#     st.session_state.guide_step = 0
# if "show_guide" not in st.session_state:
#     st.session_state.show_guide = True
# if "api_status" not in st.session_state:
#     st.session_state.api_status = "unknown"

# # [Keep your mappings]
# depot_mapping = {
#     "29": 0, "7": 1, "10": 2, "22": 3, "27": 4, "2": 5, "5": 6, "37": 7, "4": 8, "38": 9,
# }

# route_no_mapping = {
#     "PJ01": 0, "100": 1, "200": 2, "201": 3, "300": 4, "302": 5, "303": 6, "400": 7, "401": 8,
#     "500": 9, "600": 10, "601": 11, "T504": 12, "T505": 13, "T506": 14, "T507": 15, "T508": 16,
#     "T509": 17, "T510": 18, "T511": 19, "T512": 20, "T545": 21, "301": 22, "T201": 23, "SJ01": 24,
#     "SA03": 25, "SEWA1": 26, "T715": 27, "T753": 28, "T754": 29, "T756": 30, "T774": 31, "T776": 32,
#     "T778": 33, "T780": 34, "T781": 35, "T782": 36, "T786": 37, "T787": 38, "T788": 39, "T789": 40,
#     "T790": 41, "T791": 42, "402": 43, "751": 44, "752": 45, "754": 46, "770": 47, "771": 48, "783": 49,
#     "SA08": 50, "Subang HQ": 51, "T783": 52, "T850": 53, "708": 54, "750": 55, "753": 56, "782": 57,
#     "T562": 58, "T785": 59, "SA01": 60, "SA02": 61, "SU13A": 62, "T300": 63, "T301": 64, "T302": 65,
#     "MAHA1": 66, "T304": 67, "T406": 68, "T571": 69, "T406B": 70, "T602": 71, "506": 72, "540": 73,
#     "T250": 74, "T450": 75, "T580": 76, "T604": 77, "202": 78, "250": 79, "251": 80, "253": 81,
#     "590": 82, "602": 83, "641": 84, "650": 85, "651": 86, "652": 87, "T581": 88, "T601": 89,
#     "T605": 90, "T640": 91, "580": 92, "581": 93, "541": 94, "T221": 95, "151": 96, "173": 97,
#     "180": 98, "191": 99, "220": 100, "222": 101, "780": 102, "801": 103, "802": 104, "T202": 105,
#     "T203": 106, "T203B": 107, "T222": 108, "254": 109, "772": 110, "821": 111, "T251": 112,
#     "BET16": 113, "PBD1": 114, "T200": 115, "190": 116, "SEWA 2": 117, "T603": 118, "170": 119,
#     "822": 120, "851": 121, "T600": 122, "171": 123, "T582": 124, "640": 125, "T221B": 126,
#     "BET17": 127, "P101": 128, "P102": 129, "P103": 130, "P105": 131, "P106": 132, "P108": 133,
#     "T407": 134, "T569": 135, "T567": 136, "T568": 137, "MS01": 138, "DS01": 139, "DS01(PM)": 140,
#     "420": 141, "450": 142, "PAVILION BUKI": 143, "AJ2A": 144, "T783B": 145, "AJ03": 146,
#     "T785B": 147, "SEWA3": 148, "PTPM": 149, "GP03": 150, "T566": 151, "T559": 152, "T543": 153,
#     "T582B": 154, "T757B": 155, "T774B": 156, "T786B": 157, "KJ03": 158, "T542": 159, "T757": 160,
#     "T778B": 161, "421": 162, "T224": 163, "T223": 164, "T350": 165, "KLG2A": 166, "T303": 167,
#     "T351": 168, "SA04": 169, "T120": 170, "HLB1": 171, "451": 172, "SA06": 173, "T173": 174,
# }

# # [Keep your disruption_places dictionary]
# disruption_places = {
#     "Putrajaya Sentral to Bandar Utama (PKNS Sports Complex)": {
#         "lat": 3.0943772, "lng": 101.6009691, "route_no": "506", "depot": "5"
#     },
#     "Puchong Utama to Hab Pasar Seni (Parklane OUG)": {
#         "lat": 3.0680075, "lng": 101.6572824, "route_no": "600", "depot": "5"
#     },
#     "Taman Sri Muda to Hab Pasar Seni (Shell Bangsar)": {
#         "lat": 3.1294715, "lng": 101.6801838, "route_no": "751", "depot": "7"
#     },
#     "Stesen LRT Bangsar to Bangsar Shopping Centre (SMK Bkt. Bandaraya)": {
#         "lat": 3.1425927, "lng": 101.6717024, "route_no": "822", "depot": "2"
#     },
#     "Subang Mewah USJ 1 to Desa Mentari (LRT SS15)": {
#         "lat": 3.0751943, "lng": 101.585986, "route_no": "420", "depot": "4"
#     },
#     "Lembah Jaya Utara to Ampang Point (SJK (T) Ampang)": {
#         "lat": 3.1481176, "lng": 101.7632468, "route_no": "771", "depot": "37"
#     },
#     "Puchong Prima to IOI Puchong (Flat Puchong Perdana)": {
#         "lat": 3.0076847, "lng": 101.6017663, "route_no": "602", "depot": "38"
#     }
# }

# # Sidebar setup
# prasarana_logo = Image.open('assets/Rapid_KL_Logo.png')
# new_icon = Image.open('assets/icons8-driver-100.png')
# icon = Image.open('assets/icons8-bus-100.png')

# col_icon, col_title, col_logo = st.sidebar.columns([1, 6, 2])
# with col_icon:
#     st.image(new_icon, width=400)
# with col_title:
#     st.markdown(
#         """
#         <div style="display: flex; align-items: center; height: 60px;
#             font-size: 1.3rem; font-weight: 600; color: white;">
#             Bus Captain 
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )
# with col_logo:
#     st.image(prasarana_logo, width=400)

# # Disruption inputs
# selected_place = st.sidebar.selectbox(
#     "Choose Disruption Location",
#     list(disruption_places.keys()),
#     help="Select a disruption location"
# )
# defaults = disruption_places[selected_place]

# lat = st.sidebar.number_input(
#     "Latitude of Disruption", 
#     min_value=-90.0, max_value=90.0,
#     value=defaults["lat"], format="%.5f",
#     help="Enter disruption latitude"
# )
# lng = st.sidebar.number_input(
#     "Longitude of Disruption", 
#     min_value=-180.0, max_value=180.0,
#     value=defaults["lng"], format="%.5f",
#     help="Enter disruption longitude"
# )
# disruption_route_no_orig = st.sidebar.text_input(
#     "Disrupted Route No",
#     value=defaults["route_no"],
#     help="e.g. 300"
# )
# disruption_depot_orig = st.sidebar.text_input(
#     "Disrupted Depot",
#     value=defaults["depot"],
#     help="e.g. 10"
# )

# passenger_on_bus = st.sidebar.slider(
#     "Passengers on Bus", min_value=0, max_value=60, value=10, step=1,
#     help="Number of passengers currently on the bus"
# )

# weather_impact_code = 2

# disruption_route_no_enc = route_no_mapping.get(disruption_route_no_orig.strip(), -1)
# disruption_depot_enc = depot_mapping.get(disruption_depot_orig.strip(), -1)

# # Bus Control Center section
# col_icon_cc, col_title_cc = st.sidebar.columns([1, 8])
# with col_icon_cc:
#     st.image(icon, width=400)
# with col_title_cc:
#     st.markdown(
#         """
#         <div style="display: flex; align-items: center; height: 60px;
#             font-size: 1.3rem; font-weight: 600; color: white;">
#             Bus Control Center
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

# # Priority calculation
# if passenger_on_bus >= 30:
#     bus_replacement_priority_code = 1
# elif passenger_on_bus >= 20:
#     bus_replacement_priority_code = 2
# else:
#     bus_replacement_priority_code = 3

# bus_priority_map = {1: "P1", 2: "P2", 3: "P3"}

# # Get Malaysia time
# client_time_malaysia = st_javascript(js_code, key="get_malaysia_time")

# today = None
# try:
#     import zoneinfo
#     malaysia_tz = zoneinfo.ZoneInfo("Asia/Kuala_Lumpur")
# except ImportError:
#     import pytz
#     malaysia_tz = pytz.timezone("Asia/Kuala_Lumpur")

# if today is None:
#     now_utc = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
#     today = now_utc.astimezone(malaysia_tz).replace(tzinfo=None)
#     current_hour = today.hour

# if isinstance(client_time_malaysia, str) and client_time_malaysia.strip():
#     datetime_str = client_time_malaysia.strip()
#     iso_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"
#     if re.match(iso_pattern, datetime_str):
#         try:
#             today = datetime.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")
#         except Exception:
#             pass

# if today is not None:
#     disrupted_day_of_week = today.weekday()
#     disrupted_month = today.month
#     disrupted_is_holiday = 1 if disrupted_day_of_week >= 5 else 0
#     current_hour = today.hour
#     disrupted_hours_left = max(0, 24 - current_hour)

#     col_priority, col_datetime = st.sidebar.columns([2, 3])
#     with col_priority:
#         st.markdown(f"**Bus Replacement Priority:** {bus_priority_map[bus_replacement_priority_code]}")
#     with col_datetime:
#         st.markdown(f"**Date:** {today.strftime('%d-%m-%Y')}, {today.strftime('%H:%M')}")

# geo_distance_to_disruption = 1.0
# deadmileage_to_disruption = 1.0
# travel_time_min_from_hub = 15.0

# # Enhanced forecast function
# def forecast_ridership_api(route_no_enc, depot_enc):
#     """Forecast ridership with enhanced error handling"""
#     payload = {
#         "route_no_enc": route_no_enc,
#         "day_of_week": disrupted_day_of_week,
#         "month": disrupted_month,
#         "depot_enc": depot_enc,
#         "is_holiday": disrupted_is_holiday,
#         "hours_left": disrupted_hours_left
#     }
#     try:
#         resp = requests.post(f"{API_URL}/forecast_ridership", json=payload, timeout=30)
#         resp.raise_for_status()
#         data = resp.json()
#         return {
#             "ridership": data.get("forecasted_ridership"),
#             "confidence": data.get("confidence_score", 0.90),
#             "range": data.get("prediction_range", {})
#         }
#     except Exception as e:
#         st.error(f"Forecast ridership API call failed for route_enc={route_no_enc}: {e}")
#         return None

# def get_disrupted_route_forecast():
#     """Get forecast for the disrupted route"""
#     if disruption_route_no_enc == -1 or disruption_depot_enc == -1:
#         return None
#     result = forecast_ridership_api(disruption_route_no_enc, disruption_depot_enc)
#     return result["ridership"] if result else None

# # Main prediction button with enhanced logic
# # Main prediction button with enhanced logic
# if st.sidebar.button("🚀 Forecast Ridership for Predicted Routes"):
#     errors = []
#     if disruption_route_no_enc == -1:
#         errors.append(f"Route No '{disruption_route_no_orig}' not found in mapping.")
#     if disruption_depot_enc == -1:
#         errors.append(f"Depot '{disruption_depot_orig}' not found in mapping.")
    
#     if errors:
#         st.error("\n".join(errors))
#     else:
#         with st.spinner("🔍 Analyzing disruption and forecasting optimal routes..."):
#             payload_route = {
#                 "lat_disruption": float(lat),
#                 "lng_disruption": float(lng),
#                 "passenger_on_bus": int(passenger_on_bus),
#                 "disruption_weather_impact": int(weather_impact_code),
#                 "is_weekend": 1 if today.weekday() >= 5 else 0,
#                 "is_peak_hour_encoded": 1 if (7 <= current_hour <= 9 or 17 <= current_hour <= 19) else 0,
#                 "bus_replacement_priority": int(bus_replacement_priority_code),
#                 "bus_replacement_route_type_encoded": 1,
#                 "deadmileage_to_disruption": float(deadmileage_to_disruption),
#                 "geo_distance_to_disruption": float(geo_distance_to_disruption),
#                 "travel_time_min_from_hub": float(travel_time_min_from_hub)
#             }

#             try:
#                 # Call route prediction API
#                 response_route = requests.post(f"{API_URL}/predict_best_route", json=payload_route, timeout=30)
#                 response_route.raise_for_status()
#                 result_route = response_route.json()

#                 st.session_state["best_route_result"] = result_route.get('best_route', 'N/A')
#                 candidate_routes = result_route.get('candidate_routes', [])
#                 candidate_routes = list(dict.fromkeys(candidate_routes))
#                 st.session_state["candidate_routes_result"] = candidate_routes
                
#                 # Store model metrics in session state
#                 st.session_state["model_accuracy"] = result_route.get('model_accuracy', 86.0)
#                 st.session_state["model_precision"] = result_route.get('model_precision', 76.0)
#                 st.session_state["model_recall"] = result_route.get('model_recall', 85.0)
#                 st.session_state["model_improvements"] = result_route.get('improvements', '')
                
#                 # Show API info
#                 st.sidebar.success(f"✅ Found {len(candidate_routes)} candidate routes")
                
#                 # Show improved metrics
#                 model_accuracy = result_route.get('model_accuracy', 86.0)
#                 model_recall = result_route.get('model_recall', 85.0)
#                 model_precision = result_route.get('model_precision', 76.0)
                
#                 st.sidebar.success(f"🎯 **Model Accuracy: {model_accuracy:.1f}%**")
#                 st.sidebar.info(
#                     f"📊 **Model Performance:**\n\n"
#                     f"- **Recall:** {model_recall:.1f}% (finds {model_recall:.0f}% of best routes)\n"
#                     f"- **Precision:** {model_precision:.1f}%\n"
#                 )
                
#                 # Show what changed
#                 if 'improvements' in result_route:
#                     st.sidebar.markdown(f"🚀 **Enhancement:** {result_route['improvements']}")

#                 # Forecast riderships for candidate routes
#                 ridership_forecasts = {}
#                 ridership_details = {}
                
#                 progress_bar = st.sidebar.progress(0)
#                 for idx, route_no in enumerate(candidate_routes):
#                     route_no_enc = route_no_mapping.get(route_no.strip(), -1)
#                     if route_no_enc == -1:
#                         st.warning(f"Route '{route_no}' not found in encoding map, skipping forecast.")
#                         continue
                    
#                     result = forecast_ridership_api(route_no_enc, disruption_depot_enc)
#                     if result:
#                         ridership_forecasts[route_no] = result["ridership"]
#                         ridership_details[route_no] = result
                    
#                     progress_bar.progress((idx + 1) / len(candidate_routes))
                
#                 progress_bar.empty()

#                 st.session_state["ridership_forecasts"] = ridership_forecasts
#                 st.session_state["ridership_details"] = ridership_details

#                 # Get disrupted route forecast
#                 disrupted_route_forecast = get_disrupted_route_forecast()
#                 st.session_state["disrupted_route_forecast"] = disrupted_route_forecast

#             except requests.exceptions.RequestException as e:
#                 st.error(f"❌ API request failed: {e}")
#                 st.session_state.api_status = "error"

# # Display results in sidebar (OUTSIDE the button click)
# ridership_forecasts = st.session_state.get("ridership_forecasts", {})
# ridership_details = st.session_state.get("ridership_details", {})
# disrupted_forecast = st.session_state.get("disrupted_route_forecast")

# if disrupted_forecast is not None:
#     # st.sidebar.markdown("---")
#     st.sidebar.markdown(f"**📊 Disrupted Route {disruption_route_no_orig} Forecasted Ridership:** {disrupted_forecast:.0f} passengers")

# if ridership_forecasts:
#     valid_forecasts = {r: v for r, v in ridership_forecasts.items() if v is not None}
#     if valid_forecasts:
#         sorted_forecasts = sorted(valid_forecasts.items(), key=lambda x: x[1])
        
#         best_suggested_route, best_suggested_ridership = sorted_forecasts[0]
#         if best_suggested_route == disruption_route_no_orig and len(sorted_forecasts) > 1:
#             best_suggested_route, best_suggested_ridership = sorted_forecasts[1]

#         if disrupted_forecast is not None:
#             if disrupted_forecast > best_suggested_ridership:
#                 st.sidebar.warning(
#                     f"⚠️ **Recommendation:** Reroute a bus from **{best_suggested_route}** "
#                     f"({best_suggested_ridership:.0f} pax) to disrupted route **{disruption_route_no_orig}** "
#                     f"({disrupted_forecast:.0f} pax) for better coverage."
#                 )
#             elif disrupted_forecast < best_suggested_ridership:
#                 st.sidebar.info(
#                     f"✅ No rerouting needed. Disrupted route demand ({disrupted_forecast:.0f} pax) "
#                     f"is lower than suggested route ({best_suggested_ridership:.0f} pax)."
#                 )
#             else:
#                 st.sidebar.info("⚖️ Demand is equal. Review operational priorities carefully.")


# # Map visualization
# depots = {
#     "Depoh Sentul": (3.1880, 101.6900),
#     "Depoh Batu Caves": (3.2375, 101.6815),
#     "Depoh Melawati": (3.2200, 101.7150),
#     "Depoh Maluri": (3.1430, 101.7450),
#     "Depoh Cheras Selatan": (3.0560, 101.7400),
#     "Depoh OKR": (3.1100, 101.6800),
#     "Depoh Shah Alam": (3.0730, 101.5180),
#     "Depoh Asia Jaya": (3.1090, 101.6000),
#     "Depoh Kepong": (3.2100, 101.6200),
#     "Depoh Central Workshop CHS": (3.1400, 101.6900),
#     "Depoh MRT Serdang": (2.9870, 101.7400),
#     "Depoh Putrajaya": (2.9280, 101.7000),
# }

# m = folium.Map(location=[lat, lng], zoom_start=16, tiles='OpenStreetMap')

# for depot_name, coords in depots.items():
#     folium.Marker(
#         location=coords,
#         popup=depot_name,
#         tooltip=depot_name,
#         icon=folium.Icon(color='blue', icon='warehouse', prefix='fa')
#     ).add_to(m)

# folium.Marker(
#     location=[lat, lng],
#     popup="Disruption Location",
#     tooltip="Disruption Location",
#     icon=folium.Icon(color='red', icon='bus', prefix='fa')
# ).add_to(m)

# folium_static(m, width=1330, height=370)

# # Guided tour
# guide_steps = [
#     "1/6 - Welcome to SwiftRoute AI v2.0! Enhanced with multi-objective optimization. Click 'Next' to start.",
#     "2/6 - The map shows current bus breakdowns and nearby depots for quick response.",
#     "3/6 - Use sidebar controls to choose and simulate a breakdown scenario.",
#     "4/6 - Click 'Forecast Ridership' to get AI predictions with enhanced accuracy.",
#     "5/6 - Bar charts show forecasted ridership. Red bars indicate high ridership routes to avoid.",
#     "6/6 - Done! Uncheck 'Show Guided Tour' in the sidebar to use the full system.",
# ]

# if st.session_state.guide_step > len(guide_steps) - 1:
#     st.session_state.guide_step = 0

# with st.sidebar:
#     st.checkbox("Show Guided Tour", value=st.session_state.show_guide, key="show_guide")

# if st.session_state.show_guide:
#     with st.container():
#         st.markdown("### 📘 Guided Tour")
#         st.write(guide_steps[st.session_state.guide_step])
        
#         cols = st.columns([1,1])
#         with cols[0]:
#             if st.button("⬅️ Back", disabled=st.session_state.guide_step == 0):
#                 st.session_state.guide_step -= 1
#                 st.rerun()
#         with cols[1]:
#             if st.button("Next ➡️", disabled=st.session_state.guide_step == len(guide_steps) - 1):
#                 st.session_state.guide_step += 1
#                 st.rerun()

# # Enhanced visualization with confidence intervals
# if ridership_forecasts:
#     df = pd.DataFrame({
#         "Route": list(ridership_forecasts.keys()),
#         "Forecasted Ridership": [v if v is not None else 0 for v in ridership_forecasts.values()]
#     })

#     df["Route"] = df["Route"].astype(str)
    
#     # Add confidence intervals if available
#     if ridership_details:
#         df["Lower Bound"] = df["Route"].apply(
#             lambda r: ridership_details.get(r, {}).get("range", {}).get("lower_bound", df[df["Route"]==r]["Forecasted Ridership"].values[0] * 0.85)
#         )
#         df["Upper Bound"] = df["Route"].apply(
#             lambda r: ridership_details.get(r, {}).get("range", {}).get("upper_bound", df[df["Route"]==r]["Forecasted Ridership"].values[0] * 1.15)
#         )

#     # Color coding
#     max_ridership = df['Forecasted Ridership'].max()
#     threshold = max_ridership * 0.9

#     df['Color'] = df['Forecasted Ridership'].apply(
#         lambda x: "crimson" if x >= threshold else "steelblue"
#     )

#     # Create enhanced chart
#     fig = go.Figure()

#     # Add bars
#     for idx, row in df.iterrows():
#         fig.add_trace(go.Bar(
#             x=[row['Route']],
#             y=[row['Forecasted Ridership']],
#             name=row['Route'],
#             marker_color=row['Color'],
#             text=f"{row['Forecasted Ridership']:.0f}",
#             textposition='outside',
#             showlegend=False,
#             error_y=dict(
#                 type='data',
#                 symmetric=False,
#                 array=[row.get('Upper Bound', row['Forecasted Ridership'] * 1.15) - row['Forecasted Ridership']],
#                 arrayminus=[row['Forecasted Ridership'] - row.get('Lower Bound', row['Forecasted Ridership'] * 0.85)],
#                 visible=True,
#                 color='gray',
#                 thickness=1.5,
#                 width=4
#             ) if 'Upper Bound' in row else None
#         ))

#     fig.update_layout(
#         title={
#             'text': '📊 Forecasted Ridership for Candidate Routes (with Confidence Intervals)',
#             'x': 0.5,
#             'xanchor': 'center',
#             'font': dict(family='Arial, sans-serif', size=24, color='darkblue')
#         },
#         xaxis_title="Route Number",
#         yaxis_title="Forecasted Passengers",
#         width=1330,
#         height=450,
#         plot_bgcolor='rgba(0,0,0,0)',
#         margin=dict(t=80, b=50, l=50, r=50),
#         hovermode='x unified'
#     )

#     fig.update_xaxes(type='category', tickangle=-45)
    
#     st.plotly_chart(fig, use_container_width=False)
    
#     # Add summary metrics
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Total Routes Analyzed", len(df))
#     with col2:
#         st.metric("Avg Ridership", f"{df['Forecasted Ridership'].mean():.1f}")
#     with col3:
#         st.metric("Lowest Ridership", f"{df['Forecasted Ridership'].min():.1f}")
#     with col4:
#         st.metric("Highest Ridership", f"{df['Forecasted Ridership'].max():.1f}")
    
#     # Recommendation table
#     st.markdown("### 🎯 Top 5 Recommended Routes (Lowest Ridership)")
#     top_5 = df.nsmallest(5, 'Forecasted Ridership')[['Route', 'Forecasted Ridership']]
#     top_5['Recommendation'] = top_5['Forecasted Ridership'].apply(
#         lambda x: "✅ Excellent choice" if x < 15 else "⚠️ Moderate" if x < 25 else "❌ Avoid"
#     )
#     st.dataframe(top_5, use_container_width=True, hide_index=True)

import streamlit as st
import requests
import datetime
import folium
from streamlit_folium import folium_static
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
from streamlit_javascript import st_javascript
import re
import base64

API_URL = "https://prasarana-swiftroute-e21358fcb5f7.herokuapp.com"

# JavaScript to get Malaysia timezone datetime string
js_code = """
(() => {
    const dtf = new Intl.DateTimeFormat('en-CA', {
      timeZone: 'Asia/Kuala_Lumpur',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });
    const [
      { value: year },,
      { value: month },,
      { value: day },,
      { value: hour },,
      { value: minute },,
      { value: second }
    ] = dtf.formatToParts(new Date());
    return `${year}-${month}-${day}T${hour}:${minute}:${second}`;
})()
"""

st.set_page_config(page_title="SwiftRoute AI v2.0", layout="wide")

# Custom CSS styles
st.markdown(
    """
    <style>
    body, .css-1d391kg, .css-1v3fvcr {
        color: #c4c0c0 !important;  
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    div.block-container {
        padding-top: 0.25rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    h1, .css-1v3fvcr h1 {
        margin-top: 0.1rem !important;
        margin-bottom: 0.3rem !important;
    }
    .stButton button {
        margin-top: 0.15rem !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        background-color: #D50000 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #FF3333 !important;
        color: white !important;
    }
    [data-testid="stSidebar"] {
        position: fixed !important;
        top: 0 !important;
        right: 0 !important;
        width: 480px !important;
        height: 100vh !important;
        overflow-y: auto !important;
        background-color: #002060 !important;
        border-left: 3px solid #11134b !important;
        box-shadow: -2px 0 8px rgba(0,0,0,0.09);
        z-index: 9999 !important;
    }
    main > div.block-container {
        max-width: 1300px !important;
        margin-left: auto !important;
        margin-right: 480px !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div[data-baseweb="input"] > input,
    [data-testid="stSidebar"] div[data-baseweb="select"] > div > div,
    [data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: #fcfafa !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] > div > div,
    [data-testid="stSidebar"] div[data-baseweb="select"]:focus > div,
    [data-testid="stSidebar"] div[data-baseweb="input"] > input:focus {
        color: black !important;
        background-color: #ffffff !important;
    }
    [data-testid="stSidebar"] input::placeholder {
        color: #666666 !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #fff !important;
    }
    .stApp {
        background-color: #f9f6ef;  
    }
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Logo and header
with open("assets/Prasarana_vertical-01-scaled-removebg-preview.jpg", "rb") as img_file:
    logo_base64 = base64.b64encode(img_file.read()).decode()

st.markdown(f"""
    <div style='position: relative; top: 0px; left: 0; right: 0;
                width: 100%; max-width: 100%; margin: 0 auto;
                display: flex; align-self: normal; justify-content: space-between;
                padding: 14px 30px 14px 30px;
                background-color: #ffffffee;
                border-bottom: 1px solid #ccc;
                box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.07);
                border-radius: 16px 16px 0 0;
                z-index: 1000;'>
        <div style='display: flex; align-items: center; gap: 24px;'>
            <img src='image/png;base64,{logo_base64}' width='120' style='border-radius: 12px; box-shadow: 0 0 8px #eee;' />
            <div>
                <h2 style='margin: 0; color: #1f4e79;'>AI Replacement Bus Launch Point Optimisation v2.0</h2>
                <p style='margin: 0; font-size: 19px; color: #444;'>Enhanced with Multi-Objective Optimization & Advanced Ridership Forecasting</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Initialize session state variables
for key in [
    "best_route_result", "candidate_routes_result", "ridership_forecasts", "disrupted_route_forecast", 
    "guide_step", "show_guide", "api_status", "model_accuracy", "model_precision", "model_recall", "model_improvements",
    "ridership_details"
]:
    if key not in st.session_state:
        if key == "ridership_forecasts" or key == "ridership_details":
            st.session_state[key] = {}
        elif key == "guide_step":
            st.session_state[key] = 0
        elif key == "show_guide":
            st.session_state[key] = True
        else:
            st.session_state[key] = None

# Mappings (kept from original for backward compatibility)
depot_mapping = {
    "29": 0, "7": 1, "10": 2, "22": 3, "27": 4, "2": 5, "5": 6, "37": 7, "4": 8, "38": 9,
}

route_no_mapping = {
    "PJ01": 0, "100": 1, "200": 2, "201": 3, "300": 4, "302": 5, "303": 6, "400": 7, "401": 8,
    "500": 9, "600": 10, "601": 11, "T504": 12, "T505": 13, "T506": 14, "T507": 15, "T508": 16,
    "T509": 17, "T510": 18, "T511": 19, "T512": 20, "T545": 21, "301": 22, "T201": 23, "SJ01": 24,
    "SA03": 25, "SEWA1": 26, "T715": 27, "T753": 28, "T754": 29, "T756": 30, "T774": 31, "T776": 32,
    "T778": 33, "T780": 34, "T781": 35, "T782": 36, "T786": 37, "T787": 38, "T788": 39, "T789": 40,
    "T790": 41, "T791": 42, "402": 43, "751": 44, "752": 45, "754": 46, "770": 47, "771": 48, "783": 49,
    "SA08": 50, "Subang HQ": 51, "T783": 52, "T850": 53, "708": 54, "750": 55, "753": 56, "782": 57,
    "T562": 58, "T785": 59, "SA01": 60, "SA02": 61, "SU13A": 62, "T300": 63, "T301": 64, "T302": 65,
    "MAHA1": 66, "T304": 67, "T406": 68, "T571": 69, "T406B": 70, "T602": 71, "506": 72, "540": 73,
    "T250": 74, "T450": 75, "T580": 76, "T604": 77, "202": 78, "250": 79, "251": 80, "253": 81,
    "590": 82, "602": 83, "641": 84, "650": 85, "651": 86, "652": 87, "T581": 88, "T601": 89,
    "T605": 90, "T640": 91, "580": 92, "581": 93, "541": 94, "T221": 95, "151": 96, "173": 97,
    "180": 98, "191": 99, "220": 100, "222": 101, "780": 102, "801": 103, "802": 104, "T202": 105,
    "T203": 106, "T203B": 107, "T222": 108, "254": 109, "772": 110, "821": 111, "T251": 112,
    "BET16": 113, "PBD1": 114, "T200": 115, "190": 116, "SEWA 2": 117, "T603": 118, "170": 119,
    "822": 120, "851": 121, "T600": 122, "171": 123, "T582": 124, "640": 125, "T221B": 126,
    "BET17": 127, "P101": 128, "P102": 129, "P103": 130, "P105": 131, "P106": 132, "P108": 133,
    "T407": 134, "T569": 135, "T567": 136, "T568": 137, "MS01": 138, "DS01": 139, "DS01(PM)": 140,
    "420": 141, "450": 142, "PAVILION BUKI": 143, "AJ2A": 144, "T783B": 145, "AJ03": 146,
    "T785B": 147, "SEWA3": 148, "PTPM": 149, "GP03": 150, "T566": 151, "T559": 152, "T543": 153,
    "T582B": 154, "T757B": 155, "T774B": 156, "T786B": 157, "KJ03": 158, "T542": 159, "T757": 160,
    "T778B": 161, "421": 162, "T224": 163, "T223": 164, "T350": 165, "KLG2A": 166, "T303": 167,
    "T351": 168, "SA04": 169, "T120": 170, "HLB1": 171, "451": 172, "SA06": 173, "T173": 174,
}

disruption_places = {
    "Putrajaya Sentral to Bandar Utama (PKNS Sports Complex)": {
        "lat": 3.0943772, "lng": 101.6009691, "route_no": "506", "depot": "5"
    },
    "Puchong Utama to Hab Pasar Seni (Parklane OUG)": {
        "lat": 3.0680075, "lng": 101.6572824, "route_no": "600", "depot": "5"
    },
    "Taman Sri Muda to Hab Pasar Seni (Shell Bangsar)": {
        "lat": 3.1294715, "lng": 101.6801838, "route_no": "751", "depot": "7"
    },
    "Stesen LRT Bangsar to Bangsar Shopping Centre (SMK Bkt. Bandaraya)": {
        "lat": 3.1425927, "lng": 101.6717024, "route_no": "822", "depot": "2"
    },
    "Subang Mewah USJ 1 to Desa Mentari (LRT SS15)": {
        "lat": 3.0751943, "lng": 101.585986, "route_no": "420", "depot": "4"
    },
    "Lembah Jaya Utara to Ampang Point (SJK (T) Ampang)": {
        "lat": 3.1481176, "lng": 101.7632468, "route_no": "771", "depot": "37"
    },
    "Puchong Prima to IOI Puchong (Flat Puchong Perdana)": {
        "lat": 3.0076847, "lng": 101.6017663, "route_no": "602", "depot": "38"
    }
}

# Sidebar header with icons
prasarana_logo = Image.open('assets/Rapid_KL_Logo.png')
new_icon = Image.open('assets/icons8-driver-100.png')
icon_bus = Image.open('assets/icons8-bus-100.png')

col_icon, col_title, col_logo = st.sidebar.columns([1, 6, 2])
with col_icon:
    st.image(new_icon, width=75)
with col_title:
    st.markdown(
        """
        <div style="display: flex; align-items: center; height: 60px;
            font-size: 1.3rem; font-weight: 600; color: white;">
            Bus Captain 
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_logo:
    st.image(prasarana_logo, width=100)

# Disruption inputs sidebar
selected_place = st.sidebar.selectbox(
    "Choose Disruption Location",
    list(disruption_places.keys()),
    help="Select a disruption location"
)
defaults = disruption_places[selected_place]

lat = st.sidebar.number_input(
    "Latitude of Disruption", 
    min_value=-90.0, max_value=90.0,
    value=defaults["lat"], format="%.5f",
    help="Enter disruption latitude"
)
lng = st.sidebar.number_input(
    "Longitude of Disruption", 
    min_value=-180.0, max_value=180.0,
    value=defaults["lng"], format="%.5f",
    help="Enter disruption longitude"
)
disruption_route_no_orig = st.sidebar.text_input(
    "Disrupted Route No",
    value=defaults["route_no"],
    help="e.g. 300"
)
disruption_depot_orig = st.sidebar.text_input(
    "Disrupted Depot",
    value=defaults["depot"],
    help="e.g. 10"
)

passenger_on_bus = st.sidebar.slider(
    "Passengers on Bus", min_value=0, max_value=60, value=10, step=1,
    help="Number of passengers currently on the bus"
)

weather_impact_code = 2  # Fixed for example

disruption_route_no_enc = route_no_mapping.get(disruption_route_no_orig.strip(), -1)
disruption_depot_enc = depot_mapping.get(disruption_depot_orig.strip(), -1)

# Bus Control Center sidebar section
col_icon_cc, col_title_cc = st.sidebar.columns([1, 8])
with col_icon_cc:
    st.image(icon_bus, width=75)
with col_title_cc:
    st.markdown(
        """
        <div style="display: flex; align-items: center; height: 60px;
            font-size: 1.3rem; font-weight: 600; color: white;">
            Bus Control Center
        </div>
        """,
        unsafe_allow_html=True,
    )

# Bus replacement priority code calculation
if passenger_on_bus >= 30:
    bus_replacement_priority_code = 1
elif passenger_on_bus >= 20:
    bus_replacement_priority_code = 2
else:
    bus_replacement_priority_code = 3

bus_priority_map = {1: "P1", 2: "P2", 3: "P3"}

# Get Malaysia time from client JavaScript or fallback to server time
client_time_malaysia = st_javascript(js_code, key="get_malaysia_time")

today = None
try:
    import zoneinfo
    malaysia_tz = zoneinfo.ZoneInfo("Asia/Kuala_Lumpur")
except ImportError:
    import pytz
    malaysia_tz = pytz.timezone("Asia/Kuala_Lumpur")

if today is None:
    now_utc = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    today = now_utc.astimezone(malaysia_tz).replace(tzinfo=None)
    current_hour = today.hour

if isinstance(client_time_malaysia, str) and client_time_malaysia.strip():
    datetime_str = client_time_malaysia.strip()
    iso_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"
    if re.match(iso_pattern, datetime_str):
        try:
            today = datetime.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")
        except Exception:
            pass

if today is not None:
    disrupted_day_of_week = today.weekday()
    disrupted_month = today.month
    disrupted_is_holiday = 1 if disrupted_day_of_week >= 5 else 0
    current_hour = today.hour
    disrupted_hours_left = max(0, 24 - current_hour)

    col_priority, col_datetime = st.sidebar.columns([2, 3])
    with col_priority:
        st.markdown(f"**Bus Replacement Priority:** {bus_priority_map[bus_replacement_priority_code]}")
    with col_datetime:
        st.markdown(f"**Date:** {today.strftime('%d-%m-%Y')}, {today.strftime('%H:%M')}")

geo_distance_to_disruption = 1.0
deadmileage_to_disruption = 1.0
travel_time_min_from_hub = 15.0

# API function: Forecast ridership via existing endpoint
def forecast_ridership_api(route_no_enc, depot_enc):
    """Forecast ridership with enhanced error handling"""
    payload = {
        "route_no_enc": route_no_enc,
        "day_of_week": disrupted_day_of_week,
        "month": disrupted_month,
        "depot_enc": depot_enc,
        "is_holiday": disrupted_is_holiday,
        "hours_left": disrupted_hours_left
    }
    try:
        resp = requests.post(f"{API_URL}/forecast_ridership", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return {
            "ridership": data.get("forecasted_ridership"),
            "confidence": data.get("confidence_score", 0.90),
            "range": data.get("prediction_range", {})
        }
    except Exception as e:
        st.error(f"Forecast ridership API call failed for route_enc={route_no_enc}: {e}")
        return None

def get_disrupted_route_forecast():
    """Get forecast for the disrupted route using forecast API"""
    if disruption_route_no_enc == -1 or disruption_depot_enc == -1:
        return None
    result = forecast_ridership_api(disruption_route_no_enc, disruption_depot_enc)
    return result["ridership"] if result else None

# New API function to call hourly ridership prediction endpoint
def predict_disruption_ridership_api(route_no, disruption_datetime, depot):
    """Call /predict_disruption_ridership API endpoint"""
    payload = {
        "route_no": route_no,
        "disruption_datetime": disruption_datetime,
        "depot": depot
    }
    try:
        resp = requests.post(f"{API_URL}/predict_disruption_ridership", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as http_err:
        st.error(f"HTTP error: {http_err.response.status_code} - {http_err.response.json().get('detail', '')}")
    except Exception as e:
        st.error(f"Failed to get hourly ridership prediction: {e}")
    return None

# New API function to get available routes
def get_available_routes_api():
    try:
        resp = requests.get(f"{API_URL}/available_routes", timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to get available routes: {e}")
        return None

# New API function to get route info
def get_route_info_api(route_no):
    try:
        resp = requests.get(f"{API_URL}/route_info/{route_no}", timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as http_err:
        if http_err.response.status_code == 404:
            st.warning(f"Route info for '{route_no}' not found.")
            return None
        st.error(f"HTTP Error {http_err.response.status_code}: {http_err.response.json().get('detail', '')}")
    except Exception as e:
        st.error(f"Failed to get route info for {route_no}: {e}")
        return None

# Sidebar button: Forecast ridership for candidate routes as existing
if st.sidebar.button("🚀 Forecast Ridership for Predicted Routes"):
    errors = []
    if disruption_route_no_enc == -1:
        errors.append(f"Route No '{disruption_route_no_orig}' not found in mapping.")
    if disruption_depot_enc == -1:
        errors.append(f"Depot '{disruption_depot_orig}' not found in mapping.")
    
    if errors:
        st.error("\n".join(errors))
    else:
        with st.spinner("🔍 Analyzing disruption and forecasting optimal routes..."):
            payload_route = {
                "lat_disruption": float(lat),
                "lng_disruption": float(lng),
                "passenger_on_bus": int(passenger_on_bus),
                "disruption_weather_impact": int(weather_impact_code),
                "is_weekend": 1 if today.weekday() >= 5 else 0,
                "is_peak_hour_encoded": 1 if (7 <= current_hour <= 9 or 17 <= current_hour <= 19) else 0,
                "bus_replacement_priority": int(bus_replacement_priority_code),
                "bus_replacement_route_type_encoded": 1,
                "deadmileage_to_disruption": float(deadmileage_to_disruption),
                "geo_distance_to_disruption": float(geo_distance_to_disruption),
                "travel_time_min_from_hub": float(travel_time_min_from_hub)
            }

            try:
                response_route = requests.post(f"{API_URL}/predict_best_route", json=payload_route, timeout=30)
                response_route.raise_for_status()
                result_route = response_route.json()

                # Store results in session
                st.session_state["best_route_result"] = result_route.get('best_route', 'N/A')
                candidate_routes = result_route.get('candidate_routes', [])
                candidate_routes = list(dict.fromkeys(candidate_routes))
                st.session_state["candidate_routes_result"] = candidate_routes
                
                # Store model performance metrics
                st.session_state["model_accuracy"] = result_route.get('model_accuracy', 86.0)
                st.session_state["model_precision"] = result_route.get('model_precision', 76.0)
                st.session_state["model_recall"] = result_route.get('model_recall', 85.0)
                st.session_state["model_improvements"] = result_route.get('improvements', '')
                
                st.sidebar.success(f"✅ Found {len(candidate_routes)} candidate routes")
                st.sidebar.success(f"🎯 **Model Accuracy: {st.session_state['model_accuracy']:.1f}%**")
                st.sidebar.info(
                    f"📊 **Model Performance:**\n\n"
                    f"- **Recall:** {st.session_state['model_recall']:.1f}% (finds {st.session_state['model_recall']:.0f}% of best routes)\n"
                    f"- **Precision:** {st.session_state['model_precision']:.1f}%\n"
                )
                if st.session_state["model_improvements"]:
                    st.sidebar.markdown(f"🚀 **Enhancement:** {st.session_state['model_improvements']}")

                ridership_forecasts = {}
                ridership_details = {}
                progress_bar = st.sidebar.progress(0)
                for idx, route_no in enumerate(candidate_routes):
                    route_no_enc = route_no_mapping.get(route_no.strip(), -1)
                    if route_no_enc == -1:
                        st.warning(f"Route '{route_no}' not found in encoding map, skipping forecast.")
                        continue
                    result = forecast_ridership_api(route_no_enc, disruption_depot_enc)
                    if result:
                        ridership_forecasts[route_no] = result["ridership"]
                        ridership_details[route_no] = result
                    progress_bar.progress((idx + 1) / len(candidate_routes))
                progress_bar.empty()

                st.session_state["ridership_forecasts"] = ridership_forecasts
                st.session_state["ridership_details"] = ridership_details

                # Get disrupted route forecast
                disrupted_route_forecast = get_disrupted_route_forecast()
                st.session_state["disrupted_route_forecast"] = disrupted_route_forecast

            except requests.exceptions.RequestException as e:
                st.error(f"❌ API request failed: {e}")
                st.session_state.api_status = "error"

# New section: Hourly Ridership Prediction at Disruption Time
st.sidebar.markdown("---")
st.sidebar.markdown("### ⏰ Hourly Ridership Prediction")

hourly_route_no = st.sidebar.text_input(
    "Route No for Hourly Prediction", value=disruption_route_no_orig, help="Route number for hourly ridership prediction"
)
hourly_depot = st.sidebar.text_input(
    "Depot for Hourly Prediction", value=disruption_depot_orig, help="Depot for hourly ridership prediction"
)
hourly_datetime = st.sidebar.text_input(
    "Disruption DateTime (YYYY-MM-DDTHH:MM:SS)",
    value=today.strftime("%Y-%m-%dT%H:%M:%S") if today else "",
    help="ISO 8601 format datetime for disruption"
)

if st.sidebar.button("Predict Hourly Ridership at Disruption"):
    if not hourly_route_no.strip():
        st.sidebar.error("Route No is required for hourly prediction.")
    elif not hourly_depot.strip():
        st.sidebar.error("Depot is required for hourly prediction.")
    elif not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", hourly_datetime.strip()):
        st.sidebar.error("Disruption DateTime must be in ISO 8601 format: YYYY-MM-DDTHH:MM:SS.")
    else:
        with st.spinner("Predicting hourly ridership..."):
            pred_result = predict_disruption_ridership_api(
                hourly_route_no.strip(),
                hourly_datetime.strip(),
                hourly_depot.strip()
            )
            if pred_result is not None:
                st.sidebar.markdown("#### Hourly Ridership Prediction Result")
                for k, v in pred_result.items():
                    st.sidebar.write(f"**{k.replace('_', ' ').title()}:** {v}")

# Display best route and candidate routes info in sidebar
if st.session_state.get("best_route_result"):
    st.sidebar.markdown(f"### 🚍 Best Route Recommended")
    st.sidebar.markdown(f"**{st.session_state['best_route_result']}**")

if st.session_state.get("candidate_routes_result"):
    st.sidebar.markdown("### 🛣️ Candidate Alternative Routes")
    for route in st.session_state["candidate_routes_result"]:
        st.sidebar.write(route)

# Display disrupted route and forecasts
ridership_forecasts = st.session_state.get("ridership_forecasts", {})
ridership_details = st.session_state.get("ridership_details", {})
disrupted_forecast = st.session_state.get("disrupted_route_forecast")

if disrupted_forecast is not None:
    st.sidebar.markdown(f"**📊 Disrupted Route {disruption_route_no_orig} Forecasted Ridership:** {disrupted_forecast:.0f} passengers")

if ridership_forecasts:
    valid_forecasts = {r: v for r, v in ridership_forecasts.items() if v is not None}
    if valid_forecasts:
        sorted_forecasts = sorted(valid_forecasts.items(), key=lambda x: x[1])
        
        best_suggested_route, best_suggested_ridership = sorted_forecasts[0]
        if best_suggested_route == disruption_route_no_orig and len(sorted_forecasts) > 1:
            best_suggested_route, best_suggested_ridership = sorted_forecasts[1]

        if disrupted_forecast is not None:
            if disrupted_forecast > best_suggested_ridership:
                st.sidebar.warning(
                    f"⚠️ **Recommendation:** Reroute a bus from **{best_suggested_route}** "
                    f"({best_suggested_ridership:.0f} pax) to disrupted route **{disruption_route_no_orig}** "
                    f"({disrupted_forecast:.0f} pax) for better coverage."
                )
            elif disrupted_forecast < best_suggested_ridership:
                st.sidebar.info(
                    f"✅ No rerouting needed. Disrupted route demand ({disrupted_forecast:.0f} pax) "
                    f"is lower than suggested route ({best_suggested_ridership:.0f} pax)."
                )
            else:
                st.sidebar.info("⚖️ Demand is equal. Review operational priorities carefully.")

# Map with depots and disruption location
depots = {
    "Depoh Sentul": (3.1880, 101.6900),
    "Depoh Batu Caves": (3.2375, 101.6815),
    "Depoh Melawati": (3.2200, 101.7150),
    "Depoh Maluri": (3.1430, 101.7450),
    "Depoh Cheras Selatan": (3.0560, 101.7400),
    "Depoh OKR": (3.1100, 101.6800),
    "Depoh Shah Alam": (3.0730, 101.5180),
    "Depoh Asia Jaya": (3.1090, 101.6000),
    "Depoh Kepong": (3.2100, 101.6200),
    "Depoh Central Workshop CHS": (3.1400, 101.6900),
    "Depoh MRT Serdang": (2.9870, 101.7400),
    "Depoh Putrajaya": (2.9280, 101.7000),
}

m = folium.Map(location=[lat, lng], zoom_start=16, tiles='OpenStreetMap')

for depot_name, coords in depots.items():
    folium.Marker(
        location=coords,
        popup=depot_name,
        tooltip=depot_name,
        icon=folium.Icon(color='blue', icon='warehouse', prefix='fa')
    ).add_to(m)

folium.Marker(
    location=[lat, lng],
    popup="Disruption Location",
    tooltip="Disruption Location",
    icon=folium.Icon(color='red', icon='bus', prefix='fa')
).add_to(m)

folium_static(m, width=1330, height=370)

# Guided tour steps
guide_steps = [
    "1/6 - Welcome to SwiftRoute AI v2.0! Enhanced with multi-objective optimization. Click 'Next' to start.",
    "2/6 - The map shows current bus breakdowns and nearby depots for quick response.",
    "3/6 - Use sidebar controls to choose and simulate a breakdown scenario.",
    "4/6 - Click 'Forecast Ridership' to get AI predictions with enhanced accuracy.",
    "5/6 - Bar charts show forecasted ridership. Red bars indicate high ridership routes to avoid.",
    "6/6 - Done! Uncheck 'Show Guided Tour' in the sidebar to use the full system.",
]

if st.session_state.guide_step > len(guide_steps) - 1:
    st.session_state.guide_step = 0

with st.sidebar:
    show_guide_checkbox = st.checkbox("Show Guided Tour", value=st.session_state.show_guide, key="show_guide")
    st.session_state.show_guide = show_guide_checkbox

if st.session_state.show_guide:
    with st.container():
        st.markdown("### 📘 Guided Tour")
        st.write(guide_steps[st.session_state.guide_step])
        
        cols = st.columns([1,1])
        with cols[0]:
            if st.button("⬅️ Back", disabled=st.session_state.guide_step == 0):
                st.session_state.guide_step -= 1
                st.experimental_rerun()
        with cols[1]:
            if st.button("Next ➡️", disabled=st.session_state.guide_step == len(guide_steps) - 1):
                st.session_state.guide_step += 1
                st.experimental_rerun()

# Enhanced visualization with confidence intervals for ridership forecasts
if ridership_forecasts:
    df = pd.DataFrame({
        "Route": list(ridership_forecasts.keys()),
        "Forecasted Ridership": [v if v is not None else 0 for v in ridership_forecasts.values()]
    })

    df["Route"] = df["Route"].astype(str)
    
    # Add confidence intervals if available
    if ridership_details:
        df["Lower Bound"] = df["Route"].apply(
            lambda r: ridership_details.get(r, {}).get("range", {}).get("lower_bound", df[df["Route"]==r]["Forecasted Ridership"].values[0] * 0.85)
        )
        df["Upper Bound"] = df["Route"].apply(
            lambda r: ridership_details.get(r, {}).get("range", {}).get("upper_bound", df[df["Route"]==r]["Forecasted Ridership"].values[0] * 1.15)
        )

    max_ridership = df['Forecasted Ridership'].max()
    threshold = max_ridership * 0.9

    df['Color'] = df['Forecasted Ridership'].apply(
        lambda x: "crimson" if x >= threshold else "steelblue"
    )

    fig = go.Figure()

    for idx, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row['Route']],
            y=[row['Forecasted Ridership']],
            name=row['Route'],
            marker_color=row['Color'],
            text=f"{row['Forecasted Ridership']:.0f}",
            textposition='outside',
            showlegend=False,
            error_y=dict(
                type='data',
                symmetric=False,
                array=[row.get('Upper Bound', row['Forecasted Ridership'] * 1.15) - row['Forecasted Ridership']],
                arrayminus=[row['Forecasted Ridership'] - row.get('Lower Bound', row['Forecasted Ridership'] * 0.85)],
                visible=True,
                color='gray',
                thickness=1.5,
                width=4
            ) if 'Upper Bound' in row else None
        ))

    fig.update_layout(
        title={
            'text': '📊 Forecasted Ridership for Candidate Routes (with Confidence Intervals)',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(family='Arial, sans-serif', size=24, color='darkblue')
        },
        xaxis_title="Route Number",
        yaxis_title="Forecasted Passengers",
        width=1330,
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=80, b=50, l=50, r=50),
        hovermode='x unified'
    )

    fig.update_xaxes(type='category', tickangle=-45)
    
    st.plotly_chart(fig, use_container_width=False)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Routes Analyzed", len(df))
    with col2:
        st.metric("Avg Ridership", f"{df['Forecasted Ridership'].mean():.1f}")
    with col3:
        st.metric("Lowest Ridership", f"{df['Forecasted Ridership'].min():.1f}")
    with col4:
        st.metric("Highest Ridership", f"{df['Forecasted Ridership'].max():.1f}")
    
    # Recommendation table
    st.markdown("### 🎯 Top 5 Recommended Routes (Lowest Ridership)")
    top_5 = df.nsmallest(5, 'Forecasted Ridership')[['Route', 'Forecasted Ridership']].copy()
    top_5['Recommendation'] = top_5['Forecasted Ridership'].apply(
        lambda x: "✅ Excellent choice" if x < 15 else "⚠️ Moderate" if x < 25 else "❌ Avoid"
    )
    st.dataframe(top_5, use_container_width=True, hide_index=True)
