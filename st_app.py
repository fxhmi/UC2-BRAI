import streamlit as st
import requests
import datetime
import folium
from streamlit_folium import folium_static
from PIL import Image
import plotly.express as px
import pandas as pd
from streamlit_javascript import st_javascript

API_URL = "https://prasarana-swiftroute-e21358fcb5f7.herokuapp.com"

st.set_page_config(page_title="SwiftRoute", layout="wide")
st.markdown(
    """
    <style>
    body, .css-1d391kg, .css-1v3fvcr {
        color: #c4c0c0 !important;  
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Reduce main container padding, especially top */
    div.block-container {
        padding-top: 0.25rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    /* Reduce margin for page title */
    h1, .css-1v3fvcr h1 {
        margin-top: 0.1rem !important;
        margin-bottom: 0.3rem !important;
    }
    /* Reduce margin under title and above the first element */
    section.main > div.block-container > div > div:first-child > div {
        margin-top: 0.3rem !important;
    }
    /* Button styling and margin reduction */
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
    /* Fix sidebar on the right with fixed width */
    [data-testid="stSidebar"] {
        position: fixed !important;
        top: 0 !important;
        right: 0 !important;
        width: 480px !important;  /* Sidebar width */
        height: 100vh !important;
        overflow-y: auto !important;
        background-color: #002060 !important;
        border-left: 3px solid #11134b !important;
        box-shadow: -2px 0 8px rgba(0,0,0,0.09);
        z-index: 9999 !important;
    }

    /* Main content container - set max width, center, and margin for sidebar */
    main > div.block-container {
        max-width: 1300px !important;  /* Fixed max width */
        margin-left: auto !important;
        margin-right: 480px !important; /* Reserve space for sidebar */
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Responsive adjustment for smaller screens */
    @media (max-width: 1300px) {
        main > div.block-container {
            max-width: 95% !important;
            margin-right: 480px !important;
        }
    }

    @media (max-width: 1200px) {
        .main {
            margin-right: 16rem !important;
        }
    }
    /* Sidebar input fields, number inputs, and dropdown text color */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div[data-baseweb="input"] > input,
    [data-testid="stSidebar"] div[data-baseweb="select"] > div > div,
    [data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: #fcfafa !important;
    }
    /* Placeholder text color */
    [data-testid="stSidebar"] input::placeholder,
    [data-testid="stSidebar"] textarea::placeholder {
        color: #fcfafa !important;  
    }
    /* Selectbox container background */
    [data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #002060 !important;
        border-radius: 6px !important;
        padding-left: 8px !important;
        padding-right: 8px !important;
        border: none !important;
    }
    /* Selected value transparent background */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div > div {
        background-color: transparent !important;
        color: #c4c0c0 !important;
    }
    /* Dropdown arrow color */
    [data-testid="stSidebar"] div[data-baseweb="select"] svg {
        fill: #c4c0c0 !important;
    }
    /* Dropdown menu background and text */
    div[role="listbox"] {
        background-color: #002060 !important;
        color: #c4c0c0 !important;
        border-radius: 6px !important;
    }
    div[role="option"] {
        background-color: #002060 !important;
        color: #c4c0c0 !important;
    }
    div[role="option"]:hover {
        background-color: #001840 !important;
        color: #eeeeee !important;
    }
    /* Focus ring */
    [data-testid="stSidebar"] div[data-baseweb="input"] > input:focus,
    [data-testid="stSidebar"] div[data-baseweb="select"]:focus,
    [data-testid="stSidebar"] div[data-baseweb="select"] > div:focus {
        border-color: #D50000 !important;
        outline: none !important;
        box-shadow: 0 0 0 2px #D50000 !important;
    }
        /* Make sidebar text white */
    [data-testid="stSidebar"] {
        color: white !important;
    }
    /* Specifically make the Bus Replacement Priority and Date text white */
    # [data-testid="stSidebar"] > div > div > div > div > div > div > div > div {
    #     color: black !important;
    # }
    /* Also apply to any markdown or span inside sidebar */
    [data-testid="stSidebar"] span {
        color: white !important;
    }
    [data-testid="stSidebar"] label {
        color: white !important;
    } 
    # [data-testid="stSidebar"] div {
    #     color: black !important;
    # } 
    [data-testid="stSidebar"] p {
        color: white !important;
    }
    /* Change main app background color only */
    .stApp {
        background-color: #f9f6ef;  
    }
    /* Add the fixes below */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div > div,
    [data-testid="stSidebar"] div[data-baseweb="select"]:focus > div,
    [data-testid="stSidebar"] div[data-baseweb="select"] > div:focus,
    [data-testid="stSidebar"] div[data-baseweb="input"] > input:focus {
        color: black !important;
        background-color: #ffffff !important;
    }

    [data-testid="stSidebar"] input::placeholder,
    [data-testid="stSidebar"] textarea::placeholder {
        color: #666666 !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #fff !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"]:focus,
    [data-testid="stSidebar"] div[data-baseweb="input"] > input:focus {
        box-shadow: 0 0 0 2px #007bff !important;
        border-color: #007bff !important;
    }
    /* Sidebar help tooltip icon (the '?' ) */
    [data-testid="stSidebar"] button[aria-label="Show help"] {
        color: white !important;           /* text color fallback */
        filter: none !important;           /* remove grayscale or opacity filters */
        opacity: 1 !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        cursor: pointer;
    }

    /* The SVG inside the help button */
    [data-testid="stSidebar"] button[aria-label="Show help"] svg {
        fill: white !important;
        stroke: white !important;
    }

    /* Hover and focus states */
    [data-testid="stSidebar"] button[aria-label="Show help"]:hover,
    [data-testid="stSidebar"] button[aria-label="Show help"]:focus {
        background-color: rgba(255, 255, 255, 0.1) !important;
        outline: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
    /* Ensure main content avoids sidebar overlay */
    .main {
        margin-right: 480px !important;
        margin-left: 0 !important;
    }
    [data-testid="stSidebar"] {
        width: 480px !important;
        position: fixed !important;
        right: 0 !important;
        height: 100vh !important;
        overflow-y: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

import base64


# st.title("SwiftRoute: Smart Route Prediction & Disruption Map ")
with open("assets/Prasarana_vertical-01-scaled-removebg-preview.jpg", "rb") as img_file:
    logo_base64 = base64.b64encode(img_file.read()).decode()

st.markdown("""
    <style>
        /* Remove default Streamlit top padding */
        .main > div:first-child {{
            padding-top: 0rem !important;
        }}
    </style>
    <div style='position: relative; top: -20px; left: 0; right: 0;
                width: 1300px; max-width: 95vw; margin: 0 auto;
                display: flex; align-items: center; justify-content: space-between;
                padding: 14px 30px 14px 30px;
                background-color: #ffffffee;
                border-bottom: 1px solid #ccc;
                box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.07);
                border-radius: 16px 16px 0 0;
                z-index: 1000;'>
        <div style='display: flex; align-items: center; gap: 24px;'>
            <img src='data:image/png;base64,{0}' width='80' style='border-radius: 12px; box-shadow: 0 0 8px #eee;' />
            <div>
                <h2 style='margin: 0; color: #1f4e79;'>SwiftRoute: Smart Route Prediction & Disruption Map</h2>
                <p style='margin: 0; font-size: 17px; color: #444;'>Real-time disruption intelligence for Bus Captain and Operations</p>
            </div>
        </div>
    </div>
""".format(logo_base64), unsafe_allow_html=True)


prasarana_logo_path = 'assets/Rapid_KL_Logo.png'
# prasarana_logo_path = '/Users/fahmi.taib/Desktop/Deployment Code Test/Prasarana_vertical-01-scaled-removebg-preview.jpg'
prasarana_logo = Image.open(prasarana_logo_path)
prasarana_logo_width = 400

new_icon_path = 'assets/icons8-driver-100.png'
new_icon = Image.open(new_icon_path)
new_icon_width = 400

icon_path = 'assets/icons8-bus-100.png'
icon = Image.open(icon_path)
icon_width = 400

col_icon, col_title, col_logo = st.sidebar.columns([1, 6, 2])
with col_icon:
    st.image(new_icon, width=new_icon_width)
with col_title:
    st.markdown(
        """
        <div style="
            display: flex;
            align-items: center;
            height: 60px;
            font-size: 1.3rem;
            font-weight: 600;
            color: white;
            ">
            Bus Captain 
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_logo:
    st.image(prasarana_logo, width=prasarana_logo_width)

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
        "lat": 3.0943772,
        "lng": 101.6009691,
        "route_no": "506",
        "depot": "5"
    },
    "Puchong Utama to Hab Pasar Seni (Parklane OUG)": {
        "lat": 3.0680075,
        "lng": 101.6572824,
        "route_no": "600",
        "depot": "5"
    },
    "Taman Sri Muda to Hab Pasar Seni (Shell Bangsar)": {
        "lat": 3.1294715,
        "lng": 101.6801838,
        "route_no": "751",
        "depot": "7"
    },
    "Stesen LRT Bangsar to Bangsar Shopping Centre (SMK Bkt. Bandaraya)": {
        "lat": 3.1425927,
        "lng": 101.6717024,
        "route_no": "822",
        "depot": "2"
    },
    # "Stesen LRT Kelana Jaya to Subang Parade (Kuil Sri Subramaniar)": {
    #     "lat": 3.6613149,
    #     "lng": 101.5366013,
    #     "route_no": "783",
    #     "depot": "7"
    # },
    # "Klang to Sunway Pyramid (Petron SS17)": {
    #     "lat": 3.0793914,
    #     "lng": 101.5815142,
    #     "route_no": "708",
    #     "depot": "37"
    # },
    "Subang Mewah USJ 1 to Desa Mentari (LRT SS15)": {
        "lat": 3.0751943,
        "lng": 101.585986,
        "route_no": "420",
        "depot": "4"
    },
    "Lembah Jaya Utara to Ampang Point (SJK (T) Ampang)": {
        "lat": 3.1481176,
        "lng": 101.7632468,
        "route_no": "771",
        "depot": "37"
    },
    # "Stesen MRT Kajang to Stesen MRT Putrajaya Sentral": {
    #     "lat": 2.9855,
    #     "lng": 101.7913,
    #     "route_no": "451",
    #     "depot": "5"
    # },
    # "Stesen LRT Subang Jaya to Pearl Point": {
    #     "lat": 3.0400,
    #     "lng": 101.5700,
    #     "route_no": "641",
    #     "depot": "5"
    # },
    # "Klang to Sunway Pyramid via Hentian Bandar": {
    #     "lat": 3.0000,
    #     "lng": 101.4400,
    #     "route_no": "708",
    #     "depot": "7"
    # },
    # "UiTM Puncak Alam to Bandar Puncak Alam": {
    #     "lat": 3.1500,
    #     "lng": 101.3800,
    #     "route_no": "T715",
    #     "depot": "7"
    # },
    # "Stesen LRT Glenmarie to MSU, Stadium Shah Alam": {
    #     "lat": 3.0700,
    #     "lng": 101.4800,
    #     "route_no": "T774",
    #     "depot": "7"
    # },
    # "MRT Putrajaya Sentral to Presint 10": {
    #     "lat": 2.9250,
    #     "lng": 101.6900,
    #     "route_no": "T508",
    #     "depot": "22"
    # },
    "Stesen MRT Tun Razak Exchange to Taman Maluri / Desa Pandan": {
        "lat": 3.1600,
        "lng": 101.7000,
        "route_no": "T407",
        "depot": "27"
    },
    # "Terminal Maluri to Titiwangsa": {
    #     "lat": 3.2000,
    #     "lng": 101.7000,
    #     "route_no": "402",
    #     "depot": "29"
    # },
    # "Hentian Bandar Shah Alam to UiTM Puncak Alam": {
    #     "lat": 3.0800,
    #     "lng": 101.4000,
    #     "route_no": "753",
    #     "depot": "37"
    # },
    "Puchong Prima to IOI Puchong (Flat Puchong Perdana)": {
        "lat": 3.0076847,
        "lng": 101.6017663,
        "route_no": "602",
        "depot": "38"
    }
}


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
weather_impact_map = {1: "Bad Weather", 2: "Clear Weather"}
weather_impact_str = st.sidebar.selectbox(
    "Weather Condition",
    options=list(weather_impact_map.values()), index=1,
    help="Select current weather impact at disruption site"
)
weather_impact_code = [k for k, v in weather_impact_map.items() if v == weather_impact_str][0]

disruption_route_no_enc = route_no_mapping.get(disruption_route_no_orig.strip(), -1)
disruption_depot_enc = depot_mapping.get(disruption_depot_orig.strip(), -1)


col_icon_cc, col_title_cc = st.sidebar.columns([1, 8])
with col_icon_cc:
    st.image(icon, width=icon_width)
with col_title_cc:
    st.markdown(
        """
        <div style="
            display: flex;
            align-items: center;
            height: 60px;
            font-size: 1.3rem;
            font-weight: 600;
            color: white;
            ">
            Bus Control Center
        </div>
        """,
        unsafe_allow_html=True,
    )

route_type_map = {0: "UTAMA", 1: "TEMPATAN", 2: "SS", 3: "OTHERS", 4: "PJCT", 5: "PPJ"}

bus_priority_map = {1: "P1", 2: "P2", 3: "P3"}

if passenger_on_bus >= 30:
    bus_replacement_priority_code = 1
elif passenger_on_bus >= 20:
    bus_replacement_priority_code = 2
else:
    bus_replacement_priority_code = 3

col_priority, col_datetime = st.sidebar.columns([2, 3])

#Time 1
# today = datetime.datetime.now()
# disruption_date = datetime.datetime.now()
# disrupted_day_of_week = disruption_date.weekday()
# disrupted_month = disruption_date.month
# disrupted_is_holiday = 1 if disrupted_day_of_week >= 5 else 0
# current_hour = disruption_date.hour
# disrupted_hours_left = max(0, 24 - current_hour)

#Time 2
# Run JavaScript in user's browser to get local time ISO string
# This returns browser timestamp as ISO 8601 string
# js_code = "new Date().toISOString()"
# client_time_iso = st_javascript(js_code, key="get_time")

# if client_time_iso is not None:
#     today = datetime.datetime.fromisoformat(client_time_iso.replace("Z", "+00:00"))

#     disrupted_day_of_week = today.weekday()
#     disrupted_month = today.month
#     disrupted_is_holiday = 1 if disrupted_day_of_week >= 5 else 0
#     current_hour = today.hour
#     disrupted_hours_left = max(0, 24 - current_hour)

#     with col_priority:
#         st.markdown(f"**Bus Replacement Priority:** {bus_priority_map[bus_replacement_priority_code]}")

#     with col_datetime:
#         st.markdown(f"**Date:** {today.strftime('%d-%m-%Y')}, {today.strftime('%H:%M')}")
# else:
#     st.write("Fetching client local time...")

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

client_time_malaysia = st_javascript(js_code, key="get_malaysia_time")

if isinstance(client_time_malaysia, str) and client_time_malaysia:
    # safe to strip and parse
    client_time_malaysia = client_time_malaysia.strip()
    try:
        today = datetime.datetime.strptime(client_time_malaysia, "%Y-%m-%dT%H:%M:%S")
    except ValueError as e:
        st.error(f"Time parsing error: {e}")
        today = None  # or handle fallback
else:
    st.write(f"Waiting for valid Malaysia time, received: {repr(client_time_malaysia)}")
    today = None

if today is not None:
    disrupted_day_of_week = today.weekday()
    disrupted_month = today.month
    disrupted_is_holiday = 1 if disrupted_day_of_week >= 5 else 0
    current_hour = today.hour
    disrupted_hours_left = max(0, 24 - current_hour)

    with col_priority:
        st.markdown(f"**Bus Replacement Priority:** {bus_priority_map[bus_replacement_priority_code]}")

    with col_datetime:
        st.markdown(f"**Date:** {today.strftime('%d-%m-%Y')}, {today.strftime('%H:%M')}")




geo_distance_to_disruption = 1.0
deadmileage_to_disruption = 1.0
travel_time_min_from_hub = 15.0

# Function to forecast ridership for a given route encoding
def forecast_ridership_api(route_no_enc, depot_enc):
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
        return data.get("forecasted_ridership", None)
    except Exception as e:
        st.error(f"Forecast ridership API call failed for route_enc={route_no_enc}: {e}")
        return None

# Function to get disrupted route ridership forecast 
def get_disrupted_route_forecast():
    if disruption_route_no_enc == -1 or disruption_depot_enc == -1:
        return None
    return forecast_ridership_api(disruption_route_no_enc, disruption_depot_enc)

# Button to call forecast best route and ridership forecasts
if st.sidebar.button("Forecast Ridership for Predicted Routes"):
    errors = []
    if disruption_route_no_enc == -1:
        errors.append(f"Route No '{disruption_route_no_orig}' not found in mapping.")
    if disruption_depot_enc == -1:
        errors.append(f"Depot '{disruption_depot_orig}' not found in mapping.")
    if errors:
        st.error("\n".join(errors))
    else:
        with st.spinner("Calling /predict_best_route to get candidate routes..."):
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

                st.session_state["best_route_result"] = result_route.get('best_route', 'N/A')
                candidate_routes = result_route.get('candidate_routes', [])
                candidate_routes = list(dict.fromkeys(candidate_routes))
                st.session_state["candidate_routes_result"] = candidate_routes

                # Forecast riderships for candidate routes
                ridership_forecasts = {}
                for route_no in candidate_routes:
                    route_no_enc = route_no_mapping.get(route_no.strip(), -1)
                    if route_no_enc == -1:
                        st.warning(f"Route '{route_no}' not found in encoding map, skipping forecast.")
                        continue
                    ridership = forecast_ridership_api(route_no_enc, disruption_depot_enc)
                    ridership_forecasts[route_no] = ridership

                st.session_state["ridership_forecasts"] = ridership_forecasts

                # Get disrupted route forecast too and save in session state
                disrupted_route_forecast = get_disrupted_route_forecast()
                st.session_state["disrupted_route_forecast"] = disrupted_route_forecast

            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")



ridership_forecasts = st.session_state.get("ridership_forecasts", {})
best_replacement_route = None
lowest_ridership = None

# Find route with lowest forecasted ridership among valid forecasts
valid_forecasts = {route: ridership for route, ridership in ridership_forecasts.items() if ridership is not None}

if valid_forecasts:
    best_replacement_route = min(valid_forecasts, key=valid_forecasts.get)
    lowest_ridership = valid_forecasts[best_replacement_route]


candidate_routes_result = st.session_state.get("candidate_routes_result", [])
ridership_forecasts = st.session_state.get("ridership_forecasts", {})
disrupted_forecast = st.session_state.get("disrupted_route_forecast")

# Show disrupted route forecast in sidebar
if disrupted_forecast is not None:
    st.sidebar.markdown(f"**Disrupted Route {disruption_route_no_orig} Forecasted Ridership:** {disrupted_forecast:.0f} passengers")
else:
    st.sidebar.markdown("")

if ridership_forecasts:
    # Filter out routes with None as a forecast
    valid_forecasts = {r: v for r, v in ridership_forecasts.items() if v is not None}
    if valid_forecasts:
        # Sort routes by forecasted ridership (ascending: lowest first)
        sorted_forecasts = sorted(valid_forecasts.items(), key=lambda x: x[1])

        # Pick the lowest, but if it's the disrupted route, use the second lowest if available
        best_suggested_route, best_suggested_ridership = sorted_forecasts[0]
        if best_suggested_route == disruption_route_no_orig and len(sorted_forecasts) > 1:
            best_suggested_route, best_suggested_ridership = sorted_forecasts[1]

        # If you want, display this suggestion in UI:
        # st.sidebar.markdown(f"**Suggested Lower Ridership Route:** {best_suggested_route} (Forecasted {best_suggested_ridership:.0f} passengers)")

        if disrupted_forecast is not None:
            if disrupted_forecast > best_suggested_ridership:
                st.sidebar.warning(
                    f"Consider rerouting a bus from **{best_suggested_route}** to disrupted route **{disruption_route_no_orig}** for better coverage."
                )
            elif disrupted_forecast < best_suggested_ridership:
                st.sidebar.info(
                    f"No need to reroute bus; disrupted route ({disrupted_forecast:.0f} passengers) demand is lower than the suggested route ({best_suggested_ridership:.0f} passengers)."
                )
            else:
                st.sidebar.info(
                    "Demand is equal. Carefully review operational priorities before transferring a bus."
                )


st.markdown(
    """
    <style>
    [title~="st.iframe"] {
        width: 100% !important;
        height: 500px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    "(OS) Maluri": (3.1430, 101.7450),
    "(OS) Shah Alam": (3.0730, 101.5180),
    "(OS) Cheras Selatan": (3.0560, 101.7400),
    "(OS) Batu Caves": (3.2375, 101.6815),
    "(ROD) Batu Caves": (3.2375, 101.6815),
    "(ROD) Maluri": (3.1430, 101.7450),
    "(ROD) Cheras Selatan": (3.0560, 101.7400),
    "(ROD) Shah Alam": (3.0730, 101.5180),
    "(ROD) Sungai Buloh": (3.2100, 101.5700),
    "(ROD) Kajang": (2.9930, 101.7900),
    "(ROD) Jinjang": (3.2200, 101.6300),
    "(ROD) Serdang": (2.9930, 101.7400),
    "(ROD) Desa Tun Razak": (3.0600, 101.7200),
    "(SAL) Shah Alam": (3.0730, 101.5180),
    "(SAL) Sungai Buloh": (3.2100, 101.5700),
}

m = folium.Map(location=[lat, lng], zoom_start=14, tiles='OpenStreetMap', width=1400, height=500)

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

folium_static(m, width=1300, height=450)



riderships = st.session_state.get("ridership_forecasts")
if riderships:
    df = pd.DataFrame({
        "Route": list(riderships.keys()),
        "Forecasted Ridership": [v if v is not None else 0 for v in riderships.values()]
    })

    df["Route"] = df["Route"].astype(str)

    # Compute threshold for high ridership
    max_ridership = df['Forecasted Ridership'].max()
    threshold = max_ridership * 0.9  

    df['Color'] = df['Forecasted Ridership'].apply(
        lambda x: "crimson" if x >= threshold else "steelblue"
    )

    fig = px.bar(
        df,
        x='Route',
        y='Forecasted Ridership',
        color='Color',
        color_discrete_map={'crimson': 'crimson', 'steelblue': 'steelblue'},
        title='Ridership Forecasts, Highlighting High Ridership',
        labels={'Forecasted Ridership': 'Forecasted Passengers'},
        text='Forecasted Ridership',
        width=1300,
        height=400
    )
    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    fig.update_layout(
        margin=dict(t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False  # Optional: hide color legend since colors are self-explanatory
    )

    fig.update_xaxes(type='category')

    st.plotly_chart(fig, use_container_width=False, width=1300, height=400)
else:
    st.write(" ")




# disruption_date = datetime.datetime.now()
# current_hour = disruption_date.hour

if "best_route_result" not in st.session_state:
    st.session_state["best_route_result"] = None
if "candidate_routes_result" not in st.session_state:
    st.session_state["candidate_routes_result"] = None

# Prepare payload for /predict_best_route
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
#"bus_replacement_route_type_encoded": int(bus_replacement_route_type_code),
# Prepare payload for /forecast_ridership (for disrupted route)
payload_ridership = {
    "route_no_enc": disruption_route_no_enc,
    "day_of_week": disrupted_day_of_week,
    "month": disrupted_month,
    "depot_enc": disruption_depot_enc,
    "is_holiday": disrupted_is_holiday,
    "hours_left": disrupted_hours_left
}



