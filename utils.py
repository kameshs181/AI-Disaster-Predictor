import os
import requests
import pandas as pd
from geopy.distance import geodesic

# Load API key from environment (.env)
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Load your datasets (for nearest historical features)
flood_data = pd.read_csv("data/flood_risk_dataset_india.csv")
cyclone_data = pd.read_csv("data/cyclone_dataset.csv")

# ------------------------
# Live weather
# ------------------------
def get_live_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    r = requests.get(url)
    if r.status_code != 200:
        return None
    data = r.json()
    return {
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "rainfall": data.get("rain", {}).get("1h", 0.0),
        "lat": data["coord"]["lat"],
        "lon": data["coord"]["lon"]
    }

# ------------------------
# Nearest flood/cyclone features
# ------------------------
def nearest_flood_features(lat, lon):
    flood_data['dist'] = flood_data.apply(
        lambda row: geodesic((lat, lon), (row['Latitude'], row['Longitude'])).km, axis=1
    )
    return flood_data.loc[flood_data['dist'].idxmin()]

def nearest_cyclone_features(lat):
    cyclone_data['dist'] = abs(cyclone_data['Latitude'] - lat)
    return cyclone_data.loc[cyclone_data['dist'].idxmin()]

# ------------------------
# Feature preparation
# ------------------------
def prepare_flood_features(weather, flood_model):
    nearest = nearest_flood_features(weather["lat"], weather["lon"])
    cols = flood_model.feature_names_in_
    features = []
    for col in cols:
        if col == 'Rainfall (mm)': features.append(weather['rainfall'])
        elif col == 'Temperature (°C)': features.append(weather['temp'])
        elif col == 'Humidity (%)': features.append(weather['humidity'])
        elif col == 'Elevation (m)': features.append(0)  # optional elevation API
        elif col == 'Historical Floods': features.append(nearest['Historical Floods'])
        else: features.append(nearest[col])
    return pd.DataFrame([features], columns=cols)

def prepare_cyclone_features(weather, cyclone_model):
    nearest = nearest_cyclone_features(weather["lat"])
    cols = cyclone_model.feature_names_in_
    features = []
    for col in cols:
        if col in ['Atmospheric_Pressure', 'Pressure', 'Pressure (hPa)']: features.append(weather['pressure'])
        elif col in ['Humidity', 'Humidity (%)']: features.append(weather['humidity'])
        elif col in ['Sea_Surface_Temperature', 'Sea_Surface_Temperature (°C)']: features.append(nearest['Sea_Surface_Temperature'])
        else: features.append(nearest[col])
    return pd.DataFrame([features], columns=cols)

def risk_level(prob):
    if prob < 0.4: return "Low"
    elif prob < 0.7: return "Medium"
    else: return "High"
