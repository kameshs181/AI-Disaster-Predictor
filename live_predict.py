import pandas as pd
import requests
import joblib
from geopy.distance import geodesic

# -----------------------------
# 1️⃣ Load Models & Datasets
# -----------------------------
flood_model = joblib.load("models/flood_model.pkl")
cyclone_model = joblib.load("models/cyclone_model.pkl")

flood_data = pd.read_csv("data/flood_risk_dataset_india.csv")
cyclone_data = pd.read_csv("data/cyclone_dataset.csv")

API_KEY = "c1fed68d02f226d73e811e6cce276bf6"

# -----------------------------
# 2️⃣ Helper Functions
# -----------------------------
def get_live_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    r = requests.get(url)
    if r.status_code != 200:
        print("❌ Error fetching weather:", r.json())
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

def get_elevation(lat, lon):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    r = requests.get(url)
    if r.status_code != 200:
        return 0
    return r.json()['results'][0]['elevation']

def nearest_flood_features(lat, lon):
    flood_data['dist'] = flood_data.apply(
        lambda row: geodesic((lat, lon), (row['Latitude'], row['Longitude'])).km, axis=1
    )
    return flood_data.loc[flood_data['dist'].idxmin()]

def nearest_cyclone_features(lat):
    # Cyclone dataset has no Longitude, use Latitude only
    cyclone_data['dist'] = abs(cyclone_data['Latitude'] - lat)
    return cyclone_data.loc[cyclone_data['dist'].idxmin()]

# -----------------------------
# 3️⃣ Prepare Features (auto match training columns)
# -----------------------------
def prepare_flood_features(weather):
    nearest = nearest_flood_features(weather["lat"], weather["lon"])
    
    # Get trained feature names from model
    flood_columns = flood_model.feature_names_in_
    
    # Map live/historical values to training columns
    feature_values = []
    for col in flood_columns:
        if col == 'Rainfall (mm)':
            feature_values.append(weather['rainfall'])
        elif col == 'Temperature (°C)':
            feature_values.append(weather['temp'])
        elif col == 'Humidity (%)':
            feature_values.append(weather['humidity'])
        elif col == 'Elevation (m)':
            feature_values.append(get_elevation(weather['lat'], weather['lon']))
        elif col == 'Historical Floods':
            feature_values.append(nearest['Historical Floods'])
        else:
            # For other columns, take nearest historical value
            feature_values.append(nearest[col])
    
    return pd.DataFrame([feature_values], columns=flood_columns)

def prepare_cyclone_features(weather):
    nearest = nearest_cyclone_features(weather["lat"])
    
    # Get trained feature names from model
    cyclone_columns = cyclone_model.feature_names_in_
    
    feature_values = []
    for col in cyclone_columns:
        if col in ['Atmospheric_Pressure', 'Pressure', 'Pressure (hPa)']:
            feature_values.append(weather['pressure'])
        elif col in ['Humidity', 'Humidity (%)']:
            feature_values.append(weather['humidity'])
        elif col in ['Sea_Surface_Temperature', 'Sea_Surface_Temperature (°C)']:
            feature_values.append(nearest['Sea_Surface_Temperature'])
        else:
            feature_values.append(nearest[col])
    
    return pd.DataFrame([feature_values], columns=cyclone_columns)

# -----------------------------
# 4️⃣ Risk Level Helper
# -----------------------------
def risk_level(prob):
    if prob < 0.4:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"

# -----------------------------
# 5️⃣ Live Prediction
# -----------------------------
def live_smart_prediction(city):
    weather = get_live_weather(city)
    if not weather:
        return
    
    print("\n🌦️ Live Weather Data:", weather)
    
    flood_features_df = prepare_flood_features(weather)
    cyclone_features_df = prepare_cyclone_features(weather)
    
    flood_prob = flood_model.predict_proba(flood_features_df)[0][1]
    cyclone_prob = cyclone_model.predict_proba(cyclone_features_df)[0][1]
    
    print("\n🌊 Flood Risk Probability: {:.2f} ({})".format(flood_prob, risk_level(flood_prob)))
    print("🌪 Cyclone Risk Probability: {:.2f} ({})".format(cyclone_prob, risk_level(cyclone_prob)))
    
    if flood_prob >= 0.4:
        print("⚠️ Flood Alert!")
    if cyclone_prob >= 0.4:
        print("⚠️ Cyclone Alert!")
    if flood_prob < 0.4 and cyclone_prob < 0.4:
        print("✅ No significant disaster risk detected.")

# -----------------------------
# 6️⃣ Run Script
# -----------------------------
if __name__ == "__main__":
    city = input("🏙️ Enter city name: ")
    live_smart_prediction(city)
