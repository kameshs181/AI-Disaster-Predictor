# =====================================================
# Flood & Cyclone Prediction System
# (Historical + Geo Features)
# =====================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib, os

# -----------------------------
# 1️⃣ Load Datasets
# -----------------------------
flood_data = pd.read_csv("data/flood_risk_dataset_india.csv")
cyclone_data = pd.read_csv("data/cyclone_dataset.csv")

print("Flood Columns:", flood_data.columns.tolist())
print("Cyclone Columns:", cyclone_data.columns.tolist())

# -----------------------------
# 2️⃣ Select Features and Targets
# -----------------------------
# Flood dataset
flood_features = ['Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)', 
                  'Elevation (m)', 'Historical Floods']
flood_target = 'Flood Occurred'

# Cyclone dataset
cyclone_features = ['Wind_Shear', 'Atmospheric_Pressure', 'Humidity', 'Sea_Surface_Temperature']
cyclone_target = 'Cyclone'

# -----------------------------
# 3️⃣ Drop missing rows
# -----------------------------
flood_data = flood_data.dropna(subset=flood_features + [flood_target])
cyclone_data = cyclone_data.dropna(subset=cyclone_features + [cyclone_target])

# -----------------------------
# 4️⃣ Train Flood Model
# -----------------------------
Xf = flood_data[flood_features]
yf = flood_data[flood_target]

Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    Xf, yf, test_size=0.2, random_state=42
)

flood_model = RandomForestClassifier(n_estimators=100, random_state=42)
flood_model.fit(Xf_train, yf_train)
print("\n🌊 Flood Model Accuracy:", accuracy_score(yf_test, flood_model.predict(Xf_test)))

# -----------------------------
# 5️⃣ Train Cyclone Model
# -----------------------------
Xc = cyclone_data[cyclone_features]
yc = cyclone_data[cyclone_target]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, random_state=42
)

cyclone_model = RandomForestClassifier(n_estimators=100, random_state=42)
cyclone_model.fit(Xc_train, yc_train)
print("🌪 Cyclone Model Accuracy:", accuracy_score(yc_test, cyclone_model.predict(Xc_test)))

# -----------------------------
# 6️⃣ Save Models
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(flood_model, "models/flood_model.pkl")
joblib.dump(cyclone_model, "models/cyclone_model.pkl")
print("\n✅ Flood & Cyclone models trained and saved successfully!")
