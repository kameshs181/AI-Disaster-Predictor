from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from dotenv import load_dotenv
import os, sqlite3, pandas as pd, joblib, requests
from geopy.distance import geodesic

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

app = Flask(__name__)
app.secret_key = SECRET_KEY
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ----------------------------- Database -----------------------------
DB = "database.db"
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, email TEXT, password TEXT, is_admin INTEGER)''')
    conn.commit(); conn.close()
init_db()

# ----------------------------- User Loader -----------------------------
class User(UserMixin):
    def __init__(self, id, username, email, password, is_admin):
        self.id = id
        self.username = username
        self.email = email
        self.password = password
        self.is_admin = is_admin

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return User(*row)
    return None

# ----------------------------- Load Models -----------------------------
flood_model = joblib.load("models/flood_model.pkl")
cyclone_model = joblib.load("models/cyclone_model.pkl")
flood_data = pd.read_csv("data/flood_risk_dataset_india.csv")
cyclone_data = pd.read_csv("data/cyclone_dataset.csv")

# ----------------------------- Routes -----------------------------
@app.route('/')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method=='POST':
        username = request.form['username']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("INSERT INTO users (username,email,password,is_admin) VALUES (?,?,?,0)", (username,email,password))
        conn.commit(); conn.close()
        flash("Registration Successful!", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        email = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=?", (email,))
        row = c.fetchone()
        conn.close()
        if row and bcrypt.check_password_hash(row[3], password):
            user = User(*row)
            login_user(user)
            return redirect(url_for('dashboard'))
        flash("Invalid Credentials", "danger")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ----------------------------- Live Prediction API -----------------------------
def get_live_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    r = requests.get(url)
    if r.status_code != 200:
        return None
    data = r.json()
    return {"temp":data["main"]["temp"],"humidity":data["main"]["humidity"],
            "pressure":data["main"]["pressure"],"rainfall":data.get("rain",{}).get("1h",0.0),
            "lat":data["coord"]["lat"],"lon":data["coord"]["lon"]}

@app.route('/predict')
@login_required
def predict():
    city = request.args.get("city")
    weather = get_live_weather(city)
    if not weather: return jsonify({"error":"City not found"}), 404

    # Flood Prediction (simplified)
    nearest = flood_data.iloc[0]  # for demo, pick nearest
    flood_features = ['Rainfall (mm)','Temperature (°C)','Humidity (%)','Elevation (m)','Historical Floods']
    Xf = pd.DataFrame([[weather['rainfall'], weather['temp'], weather['humidity'], nearest['Elevation (m)'], nearest['Historical Floods']]], columns=flood_features)
    flood_prob = float(flood_model.predict_proba(Xf)[0][1])
    flood_risk = "High" if flood_prob>=0.7 else ("Medium" if flood_prob>=0.4 else "Low")

    # Cyclone Prediction (simplified)
    nearest_c = cyclone_data.iloc[0]
    cyclone_features = ['Wind_Shear','Atmospheric_Pressure','Humidity','Sea_Surface_Temperature']
    Xc = pd.DataFrame([[nearest_c['Wind_Shear'], weather['pressure'], weather['humidity'], nearest_c['Sea_Surface_Temperature']]], columns=cyclone_features)
    cyclone_prob = float(cyclone_model.predict_proba(Xc)[0][1])
    cyclone_risk = "High" if cyclone_prob>=0.7 else ("Medium" if cyclone_prob>=0.4 else "Low")

    # Example multiple cyclone tracks for map animation
    cyclones = [{"lat":weather['lat'], "lon":weather['lon'], "risk":cyclone_risk, "track":[
                    [weather['lat']-1,weather['lon']-2],
                    [weather['lat'],weather['lon']]
                ]}]

    return jsonify({
        "city":city,
        "weather":weather,
        "flood_prob":flood_prob,
        "flood_risk":flood_risk,
        "cyclone_prob":cyclone_prob,
        "cyclone_risk":cyclone_risk,
        "cyclones":cyclones
    })

if __name__=="__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
