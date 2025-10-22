from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import joblib, requests, pandas as pd, sqlite3, os
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt

load_dotenv()

app = Flask(__name__)
bcrypt = Bcrypt(app)
app.secret_key = os.getenv("SECRET_KEY")

DB = "users.db"
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# -----------------------------
# Database setup
# -----------------------------
def init_db():
    if not os.path.exists(DB):
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT UNIQUE,
                password TEXT
            )
        """)
        conn.commit()
        conn.close()
init_db()

# -----------------------------
# Load Models
# -----------------------------
flood_model = joblib.load("models/flood_model.pkl")
cyclone_model = joblib.load("models/cyclone_model.pkl")

# -----------------------------
# Helper functions
# -----------------------------
def get_live_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    r = requests.get(url)
    if r.status_code != 200:
        return None
    d = r.json()
    return {
        "temp": d["main"]["temp"],
        "humidity": d["main"]["humidity"],
        "pressure": d["main"]["pressure"],
        "rainfall": d.get("rain", {}).get("1h", 0.0),
        "lat": d["coord"]["lat"],
        "lon": d["coord"]["lon"]
    }

def risk_level(prob):
    if prob < 0.4: return "Low"
    elif prob < 0.7: return "Medium"
    else: return "High"

# -----------------------------
# Auth Routes
# -----------------------------
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=?", (email,))
        user = c.fetchone()
        conn.close()
        if user and bcrypt.check_password_hash(user[3], password):
            session['user'] = user[1]
            session['email'] = user[2]
            # Redirect admin email to admin page
            if email == "admin@example.com":
                return redirect(url_for('admin_page'))

            return redirect(url_for('dashboard'))
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (name, email, password) VALUES (?,?,?)", (name, email, password))
            conn.commit()
        except:
            return render_template('register.html', error="Email already registered")
        conn.close()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/admin')
def admin_page():
    if 'user' not in session or session.get('email') != 'admin@example.com':
        return redirect(url_for('login'))

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT id, name, email FROM users")
    users = c.fetchall()

    # Count total users
    total_users = len(users)

    # Dummy monthly user registration data (replace later with real timestamps)
    monthly_users = [
        {"month": "Jan", "count": 4},
        {"month": "Feb", "count": 6},
        {"month": "Mar", "count": 8},
        {"month": "Apr", "count": 10},
        {"month": "May", "count": 12},
        {"month": "Jun", "count": 14},
        {"month": "Jul", "count": 15},
        {"month": "Aug", "count": 18},
        {"month": "Sep", "count": 20},
        {"month": "Oct", "count": total_users},
    ]

    conn.close()
    return render_template('admin.html', users=users, total_users=total_users, monthly_users=monthly_users)

@app.route('/delete_user/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    if 'user' not in session or session.get('email') != 'admin@example.com':
        return jsonify({'error': 'Unauthorized'}), 403
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'User deleted successfully'})


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# -----------------------------
# Dashboard
# -----------------------------
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'], openweather_api_key=API_KEY)



# -----------------------------
# Prediction API
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    city = data.get("city")
    weather = get_live_weather(city)
    if not weather:
        return jsonify({"error": "City not found"})

    # Simple live model inference (replace with real model if needed)
    flood_prob = round((weather["humidity"]/100) * (weather["rainfall"]/10 + 0.3), 2)
    cyclone_prob = round((weather["temp"]/40) * (weather["pressure"]/1000 + 0.2), 2)

    result = {
        "city": city.capitalize(),
        "weather": weather,
        "flood_prob": flood_prob,
        "flood_risk": risk_level(flood_prob),
        "cyclone_prob": cyclone_prob,
        "cyclone_risk": risk_level(cyclone_prob)
    }
    return jsonify(result)
# ========================
# Historical Stats API
# ========================
@app.route('/historical_stats')
def historical_stats():
    # Example dummy historical data for visualization
    # In a real setup, this could come from your database or CSV files
    stats = [
        {"year": "2019", "floods": 12, "cyclones": 5},
        {"year": "2020", "floods": 15, "cyclones": 7},
        {"year": "2021", "floods": 10, "cyclones": 8},
        {"year": "2022", "floods": 18, "cyclones": 6},
        {"year": "2023", "floods": 14, "cyclones": 9},
        {"year": "2024", "floods": 20, "cyclones": 11},
    ]
    return jsonify(stats)


if __name__ == '__main__':
    # Use Render-assigned port or default to 5000
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
