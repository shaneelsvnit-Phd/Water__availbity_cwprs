import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Page Config ---
st.set_page_config(page_title="CWPRS HIF Smart Water Management (Live)", layout="wide")

# --- Configuration ---
# REPLACE 'YOUR_API_KEY' WITH YOUR ACTUAL OPENWEATHERMAP API KEY
API_KEY = "fe9989bf293345cb172b17490a704651" 
CITY_NAME = "Pune"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"

# --- Title & Context ---
st.title("💧 CWPRS Hybrid Intelligence Framework (HIF) - Pune Region")
st.markdown(f"""
**By Shaneel S SAO, AI Model Developer Scientist Cadre Dashboard** This system predicts surface/ground water availability by fusing **Historical Data** with **Live Weather API Data** for **{CITY_NAME}**.
""")

# --- 1. Data Loading & Pre-processing (Layer 2) ---
@st.cache_data
def load_and_prep_data():
    try:
        df = pd.read_csv("water_availability_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run the data generation script first.")
        return None

# --- Helper: Fetch Live Weather ---
def get_live_weather():
    if API_KEY == "fe9989bf293345cb172b17490a704651":
        return None
    
    try:
        url = f"{BASE_URL}q={CITY_NAME}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200:
            weather_data = {
                "Temperature_C": data["main"]["temp"],
                "Humidity_pct": data["main"]["humidity"],
                # OpenWeatherMap returns rain field only if it rained recently
                "Rainfall_mm": data.get("rain", {}).get("1h", 0) 
            }
            return weather_data
        else:
            st.sidebar.error(f"API Error: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        st.sidebar.error(f"Connection Error: {e}")
        return None

df = load_and_prep_data()

if df is not None:
    # --- 2. AI/ML Core (Model Training - Layer 3) ---
    st.sidebar.header("🛠️ Model Configuration")
    
    # Define features used for training
    feature_cols = ["Rainfall_mm", "Temperature_C", "Humidity_pct", "pH_Level", "Turbidity_NTU", "Flow_Rate_m3h", "Daily_Usage_m3"]
    target_col = "Water_Availability_Level"

    # Train Model
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- 3. Live Data Integration (Layer 1) ---
    st.sidebar.subheader("📡 Live Data Feed")
    
    # Session state to hold input values (default to averages)
    if 'inputs' not in st.session_state:
        st.session_state.inputs = {col: float(df[col].mean()) for col in feature_cols}

    # Button to fetch live weather
    if st.sidebar.button(f"Fetch Live Weather ({CITY_NAME})"):
        live_weather = get_live_weather()
        if live_weather:
            st.session_state.inputs["Temperature_C"] = live_weather["Temperature_C"]
            st.session_state.inputs["Humidity_pct"] = live_weather["Humidity_pct"]
            st.session_state.inputs["Rainfall_mm"] = live_weather["Rainfall_mm"]
            st.sidebar.success(f"Updated: {live_weather}")
        else:
            st.sidebar.warning("Using historical averages (API fetch failed).")

    # --- 4. User Interface & Decision Support (Layer 4 & 5) ---
    
    # Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🎛️ Prediction Parameters")
        st.info("Values auto-update from API, or adjust manually below.")
        
        # Create input form
        current_inputs = {}
        for col in feature_cols:
            # Determine min/max for sliders
            min_v = float(df[col].min())
            max_v = float(df[col].max())
            
            # Use session state (live data) as the default value for the slider
            default_v = st.session_state.inputs[col]
            
            # Ensure default is within bounds
            default_v = max(min_v, min(default_v, max_v))
            
            current_inputs[col] = st.slider(f"{col}", min_v, max_v, default_v)

        # Predict Button
        predict_btn = st.button("Run Prediction Model")

    with col2:
        st.subheader(f"📊 Real-time Water Availability: {CITY_NAME}")
        
        if predict_btn:
            # Create dataframe for prediction
            input_df = pd.DataFrame([current_inputs])
            prediction = model.predict(input_df)[0]
            
            # Decision Support Logic (Layer 4)
            st.markdown("### AI Analysis Result")
            
            # Visual Gauge for Water Level
            fig_gauge = px.bar(
                x=[prediction], 
                y=["Availability"], 
                orientation='h', 
                range_x=[0, 15000],
                title="Predicted Water Availability Level (Index)",
                labels={'x': 'Water Level Index', 'y': ''}
            )
            
            # Color logic based on scarcity
            if prediction < 5000:
                color = "red"
                status = "CRITICAL SCARCITY"
                advice = "⚠️ ACTION REQUIRED: Activate emergency rationing. Divert non-essential usage."
            elif prediction < 7500:
                color = "orange"
                status = "WARNING"
                advice = "⚠️ NOTE: Reduce pressure in distribution network. Monitor usage closely."
            else:
                color = "green"
                status = "OPTIMAL"
                advice = "✅ SYSTEM NORMAL: Maintain standard distribution schedule."

            fig_gauge.update_traces(marker_color=color)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown(f"**Status:** <span style='color:{color};font-size:20px'>**{status}**</span>", unsafe_allow_html=True)
            st.info(advice)
            
            # Comparison with Historical Average
            avg_avail = df[target_col].mean()
            delta = prediction - avg_avail
            st.metric("Deviation from Historical Average", f"{delta:.2f}", delta_color="normal")

        else:
            st.write("👈 Adjust parameters or fetch live weather to run the prediction.")
            
            # Show Model Metrics
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            st.markdown("---")
            st.metric("Model Reliability (R² Score)", f"{r2:.2f}")
            st.caption("Based on Random Forest Regression on historical datasets.")
