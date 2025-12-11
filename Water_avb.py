import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Page Config ---
st.set_page_config(page_title="HIF Smart Water Management", layout="wide")

# --- Title & Context  ---
st.title("💧CWPRS Hybrid Intelligence Framework (HIF) for Smart Water Management")
st.markdown("""
**By Shaneel S SAO, AI Model Developer Dashboard**
This system implements the **Five-Layer Architecture** for predicting water availability.
It utilizes **Random Forest Regression** for predictive analytics  and simulates **IoT/Meteorological data integration**[cite: 32, 37].
""")

# --- 1. Data Loading & Layer 2: Pre-processing  ---
@st.cache_data
def load_and_prep_data():
    try:
        df = pd.read_csv("water_availability_data.csv")
        # Feature Engineering: Extract month for seasonality
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run the data generation script first.")
        return None

df = load_and_prep_data()

if df is not None:
    # --- 2. Layer 3: AI/ML Core (Model Training) [cite: 45] ---
    st.sidebar.header("🛠️ Model Configuration")
    
    # User selects features based on available sensors 
    feature_cols = st.sidebar.multiselect(
        "Select Input Features (Sensors/Weather)",
        ["Rainfall_mm", "Temperature_C", "Humidity_pct", "pH_Level", "Turbidity_NTU", "Flow_Rate_m3h", "Daily_Usage_m3"],
        default=["Rainfall_mm", "Temperature_C", "Daily_Usage_m3", "Flow_Rate_m3h"]
    )
    
    target_col = "Water_Availability_Level"

    # Train/Test Split
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model: Random Forest (Ensemble method recommended for maintenance/prediction )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)

    # --- 3. Layer 5: User Interface / Visualization  ---
    
    # Top Metrics Row
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy (R²)", f"{r2:.2f}", help="Indicates how well the AI explains the variance in water levels.")
    col2.metric("Prediction Error (MSE)", f"{mse:.0f}", help="Mean Squared Error of the prediction.")
    col3.metric("Data Points Processed", f"{len(df)}", help="Total historical records analyzed.")

    st.markdown("---")

    # Layout: Control Panel (Left) & Predictions (Right)
    row1_col1, row1_col2 = st.columns([1, 2])

    with row1_col1:
        st.subheader("🎛️ Real-time Scenario Simulation")
        st.write("Adjust parameters to predict Water Availability (Decision Support)[cite: 10].")
        
        input_data = {}
        for col in feature_cols:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            avg_val = float(df[col].mean())
            input_data[col] = st.slider(f"{col}", min_val, max_val, avg_val)
        
        # Prediction Action
        if st.button("Predict Availability"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            
            # Layer 4: Decision Support Logic 
            status = "Normal"
            color = "green"
            if prediction < 5000: # Threshold example
                status = "CRITICAL SCARCITY"
                color = "red"
            elif prediction < 7500:
                status = "Warning: Low Levels"
                color = "orange"
                
            st.markdown(f"### Predicted Level: **{prediction:.2f}**")
            st.markdown(f"Status: <span style='color:{color};font-weight:bold'>{status}</span>", unsafe_allow_html=True)
            
            if status == "CRITICAL SCARCITY":
                st.warning("⚠️ ALERT: Initiate water rationing protocols immediately. (Optimization Engine Triggered) [cite: 52]")

    with row1_col2:
        st.subheader("📈 System Trends & Analysis")
        
        # Visualization 1: Actual vs Predicted (Test Set)
        results_df = pd.DataFrame({"Actual": y_test, "Predicted": preds}).sort_index()
        fig_perf = px.line(results_df.head(100), title="Model Performance: Actual vs Predicted Water Levels (Subset)")
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Visualization 2: Feature Importance
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        fig_imp = px.bar(feat_importances, orientation='h', title="Feature Importance: What drives Water Availability?")
        st.plotly_chart(fig_imp, use_container_width=True)

    # Layer 4: Decision Support - Historical Context
    st.subheader("📊 Historical Data Overview")
    st.dataframe(df.head())
    
else:
    st.info("Awaiting Dataset...")
