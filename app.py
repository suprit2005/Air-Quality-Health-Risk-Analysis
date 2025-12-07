import streamlit as st
import numpy as np
import joblib
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go

# ==========================
# Page Setup
# ==========================
st.set_page_config(page_title="Air Quality Health Risk", layout="wide")

# ==========================
# Load Model
# ==========================
model = joblib.load("aqi_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# ==========================
# Background + UI Theme
# ==========================
st.markdown("""
<style>
body {
    background: linear-gradient(160deg, #0f172a, #020617);
}
.big-title {
    font-size: 44px;
    font-weight: bold;
    color: #e5e7eb;
}
.sub-text {
    color: #cbd5e1;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Title
# ==========================
st.markdown("<div class='big-title'>🌫️ Air Quality Health Risk Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Enter air pollution levels and other factors to assess health risk.</div><br>", unsafe_allow_html=True)

# ==========================
# Layout
# ==========================
left, right = st.columns([1.3, 1])

# ==========================
# SLIDERS (LEFT SIDE)
# ==========================
with left:
    PM25 = st.slider("PM2.5", 0, 500, 124)
    st.write("Value:", PM25)

    PM10 = st.slider("PM10", 0, 600, 210)
    st.write("Value:", PM10)

    NO2 = st.slider("NO₂", 0, 300, 75)
    st.write("Value:", NO2)

    O3 = st.slider("Ozone", 0, 300, 100)
    st.write("Value:", O3)

    CO = st.slider("CO", 0.0, 10.0, 3.6)
    st.write("Value:", CO)

    Temp = st.slider("Temperature (°C)", -10, 50, 32)
    st.write("Value:", Temp)

    Humidity = st.slider("Humidity (%)", 0, 100, 70)
    st.write("Value:", Humidity)

    predict_btn = st.button("🚀 Predict Health Risk")

# ==========================
# If button NOT clicked yet → show simple message on right
# ==========================
if not predict_btn:
    with right:
        st.markdown("## 🏥 Health Risk Level")
        st.markdown(
            "<p style='color:#94a3b8;text-align:center;'>Move sliders and click <b>Predict Health Risk</b>.</p>",
            unsafe_allow_html=True
        )

# ==========================
# If button clicked → compute prediction + show gauge
# ==========================
if predict_btn:

    # ----- Hidden Auto Features -----
    NOx = NO2 * 2
    SO2 = NO2 * 0.3
    WindSpeed = 2
    WindDirection = 90
    Pressure = 1013

    Temp_Humidity_Index = Temp + (Humidity / 100) * Temp
    CO_NOx_Ratio = CO / (NOx + 1)
    NOx_NO2_Ratio = NOx / (NO2 + 1)

    now = datetime.now()
    DayOfWeek = now.weekday()
    Hour = now.hour

    AQI = PM25 + PM10 + NO2 + O3

    input_data = np.array([[ 
        CO, NOx, NO2, O3, SO2, PM25, PM10,
        Temp, Humidity, Pressure, WindSpeed, WindDirection,
        CO_NOx_Ratio, NOx_NO2_Ratio, Temp_Humidity_Index,
        AQI, DayOfWeek, Hour
    ]])

    columns = [
        "CO(GT)", "NOx(GT)", "NO2(GT)", "O3(GT)", "SO2(GT)", "PM2.5", "PM10",
        "Temperature", "Humidity", "Pressure", "WindSpeed", "WindDirection",
        "CO_NOx_Ratio", "NOx_NO2_Ratio", "Temp_Humidity_Index",
        "AirQualityIndex", "DayOfWeek", "Hour"
    ]

    input_df = pd.DataFrame(input_data, columns=columns)
    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)
    label = le.inverse_transform(pred)[0]

    # ==========================
    # RIGHT SIDE: BIG TEXT + SIMPLE GAUGE
    # ==========================
    with right:
        st.markdown("## 🏥 Health Risk Level")

        color_map = {
            "Good": "#22c55e",
            "Moderate": "#eab308",
            "Poor": "#fb923c",
            "Very Poor": "#ef4444",
            "Hazardous": "#7f1d1d"
        }
        text_color = color_map.get(label, "white")

        st.markdown(
            f"<h1 style='color:{text_color}; text-align:center; font-size:48px;'>{label.upper()}</h1>",
            unsafe_allow_html=True
        )

        # Map label to 0–100 gauge value
        gauge_map = {
            "Good": 10,
            "Moderate": 30,
            "Poor": 50,
            "Very Poor": 75,
            "Hazardous": 95
        }
        value = gauge_map.get(label, 0)

        # Clean angular gauge (no weird triple arc)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number={'suffix': "%"},
            title={'text': "Risk Level"},
            gauge={
                'shape': "angular",   # allowed: 'angular' or 'bullet'
                'axis': {'range': [0, 100]},
                'bar': {'color': text_color, 'thickness': 0.3},
                'bgcolor': "rgba(0,0,0,0)",
                'steps': [
                    {'range': [0, 20], 'color': '#064e3b'},     # Good zone
                    {'range': [20, 40], 'color': '#4b5563'},   # Moderate
                    {'range': [40, 60], 'color': '#78350f'},   # Poor
                    {'range': [60, 80], 'color': '#7f1d1d'},   # Very Poor
                    {'range': [80, 100], 'color': '#450a0a'}   # Hazardous
                ],
            }
        ))

        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )

        st.plotly_chart(fig, use_container_width=True)

        # Simple legend so you understand the gauge
        st.markdown("#### Gauge Meaning")
        st.markdown("- **0–20% → Good**  \n- **20–40% → Moderate**  \n- **40–60% → Poor**  \n- **60–80% → Very Poor**  \n- **80–100% → Hazardous**")
