import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="EV Charging Demand Prediction",
    page_icon="⚡",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("rf_ev_charging_demand.pkl")

model = load_model()

# ---------------- UI ----------------
st.title("⚡ EV Charging Demand Prediction")
st.write("Predict hourly EV charging demand (kWh) using real UrbanEV data")

st.divider()

# ---------------- INPUT FEATURES ----------------
hour = st.slider("⏰ Hour of Day", 0, 23, 12)

day = st.selectbox(
    "📅 Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

month = st.selectbox("🗓 Month", list(range(1, 13)))

# 🔥 NEW: WEEKDAY / WEEKEND TOGGLE
day_type = st.radio(
    "🛑 Day Type",
    ["Weekday", "Weekend"]
)

# ---------------- ENCODING ----------------
day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}

day_of_week = day_map[day]
is_weekend = 1 if day_type == "Weekend" else 0

# ---------------- FEATURE VECTOR ----------------
features = np.array([[hour, day_of_week, is_weekend, month]])

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Charging Demand"):
    prediction = model.predict(features)[0]

    st.success(f"⚡ Predicted Charging Demand: **{prediction:.2f} kWh**")

  
   #  BEHAVIORAL INSIGHT (WOW Factor)
if is_weekend == 1:
    st.info(
        "📊 Weekend behavior: Charging demand shifts to flexible hours "
        "(11 AM–3 PM & 6 PM–10 PM), with approximately 15–25% higher "
        "residential charging compared to weekdays."
    )
else:
    st.info(
        "📊 Weekday behavior: Charging demand peaks during commute hours "
        "(7–9 AM & 6–8 PM) due to office travel patterns."
    )


st.divider()
st.caption("Random Forest | UrbanEV Dataset | Real EV Charging Demand")


