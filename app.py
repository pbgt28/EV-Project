import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="EV Resale Value Predictor", page_icon="🔋", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, encoders, feature_columns


model, scaler, encoders, feature_columns = load_artifacts()

st.title("🔋 EV Resale Value Predictor")
st.write("Enter the vehicle's details to estimate its resale value (USD).")

with st.form("prediction_form"):
    st.subheader("Vehicle Details")
    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox("Company", encoders["Company"].classes_)
        model_name = st.selectbox("Model", encoders["Model"].classes_)
        year = st.number_input("Year", min_value=2015, max_value=2026, value=2020, step=1)
        region = st.selectbox("Region", encoders["Region"].classes_)
    with col2:
        vehicle_type = st.selectbox("Vehicle Type", encoders["Vehicle_Type"].classes_)
        usage_type = st.selectbox("Usage Type", encoders["Usage_Type"].classes_)
        temperature = st.slider("Avg Operating Temperature (°C)", -10.0, 40.0, 15.0)

    st.subheader("Battery & Range")
    col3, col4 = st.columns(2)
    with col3:
        battery_capacity = st.slider("Battery Capacity (kWh)", 30.0, 120.0, 75.0)
        battery_health = st.slider("Battery Health (%)", 70.0, 100.0, 90.0)
        range_km = st.slider("Range (km)", 121, 713, 375)
        charge_cycles = st.slider("Charge Cycles", 200, 2000, 1000)
    with col4:
        charging_power = st.slider("Charging Power (kW)", 11.1, 250.0, 100.0)
        charging_time = st.slider("Charging Time (hr)", 0.1, 12.5, 1.2)
        energy_consumption = st.slider("Energy Consumption (kWh/100km)", 12.0, 25.0, 18.5)

    st.subheader("Usage & Performance")
    col5, col6 = st.columns(2)
    with col5:
        distance_covered = st.number_input("Total Distance Covered (km)", min_value=0, max_value=300000, value=100000, step=1000)
        avg_speed = st.slider("Average Speed (km/h)", 30.0, 100.0, 65.0)
        max_speed = st.slider("Max Speed (km/h)", 130, 250, 190)
    with col6:
        acceleration = st.slider("0–100 km/h Acceleration (sec)", 3.5, 10.0, 6.5)
        co2_saved = st.slider("CO2 Saved (tons)", 0.0, 30.0, 15.0)

    st.subheader("Costs")
    col7, col8 = st.columns(2)
    with col7:
        maintenance_cost = st.number_input("Maintenance Cost (USD)", min_value=0, max_value=5000, value=1000, step=50)
        insurance_cost = st.number_input("Insurance Cost (USD)", min_value=0, max_value=5000, value=1500, step=50)
    with col8:
        electricity_cost = st.slider("Electricity Cost (USD/kWh)", 0.08, 0.35, 0.22)
        monthly_charging_cost = st.number_input("Monthly Charging Cost (USD)", min_value=0.0, max_value=2000.0, value=400.0, step=10.0)

    submitted = st.form_submit_button("Predict Resale Value")

if submitted:
    raw_input = {
        "Company": company,
        "Model": model_name,
        "Year": year,
        "Region": region,
        "Vehicle_Type": vehicle_type,
        "Battery_Capacity_kWh": battery_capacity,
        "Battery_Health_%": battery_health,
        "Range_km": range_km,
        "Charging_Power_kW": charging_power,
        "Charging_Time_hr": charging_time,
        "Charge_Cycles": charge_cycles,
        "Energy_Consumption_kWh_per_100km": energy_consumption,
        "Total_Distance_Covered_km": distance_covered,
        "Avg_Speed_kmh": avg_speed,
        "Max_Speed_kmh": max_speed,
        "Acceleration_0_100_kmh_sec": acceleration,
        "Temperature_C": temperature,
        "Usage_Type": usage_type,
        "CO2_Saved_tons": co2_saved,
        "Maintenance_Cost_USD": maintenance_cost,
        "Insurance_Cost_USD": insurance_cost,
        "Electricity_Cost_USD_per_kWh": electricity_cost,
        "Monthly_Charging_Cost_USD": monthly_charging_cost,
    }

    input_df = pd.DataFrame([raw_input])

    # Apply the same label encoders used at training time
    for col in ["Company", "Model", "Region", "Vehicle_Type", "Usage_Type"]:
        input_df[col] = encoders[col].transform(input_df[col])

    # Ensure column order matches training
    input_df = input_df[feature_columns]

    # Scale, then predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.success(f"### Estimated Resale Value: ${prediction:,.2f}")
