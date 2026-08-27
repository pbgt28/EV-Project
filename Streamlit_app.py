import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="EV Resale Value Predictor",
    page_icon="🔋",
    layout="centered",
)

# ----------------------------------------------------------------------
# Load model artifacts (cached so they only load once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")              # dict: {column_name: fitted LabelEncoder}
    feature_columns = joblib.load("feature_columns.pkl")  # list of column names, in training order
    return model, scaler, encoders, feature_columns

model, scaler, encoders, feature_columns = load_artifacts()

# Columns that were label-encoded during training
CATEGORICAL_COLS = ["Company", "Model", "Region", "Vehicle_Type", "Usage_Type"]

st.title("🔋 EV Resale Value Predictor")
st.write(
    "Enter a vehicle's specs below to estimate its resale value (USD), "
    "based on a model trained on EV analytics data."
)

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Vehicle Identity")
    col1, col2 = st.columns(2)
    with col1:
        vehicle_id = st.number_input("Vehicle ID", min_value=1, value=1, step=1)
        company = st.selectbox("Company (Make)", options=list(encoders["Company"].classes_))
        region = st.selectbox("Region", options=list(encoders["Region"].classes_))
    with col2:
        model_name = st.selectbox("Model", options=list(encoders["Model"].classes_))
        vehicle_type = st.selectbox("Vehicle Type", options=list(encoders["Vehicle_Type"].classes_))
        usage_type = st.selectbox("Usage Type", options=list(encoders["Usage_Type"].classes_))

    year = st.slider("Year", min_value=2015, max_value=2026, value=2021, step=1)

    st.subheader("Battery & Charging")
    col3, col4 = st.columns(2)
    with col3:
        battery_capacity = st.number_input("Battery Capacity (kWh)", min_value=20.0, max_value=150.0, value=75.0)
        battery_health = st.slider("Battery Health (%)", min_value=50.0, max_value=100.0, value=90.0)
        charging_power = st.number_input("Charging Power (kW)", min_value=5.0, max_value=350.0, value=130.0)
    with col4:
        charging_time = st.number_input("Charging Time (hr)", min_value=0.1, max_value=24.0, value=1.2)
        charge_cycles = st.number_input("Charge Cycles", min_value=0, max_value=3000, value=1100, step=10)
        energy_consumption = st.number_input("Energy Consumption (kWh/100km)", min_value=5.0, max_value=40.0, value=18.5)

    st.subheader("Performance & Usage")
    col5, col6 = st.columns(2)
    with col5:
        range_km = st.number_input("Range (km)", min_value=50, max_value=800, value=375)
        total_distance = st.number_input("Total Distance Covered (km)", min_value=0, max_value=400000, value=125000, step=1000)
        avg_speed = st.number_input("Avg Speed (km/h)", min_value=10.0, max_value=150.0, value=65.0)
        max_speed = st.number_input("Max Speed (km/h)", min_value=80, max_value=300, value=190)
    with col6:
        acceleration = st.number_input("0–100 km/h Acceleration (sec)", min_value=1.0, max_value=15.0, value=6.7)
        temperature = st.number_input("Operating Temperature (°C)", min_value=-20.0, max_value=50.0, value=15.0)
        co2_saved = st.number_input("CO2 Saved (tons)", min_value=0.0, max_value=50.0, value=15.0)

    st.subheader("Costs")
    col7, col8 = st.columns(2)
    with col7:
        maintenance_cost = st.number_input("Maintenance Cost (USD)", min_value=0, max_value=5000, value=1100, step=10)
        insurance_cost = st.number_input("Insurance Cost (USD)", min_value=0, max_value=5000, value=1500, step=10)
    with col8:
        electricity_cost = st.number_input("Electricity Cost (USD/kWh)", min_value=0.01, max_value=1.0, value=0.22)
        monthly_charging_cost = st.number_input("Monthly Charging Cost (USD)", min_value=0.0, max_value=3000.0, value=420.0)

    submitted = st.form_submit_button("Predict Resale Value")

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if submitted:
    raw_input = {
        "Vehicle_ID": vehicle_id,
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
        "Total_Distance_Covered_km": total_distance,
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

    try:
        # Apply the same label encoders used during training
        for col in CATEGORICAL_COLS:
            le = encoders[col]
            input_df[col] = le.transform(input_df[col])

        # Enforce the exact column order used during training
        input_df = input_df[feature_columns]

        # Scale, then predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.success(f"### Estimated Resale Value: ${prediction:,.2f}")

    except KeyError as e:
        st.error(
            f"Column mismatch between the form and the trained model's expected "
            f"features: {e}. Check that feature_columns.pkl matches these inputs."
        )
    except ValueError as e:
        st.error(
            f"One of the selected categorical values wasn't seen during training: {e}"
        )

st.caption(
    "Model predictions are estimates based on historical data and may not reflect "
    "actual market resale value."
)
