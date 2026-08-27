import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="EV Resale Value Predictor",
    page_icon="🚗",
    layout="wide"
)


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🚗 Electric Vehicle Resale Value Predictor")
st.write(
    "Predict the resale value of an electric vehicle using "
    "Machine Learning models."
)

st.divider()


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("electric_vehicle_analytics(10).csv")

    # Rename columns as done in the notebook
    df = df.rename(
        columns={
            "Mileage_km": "Total_Distance_Covered_km",
            "Make": "Company"
        }
    )

    return df


df = load_data()


# ---------------------------------------------------
# DATA PREPROCESSING
# ---------------------------------------------------

@st.cache_resource
def train_models(df):

    data = df.copy()

    # Drop Vehicle ID
    if "Vehicle_ID" in data.columns:
        data = data.drop("Vehicle_ID", axis=1)

    # Categorical columns
    categorical_cols = [
        "Company",
        "Model",
        "Region",
        "Vehicle_Type",
        "Usage_Type"
    ]

    # Store encoders
    encoders = {}

    # Label Encoding
    for col in categorical_cols:

        le = LabelEncoder()

        data[col] = le.fit_transform(
            data[col].astype(str)
        )

        encoders[col] = le

    # Features and target
    X = data.drop("Resale_Value_USD", axis=1)
    y = data["Resale_Value_USD"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Standard Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------
    # Linear Regression
    # ------------------------------------------------

    lr = LinearRegression()

    lr.fit(
        X_train_scaled,
        y_train
    )

    lr_pred = lr.predict(X_test_scaled)

    # ------------------------------------------------
    # Decision Tree
    # ------------------------------------------------

    dt = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    dt.fit(
        X_train_scaled,
        y_train
    )

    dt_pred = dt.predict(X_test_scaled)

    # ------------------------------------------------
    # Random Forest
    # ------------------------------------------------

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    )

    rf.fit(
        X_train_scaled,
        y_train
    )

    rf_pred = rf.predict(X_test_scaled)

    # ------------------------------------------------
    # Model Metrics
    # ------------------------------------------------

    metrics = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest"
        ],

        "R²": [
            r2_score(y_test, lr_pred),
            r2_score(y_test, dt_pred),
            r2_score(y_test, rf_pred)
        ],

        "MAE": [
            mean_absolute_error(y_test, lr_pred),
            mean_absolute_error(y_test, dt_pred),
            mean_absolute_error(y_test, rf_pred)
        ],

        "RMSE": [
            np.sqrt(mean_squared_error(y_test, lr_pred)),
            np.sqrt(mean_squared_error(y_test, dt_pred)),
            np.sqrt(mean_squared_error(y_test, rf_pred))
        ]
    })

    return (
        X,
        y,
        X_test,
        y_test,
        encoders,
        scaler,
        lr,
        dt,
        rf,
        metrics
    )


(
    X,
    y,
    X_test,
    y_test,
    encoders,
    scaler,
    lr,
    dt,
    rf,
    metrics
) = train_models(df)


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Prediction",
        "Dataset",
        "Model Performance",
        "Feature Importance"
    ]
)


# ===================================================
# PREDICTION PAGE
# ===================================================

if page == "Prediction":

    st.header("🔮 Predict EV Resale Value")

    st.write(
        "Enter the vehicle specifications below."
    )

    col1, col2 = st.columns(2)

    # ------------------------------------------------
    # CATEGORICAL INPUTS
    # ------------------------------------------------

    with col1:

        company = st.selectbox(
            "Company",
            sorted(df["Company"].astype(str).unique())
        )

        model = st.selectbox(
            "Model",
            sorted(df["Model"].astype(str).unique())
        )

        region = st.selectbox(
            "Region",
            sorted(df["Region"].astype(str).unique())
        )

        vehicle_type = st.selectbox(
            "Vehicle Type",
            sorted(df["Vehicle_Type"].astype(str).unique())
        )

        usage_type = st.selectbox(
            "Usage Type",
            sorted(df["Usage_Type"].astype(str).unique())
        )

    # ------------------------------------------------
    # NUMERICAL INPUTS
    # ------------------------------------------------

    with col2:

        battery_capacity = st.number_input(
            "Battery Capacity (kWh)",
            min_value=float(df["Battery_Capacity_kWh"].min()),
            max_value=float(df["Battery_Capacity_kWh"].max()),
            value=float(df["Battery_Capacity_kWh"].median())
        )

        range_km = st.number_input(
            "Driving Range (km)",
            min_value=float(df["Range_km"].min()),
            max_value=float(df["Range_km"].max()),
            value=float(df["Range_km"].median())
        )

        battery_health = st.number_input(
            "Battery Health (%)",
            min_value=float(df["Battery_Health_%"].min()),
            max_value=float(df["Battery_Health_%"].max()),
            value=float(df["Battery_Health_%"].median())
        )

        charging_power = st.number_input(
            "Charging Power (kW)",
            min_value=float(df["Charging_Power_kW"].min()),
            max_value=float(df["Charging_Power_kW"].max()),
            value=float(df["Charging_Power_kW"].median())
        )

        charging_time = st.number_input(
            "Charging Time (hr)",
            min_value=float(df["Charging_Time_hr"].min()),
            max_value=float(df["Charging_Time_hr"].max()),
            value=float(df["Charging_Time_hr"].median())
        )

        total_distance = st.number_input(
            "Total Distance Covered (km)",
            min_value=float(df["Total_Distance_Covered_km"].min()),
            max_value=float(df["Total_Distance_Covered_km"].max()),
            value=float(df["Total_Distance_Covered_km"].median())
        )


    # ------------------------------------------------
    # PREDICT BUTTON
    # ------------------------------------------------

    if st.button(
        "🚀 Predict Resale Value",
        use_container_width=True
    ):

        # Encode categorical values

        company_encoded = encoders["Company"].transform(
            [company]
        )[0]

        model_encoded = encoders["Model"].transform(
            [model]
        )[0]

        region_encoded = encoders["Region"].transform(
            [region]
        )[0]

        vehicle_type_encoded = encoders["Vehicle_Type"].transform(
            [vehicle_type]
        )[0]

        usage_type_encoded = encoders["Usage_Type"].transform(
            [usage_type]
        )[0]


        # Create input dataframe
        input_data = pd.DataFrame({

            "Company": [company_encoded],

            "Model": [model_encoded],

            "Region": [region_encoded],

            "Vehicle_Type": [vehicle_type_encoded],

            "Usage_Type": [usage_type_encoded],

            "Battery_Capacity_kWh": [
                battery_capacity
            ],

            "Range_km": [
                range_km
            ],

            "Battery_Health_%": [
                battery_health
            ],

            "Charging_Power_kW": [
                charging_power
            ],

            "Charging_Time_hr": [
                charging_time
            ],

            "Total_Distance_Covered_km": [
                total_distance
            ]
        })


        # Make sure feature order is identical
        input_data = input_data[X.columns]


        # Scale input
        input_scaled = scaler.transform(
            input_data
        )


        # Predictions
        prediction_lr = lr.predict(
            input_scaled
        )[0]

        prediction_dt = dt.predict(
            input_scaled
        )[0]

        prediction_rf = rf.predict(
            input_scaled
        )[0]


        # ------------------------------------------------
        # DISPLAY RESULTS
        # ------------------------------------------------

        st.success("Prediction completed successfully!")

        st.subheader("💰 Estimated Resale Value")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Linear Regression",
                f"${prediction_lr:,.2f}"
            )

        with result_col2:

            st.metric(
                "Decision Tree",
                f"${prediction_dt:,.2f}"
            )

        with result_col3:

            st.metric(
                "Random Forest",
                f"${prediction_rf:,.2f}"
            )


        st.divider()

        # Random Forest is generally the preferred model
        st.subheader("⭐ Recommended Prediction")

        st.success(
            f"Estimated EV Resale Value: "
            f"${prediction_rf:,.2f}"
        )


# ===================================================
# DATASET PAGE
# ===================================================

elif page == "Dataset":

    st.header("📊 EV Dataset")

    st.write(
        f"Dataset contains **{df.shape[0]} rows** "
        f"and **{df.shape[1]} columns**."
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Dataset Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# ===================================================
# MODEL PERFORMANCE
# ===================================================

elif page == "Model Performance":

    st.header("📈 Model Performance")

    st.write(
        "Comparison of the three machine learning models "
        "on the test dataset."
    )

    display_metrics = metrics.copy()

    display_metrics["R²"] = display_metrics["R²"].round(3)
    display_metrics["MAE"] = display_metrics["MAE"].round(2)
    display_metrics["RMSE"] = display_metrics["RMSE"].round(2)

    st.dataframe(
        display_metrics,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("R² Score Comparison")

    chart_data = metrics.set_index("Model")["R²"]

    st.bar_chart(chart_data)

    st.subheader("RMSE Comparison")

    rmse_data = metrics.set_index("Model")["RMSE"]

    st.bar_chart(rmse_data)


# ===================================================
# FEATURE IMPORTANCE
# ===================================================

elif page == "Feature Importance":

    st.header("🌟 Random Forest Feature Importance")

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": rf.feature_importances_

    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.dataframe(
        importance,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Feature Importance Chart")

    chart_data = importance.set_index(
        "Feature"
    )["Importance"]

    st.bar_chart(chart_data)


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.caption(
    "Electric Vehicle Analytics | Machine Learning Project"
)
