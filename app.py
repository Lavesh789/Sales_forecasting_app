import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sales Forecasting Dashboard", layout="wide"
)
st.title("📈 Sales Demand & Revenue Forecasting")


@st.cache_resource
def load_forecasting_pipeline():
    # Load your trained time-series / regression model pipeline
    return joblib.load("sales_forecast_model.pkl")


try:
    pipeline = load_forecasting_pipeline()
    st.sidebar.success("Forecasting Model Loaded!")
except Exception as e:
    st.sidebar.error(f"Error loading model file: {e}")
    st.stop()

# --- Input Parameters Section ---
st.subheader("Select Parameters for Forecast")
col1, col2, col3 = st.columns(3)

with col1:
    store_id = st.selectbox("Store ID", ["S001", "S002", "S003"])
    product_category = st.selectbox(
        "Category", ["Electronics", "Apparel", "Home & Kitchen"]
    )
    forecast_date = st.date_input("Target Forecast Date")

with col2:
    unit_price = st.number_input("Unit Price ($)", value=49.99, min_value=0.0)
    discount_pct = st.slider("Discount (%)", 0, 50, 10)
    is_promo = st.checkbox("Active Promotion / Campaign", value=True)

with col3:
    lag_1_sales = st.number_input(
        "Sales Yesterday (Units)", value=120, min_value=0
    )
    lag_7_sales = st.number_input(
        "Sales Same Day Last Week (Units)", value=115, min_value=0
    )
    rolling_mean_7 = st.number_input(
        "7-Day Avg Sales (Units)", value=118.0, min_value=0.0
    )

if st.button("Generate Forecast", type="primary"):
    # Convert date inputs into datetime temporal features
    dt = pd.to_datetime(forecast_date)
    month = dt.month
    day_of_week = dt.day_name()
    day_of_year = dt.dayofyear
    quarter = f"Q{dt.quarter}"

    # Cyclic encoding for seasonality
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)

    # Build feature DataFrame expected by the pipeline
    input_df = pd.DataFrame(
        [
            {
                "Store_ID": store_id,
                "Category": product_category,
                "Unit_Price": unit_price,
                "Discount_Percentage": discount_pct,
                "Is_Promo": int(is_promo),
                "Day_of_Week": day_of_week,
                "Month": month,
                "Quarter": quarter,
                "DayOfYear": day_of_year,
                "Month_sin": month_sin,
                "Month_cos": month_cos,
                "Sales_Lag_1": lag_1_sales,
                "Sales_Lag_7": lag_7_sales,
                "Sales_RollMean_7": rolling_mean_7,
            }
        ]
    )

    # Predict forecasted units and revenue
    predicted_units = pipeline.predict(input_df)[0]
    effective_price = unit_price * (1 - discount_pct / 100.0)
    predicted_revenue = predicted_units * effective_price

    # Display key metrics
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Forecasted Demand", f"{int(np.round(predicted_units))} Units")
    res_col2.metric("Projected Revenue", f"${predicted_revenue:,.2f}")
