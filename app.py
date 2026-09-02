import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page setup
st.set_page_config(
    page_title="Sales Forecasting & Inventory System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting & Business Insights Dashboard")
st.markdown("Enter pricing, promotional, date, and seasonal parameters to compute model forecasts.")

# Load Model
@st.cache_resource
def load_trained_model():
    model_files = ['sales_forecast_model.pkl', 'sales_model.pkl']
    for file in model_files:
        try:
            return joblib.load(file)
        except Exception:
            continue
    st.error("Could not load 'sales_forecast_model.pkl' or 'sales_model.pkl'. Ensure the model file is uploaded to your GitHub repository root.")
    return None

model = load_trained_model()

# --- SIDEBAR INPUTS ---
st.sidebar.header("📊 Input Features")

# Core Pricing & Spend Inputs
price = st.sidebar.number_input("Unit Price ($)", min_value=1.0, max_value=2000.0, value=25.0, step=0.5)
competitor_price = st.sidebar.number_input("Competitor Price ($)", min_value=1.0, max_value=2000.0, value=26.5, step=0.5)
discount_pct = st.sidebar.slider("Discount Percentage (%)", min_value=0, max_value=100, value=10)
marketing_spend = st.sidebar.number_input("Marketing Spend ($)", min_value=0.0, max_value=50000.0, value=500.0, step=50.0)

st.sidebar.markdown("---")
# Categorical Information
category = st.sidebar.selectbox("Category", ["Electronics", "Clothing", "Home & Kitchen", "Groceries", "Beauty"])
product_name = st.sidebar.text_input("Product Name", value="Standard_Product")
customer_segment = st.sidebar.selectbox("Customer Segment", ["Standard", "Premium", "VIP"])
weather = st.sidebar.selectbox("Weather", ["Sunny", "Rainy", "Snowy", "Cloudy", "Overcast"])

st.sidebar.markdown("---")
# Date & Time Features
month = st.sidebar.slider("Month", 1, 12, 6)
day_of_week = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
day_of_year = st.sidebar.slider("Day of Year", 1, 365, 160)
Quarter = st.sidebar.slider("Quarter", 1, 2, 3, 4)

st.sidebar.markdown("---")
# Flags & Operational Indicators
promotion_flag = st.sidebar.radio("Promotion Active?", [0, 1], index=1)
local_event_flag = st.sidebar.radio("Local Event Active?", [0, 1], index=0)
holiday_flag = st.sidebar.radio("Holiday Active?", [0, 1], index=0)
has_holiday_name = st.sidebar.radio("Has Specific Holiday Name?", [0, 1], index=0)
is_weekend = st.sidebar.radio("Is Weekend?", [0, 1], index=1 if day_of_week in [5, 6] else 0)
stock_avail = st.sidebar.radio("Stock Available?", [0, 1], index=1)
economic_indicator = st.sidebar.number_input("Economic Indicator Index", value=1.0, step=0.05)

st.sidebar.markdown("---")
# Historical Lags
st.sidebar.subheader("Historical Lags")
units_sold_lag1 = st.sidebar.number_input("Yesterday Sales (Units_Sold_Lag1)", min_value=0, value=150)
units_sold_lag7 = st.sidebar.number_input("Last Week Sales (Units_Sold_Lag7)", min_value=0, value=145)
units_sold_rollstd_4 = st.sidebar.number_input("Rolling Std Dev (Units_Sold_RollStd_4)", min_value=0.0, value=12.5)

# --- FORECAST COMPUTATION ---
st.subheader("🤖 Predict Sales Volume & Revenue")

if st.button("Generate Forecast", type="primary"):
    if model is not None:
        # Recreate exact dataset column feature transformations
        discount_amount = price * (discount_pct / 100.0)
        discounted_price = price - discount_amount
        price_ratio = price / competitor_price if competitor_price > 0 else 1.0
        mkt_per_price = marketing_spend / price if price > 0 else 0.0

        # Cyclical transformations
        month_sin = np.sin(2 * np.pi * month / 12.0)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7.0)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7.0)

        # Flag interactions
        promo_and_weekend = promotion_flag * is_weekend
        instock_and_promo = stock_avail * promotion_flag

        # Construct DataFrame matching model schema
        input_data = pd.DataFrame([{
            'Price': price,
            'Competitor_Price': competitor_price,
            'Discounted_Price': discounted_price,
            'Discount_Amount': discount_amount,
            'Discount_Percentage': discount_pct,
            'Price_Ratio_vs_Competitor': price_ratio,
            'Marketing_Spend': marketing_spend,
            'Marketing_Spend_per_Unit_Price': mkt_per_price,
            'Weather': weather,
            'Category': category,
            'Product_Name': product_name,
            'Customer_Segment': customer_segment,
            'Month': month,
            'Day_of_Week': day_of_week,
            'DayOfYear': day_of_year,
            'Month_sin': month_sin,
            'DOW_sin': dow_sin,
            'DOW_cos': dow_cos,
            'Promotion_Flag': promotion_flag,
            'Quarter': Quarter,
            'Local_Event_Flag': local_event_flag,
            'Holiday_Flag': holiday_flag,
            'Has_Holiday_Name': has_holiday_name,
            'Is_Weekend': is_weekend,
            'Stock_Avail': stock_avail,
            'Economic_Indicator': economic_indicator,
            'Promo_and_Weekend': promo_and_weekend,
            'InStock_and_Promo': instock_and_promo,
            'Units_Sold_Lag1': units_sold_lag1,
            'Units_Sold_Lag7': units_sold_lag7,
            'Units_Sold_RollStd_4': units_sold_rollstd_4
        }])

        try:
            raw_prediction = model.predict(input_data)[0]
            predicted_units = max(0, int(round(raw_prediction)))
            expected_revenue = predicted_units * discounted_price

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Units Sold", f"{predicted_units:,} units")
            col2.metric("Effective Unit Price", f"${discounted_price:.2f}")
            col3.metric("Projected Revenue", f"${expected_revenue:,.2f}")

            st.success("✅ Prediction generated successfully!")

            st.subheader("💡 Inventory Guidelines")
            safety_stock = int(predicted_units * 0.15)
            st.info(f"""
            - **Target Inventory Stock:** {predicted_units + safety_stock:,} units 
            - **Safety Buffer (15%):** {safety_stock:,} units
            """)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
