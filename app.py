import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting & Inventory System")
st.markdown("Enter product, pricing, and seasonal parameters to generate model forecasts.")

# Load Model
@st.cache_resource
def load_trained_model():
    model_files = ['sales_forecast_model.pkl', 'sales_model.pkl']
    for file in model_files:
        try:
            return joblib.load(file)
        except Exception:
            continue
    st.error("Model file could not be found. Ensure 'sales_forecast_model.pkl' or 'sales_model.pkl' is uploaded to your GitHub repository root.")
    return None

model = load_trained_model()

# --- SIDEBAR INPUTS ---
st.sidebar.header("📊 Model Input Features")

# Core Pricing & Promotions
price = st.sidebar.number_input("Price ($)", min_value=1.0, max_value=2000.0, value=25.0, step=0.5)
competitor_price = st.sidebar.number_input("Competitor Price ($)", min_value=1.0, max_value=2000.0, value=26.5, step=0.5)
discount_pct = st.sidebar.slider("Discount Percentage (%)", min_value=0, max_value=100, value=10)
marketing_spend = st.sidebar.number_input("Marketing Spend ($)", min_value=0.0, max_value=50000.0, value=500.0, step=50.0)

st.sidebar.markdown("---")
# Entities & Categories
product_id = st.sidebar.text_input("Product_ID", value="PROD_101")
product_name = st.sidebar.text_input("Product_Name", value="Product_A")
category = st.sidebar.selectbox("Category", ["Electronics", "Clothing", "Home & Kitchen", "Groceries", "Beauty"])
store_id = st.sidebar.text_input("Store_ID", value="STORE_01")
store_location = st.sidebar.selectbox("Store_Location", ["Urban", "Suburban", "Rural"])
sales_channel = st.sidebar.selectbox("Sales_Channel", ["In-Store", "Online"])
customer_segment = st.sidebar.selectbox("Customer_Segment", ["Standard", "Premium", "VIP"])

st.sidebar.markdown("---")
# Seasonal & Temporal Context
season = st.sidebar.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
weather = st.sidebar.selectbox("Weather", ["Sunny", "Rainy", "Snowy", "Cloudy", "Overcast"])
holiday_name = st.sidebar.selectbox("Holiday_Name", ["None", "New Year", "Christmas", "Thanksgiving", "Labor Day"])

month = st.sidebar.slider("Month", 1, 12, 6)
day = st.sidebar.slider("Day", 1, 31, 15)
day_of_week = st.sidebar.slider("Day_of_Week (0=Mon, 6=Sun)", 0, 6, 2)
day_of_year = st.sidebar.slider("DayOfYear", 1, 365, 160)
week_of_year = st.sidebar.slider("WeekOfYear", 1, 52, 24)
year = st.sidebar.number_input("Year", min_value=2020, max_value=2030, value=2025)

st.sidebar.markdown("---")
# Environmental & Operational Flags
promotion_flag = st.sidebar.radio("Promotion_Flag", [0, 1], index=1)
local_event_flag = st.sidebar.radio("Local_Event_Flag", [0, 1], index=0)
holiday_flag = st.sidebar.radio("Holiday_Flag", [0, 1], index=1 if holiday_name != "None" else 0)
has_holiday_name = st.sidebar.radio("Has_Holiday_Name", [0, 1], index=1 if holiday_name != "None" else 0)
is_weekend = st.sidebar.radio("Is_Weekend", [0, 1], index=1 if day_of_week in [5, 6] else 0)
stock_availability = st.sidebar.radio("Stock_Availability", [0, 1], index=1)
economic_indicator = st.sidebar.number_input("Economic_Indicator", value=1.0, step=0.05)

st.sidebar.markdown("---")
# Historical Lag & Rolling Statistics
st.sidebar.subheader("Lag & Rolling Features")
units_sold_lag1 = st.sidebar.number_input("Units_Sold_Lag1", min_value=0, value=150)
units_sold_lag7 = st.sidebar.number_input("Units_Sold_Lag7", min_value=0, value=145)
units_sold_rollmean_4 = st.sidebar.number_input("Units_Sold_RollMean_4", min_value=0.0, value=148.0)
units_sold_rollstd_4 = st.sidebar.number_input("Units_Sold_RollStd_4", min_value=0.0, value=12.5)

# --- INFERENCE EXECUTION ---
st.subheader("🤖 Predict Sales Volume & Revenue")

if st.button("Generate Forecast", type="primary"):
    if model is not None:
        # Re-compute Feature Engineering Variables required by model
        discount_amount = price * (discount_pct / 100.0)
        discounted_price = price - discount_amount
        price_diff_vs_competitor = price - competitor_price
        price_ratio_vs_competitor = price / competitor_price if competitor_price > 0 else 1.0
        marketing_spend_per_unit_price = marketing_spend / price if price > 0 else 0.0

        # Mathematical Cyclical Transformations
        month_sin = np.sin(2 * np.pi * month / 12.0)
        month_cos = np.cos(2 * np.pi * month / 12.0)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7.0)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7.0)

        # Cross Interaction Terms
        promo_and_weekend = promotion_flag * is_weekend
        promo_and_holiday = promotion_flag * holiday_flag
        instock_and_promo = stock_availability * promotion_flag

        # Construct DataFrame with exact column names required by your trained model
        input_data = pd.DataFrame([{
            'Price': price,
            'Competitor_Price': competitor_price,
            'Discount': discount_pct,
            'Discount_Percentage': discount_pct,
            'Discount_Amount': discount_amount,
            'Discounted_Price': discounted_price,
            'Price_Diff_vs_Competitor': price_diff_vs_competitor,
            'Price_Ratio_vs_Competitor': price_ratio_vs_competitor,
            'Marketing_Spend': marketing_spend,
            'Marketing_Spend_per_Unit_Price': marketing_spend_per_unit_price,
            'Category': category,
            'Product_ID': product_id,
            'Product_Name': product_name,
            'Store_ID': store_id,
            'Store_Location': store_location,
            'Sales_Channel': sales_channel,
            'Customer_Segment': customer_segment,
            'Season': season,
            'Weather': weather,
            'Holiday_Name': holiday_name,
            'Has_Holiday_Name': has_holiday_name,
            'Year': year,
            'Month': month,
            'Day': day,
            'Day_of_Week': day_of_week,
            'DayOfYear': day_of_year,
            'WeekOfYear': week_of_year,
            'Month_sin': month_sin,
            'Month_cos': month_cos,
            'DOW_sin': dow_sin,
            'DOW_cos': dow_cos,
            'Promotion_Flag': promotion_flag,
            'Local_Event_Flag': local_event_flag,
            'Holiday_Flag': holiday_flag,
            'Is_Weekend': is_weekend,
            'Stock_Availability': stock_availability,
            'Economic_Indicator': economic_indicator,
            'Promo_and_Weekend': promo_and_weekend,
            'Promo_and_Holiday': promo_and_holiday,
            'InStock_and_Promo': instock_and_promo,
            'Units_Sold_Lag1': units_sold_lag1,
            'Units_Sold_Lag7': units_sold_lag7,
            'Units_Sold_RollMean_4': units_sold_rollmean_4,
            'Units_Sold_RollStd_4': units_sold_rollstd_4
        }])

        try:
            # Predict sales volume
            raw_pred = model.predict(input_data)[0]
            predicted_units = max(0, int(round(raw_pred)))
            expected_revenue = predicted_units * discounted_price

            # Display Output Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Units Sold", f"{predicted_units:,} units")
            col2.metric("Effective Unit Price", f"${discounted_price:.2f}")
            col3.metric("Projected Revenue", f"${expected_revenue:,.2f}")

            st.success("✅ Prediction executed successfully without column mismatch errors!")

            # Supply Chain Recommendation
            st.subheader("💡 Inventory Allocation")
            safety_stock = int(predicted_units * 0.15)
            st.info(f"""
            - **Target Inventory Stock:** {predicted_units + safety_stock:,} units 
            - **Safety Buffer (15%):** {safety_stock:,} units
            """)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
