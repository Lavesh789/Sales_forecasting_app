import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting & Inventory System")
st.markdown("Input dataset feature parameters to generate sales volume and expected revenue predictions.")

# Model Loader
@st.cache_resource
def load_trained_model():
    model_files = ['sales_forecast_model.pkl']
    for file in model_files:
        try:
            return joblib.load(file)
        except Exception:
            continue
    st.error("Unable to load model file. Ensure 'sales_forecast_model.pkl' is uploaded to your GitHub root folder.")
    return None

model = load_trained_model()

# --- SIDEBAR INPUTS (Exact Dataset Features) ---
st.sidebar.header("📊 Dataset Input Features")

# Pricing & Promotional Features
price = st.sidebar.number_input("Price (₹)", min_value=1.0, max_value=2000.0, value=25.0, step=0.5)
competitor_price = st.sidebar.number_input("Competitor_Price (₹)", min_value=1.0, max_value=2000.0, value=26.5, step=0.5)
discount = st.sidebar.slider("Discount (%)", min_value=0, max_value=100, value=10)
marketing_spend = st.sidebar.number_input("Marketing_Spend (₹)", min_value=0.0, max_value=50000.0, value=500.0, step=50.0)

st.sidebar.markdown("---")
# Categorical & Entity Features
category = st.sidebar.selectbox("Category", ["Electronics", "Clothing", "Home & Kitchen", "Groceries", "Beauty"])
product_name = st.sidebar.text_input("Product_Name", value="Product_A")
store_location = st.sidebar.selectbox("Store_Location", ["Urban", "Suburban", "Rural"])
sales_channel = st.sidebar.selectbox("Sales_Channel", ["In-Store", "Online"])
customer_segment = st.sidebar.selectbox("Customer_Segment", ["Standard", "Premium", "VIP"])
weather = st.sidebar.selectbox("Weather", ["Sunny", "Rainy", "Snowy", "Cloudy", "Overcast"])
holiday_name = st.sidebar.selectbox("Holiday_Name", ["None", "New Year", "Christmas", "Thanksgiving", "Labor Day"])

st.sidebar.markdown("---")
# Date & Time Component Features
year = st.sidebar.number_input("Year", min_value=2020, max_value=2030, value=2025)
quarter = st.sidebar.slider("Quarter", 1, 4, 2)
month = st.sidebar.slider("Month", 1, 12, 6)
day = st.sidebar.slider("Day", 1, 31, 15)
day_of_week = st.sidebar.slider("Day_of_Week (0=Mon, 6=Sun)", 0, 6, 2)

st.sidebar.markdown("---")
# Binary & Environmental Flags
is_weekend = st.sidebar.radio("Is_Weekend", [0, 1], index=1 if day_of_week in [5, 6] else 0)
holiday_flag = st.sidebar.radio("Holiday_Flag", [0, 1], index=1 if holiday_name != "None" else 0)
local_event_flag = st.sidebar.radio("Local_Event_Flag", [0, 1], index=0)
stock_availability = st.sidebar.radio("Stock_Availability", [0, 1], index=1)
economic_indicator = st.sidebar.number_input("Economic_Indicator", value=1.0, step=0.05)

# --- INFERENCE EXECUTION ---
st.subheader("🤖 Predict Sales Volume & Revenue")

if st.button("Generate Forecast", type="primary"):
    if model is not None:
        # Build exact input DataFrame matching dataset column names
        input_data = pd.DataFrame([{
            'Price': price,
            'Competitor_Price': competitor_price,
            'Discount': discount,
            'Marketing_Spend': marketing_spend,
            'Category': category,
            'Product_Name': product_name,
            'Store_Location': store_location,
            'Sales_Channel': sales_channel,
            'Customer_Segment': customer_segment,
            'Weather': weather,
            'Holiday_Name': holiday_name,
            'Year': year,
            'Quarter': quarter,
            'Month': month,
            'Day': day,
            'Day_of_Week': day_of_week,
            'Is_Weekend': is_weekend,
            'Holiday_Flag': holiday_flag,
            'Local_Event_Flag': local_event_flag,
            'Stock_Availability': stock_availability,
            'Economic_Indicator': economic_indicator
        }])

        try:
            # Predict sales volume
            raw_pred = model.predict(input_data)[0]
            predicted_units = max(0, int(round(raw_pred)))
            
            effective_price = price * (1 - (discount / 100.0))
            expected_revenue = predicted_units * effective_price

            # Display Output Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Units Sold", f"{predicted_units:,} units")
            col2.metric("Effective Unit Price", f"₹{effective_price:.2f}")
            col3.metric("Projected Revenue", f"₹{expected_revenue:,.2f}")

            st.success("✅ Prediction executed successfully using exact dataset features!")

            # Inventory Recommendation
            st.subheader("💡 Supply Chain Allocation")
            safety_stock = int(predicted_units * 0.15)
            st.info(f"""
            - **Target Inventory Stock:** {predicted_units + safety_stock:,} units 
            - **Safety Stock Buffer (15%):** {safety_stock:,} units
            """)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
