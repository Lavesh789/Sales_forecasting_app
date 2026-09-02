import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting & Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting & Future Prediction Dashboard")
st.markdown("""
Enter the store, product, and sales parameters below to generate accurate future sales volume predictions.
""")

# Load Trained Model Function
@st.cache_resource
def load_trained_model():
    model_files = ['sales_forecast_model.pkl']
    for file in model_files:
        try:
            return joblib.load(file)
        except Exception:
            continue
    st.error("Model file not found. Ensure 'sales_forecast_model.pkl' is uploaded in your GitHub repository root.")
    return None

model = load_trained_model()

# Sidebar: User Input Features
st.sidebar.header("📊 Input Features")

# 1. Product & Category Selection
product_name = st.sidebar.selectbox(
    "Product Name", 
    ["Product_A", "Product_B", "Product_C", "Product_D", "Product_E", "Standard_Product"]
)

category = st.sidebar.selectbox(
    "Category", 
    ["Electronics", "Clothing", "Home & Kitchen", "Groceries", "Beauty", "Sports"]
)

# 2. Store & Sales Channel Selection
store_location = st.sidebar.selectbox(
    "Store Location", 
    ["Urban", "Suburban", "Rural"]
)

sales_channel = st.sidebar.selectbox(
    "Sales Channel", 
    ["In-Store", "Online"]
)

customer_segment = st.sidebar.selectbox(
    "Customer Segment", 
    ["Standard", "Premium", "VIP"]
)

st.sidebar.markdown("---")

# 3. Pricing & Volume Inputs
price = st.sidebar.number_input(
    "Unit Price (₹)", 
    min_value=1.0, 
    max_value=2000.0, 
    value=25.0, 
    step=0.5
)

discount_percentage = st.sidebar.slider(
    "Discount Percentage (%)", 
    min_value=0, 
    max_value=100, 
    value=10
)

# 4. Temporal Features
day_of_week = st.sidebar.selectbox(
    "Day of Week", 
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x]
)

month = st.sidebar.selectbox(
    "Month", 
    options=list(range(1, 13)),
    format_func=lambda x: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"][x-1]
)

st.sidebar.markdown("---")
# Historical Baseline Context (Units Sold)
historical_units_sold = st.sidebar.number_input(
    "Baseline / Past Units Sold", 
    min_value=0, 
    value=150, 
    step=5
)

# Main Prediction Section
st.subheader("🤖 Future Sales Prediction & Business Metrics")

if st.button("Generate Future Forecast", type="primary"):
    if model is not None:
        # Derived Feature Calculations
        discount_amount = price * (discount_percentage / 100.0)
        discounted_price = price - discount_amount
        competitor_price = price * 1.05  # Estimated benchmark competitor ratio
        price_ratio = price / competitor_price if competitor_price > 0 else 1.0
        marketing_spend = 500.0  # Standard default marketing baseline
        mkt_per_price = marketing_spend / price if price > 0 else 0.0

        # Time-based Feature Calculations
        quarter = (month - 1) // 3 + 1
        day_of_year = (month - 1) * 30 + 15  # Mid-month approximation
        is_weekend = 1 if day_of_week in [5, 6] else 0

        # Cyclical Features
        month_sin = np.sin(2 * np.pi * month / 12.0)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7.0)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7.0)

        # Operational Defaults
        promotion_flag = 1 if discount_percentage > 0 else 0
        local_event_flag = 0
        holiday_flag = 0
        has_holiday_name = 0
        stock_avail = 1
        economic_indicator = 1.0
        promo_and_weekend = promotion_flag * is_weekend
        instock_and_promo = stock_avail * promotion_flag

        # Construct DataFrame matching model pipeline schema
        input_data = pd.DataFrame([{
            'Price': price,
            'Competitor_Price': competitor_price,
            'Discounted_Price': discounted_price,
            'Discount_Amount': discount_amount,
            'Discount_Percentage': discount_percentage,
            'Price_Ratio_vs_Competitor': price_ratio,
            'Marketing_Spend': marketing_spend,
            'Marketing_Spend_per_Unit_Price': mkt_per_price,
            'Weather': 'Sunny',
            'Category': category,
            'Product_Name': product_name,
            'Customer_Segment': customer_segment,
            'Store_Location': store_location,
            'Sales_Channel': sales_channel,
            'Month': month,
            'Quarter': quarter,
            'Day_of_Week': day_of_week,
            'DayOfYear': day_of_year,
            'Month_sin': month_sin,
            'DOW_sin': dow_sin,
            'DOW_cos': dow_cos,
            'Promotion_Flag': promotion_flag,
            'Local_Event_Flag': local_event_flag,
            'Holiday_Flag': holiday_flag,
            'Has_Holiday_Name': has_holiday_name,
            'Is_Weekend': is_weekend,
            'Stock_Avail': stock_avail,
            'Economic_Indicator': economic_indicator,
            'Promo_and_Weekend': promo_and_weekend,
            'InStock_and_Promo': instock_and_promo,
            'Units_Sold_Lag1': historical_units_sold,
            'Units_Sold_Lag7': historical_units_sold,
            'Units_Sold_RollStd_4': 12.5
        }])

        try:
            # Predict Future Sales Volume
            prediction = model.predict(input_data)[0]
            predicted_units = max(0, int(round(prediction)))
            estimated_revenue = predicted_units * discounted_price

            # Display Key Business Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Future Units Sold", f"{predicted_units:,} units")
            col2.metric("Discounted Unit Price", f"₹{discounted_price:.2f}")
            col3.metric("Projected Total Revenue", f"₹{estimated_revenue:,.2f}")

            st.success("✅ Forecast generated successfully without errors!")

            # Inventory Recommendation
            st.subheader("📦 Inventory Allocation Guidance")
            safety_stock = int(predicted_units * 0.15)
            st.info(f"""
            - **Target Stock Allocation:** {predicted_units + safety_stock:,} units
            - **Safety Buffer (15%):** {safety_stock:,} units
            - **Sales Channel:** {sales_channel} | **Location:** {store_location}
            """)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
