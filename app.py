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
st.markdown("Enter store, product, and sales parameters below to generate accurate future sales volume predictions.")

# Load Trained Model
@st.cache_resource
def load_trained_model():
    model_files = ['sales_forecast_model.pkl', 'sales_model.pkl']
    for file in model_files:
        try:
            return joblib.load(file)
        except Exception:
            continue
    st.error("Model file not found. Ensure 'sales_forecast_model.pkl' is uploaded to your GitHub repository root.")
    return None

model = load_trained_model()

# Sidebar Feature Inputs
st.sidebar.header("📊 Input Features")

# 1. Product & Category
product_name = st.sidebar.selectbox("Product Name", ["Product_A", "Product_B", "Product_C", "Product_D", "Product_E", "Standard_Product"])
product_id = st.sidebar.text_input("Product ID", value="PROD_001")
category = st.sidebar.selectbox("Category", ["Electronics", "Clothing", "Home & Kitchen", "Groceries", "Beauty", "Sports"])

# 2. Store & Customer Segment
store_location = st.sidebar.selectbox("Store Location", ["Urban", "Suburban", "Rural"])
store_id = st.sidebar.text_input("Store ID", value="STORE_101")
sales_channel = st.sidebar.selectbox("Sales Channel", ["In-Store", "Online"])
customer_segment = st.sidebar.selectbox("Customer Segment", ["Standard", "Premium", "VIP"])

st.sidebar.markdown("---")

# 3. Pricing & Discounts
price = st.sidebar.number_input("Unit Price ($)", min_value=1.0, max_value=2000.0, value=25.0, step=0.5)
competitor_price = st.sidebar.number_input("Competitor Price ($)", min_value=1.0, max_value=2000.0, value=26.5, step=0.5)
discount_percentage = st.sidebar.slider("Discount Percentage (%)", min_value=0, max_value=100, value=10)
marketing_spend = st.sidebar.number_input("Marketing Spend ($)", min_value=0.0, max_value=50000.0, value=500.0, step=50.0)

st.sidebar.markdown("---")

# 4. Temporal Features
year = st.sidebar.selectbox("Year", [2024, 2025, 2026], index=2)
month = st.sidebar.selectbox(
    "Month", 
    options=list(range(1, 13)),
    format_func=lambda x: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"][x-1]
)
day = st.sidebar.slider("Day of Month", 1, 31, 15)
day_of_week = st.sidebar.selectbox(
    "Day of Week", 
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x]
)

st.sidebar.markdown("---")

# 5. Operational Status
season = st.sidebar.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
weather = st.sidebar.selectbox("Weather", ["Sunny", "Rainy", "Snowy", "Cloudy", "Overcast"])
stock_availability = st.sidebar.selectbox("Stock Availability", ["In Stock", "Out of Stock", "Low Stock"])
stock_avail = 1 if stock_availability == "In Stock" else 0

st.sidebar.markdown("---")

# Historical Volume Context
historical_units_sold = st.sidebar.number_input("Baseline / Past Units Sold", min_value=0, value=150, step=5)

# Forecast Generation
st.subheader("🤖 Future Sales Prediction & Business Metrics")

if st.button("Generate Future Forecast", type="primary"):
    if model is not None:
        # 1. Pricing Calculations
        discount_amount = price * (discount_percentage / 100.0)
        discounted_price = price - discount_amount
        price_diff = price - competitor_price
        price_ratio = price / competitor_price if competitor_price > 0 else 1.0
        mkt_per_price = marketing_spend / price if price > 0 else 0.0

        # 2. Date Calculations
        quarter = (month - 1) // 3 + 1
        day_of_year = (month - 1) * 30 + day
        week_of_year = min(52, max(1, day_of_year // 7))
        is_weekend = 1 if day_of_week in [5, 6] else 0

        # Cyclical Transformations
        month_sin = np.sin(2 * np.pi * month / 12.0)
        month_cos = np.cos(2 * np.pi * month / 12.0)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7.0)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7.0)

        # 3. Operational Flags & Interactions
        promotion_flag = 1 if discount_percentage > 0 else 0
        local_event_flag = 0
        holiday_flag = 0
        has_holiday_name = 0
        economic_indicator = 1.0

        promo_and_weekend = promotion_flag * is_weekend
        promo_and_holiday = promotion_flag * holiday_flag
        instock_and_promo = stock_avail * promotion_flag

        # 4. Construct Full Feature Map
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
            'Year': year,
            'Month': month,
            'Quarter': quarter,
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
            'Units_Sold_RollMean_4': float(historical_units_sold)
        }

        input_data = pd.DataFrame([raw_feature_dict])

        # 5. Automated Pipeline Schema Alignment
        # If the model metadata stores expected feature names, ensure exact column match & order
        try:
            expected_cols = None
            if hasattr(model, "feature_names_in_"):
                expected_cols = list(model.feature_names_in_)
            elif hasattr(model, "named_steps"):
                # Check first step or last step of Scikit-Learn pipeline
                for step in model.named_steps.values():
                    if hasattr(step, "feature_names_in_"):
                        expected_cols = list(step.feature_names_in_)
                        break

            if expected_cols:
                # Add any missing expected columns with default fallback
                for col in expected_cols:
                    if col not in input_data.columns:
                        input_data[col] = 0
                # Reorder columns to match the trained model sequence
                input_data = input_data[expected_cols]
        except Exception:
            pass

        try:
            # Predict Future Volume
            prediction = model.predict(input_data)[0]
            predicted_units = max(0, int(round(prediction)))
            estimated_revenue = predicted_units * discounted_price

            # Render Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Future Units Sold", f"{predicted_units:,} units")
            col2.metric("Discounted Unit Price", f"${discounted_price:.2f}")
            col3.metric("Projected Revenue", f"${estimated_revenue:,.2f}")

            st.success("✅ Forecast generated successfully!")

            # Inventory Recommendation
            st.subheader("📦 Supply Chain & Inventory Allocation")
            safety_stock = int(predicted_units * 0.15)
            st.info(f"""
            - **Target Inventory Stock:** {predicted_units + safety_stock:,} units
            - **Safety Buffer (15%):** {safety_stock:,} units
            - **Store:** {store_id} ({store_location}) | **Channel:** {sales_channel}
            """)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
