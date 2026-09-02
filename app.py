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

st.title("📈 Sales Forecasting & Future Prediction Dashboard")

# Load Model
@st.cache_resource
def load_trained_model():
    for name in ['sales_forecast_model.pkl', 'sales_model.pkl']:
        try:
            return joblib.load(name)
        except Exception:
            continue
    return None

model = load_trained_model()

# Sidebar Feature Inputs
st.sidebar.header("📊 Input Features")

price = float(st.sidebar.number_input("Unit Price ($)", min_value=1.0, value=25.0, step=0.5))
discount_percentage = float(st.sidebar.slider("Discount Percentage (%)", 0, 100, 10))
competitor_price = float(st.sidebar.number_input("Competitor Price ($)", min_value=1.0, value=26.5, step=0.5))
marketing_spend = float(st.sidebar.number_input("Marketing Spend ($)", min_value=0.0, value=500.0, step=50.0))

month = int(st.sidebar.selectbox("Month", list(range(1, 13))))
day_of_week = int(st.sidebar.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(0, 7))))
historical_units_sold = float(st.sidebar.number_input("Past Units Sold", min_value=0, value=150))

st.subheader("🤖 Future Sales Prediction & Business Metrics")

if st.button("Generate Future Forecast", type="primary"):
    if model is None:
        st.error("Model file not found. Place 'sales_forecast_model.pkl' in the root directory.")
    else:
        # 1. Feature Calculations (Pure Numeric Defaults)
        discount_amount = float(price * (discount_percentage / 100.0))
        discounted_price = float(price - discount_amount)
        price_diff = float(price - competitor_price)
        price_ratio = float(price / competitor_price) if competitor_price > 0 else 1.0
        mkt_per_price = float(marketing_spend / price) if price > 0 else 0.0

        quarter = float((month - 1) // 3 + 1)
        day_of_year = float((month - 1) * 30 + 15)
        week_of_year = float(min(52, max(1, day_of_year // 7)))
        is_weekend = float(1.0 if day_of_week in [5, 6] else 0.0)

        month_sin = float(np.sin(2 * np.pi * month / 12.0))
        month_cos = float(np.cos(2 * np.pi * month / 12.0))
        dow_sin = float(np.sin(2 * np.pi * day_of_week / 7.0))
        dow_cos = float(np.cos(2 * np.pi * day_of_week / 7.0))

        promotion_flag = float(1.0 if discount_percentage > 0 else 0.0)
        
        # 2. Complete Numeric Map (Category fields encoded as float codes to prevent numpy string crash)
        numeric_feature_dict = {
            'Price': price,
            'Competitor_Price': competitor_price,
            'Discounted_Price': discounted_price,
            'Discount_Amount': discount_amount,
            'Discount_Percentage': discount_percentage,
            'Price_Ratio_vs_Competitor': price_ratio,
            'Price_Diff_vs_Competitor': price_diff,
            'Marketing_Spend': marketing_spend,
            'Marketing_Spend_per_Unit_Price': mkt_per_price,
            'Weather': 0.0,
            'Season': 0.0,
            'Category': 0.0,
            'Product_Name': 0.0,
            'Product_ID': 0.0,
            'Customer_Segment': 0.0,
            'Store_Location': 0.0,
            'Store_ID': 0.0,
            'Sales_Channel': 0.0,
            'Year': 2026.0,
            'Month': float(month),
            'Quarter': quarter,
            'Day': 15.0,
            'Day_of_Week': float(day_of_week),
            'DayOfYear': day_of_year,
            'WeekOfYear': week_of_year,
            'Month_sin': month_sin,
            'Month_cos': month_cos,
            'DOW_sin': dow_sin,
            'DOW_cos': dow_cos,
            'Promotion_Flag': promotion_flag,
            'Local_Event_Flag': 0.0,
            'Holiday_Flag': 0.0,
            'Has_Holiday_Name': 0.0,
            'Is_Weekend': is_weekend,
            'Stock_Avail': 1.0,
            'Stock_Availability': 1.0,
            'Economic_Indicator': 1.0,
            'Promo_and_Weekend': promotion_flag * is_weekend,
            'Promo_and_Holiday': 0.0,
            'InStock_and_Promo': promotion_flag,
            'Units_Sold_Lag1': historical_units_sold,
            'Units_Sold_Lag7': historical_units_sold,
            'Units_Sold_RollStd_4': 12.5,
            'Units_Sold_RollMean_4': historical_units_sold
        }

        df_input = pd.DataFrame([numeric_feature_dict])

        # Align columns strictly to feature order inside the model pipeline
        expected_cols = None
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
        elif hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    expected_cols = list(step.feature_names_in_)
                    break

        if expected_cols:
            for col in expected_cols:
                if col not in df_input.columns:
                    df_input[col] = 0.0
            df_input = df_input[expected_cols]

        # Convert everything explicitly to float64
        df_input = df_input.astype(np.float64)

        # 3. Safe Dual Prediction Execution
        try:
            # Attempt 1: DataFrame Prediction
            prediction = model.predict(df_input)[0]
        except Exception:
            # Attempt 2: Array-based fallback (for models trained on raw NumPy matrices)
            prediction = model.predict(df_input.values)[0]

        predicted_units = max(0, int(round(float(prediction))))
        estimated_revenue = predicted_units * discounted_price

        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Units Sold", f"{predicted_units:,} units")
        col2.metric("Discounted Price", f"${discounted_price:.2f}")
        col3.metric("Projected Revenue", f"${estimated_revenue:,.2f}")

        st.success("✅ Forecast generated successfully!")
