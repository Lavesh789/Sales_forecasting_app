import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sales Demand & Forecasting", layout="wide"
)
st.title("📈 Sales Demand & Forecasting Dashboard")


@st.cache_resource
def load_pipeline():
    # Make sure this points to your trained model file
    return joblib.load("sales_forecast_model.pkl")


try:
    pipeline = load_pipeline()
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model file: {e}")
    st.stop()

# --- Form Inputs ---
st.subheader("Input Parameters")
col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

with col1:
    product_id = st.selectbox("Product ID", ["P001","P002","P003","P004","P005","P006","P007","P008"])
    product_name = st.selectbox("Product Name", ["Bluetooth Speaker", "LED Desk Lamp", "Running Shoes", "Notebook Pack",
 "Wireless Mouse", "Yoga Mat", "Coffee Maker", "Office Chair"])
    category = st.selectbox(
        "Category", ["Electronics", "Appliances", "Apparel"]
    )
    store_id = st.selectbox("Store ID", ["S01","S02","S03","S04"])
    store_location = st.selectbox("Store Location", ["Bengaluru","Mumbai","Delhi","Chennai"])
    sales_channel = st.selectbox("Sales Channel", ["Online", "In-Store"])
    customer_segment = st.selectbox(
        "Customer Segment", ["Retail", "Corporate", "WholSale"]
    )

with col2:
    price = st.number_input("Price (₹)", value=1500.0, min_value=0.0)
    discount_pct = st.number_input(
        "Discount (%)", value=10.0, min_value=0.0, max_value=100.0
    )
    competitor_price = st.number_input(
        "Competitor Price (₹)", value=1600.0, min_value=0.0
    )
    marketing_spend = st.number_input(
        "Marketing Spend (₹)", value=500.0, min_value=0.0
    )
    economic_indicator = st.number_input("Economic Indicator", value=1.0)
    promo_flag = st.selectbox("Promotion Active", [1, 0])
    stock_availability = st.selectbox("In Stock", [1, 0])

with col3:
    forecast_date = st.date_input("Forecast Date")
    season = st.selectbox(
        "Season", ["Winter", "Summer", "Monsoon", "Spring"]
    )
    weather = st.selectbox("Weather", ["Clear", "Rainy", "Cloudy", "Snowy"])
    holiday_flag = st.selectbox("Holiday Flag", [0, 1])
    local_event_flag = st.selectbox("Local Event Flag", [0, 1])
    units_sold_lag1 = st.number_input("Sales Lag 1 (Yesterday)", value=12.0)
    units_sold_lag7 = st.number_input("Sales Lag 7 (Last Week)", value=15.0)

# --- Process Inputs & Predict ---
if st.button("Generate Forecast", type="primary"):
    # Date Calculations
    dt = pd.to_datetime(forecast_date)
    year = dt.year
    month = dt.month
    day = dt.day
    day_of_week = dt.day_name()
    day_of_year = dt.dayofyear
    week_of_year = dt.isocalendar().week
    quarter = f"Q{dt.quarter}"
    is_weekend = 1 if dt.weekday() >= 5 else 0

    # Cyclic Encodings
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    dow_num = dt.weekday()
    dow_sin = np.sin(2 * np.pi * dow_num / 7.0)
    dow_cos = np.cos(2 * np.pi * dow_num / 7.0)

    # Price / Financial Calculations
    discount_amount = price * (discount_pct / 100.0)
    discounted_price = price - discount_amount
    price_diff = price - competitor_price
    price_ratio = (
        price / competitor_price if competitor_price > 0 else 1.0
    )
    mkt_spend_per_price = marketing_spend / price if price > 0 else 0.0

    # Interaction & Lag Features
    promo_and_weekend = promo_flag * is_weekend
    promo_and_holiday = promo_flag * holiday_flag
    instock_and_promo = stock_availability * promo_flag
    units_sold_rollmean_4 = (units_sold_lag1 + units_sold_lag7) / 2.0
    units_sold_rollstd_4 = abs(units_sold_lag1 - units_sold_lag7) / 2.0

    # Construct DataFrame with ALL 43 Expected Columns
    input_df = pd.DataFrame(
        [
            {
                "Product_ID": product_id,
                "Product_Name": product_name,
                "Category": category,
                "Store_ID": store_id,
                "Store_Location": store_location,
                "Price": price,
                "Discount_Percentage": discount_pct,
                "Promotion_Flag": promo_flag,
                "Stock_Availability": stock_availability,
                "Day_of_Week": day_of_week,
                "Month": month,
                "Quarter": quarter,
                "Holiday_Flag": holiday_flag,
                "Is_Weekend": is_weekend,
                "Season": season,
                "Weather": weather,
                "Local_Event_Flag": local_event_flag,
                "Competitor_Price": competitor_price,
                "Economic_Indicator": economic_indicator,
                "Sales_Channel": sales_channel,
                "Customer_Segment": customer_segment,
                "Marketing_Spend": marketing_spend,
                "Year": year,
                "Day": day,
                "DayOfYear": day_of_year,
                "WeekOfYear": week_of_year,
                "Month_sin": month_sin,
                "Month_cos": month_cos,
                "DOW_sin": dow_sin,
                "DOW_cos": dow_cos,
                "Price_Diff_vs_Competitor": price_diff,
                "Price_Ratio_vs_Competitor": price_ratio,
                "Discounted_Price": discounted_price,
                "Discount_Amount": discount_amount,
                "Marketing_Spend_per_Unit_Price": mkt_spend_per_price,
                "Promo_and_Weekend": promo_and_weekend,
                "Promo_and_Holiday": promo_and_holiday,
                "InStock_and_Promo": instock_and_promo,
                "Units_Sold_Lag1": units_sold_lag1,
                "Units_Sold_Lag7": units_sold_lag7,
                "Units_Sold_RollMean_4": units_sold_rollmean_4,
                "Units_Sold_RollStd_4": units_sold_rollstd_4,
                "Has_Holiday_Name": holiday_flag,
            }
        ]
    )

    try:
        prediction = pipeline.predict(input_df)[0]
        projected_revenue = prediction * discounted_price

        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Predicted Demand", f"{int(np.round(prediction))} Units")
        res_col2.metric("Projected Revenue", f"₹{projected_revenue:,.2f}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
