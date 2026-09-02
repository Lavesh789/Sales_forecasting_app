import joblib
import pandas as pd
import streamlit as st

# Set page title
st.set_page_config(page_title="Sales Prediction App", layout="wide")
st.title("Sales Demand Prediction")


# Load trained pipeline from disk
@st.cache_resource
def load_pipeline():
    # Replace 'model.pkl' with the actual path to your saved .joblib or .pkl file
    return joblib.load("sales_forecast_model.pkl")


try:
    pipeline = load_pipeline()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Form inputs for key features
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        product_id = st.text_input("Product ID", "P001")
        category = st.selectbox("Category", ["Electronics", "Appliances", "Apparel"])
        price = st.number_input("Price", value=1500.0)
        discount = st.number_input("Discount %", value=10.0)

    with col2:
        store_id = st.text_input("Store ID", "S01")
        day_of_week = st.selectbox(
            "Day of Week",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
        )
        season = st.selectbox("Season", ["Winter", "Summer", "Monsoon", "Spring"])
        channel = st.selectbox("Sales Channel", ["Online", "In-Store"])

    with col3:
        promo_flag = st.selectbox("Promotion Active", [1, 0])
        stock_availability = st.selectbox("In Stock", [1, 0])
        competitor_price = st.number_input("Competitor Price", value=1600.0)
        marketing_spend = st.number_input("Marketing Spend", value=500.0)

    submit = st.form_submit_button("Predict Sales")

if submit:
    # Prepare input payload matching expected model columns
    input_data = pd.DataFrame(
        [
            {
                "Product_ID": product_id,
                "Product_Name": "Sample Product",
                "Category": category,
                "Store_ID": store_id,
                "Store_Location": "Bengaluru",
                "Price": price,
                "Discount_Percentage": discount,
                "Promotion_Flag": promo_flag,
                "Stock_Availability": stock_availability,
                "Day_of_Week": day_of_week,
                "Month": "January",
                "Quarter": "Q1",
                "Holiday_Flag": 0,
                "Is_Weekend": 0,
                "Season": season,
                "Weather": "Clear",
                "Local_Event_Flag": 0,
                "Competitor_Price": competitor_price,
                "Economic_Indicator": 1.0,
                "Sales_Channel": channel,
                "Customer_Segment": "Retail",
                "Marketing_Spend": marketing_spend,
                "Year": 2026,
                "Day": 15,
                "DayOfYear": 15,
                "WeekOfYear": 3,
                "Month_sin": 0.5,
                "Month_cos": 0.86,
                "DOW_sin": 0.0,
                "DOW_cos": 1.0,
                "Price_Diff_vs_Competitor": price - competitor_price,
                "Price_Ratio_vs_Competitor": price / competitor_price
                if competitor_price
                else 1.0,
                "Discounted_Price": price * (1 - discount / 100),
                "Discount_Amount": price * (discount / 100),
                "Marketing_Spend_per_Unit_Price": marketing_spend / price
                if price
                else 0,
                "Promo_and_Weekend": 0,
                "Promo_and_Holiday": 0,
                "InStock_and_Promo": stock_availability * promo_flag,
                "Units_Sold_Lag1": 10.0,
                "Units_Sold_Lag7": 12.0,
                "Units_Sold_RollMean_4": 11.0,
                "Units_Sold_RollStd_4": 1.0,
                "Has_Holiday_Name": 0,
            }
        ]
    )

    prediction = pipeline.predict(input_data)[0]
    st.metric("Predicted Units Sold", f"{prediction:.2f}")
