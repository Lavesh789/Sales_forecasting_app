import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Sales Forecasting & Inventory System",
    page_icon="📈",
    layout="wide"
)

# Title & Description
st.title("📈 Sales Forecasting & Business Insights Dashboard")
st.markdown("""
This interactive web interface allows you to input store metrics, promotional data, and seasonal features 
to generate real-time predicted sales volume and expected revenue.
""")

# Load Trained Model Function
@st.cache_resource
def load_trained_model():
    try:
        model = joblib.load('sales_forecast_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model file 'sales_forecast_model.pkl': {e}")
        return None

model = load_trained_model()

# Sidebar: Input Features
st.sidebar.header("📊 Input Features")

# Feature Inputs
price = st.sidebar.number_input("Unit Price (₹)", min_value=1.0, max_value=1000.0, value=25.0, step=0.5)
marketing_spend = st.sidebar.number_input("Marketing Spend (₹)", min_value=0.0, max_value=50000.0, value=500.0, step=50.0)
competitor_price = st.sidebar.number_input("Competitor Price (₹)", min_value=1.0, max_value=1000.0, value=26.5, step=0.5)
discount = st.sidebar.slider("Discount Percentage (%)", min_value=0, max_value=50, value=10)

st.sidebar.subheader("Contextual Details")
season = st.sidebar.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
store_location = st.sidebar.selectbox("Store Location", ["Urban", "Suburban", "Rural"])
sales_channel = st.sidebar.selectbox("Sales Channel", ["In-Store", "Online"])
customer_segment = st.sidebar.selectbox("Customer Segment", ["Standard", "Premium", "VIP"])
is_weekend = st.sidebar.radio("Weekend Sale?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Main Interface: Run Prediction
st.subheader("🤖 Predict Sales Volume & Revenue")

if st.button("Generate Forecast", type="primary"):
    if model is not None:
        # Build DataFrame matching training format
        input_data = pd.DataFrame([{
            'Price': price,
            'Marketing_Spend': marketing_spend,
            'Competitor_Price': competitor_price,
            'Discount': discount,
            'Season': season,
            'Store_Location': store_location,
            'Sales_Channel': sales_channel,
            'Customer_Segment': customer_segment,
            'Is_Weekend': is_weekend
        }])

        # Note: If your model requires specific encoding, pipeline transformations,
        # ensure preprocessors are included in the saved 'sales_model.pkl' pipeline.
        
        try:
            # Prediction
            predicted_units = model.predict(input_data)[0]
            predicted_units = max(0, int(round(predicted_units)))
            
            # Metric Calculation
            effective_price = price * (1 - (discount / 100))
            expected_revenue = predicted_units * effective_price
            
            # Display Key Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Units Sold", f"{predicted_units:,} units")
            col2.metric("Effective Unit Price", f"₹{effective_price:.2f}")
            col3.metric("Estimated Revenue", f"₹{expected_revenue:,.2f}")

            # Business Insight Output
            st.success("✅ Prediction generated successfully!")
            
            # Inventory Recommendation
            st.subheader("💡 Supply Chain & Inventory Recommendations")
            safety_stock = int(predicted_units * 0.15)
            st.info(f"""
            - Target Inventory Allocation: {predicted_units + safety_stock:,} units (Includes 15% safety stock buffer: {safety_stock} units).
            - Promotional ROI: Marketing spend of ₹{marketing_spend:,.2f} yields an estimated return of ₹{expected_revenue:,.2f}.
            """)

            # Simple Visualization
            st.subheader("📊 Price vs Revenue Simulation")
            sim_prices = np.linspace(price * 0.5, price * 1.5, 10)
            sim_revenues = [p * (1 - (discount / 100)) * predicted_units for p in sim_prices]
            
            chart_data = pd.DataFrame({
                "Simulated Price (₹)": sim_prices,
                "Projected Revenue (₹)": sim_revenues
            })
            st.line_chart(chart_data.set_index("Simulated Price (₹)"))

        except Exception as e:
            st.error(f"Prediction failed. Ensure feature columns match model inputs. Error detail: {e}")
    else:
        st.warning("Model file `sales_forecastt_model.pkl` not loaded correctly.")

# Footer
st.markdown("---")
st.caption("Sales Forecasting Application | Powered by Streamlit & Scikit-Learn / XGBoost")
