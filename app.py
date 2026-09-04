import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Sales Forecast", layout="centered")
st.title("📈 Sales Forecasting Dashboard")

DATASET_PATH = "Sales_Forcasting_Dataset.xlsx"


@st.cache_data
def load_data():
    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_excel(DATASET_PATH)
            df["Date"] = pd.to_datetime(df["Date"])
            return df
        except Exception:
            pass

    # Fallback dataset if Excel is missing
    dates = pd.date_range(start="2024-01-01", end="2026-08-31", freq="D")
    np.random.seed(42)
    sales = np.sin(np.linspace(0, 20, len(dates))) * 500 + np.random.normal(
        2000, 300, len(dates)
    )
    return pd.DataFrame({"Date": dates, "Units_Sold": sales})


df = load_data()

# Controls
forecast_days = st.slider(
    "Select Forecast Horizon (Days):",
    min_value=7,
    max_value=365,
    value=90,
    step=7,
)

# Model & Prediction
Weekly_sales = (
    df.groupby("Date")["Units_Sold"].sum().reset_index().sort_values("Date")
)
Weekly_sales["Ordinal_Date"] = Weekly_sales["Date"].map(datetime.toordinal)

X = Weekly_sales[["Week"]]
y = Weekly_sales["Units_Sold"]

model = GradientBoostingRegressor()
model.fit(X, y)

last_Week = Weekly_sales["Date"].max()
future_Week = pd.date_range(
    start=last_Week + pd.Timedelta(days=7), periods=forecast_days, freq="D"
)
future_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
future_preds = np.maximum(0, model.predict(future_ordinal))

# Weekly Resampling for Clean Visualization
hist_Weekly = Weekly_sales.set_index("Date")["Units_Sold"].resample("ME").sum()
future_df = pd.DataFrame({"Date": future_dates, "Units_Sold": future_preds})
fut_Weekly = future_df.set_index("Date")["Units_Sold"].resample("ME").sum()

# Aligning Graph Data
hist_x = [d.strftime("%Y-%w") for d in hist_Weekly.index]
hist_y = [round(v, 2) for v in hist_Weekly.values]

fut_x = [hist_x[-1]] + [d.strftime("%Y-%w") for d in fut_Weekly.index]
fut_y = [hist_y[-1]] + [round(v, 2) for v in fut_Weekly.values]

# Plotting Interactive Chart
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=hist_x, y=hist_y, mode="lines", name="Historical Sales")
)
fig.add_trace(
    go.Scatter(
        x=fut_x,
        y=fut_y,
        mode="lines",
        name="Future Prediction",
        line=dict(dash="dash", color="red"),
    )
)

fig.update_layout(
    title="Historical vs Predicted Units_Sold",
    xaxis_title="Week",
    yaxis_title="Units_Sold",
)

st.plotly_chart(fig, use_container_width=True)
