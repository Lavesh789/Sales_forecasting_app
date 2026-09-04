import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

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
    return pd.DataFrame({"Date": dates, "Revenue": sales})


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
daily_sales = (
    df.groupby("Date")["Revenue"].sum().reset_index().sort_values("Date")
)
daily_sales["Ordinal_Date"] = daily_sales["Date"].map(datetime.toordinal)

X = daily_sales[["Ordinal_Date"]]
y = daily_sales["Revenue"]

model = LinearRegression()
model.fit(X, y)

last_date = daily_sales["Date"].max()
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D"
)
future_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
future_preds = np.maximum(0, model.predict(future_ordinal))

# Monthly Resampling for Clean Visualization
hist_monthly = daily_sales.set_index("Date")["Revenue"].resample("ME").sum()
future_df = pd.DataFrame({"Date": future_dates, "Revenue": future_preds})
fut_monthly = future_df.set_index("Date")["Revenue"].resample("ME").sum()

# Aligning Graph Data
hist_x = [d.strftime("%Y-%m") for d in hist_monthly.index]
hist_y = [round(v, 2) for v in hist_monthly.values]

fut_x = [hist_x[-1]] + [d.strftime("%Y-%m") for d in fut_monthly.index]
fut_y = [hist_y[-1]] + [round(v, 2) for v in fut_monthly.values]

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
    title="Historical vs Predicted Revenue",
    xaxis_title="Date",
    yaxis_title="Revenue",
)

st.plotly_chart(fig, use_container_width=True)
