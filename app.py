import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
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
            # Fallback if Units_Sold is missing in Excel
            if "Units_Sold" not in df.columns:
                df["Units_Sold"] = (df["Revenue"] / 50).astype(int)
            return df
        except Exception:
            pass

    # Fallback dataset if Excel is missing
    dates = pd.date_range(start="2024-01-01", end="2026-08-31", freq="D")
    np.random.seed(42)
    units = np.sin(np.linspace(0, 20, len(dates))) * 50 + np.random.normal(
        200, 30, len(dates)
    )
    units = np.maximum(1, units)
    return pd.DataFrame({"Date": dates, "Units_Sold": units})


df = load_data()

# Controls (Weekly Horizon)
forecast_weeks = st.slider(
    "Select Forecast Horizon (Weeks):",
    min_value=4,
    max_value=52,
    value=12,
    step=1,
)

# Resample dataset to Weekly basis
weekly_sales = (
    df.set_index("Date")["Units_Sold"].resample("W-MON").sum().reset_index()
)

# Feature Extraction directly from Dataset Dates
weekly_sales["Week"] = weekly_sales["Date"].dt.isocalendar().week.astype(int)
weekly_sales["Month"] = weekly_sales["Date"].dt.month
weekly_sales["Year"] = weekly_sales["Date"].dt.year
weekly_sales["Lag_1"] = (
    weekly_sales["Units_Sold"].shift(1).bfill()
)  # Fixed deprecation crash

X = weekly_sales[["Week", "Month", "Year", "Lag_1"]]
y = weekly_sales["Units_Sold"]

# Model & Prediction using Gradient Boosting
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Iterative Future Prediction
last_date = weekly_sales["Date"].max()
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=7), periods=forecast_weeks, freq="W-MON"
)

future_preds = []
last_units = weekly_sales["Units_Sold"].iloc[-1]

for date in future_dates:
    week = int(date.isocalendar().week)
    month = date.month
    year = date.year

    input_feat = np.array([[week, month, year, last_units]])
    pred = model.predict(input_feat)[0]
    pred = max(0, pred)

    future_preds.append(pred)
    last_units = pred

future_df = pd.DataFrame({"Date": future_dates, "Units_Sold": future_preds})

# Aligning Graph Data (Weekly Timeline)
hist_x = [d.strftime("%Y-W%U") for d in weekly_sales["Date"]]
hist_y = [round(v, 2) for v in weekly_sales["Units_Sold"]]

fut_x = [hist_x[-1]] + [d.strftime("%Y-W%U") for d in future_df["Date"]]
fut_y = [hist_y[-1]] + [round(v, 2) for v in future_df["Units_Sold"]]

# Plotting Interactive Chart (Weekly vs Units Sold)
fig = go.Figure()

# Historical Line (Solid Blue)
fig.add_trace(
    go.Scatter(
        x=hist_x,
        y=hist_y,
        mode="lines+markers",
        name="Historical Units Sold",
        line=dict(color="#007bff", width=2),
    )
)

# Future Prediction Line (Dashed Red with Visual Distinction)
fig.add_trace(
    go.Scatter(
        x=fut_x,
        y=fut_y,
        mode="lines+markers",
        name="Future Prediction",
        line=dict(dash="dash", color="red", width=2),
    )
)

fig.update_layout(
    title="Weekly Units Sold: Historical vs Future Prediction",
    xaxis_title="Timeline (Weeks)",
    yaxis_title="Units Sold",
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)
