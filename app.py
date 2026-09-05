import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Weekly Sales Forecasting", layout="wide")
st.title("📈 Weekly Sales Forecasting Dashboard")

DATASET_PATH = "Sales_Forcasting_Dataset.xlsx"

@st.cache_data
def load_and_prepare_data():
    """Loads dataset and resamples to a weekly frequency."""
    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_excel(DATASET_PATH)
            df["Date"] = pd.to_datetime(df["Date"])
        except Exception:
            df = generate_mock_data()
    else:
        df = generate_mock_data()

    # Aggregate daily data to Weekly (Starting Mondays)
    weekly_df = df.set_index("Date").resample("W-MON")["Revenue"].sum().reset_index()
    return weekly_df

def generate_mock_data():
    dates = pd.date_range(start="2023-01-01", end="2026-08-31", freq="D")
    np.random.seed(42)
    sales = np.sin(np.linspace(0, 20, len(dates))) * 500 + np.random.normal(2000, 300, len(dates))
    return pd.DataFrame({"Date": dates, "Revenue": sales})

# Load Data
weekly_df = load_and_prepare_data()

# ---------------------------------------------------------------------------
# Sidebar Input Features
# ---------------------------------------------------------------------------
st.sidebar.header("Forecast Settings & Inputs")

forecast_weeks = st.sidebar.slider(
    "Forecast Horizon (Weeks):",
    min_value=4,
    max_value=52,
    value=12,
    step=1
)

n_estimators = st.sidebar.slider("Gradient Boosting Estimators:", 50, 300, 100, step=25)
learning_rate = st.sidebar.select_slider("Learning Rate:", options=[0.01, 0.05, 0.1, 0.2], value=0.1)
growth_adjustment = st.sidebar.slider("Simulated Weekly Demand Shift (%):", -20, 20, 0, step=1)

# ---------------------------------------------------------------------------
# Feature Engineering & Model Training (Gradient Boosting)
# ---------------------------------------------------------------------------
def create_features(data):
    df_feat = data.copy()
    df_feat["Week"] = df_feat["Date"].dt.isocalendar().week.astype(int)
    df_feat["Month"] = df_feat["Date"].dt.month
    df_feat["Year"] = df_feat["Date"].dt.year
    df_feat["DayOfYear"] = df_feat["Date"].dt.dayofyear
    df_feat["Lag_1"] = df_feat["Revenue"].shift(1).fillna(method="bfill")
    return df_feat

df_features = create_features(weekly_df)

X = df_features[["Week", "Month", "Year", "DayOfYear", "Lag_1"]]
y = df_features["Revenue"]

# Train Gradient Boosting Regressor
model = GradientBoostingRegressor(
    n_estimators=n_estimators, 
    learning_rate=learning_rate, 
    random_state=42
)
model.fit(X, y)

# ---------------------------------------------------------------------------
# Iterative Weekly Forecast Generation
# ---------------------------------------------------------------------------
last_date = weekly_df["Date"].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=forecast_weeks, freq="W-MON")

future_preds = []
last_revenue = weekly_df["Revenue"].iloc[-1]

for date in future_dates:
    week = int(date.isocalendar().week)
    month = date.month
    year = date.year
    day_of_year = date.dayofyear
    
    input_features = np.array([[week, month, year, day_of_year, last_revenue]])
    pred = model.predict(input_features)[0]
    
    # Apply user-defined demand shift adjustment
    pred = pred * (1 + (growth_adjustment / 100.0))
    pred = max(0, pred)  # Ensure non-negative predictions
    
    future_preds.append(pred)
    last_revenue = pred

future_df = pd.DataFrame({"Date": future_dates, "Revenue": future_preds})

# ---------------------------------------------------------------------------
# Interactive Trend Plotting (Plotly)
# ---------------------------------------------------------------------------
fig = go.Figure()

# Historical Line (Solid Blue)
fig.add_trace(go.Scatter(
    x=weekly_df["Date"],
    y=weekly_df["Revenue"],
    mode="lines+markers",
    name="Historical Weekly Sales",
    line=dict(color="#1f77b4", width=2.5),
    marker=dict(size=4)
))

# Connecting line bridging last historic point to first prediction point
connect_dates = [weekly_df["Date"].iloc[-1]] + list(future_df["Date"])
connect_values = [weekly_df["Revenue"].iloc[-1]] + list(future_df["Revenue"])

# Forecast Line (Dashed Red with Shaded Fill)
fig.add_trace(go.Scatter(
    x=connect_dates,
    y=connect_values,
    mode="lines+markers",
    name="Gradient Boosting Forecast",
    line=dict(color="#d62728", width=2.5, dash="dash"),
    marker=dict(size=5),
    fill="tozeroy",
    fillcolor="rgba(214, 39, 40, 0.08)"
))

fig.update_layout(
    title="Weekly Sales Trend: Past Performance vs Future Gradient Boosting Forecast",
    xaxis_title="Timeline (Weekly)",
    yaxis_title="Revenue",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# Key Metrics Overview
col1, col2, col3 = st.columns(3)
col1.metric("Historical Avg Weekly Revenue", f"₹{weekly_df['Revenue'].mean():,.2f}")
col2.metric("Projected Avg Weekly Revenue", f"₹{future_df['Revenue'].mean():,.2f}")
col3.metric("Total Projected Revenue", f"₹{future_df['Revenue'].sum():,.2f}")
