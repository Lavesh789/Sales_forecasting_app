import warnings
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# 1. Page Configuration
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.title("📈 Sales Forecasting & Evaluation Dashboard")

DATASET_PATH = "Sales_Forcasting_Dataset.xlsx"


# 2. WMAPE Metric Calculation
def calculate_wmape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.clip(np.asarray(y_pred), 0, None)
    sum_actuals = np.sum(np.abs(y_true))
    if sum_actuals == 0:
        return 0.0
    return (np.sum(np.abs(y_true - y_pred)) / sum_actuals) * 100.0


# 3. Load & Process Dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_excel(DATASET_PATH)
    except Exception:
        # Fallback synthetic generator if file is not found
        dates = pd.date_range(start="2024-01-01", end="2026-08-31", freq="D")
        np.random.seed(42)
        units = np.sin(np.linspace(0, 20, len(dates))) * 50 + np.random.normal(
            200, 30, len(dates)
        )
        df = pd.DataFrame({"Date": dates, "Units_Sold": np.maximum(1, units)})

    df["Date"] = pd.to_datetime(df["Date"])

    # Aggregate to daily timeline
    daily = (
        df.groupby("Date")["Units_Sold"].sum().asfreq("D", fill_value=0).to_frame()
    )
    daily.index.name = "Date"

    # Robust Numerical Time-Series Features
    daily["DayOfWeek"] = daily.index.dayofweek
    daily["DayOfMonth"] = daily.index.day
    daily["Month"] = daily.index.month
    daily["Quarter"] = daily.index.quarter
    daily["Year"] = daily.index.year
    daily["Is_Weekend"] = daily["DayOfWeek"].apply(
        lambda x: 1 if x in [5, 6] else 0
    )

    # Lag & Rolling Features
    daily["Lag_1"] = daily["Units_Sold"].shift(1)
    daily["Lag_7"] = daily["Units_Sold"].shift(7)
    daily["Rolling_Mean_7"] = daily["Units_Sold"].shift(1).rolling(7).mean()

    return daily.bfill().ffill()


daily_df = load_data()

# 4. Split and Train Random Forest Model
X = daily_df.drop(columns=["Units_Sold"])
y = daily_df["Units_Sold"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300, max_depth=15, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# Predictions & Metrics
test_preds = model.predict(X_test)
wmape_val = calculate_wmape(y_test.values, test_preds)
accuracy = max(0.0, 100.0 - wmape_val)
mae = np.mean(np.abs(y_test.values - test_preds))

# 5. UI Controls & Top Metrics
st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    min_value=7,
    max_value=90,
    value=90,
    step=1,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🏆 Model", "Random Forest")
col2.metric("🎯 Accuracy (WMAPE)", f"{accuracy:.2f}%")
col3.metric("📉 WMAPE Error", f"{wmape_val:.2f}%")
col4.metric("📊 MAE Score", f"{mae:.2f}")

st.markdown("---")

# 6. Historical Evaluation Chart
st.subheader("📊 Historical Evaluation: Actual vs Predictions")
comp_df = (
    pd.DataFrame(
        {
            "Date": X_test.index,
            "Actual Sales": y_test.values,
            "Predicted Sales": test_preds,
        }
    )
    .sort_values("Date")
    .set_index("Date")
)

st.line_chart(comp_df)

# 7. Iterative Future Forecast Loop (90 Days)
st.subheader(f"🔮 Future {forecast_days}-Day Sales Forecast")

future_dates = pd.date_range(
    start=daily_df.index.max() + pd.Timedelta(days=1),
    periods=forecast_days,
    freq="D",
)

future_preds = []
last_data = daily_df.copy()

for date in future_dates:
    last_units = last_data["Units_Sold"]

    feat_row = pd.DataFrame(
        [
            {
                "DayOfWeek": date.dayofweek,
                "DayOfMonth": date.day,
                "Month": date.month,
                "Quarter": date.quarter,
                "Year": date.year,
                "Is_Weekend": 1 if date.dayofweek in [5, 6] else 0,
                "Lag_1": last_units.iloc[-1],
                "Lag_7": (
                    last_units.iloc[-7]
                    if len(last_data) >= 7
                    else last_units.iloc[-1]
                ),
                "Rolling_Mean_7": last_units.tail(7).mean(),
            }
        ],
        index=[date],
    )

    pred_val = max(0.0, float(model.predict(feat_row)[0]))
    future_preds.append(pred_val)

    # Append prediction to propagate lag features for the next step
    new_entry = feat_row.copy()
    new_entry["Units_Sold"] = pred_val
    last_data = pd.concat([last_data, new_entry])

forecast_df = pd.DataFrame(
    {"Forecasted Units": future_preds}, index=future_dates
)

st.line_chart(forecast_df)

with st.expander("📋 View Raw 90-Day Forecast Table"):
    st.dataframe(forecast_df)
