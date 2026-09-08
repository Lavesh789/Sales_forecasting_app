import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# Streamlit Page Setup
st.set_page_config(
    page_title="Sales Forecasting Dashboard", page_layout="wide"
)
st.title("📈 Daily Sales Forecasting & Model Evaluation")

MODEL_PATH = "rf_model.joblib"
DATASET_PATH = "Sales_Forcasting_Dataset.xlsx"


# ---------------------------------------------------------------------------
# 1. Helper Functions & Metrics
# ---------------------------------------------------------------------------
def calculate_wmape(y_true, y_pred):
    """Calculates Weighted Absolute Percentage Error (WMAPE)."""
    y_true = np.asarray(y_true)
    y_pred = np.clip(np.asarray(y_pred), 0, None)
    sum_actuals = np.sum(np.abs(y_true))
    if sum_actuals == 0:
        return 0.0
    return (np.sum(np.abs(y_true - y_pred)) / sum_actuals) * 100.0


@st.cache_resource
def load_saved_model():
    """Loads pre-trained model exported from salesforecasting_model.ipynb."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"⚠️ Could not load `{MODEL_PATH}`: {e}")
        st.stop()


@st.cache_data
def load_and_prep_data():
    """Loads dataset and prepares all features required by the trained model."""
    df = pd.read_excel(DATASET_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 1. Basic Date/Time Features
    df["Day_of_Week"] = df["Date"].dt.day_name()
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfMonth"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Year"] = df["Date"].dt.year
    df["Is_Weekend"] = df["DayOfWeek"].apply(
        lambda x: 1 if x in [5, 6] else 0
    )

    # 2. Season Mapping
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    if "Season" not in df.columns:
        df["Season"] = df["Month"].apply(get_season)

    # 3. Lag Features and Rolling Averages
    df["Lag_1"] = df["Units_Sold"].shift(1)
    df["Lag_7"] = df["Units_Sold"].shift(7)
    df["Rolling_Mean_7"] = df["Units_Sold"].shift(1).rolling(window=7).mean()

    # Clean missing values resulting from shifts/rolling windows
    df = df.bfill().ffill()
    return df


# Load model and dataset
model = load_saved_model()
df = load_and_prep_data()

# Identify features expected by the trained pipeline
if hasattr(model, "feature_names_in_"):
    expected_cols = list(model.feature_names_in_)
else:
    # Exclude Target and Identifier columns if feature_names_in_ isn't available
    exclude_cols = ["Units_Sold", "Date"]
    expected_cols = [c for c in df.columns if c not in exclude_cols]

# ---------------------------------------------------------------------------
# 2. Historical Evaluation (Matches Notebook Train/Test Split)
# ---------------------------------------------------------------------------
X = df[expected_cols]
y = df["Units_Sold"]

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
test_dates = df["Date"].iloc[split_idx:]

# Predictions on Test Data
test_preds = model.predict(X_test)

# WMAPE & Metrics Evaluation
wmape_val = calculate_wmape(y_test.values, test_preds)
accuracy = max(0.0, 100.0 - wmape_val)
mae = np.mean(np.abs(y_test.values - test_preds))

# ---------------------------------------------------------------------------
# 3. Streamlit Dashboard Layout
# ---------------------------------------------------------------------------
st.sidebar.header("Forecast Controls")
forecast_days = st.sidebar.slider(
    "Future Forecast Horizon (Days)",
    min_value=7,
    max_value=90,
    value=90,
    step=1,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🏆 Model", "Random Forest")
col2.metric("🎯 Accuracy", f"{accuracy:.2f}%")
col3.metric("📉 WMAPE Error", f"{wmape_val:.2f}%")
col4.metric("📊 MAE Score", f"{mae:.2f}")

st.markdown("---")

# Historical Actual vs Predicted Chart
st.subheader("📊 Historical Evaluation: Actual vs Model Predictions")
comp_df = pd.DataFrame(
    {"Actual Sales": y_test.values, "Model Predictions": test_preds},
    index=test_dates,
)
st.line_chart(comp_df)

# ---------------------------------------------------------------------------
# 4. Iterative 90-Day Future Forecast Loop
# ---------------------------------------------------------------------------
st.subheader(f"🔮 Future {forecast_days}-Day Sales Forecast")

last_date = df["Date"].max()
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=forecast_days,
    freq="D",
)

# Template row copied from the latest entry to keep categorical variables intact
latest_df = df.copy()
future_preds = []

for date in future_dates:
    # Copy latest available row as a template for non-time features
    next_row = latest_df.iloc[-1:].copy()
    next_row["Date"] = date

    # Update date-based features
    next_row["Day_of_Week"] = date.day_name()
    next_row["DayOfWeek"] = date.dayofweek
    next_row["DayOfMonth"] = date.day
    next_row["Month"] = date.month
    next_row["Quarter"] = date.quarter
    next_row["Year"] = date.year
    next_row["Is_Weekend"] = 1 if date.dayofweek in [5, 6] else 0

    # Update lag & rolling features
    last_units = latest_df["Units_Sold"]
    next_row["Lag_1"] = last_units.iloc[-1]
    next_row["Lag_7"] = (
        last_units.iloc[-7] if len(latest_df) >= 7 else last_units.iloc[-1]
    )
    next_row["Rolling_Mean_7"] = last_units.tail(7).mean()

    # Predict future day
    X_future = next_row[expected_cols]
    pred_val = max(0.0, float(model.predict(X_future)[0]))
    future_preds.append(pred_val)

    # Append predicted row back to latest_df for future lag propagation
    next_row["Units_Sold"] = pred_val
    latest_df = pd.concat([latest_df, next_row], ignore_index=True)

forecast_df = pd.DataFrame(
    {"Forecasted Units": future_preds}, index=future_dates
)

st.line_chart(forecast_df)

with st.expander("View Raw 90-Day Forecast Data"):
    st.dataframe(forecast_df)
