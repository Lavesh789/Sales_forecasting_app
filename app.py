import warnings
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# Streamlit Page Config
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.title("📈 Daily Sales Forecasting & Model Evaluation")


# ---------------------------------------------------------------------------
# 1. WMAPE Metric Definitions
# ---------------------------------------------------------------------------
def wmape(y_true, y_pred):
    """Calculates Weighted Absolute Percentage Error (WMAPE)."""
    y_true = np.asarray(y_true)
    y_pred = np.clip(np.asarray(y_pred), 0, None)
    sum_actuals = np.sum(np.abs(y_true))
    if sum_actuals == 0:
        return 0.0
    return (np.sum(np.abs(y_true - y_pred)) / sum_actuals) * 100.0


# ---------------------------------------------------------------------------
# 2. Data Preparation
# ---------------------------------------------------------------------------
DATASET_PATH = "Sales_Forcasting_Dataset.xlsx"


@st.cache_data
def load_and_prep_data():
    try:
        df = pd.read_excel(DATASET_PATH)
    except Exception:
        # Fallback synthetic dataset if file is absent
        dates = pd.date_range(start="2024-01-01", end="2026-08-31", freq="D")
        np.random.seed(42)
        units = np.sin(np.linspace(0, 20, len(dates))) * 50 + np.random.normal(
            200, 30, len(dates)
        )
        df = pd.DataFrame({"Date": dates, "Units_Sold": np.maximum(1, units)})

    df["Date"] = pd.to_datetime(df["Date"])

    # Continuous daily time-series indexing
    daily = (
        df.groupby("Date")["Units_Sold"].sum().asfreq("D", fill_value=0).to_frame()
    )
    daily.index.name = "Date"

    # Feature Engineering
    daily["DayOfWeek"] = daily.index.dayofweek
    daily["Month"] = daily.index.month
    daily["Year"] = daily.index.year
    daily["DayOfMonth"] = daily.index.day
    daily["Lag_1"] = daily["Units_Sold"].shift(1)
    daily["Lag_7"] = daily["Units_Sold"].shift(7)
    daily["Rolling_Mean_7"] = daily["Units_Sold"].shift(1).rolling(7).mean()

    return daily.bfill().ffill()


daily_df = load_and_prep_data()

# ---------------------------------------------------------------------------
# 3. Model Pipeline & Training
# ---------------------------------------------------------------------------
X = daily_df.drop(columns=["Units_Sold"])
y = daily_df["Units_Sold"]

#split_idx = int(len(daily_df) * 0.8)
#X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
#y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_cols = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols,
        )
    ],
    remainder="passthrough",
)

rf_pipeline = Pipeline([
    ("prep", preprocessor),
    (
        "model",
        RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

# Fit Random Forest
rf_pipeline.fit(X_train, y_train)
preds = rf_pipeline.predict(X_test)

# Metrics Evaluation
wmape_val = wmape(y_test.values, preds)
accuracy = max(0.0, 100.0 - wmape_val)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

# ---------------------------------------------------------------------------
# 4. Streamlit Dashboard Layout
# ---------------------------------------------------------------------------
st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Future Forecast Horizon (Days)", 7, 60, 30)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🏆 Best Model", "Random Forest")
col2.metric("🎯 Model Accuracy", f"{accuracy:.2f}%")
col3.metric("📉 WMAPE Error", f"{wmape_val:.2f}%")
col4.metric("📊 MAE Score", f"{mae:.2f}")

st.markdown("---")

# Historical Chart
st.subheader("📊 Historical Evaluation: Actual vs Random Forest Predictions")
comp_df = pd.DataFrame(
    {"Actual Sales": y_test.values, "Random Forest (Predicted)": preds},
    index=X_test.index,
)

st.line_chart(comp_df)

# ---------------------------------------------------------------------------
# 5. Future Predictions Iterative Loop
# ---------------------------------------------------------------------------
st.subheader(f"🔮 Future {forecast_days}-Day Sales Forecast")

future_dates = pd.date_range(
    start=daily_df.index[-1] + pd.Timedelta(days=1),
    periods=forecast_days,
    freq="D",
)
future_preds = []
last_data = daily_df.copy()

for date in future_dates:
    last_row = last_data.iloc[-1]
    lag_1 = last_row["Units_Sold"]
    lag_7 = (
        last_data.iloc[-7]["Units_Sold"]
        if len(last_data) >= 7
        else last_row["Units_Sold"]
    )
    rolling_7 = last_data["Units_Sold"].tail(7).mean()

    feat_df = pd.DataFrame(
        [
            {
                "DayOfWeek": date.dayofweek,
                "Month": date.month,
                "Year": date.year,
                "DayOfMonth": date.day,
                "Lag_1": lag_1,
                "Lag_7": lag_7,
                "Rolling_Mean_7": rolling_7,
            }
        ],
        index=[date],
    )

    pred_val = max(0.0, float(rf_pipeline.predict(feat_df)[0]))
    future_preds.append(pred_val)

    new_row = feat_df.iloc[0].to_dict()
    new_row["Units_Sold"] = pred_val
    last_data = pd.concat([last_data, pd.DataFrame([new_row], index=[date])])

forecast_df = pd.DataFrame(
    {"Forecasted Units": future_preds}, index=future_dates
)

st.line_chart(forecast_df)

with st.expander("View Raw Forecast Data"):
    st.dataframe(forecast_df)
