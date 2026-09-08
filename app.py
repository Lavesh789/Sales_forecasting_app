import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# Streamlit Page Setup
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
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
        st.error(
            f"⚠️ Could not load `{MODEL_PATH}` from GitHub repository.\n\n"
            "Please ensure you exported the model using `joblib.dump(model, 'rf_model.joblib')` "
            "in your notebook and pushed it to GitHub."
        )
        st.stop()


@st.cache_data
def load_and_prep_data():
    """Loads dataset and performs exact feature engineering."""
    df = pd.read_excel(DATASET_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    # Aggregate daily sales
    daily = (
        df.groupby("Date")["Units_Sold"].sum().asfreq("D", fill_value=0).to_frame()
    )
    daily.index.name = "Date"

    # Time-series features
    daily["DayOfWeek"] = daily.index.dayofweek
    daily["Month"] = daily.index.month
    daily["Year"] = daily.index.year
    daily["DayOfMonth"] = daily.index.day
    daily["Lag_1"] = daily["Units_Sold"].shift(1)
    daily["Lag_7"] = daily["Units_Sold"].shift(7)
    daily["Rolling_Mean_7"] = daily["Units_Sold"].shift(1).rolling(7).mean()

    return daily.bfill().ffill()


# Load model and dataset
model = load_saved_model()
daily_df = load_and_prep_data()

# ---------------------------------------------------------------------------
# 2. Historical Evaluation (Matches Notebook Train/Test Split)
# ---------------------------------------------------------------------------
X = daily_df.drop(columns=["Units_Sold"])
y = daily_df["Units_Sold"]

#split_idx = int(len(daily_df) * 0.8)
#X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
#y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    index=X_test.index,
)
st.line_chart(comp_df)

# ---------------------------------------------------------------------------
# 4. Iterative 90-Day Future Forecast Loop
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

    pred_val = max(0.0, float(model.predict(feat_df)[0]))
    future_preds.append(pred_val)

    # Append predicted value to history to drive subsequent lag features
    new_row = feat_df.iloc[0].to_dict()
    new_row["Units_Sold"] = pred_val
    last_data = pd.concat([last_data, pd.DataFrame([new_row], index=[date])])

forecast_df = pd.DataFrame(
    {"Forecasted Units": future_preds}, index=future_dates
)

st.line_chart(forecast_df)

with st.expander("View Raw 90-Day Forecast Data"):
    st.dataframe(forecast_df)
