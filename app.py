import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sktime.performance_metrics.forecasting import (
    mean_absolute_percentage_error,
)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard", page_layout="wide"
)
st.title("📈 Daily Sales Forecasting & Model Evaluation")

# ---------------------------------------------------------------------------
# 1. Data Loading & Preprocessing
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

    # Continuous daily aggregation
    daily = (
        df.groupby("Date")["Units_Sold"].sum().asfreq("D", fill_value=0)
    ).to_frame()
    daily.index.name = "Date"

    # Feature Engineering
    daily["DayOfWeek"] = daily.index.dayofweek
    daily["Month"] = daily.index.month
    daily["Year"] = daily.index.year
    daily["DayOfMonth"] = daily.index.day
    daily["Lag_1"] = daily["Units_Sold"].shift(1)
    daily["Lag_7"] = daily["Units_Sold"].shift(7)
    daily["Rolling_Mean_7"] = daily["Units_Sold"].shift(1).rolling(7).mean()

    daily = daily.bfill().ffill()
    return daily


daily_df = load_and_prep_data()

# ---------------------------------------------------------------------------
# 2. Train / Test Split & Model Pipelines
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

models = {
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
        ),
    ]),
    "XGBoost": Pipeline([
        ("prep", preprocessor),
        (
            "model",
            XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]),
    "LightGBM": Pipeline([
        ("prep", preprocessor),
        (
            "model",
            LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
    ]),
    "Gradient Boosting": Pipeline([
        ("prep", preprocessor),
        (
            "model",
            GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.05, random_state=42
            ),
        ),
    ]),
}

# ---------------------------------------------------------------------------
# 3. Model Training & Selection Loop
# ---------------------------------------------------------------------------
st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Future Forecast Horizon (Days)", 7, 60, 30)

results = {}
predictions = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    predictions[name] = preds

    # Accuracy metrics
    mask = y_test.values != 0
    mape = (
        np.mean(
            np.abs(
                (y_test.values[mask] - preds[mask]) / y_test.values[mask]
            )
        )
        * 100
    )
    acc = max(0.0, 100.0 - mape)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results[name] = {
        "Pipeline": pipe,
        "MAPE": mape,
        "Accuracy": acc,
        "MAE": mae,
        "R2": r2,
    }

# Find Best Model based on Accuracy
best_model_name = max(results, key=lambda k: results[k]["Accuracy"])
best_model_info = results[best_model_name]
best_pipeline = best_model_info["Pipeline"]

# ---------------------------------------------------------------------------
# 4. Dashboard Visualizations
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("🏆 Best Model", best_model_name)
col2.metric("🎯 Model Accuracy", f"{best_model_info['Accuracy']:.2f}%")
col3.metric("📉 MAPE Error", f"{best_model_info['MAPE']:.2f}%")

st.markdown("---")

# Historical Actuals vs Predictions Plot
st.subheader("📊 Historical Evaluation: Actual vs Model Predictions")
comp_df = pd.DataFrame(
    {
        "Actual": y_test.values,
        f"{best_model_name} (Best)": predictions[best_model_name],
    },
    index=X_test.index,
)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(
    daily_df.index[:split_idx],
    y_train.values,
    label="Training Data",
    color="gray",
    alpha=0.5,
)
ax.plot(
    comp_df.index,
    comp_df["Actual"],
    label="Actual Sales (Test)",
    color="blue",
)
ax.plot(
    comp_df.index,
    comp_df[f"{best_model_name} (Best)"],
    label=f"Predicted ({best_model_name})",
    color="green",
    linestyle="--",
)
ax.set_ylabel("Units Sold")
ax.legend()
st.pyplot(fig)

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
    # Feature extraction for single future date
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

    pred_val = max(0, float(best_pipeline.predict(feat_df)[0]))
    future_preds.append(pred_val)

    # Append to rolling temp dataset for consecutive lag generation
    new_row = feat_df.iloc[0].to_dict()
    new_row["Units_Sold"] = pred_val
    last_data = pd.concat([last_data, pd.DataFrame([new_row], index=[date])])

forecast_df = pd.DataFrame({"Forecasted_Units": future_preds}, index=future_dates)

# Future Forecast Plot
fig_fut, ax_fut = plt.subplots(figsize=(12, 4))
ax_fut.plot(
    daily_df.index[-60:],
    daily_df["Units_Sold"].tail(60),
    label="Recent Actual Sales",
    color="blue",
)
ax_fut.plot(
    forecast_df.index,
    forecast_df["Forecasted_Units"],
    label="Future Forecast",
    color="orange",
    linestyle="--",
    marker="o",
)
ax_fut.set_ylabel("Units Sold")
ax_fut.legend()
st.pyplot(fig_fut)

# Show Forecast Data Table
with st.expander("View Future Forecast Data Table"):
    st.dataframe(forecast_df)
