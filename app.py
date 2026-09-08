import warnings
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# 1. Page Configuration
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.title("📈 Interactive Daily Sales Forecasting Dashboard")

DATASET_PATH = "Sales_Forcasting_Dataset.xlsx"


# 2. Load & Process Dataset
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

    # Numerical Time-Series Features
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

# 3. Interactive Sidebar Controls for Input Features
st.sidebar.header("🎛️ Input Feature Controls")

# Dropdown: Year Filter
available_years = sorted(daily_df["Year"].unique().tolist())
selected_year = st.sidebar.selectbox(
    "Select Target Year",
    options=["All Years"] + available_years,
    index=0,
)

# Dropdown: Quarter Filter
selected_quarter = st.sidebar.selectbox(
    "Select Quarter",
    options=["All Quarters", "Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"],
    index=0,
)

# Dropdown: Day Type Filter
day_type = st.sidebar.selectbox(
    "Select Day Type",
    options=["All Days", "Weekdays Only", "Weekends Only"],
    index=0,
)

# Slider: Forecast Horizon
forecast_days = st.sidebar.slider(
    "Future Forecast Horizon (Days)",
    min_value=7,
    max_value=90,
    value=90,
    step=1,
)

# 4. Filter Dataset Based on Selected Dropdowns
filtered_df = daily_df.copy()

if selected_year != "All Years":
    filtered_df = filtered_df[filtered_df["Year"] == int(selected_year)]

if selected_quarter != "All Quarters":
    q_map = {
        "Q1 (Jan-Mar)": 1,
        "Q2 (Apr-Jun)": 2,
        "Q3 (Jul-Sep)": 3,
        "Q4 (Oct-Dec)": 4,
    }
    filtered_df = filtered_df[filtered_df["Quarter"] == q_map[selected_quarter]]

if day_type == "Weekdays Only":
    filtered_df = filtered_df[filtered_df["Is_Weekend"] == 0]
elif day_type == "Weekends Only":
    filtered_df = filtered_df[filtered_df["Is_Weekend"] == 1]

if filtered_df.empty:
    st.warning("⚠️ No data available for the selected dropdown combination. Reverting to full dataset.")
    filtered_df = daily_df.copy()

# 5. Train Random Forest Model
X = daily_df.drop(columns=["Units_Sold"])
y = daily_df["Units_Sold"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300, max_depth=15, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# Filtered Historical Evaluation Predictions
eval_X = filtered_df.drop(columns=["Units_Sold"])
eval_y = filtered_df["Units_Sold"]
test_preds = model.predict(eval_X)

st.markdown("---")

# 6. Historical Evaluation Chart
st.subheader("📊 Historical Predictions vs Actual Sales")
comp_df = (
    pd.DataFrame(
        {
            "Date": eval_X.index,
            "Actual Sales": eval_y.values,
            "Predicted Sales": test_preds,
        }
    )
    .sort_values("Date")
    .set_index("Date")
)

st.line_chart(comp_df)

# 7. Iterative Future Forecast Loop
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

    # Append prediction to update lag rolling features
    new_entry = feat_row.copy()
    new_entry["Units_Sold"] = pred_val
    last_data = pd.concat([last_data, new_entry])

forecast_df = pd.DataFrame(
    {"Forecasted Units": future_preds}, index=future_dates
)

st.line_chart(forecast_df)

with st.expander("📋 View Raw 90-Day Forecast Table"):
    st.dataframe(forecast_df)
