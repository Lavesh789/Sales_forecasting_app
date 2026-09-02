"""
Sales Forecasting — Interactive Streamlit App
================================================
Consolidates the full pipeline: EDA, feature engineering, model comparison
(Linear Regression, Random Forest, Gradient Boosting, XGBoost), feature
importance, and a recursive future-forecast chart (historical + predicted).

Only depends on: streamlit, pandas, numpy, scikit-learn, xgboost, matplotlib,
openpyxl — all of which install cleanly together, no extra charting library.

Run locally:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Sales Forecasting", page_icon="📈", layout="wide")

TARGET = "Units_Sold"
DROP_COLS = ["Row_ID", "Date", "Units_Sold", "Revenue", "Holiday_Name", "Month_Num", "DayOfWeek_Num"]
DEFAULT_DATA_PATH = "Sales_Forcasting_Dataset.xlsx"


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    required_cols = {
        "Date", "Product_ID", "Store_ID", "Units_Sold", "Price",
        "Discount_Percentage", "Promotion_Flag", "Stock_Availability",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================
@st.cache_data(show_spinner=False)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Product_ID", "Store_ID", "Date"]).reset_index(drop=True)

    df["Year"] = df["Date"].dt.year
    df["Day"] = df["Date"].dt.day
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Month_Num"] = df["Date"].dt.month
    df["DayOfWeek_Num"] = df["Date"].dt.dayofweek

    df["Month_sin"] = np.sin(2 * np.pi * df["Month_Num"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month_Num"] / 12)
    df["DOW_sin"] = np.sin(2 * np.pi * df["DayOfWeek_Num"] / 7)
    df["DOW_cos"] = np.cos(2 * np.pi * df["DayOfWeek_Num"] / 7)

    safe_price = df["Price"].replace(0, np.nan)
    safe_competitor = df["Competitor_Price"].replace(0, np.nan)
    df["Price_Diff_vs_Competitor"] = df["Price"] - df["Competitor_Price"]
    df["Price_Ratio_vs_Competitor"] = (df["Price"] / safe_competitor).fillna(0)
    df["Discounted_Price"] = df["Price"] * (1 - df["Discount_Percentage"].fillna(0) / 100)
    df["Discount_Amount"] = df["Price"] - df["Discounted_Price"]
    df["Marketing_Spend_per_Unit_Price"] = (df["Marketing_Spend"] / safe_price).fillna(0)

    df["Promo_and_Weekend"] = df["Promotion_Flag"] * df["Is_Weekend"]
    df["Promo_and_Holiday"] = df["Promotion_Flag"] * df["Holiday_Flag"]
    df["InStock_and_Promo"] = df["Stock_Availability"] * df["Promotion_Flag"]

    grp = df.groupby(["Product_ID", "Store_ID"])
    df["Units_Sold_Lag1"] = grp["Units_Sold"].shift(1)
    df["Units_Sold_Lag7"] = grp["Units_Sold"].shift(7)
    df["Units_Sold_RollMean_4"] = grp["Units_Sold"].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    df["Units_Sold_RollStd_4"] = grp["Units_Sold"].transform(lambda x: x.shift(1).rolling(4, min_periods=1).std())
    for c in ["Units_Sold_Lag1", "Units_Sold_Lag7", "Units_Sold_RollMean_4", "Units_Sold_RollStd_4"]:
        df[c] = df[c].fillna(grp["Units_Sold"].transform("mean"))
        df[c] = df[c].fillna(df["Units_Sold"].mean())

    df["Has_Holiday_Name"] = df["Holiday_Name"].notna().astype(int) if "Holiday_Name" in df.columns else 0

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def get_X_y(df: pd.DataFrame):
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df[TARGET]
    return X, y


def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )
    return preprocessor, cat_cols, num_cols


# ============================================================
# MODEL TRAINING
# ============================================================
@st.cache_resource(show_spinner=False)
def train_all_models(_df: pd.DataFrame):
    X, y = get_X_y(_df)
    preprocessor, cat_cols, num_cols = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_defs = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        model_defs["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )

    results = {}
    fitted = {}
    for name, model in model_defs.items():
        try:
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            results[name] = {
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": mean_squared_error(y_test, pred) ** 0.5,
                "R2": r2_score(y_test, pred),
            }
            fitted[name] = pipe
        except Exception as e:
            st.warning(f"Skipped {name} due to an error: {e}")

    if not fitted:
        raise RuntimeError("No models trained successfully — check the dataset format.")

    results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
    best_name = results_df["R2"].idxmax()

    importances = None
    imp_source = fitted.get(best_name)
    if imp_source is not None and hasattr(imp_source.named_steps["model"], "feature_importances_"):
        try:
            ohe = imp_source.named_steps["prep"].named_transformers_["cat"]
            feat_names = list(ohe.get_feature_names_out(cat_cols)) + num_cols
            importances = pd.Series(
                imp_source.named_steps["model"].feature_importances_, index=feat_names
            ).sort_values(ascending=False)
        except Exception:
            importances = None

    return {
        "results_df": results_df,
        "fitted": fitted,
        "best_name": best_name,
        "X_test": X_test,
        "y_test": y_test,
        "X_columns": X.columns,
        "importances": importances,
    }


# ============================================================
# RECURSIVE FUTURE FORECASTING
# ============================================================
def forecast_future(engineered_df, model, X_columns, product_id, store_id, periods, freq, assumptions):
    hist = engineered_df[
        (engineered_df["Product_ID"] == product_id) & (engineered_df["Store_ID"] == store_id)
    ].sort_values("Date")

    if hist.empty:
        raise ValueError("No history found for that Product / Store combination.")

    last_row = hist.iloc[-1].copy()
    cur_date = last_row["Date"]
    recent_units = list(hist["Units_Sold"].tail(7).values)

    future_rows = []
    for _ in range(periods):
        cur_date = cur_date + pd.tseries.frequencies.to_offset(freq)
        row = last_row.copy()
        row["Date"] = cur_date
        row["Year"] = cur_date.year
        row["Day"] = cur_date.day
        row["DayOfYear"] = cur_date.dayofyear
        row["WeekOfYear"] = int(cur_date.isocalendar()[1])
        month_num = cur_date.month
        dow_num = cur_date.dayofweek
        row["Month_sin"] = np.sin(2 * np.pi * month_num / 12)
        row["Month_cos"] = np.cos(2 * np.pi * month_num / 12)
        row["DOW_sin"] = np.sin(2 * np.pi * dow_num / 7)
        row["DOW_cos"] = np.cos(2 * np.pi * dow_num / 7)
        row["Is_Weekend"] = 1 if dow_num >= 5 else 0

        for k, v in assumptions.items():
            if k in row.index:
                row[k] = v

        price = row["Price"] if row["Price"] else 0.01
        competitor = row["Competitor_Price"] if row["Competitor_Price"] else np.nan
        row["Price_Diff_vs_Competitor"] = row["Price"] - row["Competitor_Price"]
        row["Price_Ratio_vs_Competitor"] = (row["Price"] / competitor) if competitor and not np.isnan(competitor) else 0
        row["Discounted_Price"] = row["Price"] * (1 - row["Discount_Percentage"] / 100)
        row["Discount_Amount"] = row["Price"] - row["Discounted_Price"]
        row["Marketing_Spend_per_Unit_Price"] = row["Marketing_Spend"] / price
        row["Promo_and_Weekend"] = row["Promotion_Flag"] * row["Is_Weekend"]
        row["Promo_and_Holiday"] = row["Promotion_Flag"] * row.get("Holiday_Flag", 0)
        row["InStock_and_Promo"] = row["Stock_Availability"] * row["Promotion_Flag"]

        row["Units_Sold_Lag1"] = recent_units[-1] if len(recent_units) >= 1 else hist["Units_Sold"].mean()
        row["Units_Sold_Lag7"] = recent_units[-7] if len(recent_units) >= 7 else hist["Units_Sold"].mean()
        window = recent_units[-4:] if len(recent_units) >= 1 else [hist["Units_Sold"].mean()]
        row["Units_Sold_RollMean_4"] = float(np.mean(window))
        row["Units_Sold_RollStd_4"] = float(np.std(window)) if len(window) > 1 else 0.0

        X_row = row.drop(labels=[c for c in DROP_COLS if c in row.index])
        X_row_df = pd.DataFrame([X_row])
        for col in X_columns:
            if col not in X_row_df.columns:
                X_row_df[col] = 0
        X_row_df = X_row_df[X_columns]

        pred = float(model.predict(X_row_df)[0])
        pred = max(0.0, pred)

        row["Units_Sold"] = pred
        future_rows.append(row)
        recent_units.append(pred)
        last_row = row

    return pd.DataFrame(future_rows)


# ============================================================
# SIDEBAR — DATA SOURCE
# ============================================================
st.sidebar.title("📈 Sales Forecasting")
st.sidebar.markdown("Upload your dataset, or use the bundled sample.")

uploaded_file = st.sidebar.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

data_error = None
raw_df = None
try:
    if uploaded_file is not None:
        raw_df = load_data(uploaded_file)
    else:
        try:
            raw_df = load_data(DEFAULT_DATA_PATH)
            st.sidebar.info(f"Using bundled dataset: {DEFAULT_DATA_PATH}")
        except FileNotFoundError:
            data_error = "No dataset uploaded, and no bundled file found. Please upload a .xlsx file."
except Exception as e:
    data_error = f"Could not read the dataset: {e}"

if data_error:
    st.title("📈 Sales Forecasting")
    st.error(data_error)
    st.stop()

with st.spinner("Engineering features..."):
    eng_df = engineer_features(raw_df)

with st.spinner("Training models (first run only, cached afterwards)..."):
    try:
        model_bundle = train_all_models(eng_df)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

results_df = model_bundle["results_df"]
fitted = model_bundle["fitted"]
best_name = model_bundle["best_name"]
X_test, y_test = model_bundle["X_test"], model_bundle["y_test"]
X_columns = model_bundle["X_columns"]
importances = model_bundle["importances"]
best_pipe = fitted[best_name]

st.sidebar.success(f"Best model: **{best_name}**")
st.sidebar.metric("R²", f"{results_df.loc[best_name, 'R2']:.3f}")
st.sidebar.metric("MAE", f"{results_df.loc[best_name, 'MAE']:.2f}")

if not XGBOOST_AVAILABLE:
    st.sidebar.warning("XGBoost isn't installed — comparison runs with 3 models.")


# ============================================================
# MAIN — TABS
# ============================================================
st.title("📈 Sales Forecasting Dashboard")
st.caption("EDA → Feature Engineering → Model Comparison → Feature Importance → Forecast")

tab_overview, tab_eda, tab_models, tab_importance, tab_forecast = st.tabs(
    ["📋 Overview", "🔍 EDA", "🤖 Model Comparison", "⭐ Feature Importance", "🔮 Forecast"]
)

# ---------------- OVERVIEW ----------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(raw_df):,}")
    c2.metric("Products", raw_df["Product_ID"].nunique())
    c3.metric("Stores", raw_df["Store_ID"].nunique())
    c4.metric("Date Range", f"{raw_df['Date'].min().date()} → {raw_df['Date'].max().date()}")

    st.subheader("Sample data")
    st.dataframe(raw_df.head(20), use_container_width=True)

    missing = raw_df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        st.subheader("Missing values")
        st.dataframe(missing.rename("Missing count"), use_container_width=True)
    else:
        st.success("No missing values in the raw dataset.")

# ---------------- EDA ----------------
with tab_eda:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution of Units_Sold")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(raw_df["Units_Sold"], bins=30, color="#4f8cff", edgecolor="white")
        ax.set_xlabel("Units_Sold")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        st.subheader("Monthly trend")
        monthly = raw_df.groupby(raw_df["Date"].dt.to_period("M").astype(str))["Units_Sold"].sum()
        st.line_chart(monthly)

    st.subheader("Average Units Sold by segment")
    seg_cols = [c for c in ["Category", "Store_Location", "Sales_Channel", "Customer_Segment"] if c in raw_df.columns]
    if seg_cols:
        seg_choice = st.selectbox("Segment by", seg_cols)
        seg_avg = raw_df.groupby(seg_choice)["Units_Sold"].mean().sort_values()
        st.bar_chart(seg_avg)

    st.subheader("Correlation heatmap")
    numeric_df = raw_df.select_dtypes(include=[np.number])
    if "Row_ID" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["Row_ID"])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------- MODEL COMPARISON ----------------
with tab_models:
    st.subheader("Model comparison (test set)")
    st.dataframe(
        results_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}"}),
        use_container_width=True,
    )

    st.subheader("R² by model")
    st.bar_chart(results_df["R2"])

    st.subheader(f"{best_name}: Predicted vs Actual")
    pred = best_pipe.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, pred, alpha=0.4, color="#4f8cff")
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("Actual Units_Sold")
    ax.set_ylabel("Predicted Units_Sold")
    st.pyplot(fig)
    plt.close(fig)

# ---------------- FEATURE IMPORTANCE ----------------
with tab_importance:
    if importances is not None:
        top_n = st.slider("Number of top features to show", 5, min(25, len(importances)), 15)
        top_imp = importances.head(top_n).sort_values()
        st.subheader(f"Top {top_n} Feature Importances ({best_name})")
        st.bar_chart(top_imp)
    else:
        st.info(f"{best_name} doesn't expose feature importances. Showing coefficients instead where possible.")
        try:
            coefs = best_pipe.named_steps["model"].coef_
            ohe = best_pipe.named_steps["prep"].named_transformers_["cat"]
            X, _ = get_X_y(eng_df)
            _, cat_cols, num_cols = build_preprocessor(X)
            feat_names = list(ohe.get_feature_names_out(cat_cols)) + num_cols
            coef_series = pd.Series(coefs, index=feat_names).sort_values(key=abs, ascending=False).head(15).sort_values()
            st.bar_chart(coef_series)
        except Exception:
            st.warning("Coefficients aren't available for this model either.")

# ---------------- FORECAST ----------------
with tab_forecast:
    st.subheader("Forecast future sales")
    st.caption("Historical actual sales plus a recursive future forecast, with adjustable assumptions.")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        product_id = st.selectbox("Product", sorted(eng_df["Product_ID"].unique()))
    with fc2:
        store_options = sorted(eng_df[eng_df["Product_ID"] == product_id]["Store_ID"].unique())
        store_id = st.selectbox("Store", store_options)
    with fc3:
        freq_label = st.selectbox("Forecast frequency", ["Weekly", "Daily", "Monthly"])
        freq_map = {"Weekly": "W", "Daily": "D", "Monthly": "M"}
        freq = freq_map[freq_label]

    periods = st.slider("Number of future periods to forecast", 1, 52, 8)

    hist_subset = eng_df[(eng_df["Product_ID"] == product_id) & (eng_df["Store_ID"] == store_id)].sort_values("Date")

    if hist_subset.empty:
        st.warning("No historical data for this Product/Store combination.")
    else:
        st.markdown("**Future assumptions** (defaults use recent averages — adjust as needed)")
        a1, a2, a3 = st.columns(3)
        with a1:
            price_default = float(hist_subset["Price"].tail(10).mean())
            future_price = st.number_input("Price", min_value=0.0, value=round(price_default, 2))
            discount_default = float(hist_subset["Discount_Percentage"].tail(10).mean())
            future_discount = st.slider("Discount %", 0, 50, int(round(discount_default)))
        with a2:
            competitor_default = float(hist_subset["Competitor_Price"].tail(10).mean())
            future_competitor = st.number_input("Competitor Price", min_value=0.0, value=round(competitor_default, 2))
            future_promo = st.checkbox("Promotion running?", value=bool(hist_subset["Promotion_Flag"].tail(5).mode()[0]))
        with a3:
            marketing_default = float(hist_subset["Marketing_Spend"].tail(10).mean())
            future_marketing = st.number_input("Marketing Spend", min_value=0.0, value=round(marketing_default, 2))
            future_stock = st.checkbox("In stock?", value=bool(hist_subset["Stock_Availability"].tail(5).mode()[0]))

        assumptions = {
            "Price": future_price,
            "Discount_Percentage": future_discount,
            "Competitor_Price": future_competitor,
            "Promotion_Flag": int(future_promo),
            "Marketing_Spend": future_marketing,
            "Stock_Availability": int(future_stock),
        }

        if st.button("Generate forecast", type="primary"):
            try:
                with st.spinner("Forecasting..."):
                    future_df = forecast_future(
                        eng_df, best_pipe, X_columns, product_id, store_id, periods, freq, assumptions
                    )

                hist_plot = hist_subset[["Date", "Units_Sold"]].copy()
                fut_plot = future_df[["Date", "Units_Sold"]].copy()

                fig, ax = plt.subplots(figsize=(11, 5))
                ax.plot(hist_plot["Date"], hist_plot["Units_Sold"], marker="o", markersize=3,
                        color="#4f8cff", label="Actual (historical)")
                ax.plot(fut_plot["Date"], fut_plot["Units_Sold"], marker="o", markersize=3,
                        color="#ff6b6b", linestyle="--", label="Forecast")
                ax.plot(
                    [hist_plot["Date"].iloc[-1], fut_plot["Date"].iloc[0]],
                    [hist_plot["Units_Sold"].iloc[-1], fut_plot["Units_Sold"].iloc[0]],
                    color="#ff6b6b", linestyle="--",
                )
                ax.set_title(f"Units Sold — {product_id} @ {store_id}: History + Forecast")
                ax.set_xlabel("Date")
                ax.set_ylabel("Units Sold")
                ax.legend()
                fig.autofmt_xdate()
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Forecast table")
                display_cols = ["Date", "Units_Sold"]
                st.dataframe(
                    future_df[display_cols].rename(columns={"Units_Sold": "Predicted_Units_Sold"}).round(1),
                    use_container_width=True,
                )

                csv = future_df[display_cols].rename(columns={"Units_Sold": "Predicted_Units_Sold"}).to_csv(index=False)
                st.download_button("Download forecast as CSV", csv, file_name=f"forecast_{product_id}_{store_id}.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Forecast failed: {e}")

st.divider()
st.caption(f"Model: {best_name} · Trained on {len(eng_df):,} transactions")
