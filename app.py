import base64
import os
from datetime import datetime

# Set Matplotlib to non-interactive 'Agg' backend BEFORE importing pyplot
# This prevents GUI/threading issues on cloud servers like Render
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

DATASET_PATH = "Sales_Forcasting_Dataset.xlsx"


def load_data():
    """Loads dataset from Excel or generates mock data if missing."""
    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_excel(DATASET_PATH)
            df["Date"] = pd.to_datetime(df["Date"])
            return df
        except Exception as e:
            print(f"Error loading excel file: {e}")

    # Fallback dataset generator for continuous uptime
    dates = pd.date_range(start="2023-01-01", end="2026-08-31", freq="D")
    np.random.seed(42)
    sales = np.sin(np.linspace(0, 20, len(dates))) * 500 + np.random.normal(
        2000, 300, len(dates)
    )
    return pd.DataFrame(
        {
            "Date": dates,
            "Units_Sold": np.random.randint(5, 100, size=len(dates)),
            "Revenue": sales,
            "Product_Name": np.random.choice(
                [
                    "Bluetooth Speaker",
                    "LED Desk Lamp",
                    "Running Shoes",
                    "Notebook Pack",
                    "Wireless Mouse",
                ],
                size=len(dates),
            ),
        }
    )


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sales Forecasting Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 20px; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 25px; }
        .form-group { display: flex; justify-content: center; align-items: center; gap: 15px; margin-bottom: 25px; }
        label { font-weight: bold; }
        select, input[type="number"], button { padding: 10px 15px; border-radius: 4px; border: 1px solid #ccc; font-size: 14px; }
        button { background-color: #3498db; color: white; border: none; cursor: pointer; transition: background 0.3s; }
        button:hover { background-color: #2980b9; }
        .chart-box { text-align: center; margin-top: 20px; }
        .chart-box img { max-width: 100%; height: auto; border-radius: 5px; border: 1px solid #ddd; }
        .stats { display: flex; justify-content: space-around; margin-top: 25px; background: #ecf0f1; padding: 15px; border-radius: 5px; }
        .stat-card { text-align: center; }
        .stat-card h3 { margin: 0; font-size: 20px; color: #2c3e50; }
        .stat-card p { margin: 5px 0 0 0; color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Sales Forecasting Dashboard</h1>
        <form method="POST" class="form-group">
            <label for="product">Select Product:</label>
            <select name="product" id="product">
                <option value="All" {% if selected_product == 'All' %}selected{% endif %}>All Products</option>
                {% for prod in products %}
                    <option value="{{ prod }}" {% if selected_product == prod %}selected{% endif %}>{{ prod }}</option>
                {% endfor %}
            </select>

            <label for="days">Forecast Days:</label>
            <input type="number" name="days" id="days" value="{{ forecast_days }}" min="7" max="365">

            <button type="submit">Generate Forecast</button>
        </form>

        {% if plot_url %}
            <div class="chart-box">
                <img src="data:image/png;base64,{{ plot_url }}" alt="Sales Forecast Chart">
            </div>

            <div class="stats">
                <div class="stat-card">
                    <h3>₹{{ "%.2f"|format(total_historic) }}</h3>
                    <p>Total Historical Revenue</p>
                </div>
                <div class="stat-card">
                    <h3>₹{{ "%.2f"|format(total_forecast) }}</h3>
                    <p>Predicted Revenue (Next {{ forecast_days }} Days)</p>
                </div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    df = load_data()
    products = (
        sorted(df["Product_Name"].unique().tolist())
        if "Product_Name" in df.columns
        else []
    )

    selected_product = request.form.get("product", "All")
    forecast_days = int(request.form.get("days", 90))

    if selected_product != "All" and "Product_Name" in df.columns:
        filtered_df = df[df["Product_Name"] == selected_product].copy()
    else:
        filtered_df = df.copy()

    # Data aggregation
    daily_sales = filtered_df.groupby("Date")["Revenue"].sum().reset_index()
    daily_sales = daily_sales.sort_values("Date")
    daily_sales["Ordinal_Date"] = daily_sales["Date"].map(datetime.toordinal)

    # Forecasting model
    X = daily_sales[["Ordinal_Date"]]
    y = daily_sales["Revenue"]
    model = LinearRegression()
    model.fit(X, y)

    last_date = daily_sales["Date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D"
    )
    future_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(
        -1, 1
    )
    future_predictions = np.maximum(0, model.predict(future_ordinal))

    future_df = pd.DataFrame(
        {"Date": future_dates, "Revenue": future_predictions}
    )

    # Plot generation
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)

    monthly_historic = (
        daily_sales.set_index("Date")["Revenue"].resample("ME").sum()
    )
    monthly_future = (
        future_df.set_index("Date")["Revenue"].resample("ME").sum()
    )

    ax.plot(
        monthly_historic.index,
        monthly_historic.values,
        label="Historical Sales",
        color="#2b5c8f",
        linewidth=2,
    )
    ax.plot(
        monthly_future.index,
        monthly_future.values,
        label="Future Prediction",
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
    )

    ax.set_title(
        f"Sales Analysis & Forecast ({selected_product})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Revenue", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    img_buf = pd.io.common.BytesIO()
    fig.savefig(img_buf, format="png")
    img_buf.seek(0)
    plot_url = base64.b64encode(img_buf.getvalue()).decode("utf-8")
    plt.close(fig)

    return render_template_string(
        HTML_TEMPLATE,
        products=products,
        selected_product=selected_product,
        forecast_days=forecast_days,
        plot_url=plot_url,
        total_historic=daily_sales["Revenue"].sum(),
        total_forecast=future_df["Revenue"].sum(),
    )


if __name__ == "__main__":
    # Render binds dynamically to PORT env variable; fallback to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    # debug=False and use_reloader=False prevent thread signal errors in hosted environments
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
