import os
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

DATASET_PATH = "Sales_Forcasting_Dataset.xlsx"


def load_data():
    """Loads Excel data or falls back to synthetic data if missing."""
    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_excel(DATASET_PATH)
            df["Date"] = pd.to_datetime(df["Date"])
            return df
        except Exception:
            pass

    # Basic sample data so the app never crashes
    dates = pd.date_range(start="2024-01-01", end="2026-08-31", freq="D")
    np.random.seed(42)
    sales = np.sin(np.linspace(0, 20, len(dates))) * 500 + np.random.normal(
        2000, 300, len(dates)
    )
    return pd.DataFrame({"Date": dates, "Revenue": sales})


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Sales Forecast</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 30px; background-color: #f8f9fa; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .form-box { margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
        input, button { padding: 8px 12px; font-size: 14px; }
        button { background-color: #007bff; color: white; border: none; cursor: pointer; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Sales Forecast (Past vs Future)</h2>
        <form method="POST" class="form-box">
            <label>Forecast Days:</label>
            <input type="number" name="days" value="{{ forecast_days }}" min="7" max="365">
            <button type="submit">Generate Graph</button>
        </form>
        <div id="plotlyChart"></div>
    </div>

    <script>
        var histDates = {{ hist_dates | tojson }};
        var histValues = {{ hist_values | tojson }};
        var futDates = {{ fut_dates | tojson }};
        var futValues = {{ fut_values | tojson }};

        var trace1 = {
            x: histDates,
            y: histValues,
            mode: 'lines',
            name: 'Historical Data',
            line: {color: '#007bff'}
        };

        var trace2 = {
            x: futDates,
            y: futValues,
            mode: 'lines',
            name: 'Future Prediction',
            line: {color: '#dc3545', dash: 'dash'}
        };

        var layout = {
            title: 'Historical Sales vs Future Predictions',
            xaxis: { title: 'Date' },
            yaxis: { title: 'Revenue' }
        };

        Plotly.newPlot('plotlyChart', [trace1, trace2], layout);
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    forecast_days = int(request.form.get("days", 90))

    df = load_data()
    daily_sales = (
        df.groupby("Date")["Revenue"].sum().reset_index().sort_values("Date")
    )

    # Simple Linear Regression Prediction
    daily_sales["Ordinal_Date"] = daily_sales["Date"].map(datetime.toordinal)
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
    future_preds = np.maximum(0, model.predict(future_ordinal))

    # Aggregating Monthly for fast, clean line rendering
    hist_monthly = daily_sales.set_index("Date")["Revenue"].resample("ME").sum()
    future_df = pd.DataFrame({"Date": future_dates, "Revenue": future_preds})
    fut_monthly = future_df.set_index("Date")["Revenue"].resample("ME").sum()

    # Connecting the historical line to the forecast start line
    fut_dates = [hist_monthly.index[-1].strftime("%Y-%m-%d")] + [
        d.strftime("%Y-%m-%d") for d in fut_monthly.index
    ]
    fut_vals = [round(hist_monthly.values[-1], 2)] + [
        round(v, 2) for v in fut_monthly.values
    ]

    return render_template_string(
        HTML_TEMPLATE,
        forecast_days=forecast_days,
        hist_dates=[d.strftime("%Y-%m-%d") for d in hist_monthly.index],
        hist_values=[round(v, 2) for v in hist_monthly.values],
        fut_dates=fut_dates,
        fut_values=fut_vals,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
