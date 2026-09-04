import os
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template_string, request
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

DATASET_PATH = "Sales_Forcasting_Dataset.xlsx"

def load_data():
    """Loads Excel dataset or uses quick sample data."""
    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_excel(DATASET_PATH)
            df["Date"] = pd.to_datetime(df["Date"])
            return df
        except Exception:
            pass
            
    # Quick fallback data
    dates = pd.date_range(start="2024-01-01", end="2026-08-31", freq="D")
    np.random.seed(42)
    sales = np.sin(np.linspace(0, 20, len(dates))) * 500 + np.random.normal(2000, 300, len(dates))
    return pd.DataFrame({"Date": dates, "Revenue": sales})

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sales Forecast</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f9f9f9; color: #333; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .form-group { margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
        input, button { padding: 8px 12px; font-size: 14px; }
        button { background-color: #007bff; color: white; border: none; cursor: pointer; border-radius: 4px; }
        button:hover { background-color: #0056b3; }
        .chart-box { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Sales Forecasting</h2>
        <form method="POST" class="form-group">
            <label for="days">Forecast Days:</label>
            <input type="number" name="days" id="days" value="{{ forecast_days }}" min="7" max="365">
            <button type="submit">Predict</button>
        </form>

        <div class="chart-box">
            <canvas id="salesChart"></canvas>
        </div>
    </div>

    <script>
        const historicalLabels = {{ historic_labels | tojson }};
        const historicalData = {{ historic_values | tojson }};
        const futureLabels = {{ future_labels | tojson }};
        const futureData = {{ future_values | tojson }};

        // Combine labels for continuous timeline
        const allLabels = [...historicalLabels, ...futureLabels];
        
        // Align historical data (null padding for future slots)
        const paddedHistorical = [...historicalData, ...new Array(futureData.length).fill(null)];
        
        // Align future data (null padding for historical slots, start connecting from last historical point)
        const lastHistoricVal = historicalData[historicalData.length - 1];
        const paddedFuture = [...new Array(historicalData.length - 1).fill(null), lastHistoricVal, ...futureData];

        const ctx = document.getElementById('salesChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: allLabels,
                datasets: [
                    {
                        label: 'Historical Sales',
                        data: paddedHistorical,
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'Future Prediction',
                        data: paddedFuture,
                        borderColor: '#dc3545',
                        borderDash: [5, 5],
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        fill: false,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: { title: { display: true, text: 'Date' } },
                    y: { title: { display: true, text: 'Revenue' }, beginAtZero: true }
                }
            }
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    forecast_days = int(request.form.get("days", 90))
    
    df = load_data()
    daily_sales = df.groupby("Date")["Revenue"].sum().reset_index().sort_values("Date")
    
    # Linear Regression Model
    daily_sales["Ordinal_Date"] = daily_sales["Date"].map(datetime.toordinal)
    X = daily_sales[["Ordinal_Date"]]
    y = daily_sales["Revenue"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    last_date = daily_sales["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    future_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_preds = np.maximum(0, model.predict(future_ordinal))
    
    # Prepare monthly data points for lightweight rendering
    hist_monthly = daily_sales.set_index("Date")["Revenue"].resample("ME").sum()
    future_df = pd.DataFrame({"Date": future_dates, "Revenue": future_preds})
    fut_monthly = future_df.set_index("Date")["Revenue"].resample("ME").sum()

    return render_template_string(
        HTML_TEMPLATE,
        forecast_days=forecast_days,
        historic_labels=hist_monthly.index.strftime('%Y-%m').tolist(),
        historic_values=[round(v, 2) for v in hist_monthly.values.tolist()],
        future_labels=fut_monthly.index.strftime('%Y-%m').tolist(),
        future_values=[round(v, 2) for v in fut_monthly.values.tolist()]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
