import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_percentage_error

LAG = 10
st.set_page_config(page_title="Health Forecasting", layout="wide")
st.title("📈 Forecasting Tool for Health Commodities")

# Upload CSV
file = st.file_uploader("Upload CSV with 'Date' and 'Consumption' columns", type="csv")
if not file:
    st.stop()

df = pd.read_csv(file, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

st.subheader("📋 Uploaded Data Preview")
st.dataframe(df.head())

# Sidebar: Adjustment sliders
st.sidebar.title("🔧 Adjustment Factors (2020–2027)")
adjustment_factors = {
    year: st.sidebar.slider(str(year), 0.0, 1.5, 1.0, 0.1)
    for year in range(2020, 2028)
}

# Split data
train_df = df[df['Date'] < '2019-01-01'].copy()
test_df = df[(df['Date'] >= '2019-01-01') & (df['Date'] < '2020-01-01')].copy()
future_dates = pd.date_range("2020-01-01", "2027-12-01", freq='MS')
future_steps = len(future_dates)
test_steps = len(test_df)

# Adjustment
def apply_adjustment(forecast, dates):
    years = dates.dt.year
    return [v * adjustment_factors.get(y, 1.0) for v, y in zip(forecast, years)]

# ETS (no LSTM)
def forecast_ets(data, steps):
    ets_model = ExponentialSmoothing(data['Consumption'], seasonal='add', seasonal_periods=12).fit()
    return ets_model.forecast(steps).values

# Moving Average
def forecast_moving_average(data, steps):
    avg = data['Consumption'].rolling(window=LAG).mean().iloc[-1]
    return np.full(steps, avg)

# Auto Regression
def forecast_autoreg(data, steps):
    model = AutoReg(data['Consumption'], lags=LAG).fit()
    return model.forecast(steps)

# Run forecasting
with st.spinner("⏳ Running forecasting models..."):
    test_forecasts = {
        'ETS': forecast_ets(train_df, test_steps),
        'Moving Average': forecast_moving_average(train_df, test_steps),
        'Auto Regression': forecast_autoreg(train_df, test_steps)
    }

    future_forecasts = {
        'ETS': forecast_ets(df, future_steps),
        'Moving Average': forecast_moving_average(df, future_steps),
        'Auto Regression': forecast_autoreg(df, future_steps)
    }

# Assemble forecast DataFrame
future_df = pd.DataFrame({'Date': future_dates})
for method in future_forecasts:
    future_df[method] = future_forecasts[method]
    future_df[f"{method} Adjusted"] = apply_adjustment(future_forecasts[method], future_df['Date'])

# Plotting
def plot_forecast(method):
    st.subheader(f"📊 {method} Forecast (2017–2027)")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['Date'], df['Consumption'], label="Historical (2017–2018)", color="black")
    ax.plot(test_df['Date'], test_forecasts[method], label="2019 Forecast (Test)", linestyle="--", color="blue")
    ax.plot(future_df['Date'], future_df[method], label="Forecast (2020–2027)", linestyle="--", color="orange")
    ax.plot(future_df['Date'], future_df[f"{method} Adjusted"], label="Adjusted Forecast", color="green")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consumption")
    ax.set_title(f"{method} Forecast with Adjustments")
    ax.legend()
    st.pyplot(fig)

for method in ['ETS', 'Moving Average', 'Auto Regression']:
    plot_forecast(method)

# MAPE evaluation on 2019
st.subheader("📈 MAPE (Model Accuracy for 2019 Test Set)")
for method, preds in test_forecasts.items():
    if len(preds) == len(test_df):
        mape = mean_absolute_percentage_error(test_df['Consumption'].values, preds)
        st.success(f"{method}: {mape:.2%}")
    else:
        st.warning(f"⚠️ {method}: MAPE could not be calculated (length mismatch)")

# Export forecasts
st.subheader("📥 Download Forecasts")
st.download_button(
    label="Download Forecast CSV",
    data=future_df.to_csv(index=False),
    file_name="health_forecasts_2020_2027.csv",
    mime="text/csv"
)

st.success("✅ Forecasting completed.")
st.markdown("Thank you for using the Health Forecasting app! If you have any questions or feedback, please reach out.")
