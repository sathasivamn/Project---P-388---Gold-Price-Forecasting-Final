
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

@st.cache
def load_data():
    data = pd.read_csv("Gold_data.csv")
    data['date'] = pd.to_datetime(data['date'])
    data.set_index("date", inplace=True)
    data.sort_index(inplace=True)
    return data

gold_df = load_data()

st.title("Gold Price Prediction Dashboard")
st.write("This app allows you to choose a model and forecast gold prices.")

# Sidebar for model selection and future forecast periods
model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "ARIMA", "SARIMA", "Prophet"])
future_periods = st.sidebar.number_input("Number of days to forecast", min_value=1, max_value=365, value=30)

st.subheader("Data Overview")
st.write(gold_df.head())

# Display time series plot
st.subheader("Time Series Plot")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(gold_df.index, gold_df['price'], color='gold', label='Gold Price')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Define function for each model prediction

def linear_regression_model(data, forecast_periods):
    data['time_ordinal'] = data.index.map(pd.Timestamp.toordinal)
    X = data[['time_ordinal']]
    y = data['price']
    model = LinearRegression().fit(X, y)
    # Forecast future
    last_date = data.index[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_periods+1)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    forecast = model.predict(future_ordinals)
    return future_dates, forecast

def arima_model_forecast(data, order, forecast_periods):
    train = data['price']
    model = sm.tsa.ARIMA(train, order=order).fit()
    forecast = model.forecast(steps=forecast_periods)
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_periods)
    return future_dates, forecast

def prophet_model_forecast(data, forecast_periods):
    df_prophet = data[['price']].reset_index().rename(columns={'date': 'ds', 'price': 'y'})
    model = Prophet(changepoint_prior_scale=0.05, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(forecast_periods)['ds'], forecast[['ds', 'yhat']].tail(forecast_periods)['yhat']

# Display forecast based on selected model
if model_choice == "Linear Regression":
    forecast_dates, forecast_values = linear_regression_model(gold_df.copy(), future_periods)
elif model_choice == "ARIMA":
    # For demo purposes, using a preset order; in production, use hyperparameter tuning.
    forecast_dates, forecast_values = arima_model_forecast(gold_df.copy(), order=(1,1,1), forecast_periods=future_periods)
elif model_choice == "SARIMA":
    # For demo purposes, using a preset order; adjust the seasonal order as needed.
    model = sm.tsa.statespace.SARIMAX(gold_df['price'], order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
    forecast_values = model.forecast(steps=future_periods)
    forecast_dates = pd.date_range(start=gold_df.index[-1] + pd.Timedelta(days=1), periods=future_periods)
elif model_choice == "Prophet":
    forecast_dates, forecast_values = prophet_model_forecast(gold_df.copy().reset_index().rename(columns={'index': 'date'}), future_periods)

# Plot forecast results
st.subheader(f"Forecast using {model_choice}")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(gold_df.index, gold_df['price'], label="Historical", color='gold')
ax2.plot(forecast_dates, forecast_values, label="Forecast", color='green', marker='o')
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)
