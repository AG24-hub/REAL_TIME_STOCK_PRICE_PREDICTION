import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import joblib
from tensorflow.keras.models import load_model

# ----- Page Config -----
st.set_page_config(page_title='Stock Price Prediction App', layout='wide')

# ----- Title -----
st.title("📈 Stock Price Prediction Dashboard")

# ----- date time -----
st.caption(f"📅 Today: {datetime.datetime.now().strftime('%A, %d %B %Y')}")

# ----- Sidebar Controls -----
stock_symbol = st.sidebar.selectbox("Choose a stock:", ["APPL", "RELIANCE.NS", "TCS.NS", "INFY.NS", "SBIN.NS", "ICICIBANK.NS"])
predict_days = st.sidebar.slider("Days to Predict", min_value=30, max_value=100, value=60, step=10)

# ----- Load Saved Model and Scaler -----
model = load_model("saved_models/stock_model.h5")
scaler = joblib.load("saved_models/scaler.pkl")

# ----- Function to Fetch Live Index Data -----
def get_live_index(ticker):
  try:
    df = yf.download(ticker, period="1d", interval="1m", progress=False)
    if df.empty: 
      st.warning(f"⚠️ No data found for {ticker}. It might be temporarily unavailable or delisted.")
      return None, None
    current = df['Close'].iloc[-1]
    change = ((df['Close'].iloc[-1] - df['Open'].iloc[0]) / df['Open'].iloc[0]) * 100
    return current, change
  except Exception as e:
    st.warning(f"⚠️ Error fetching data for {ticker}: {e}")
    return None, None

# ----- Display Live Indices -----
st.subheader("📊 Market Indices (Live)")
nifty, nifty_chg = get_live_index("^NSEI")
sensex, sensex_chg = get_live_index("^BSESN")

col1, col2 = st.columns(2)
if nifty:
    col1.metric("🟢 NIFTY 50", f"{nifty:.2f} ₹", f"{nifty_chg:.2f}%")
if sensex:
    col2.metric("🔵 SENSEX", f"{sensex:.2f} ₹", f"{sensex_chg:.2f}%")

# ----- Fetch Historical Data -----
st.subheader(f"📈 Historical Data for {stock_symbol}")
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=365*10)
data = yf.download(stock_symbol, start=start_date, end=end_date, auto_adjust=True)
if data.empty:
  st.error("❌ Failed to load stock data.")
  st.stop()

# ----- Prepare Data -----
df = data[['Close']].copy()
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
closing_price = df.values.reshape(-1, 1)
scaled_data = scaler.transform(closing_price)

#visualizing raw closing prices over the time
st.subheader("Closing price VS time chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df, label='Closing Price', color='blue')
plt.title(f'{stock_symbol} Stock price - Last 10 years')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.tight_layout()
plt.legend()
st.pyplot(fig)

#Zooming into last one year of prices
fig = plt.figure(figsize=(12,6))
plt.plot(df[-252:], label='Closing Price', color='blue')
plt.title(f'{stock_symbol} Stock Price - last one year')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.tight_layout()
plt.legend()
st.pyplot(fig)

#plotting moving averages
ma_100 = df.rolling(window=100).mean()
ma_200 = df.rolling(window=200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df, label='Closing Price', color='blue')
plt.plot(ma_100, label='Moving average of last 100 days', color='red')
plt.plot(ma_200, label='Moving average of last 200 days', color='green')
plt.title(f'{stock_symbol} Stock Price - Moving average')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.tight_layout()
st.pyplot(fig)

# ----- future prediction -----
with st.spinner("⏳ Predicting future stock prices..."):
  sequence_length = 60
  x_input = scaled_data[-sequence_length:]
  predictions = []

  for _ in range(predict_days):
    x = x_input.reshape(1, sequence_length, 1)
    pred = model.predict(x, verbose=0)
    predictions.append(pred[0])
    x_input = np.append(x_input[1:], pred, axis=0)

future_predictions_60_real = scaler.inverse_transform(predictions)

# ----- Create Prediction Date Range -----
last_date = data.index[-1]
future_dates = pd.bdate_range(start=last_date+pd.Timedelta(days=1), periods=60)

prediction_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted price': future_predictions_60_real.flatten()
})
prediction_df.set_index('Date', inplace=True)

# ----- Plot Predicted Prices -----
st.subheader("🔮 Predicted Future Prices")
st.line_chart(prediction_df)

# ----- Show Data Table and Download Option -----
st.subheader("📅 Prediction Table")
st.dataframe(prediction_df.tail(10))
st.download_button("Download Full Prediction as CSV", data=prediction_df.to_csv(), file_name="future_predictions.csv")

