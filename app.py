import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Silver Price Predictor",
    page_icon="🪙",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Title */
.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: white;
    margin-top: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #d9f7ff;
    margin-bottom: 25px;
}

/* White Box */
.box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
}

/* KPI Card */
.kpi-card {
    background: linear-gradient(135deg, #004d4d, #008080);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.3);
}

/* KPI Title */
.kpi-title {
    font-size: 18px;
    font-weight: bold;
}

/* KPI Value */
.kpi-value {
    font-size: 30px;
    margin-top: 10px;
    font-weight: bold;
}

/* Button */
.stButton>button {
    background-color: #00b894;
    color: white;
    border-radius: 10px;
    padding: 12px 25px;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #00cec9;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.markdown(
    '<div class="title">🪙 AI Silver Price Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Live Silver Market Analysis & Future Forecast using LSTM</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Dashboard Controls")

years = st.sidebar.slider(
    "Select Historical Data (Years)",
    1,
    10,
    5
)

prediction_days = st.sidebar.selectbox(
    "Select Future Prediction Days",
    [10, 30, 60, 90, 120]
)

# =========================================================
# FUNCTION TO FETCH SILVER DATA FROM YAHOO FINANCE
# =========================================================

def load_silver_data(years):

    import time

    ticker = "SI=F"

    for attempt in range(3):

        try:

            df = yf.download(
                ticker,
                period=f"{years}y",
                interval="1d",
                progress=False,
                auto_adjust=True
            )

            if not df.empty:

                df.reset_index(inplace=True)

                return df

        except Exception:

            time.sleep(5)

    return pd.DataFrame()

# =========================================================
# LOAD LIVE SILVER DATA
# =========================================================

st.info("Fetching LIVE silver market data from Yahoo Finance...")

df = load_silver_data(years)


if df.empty:

    st.warning(
        "⚠️ Yahoo Finance API is temporarily unavailable. Loading backup dataset..."
    )

    df = pd.read_csv("silver_data.csv")

else:

    st.success("✅ Live Yahoo Finance data loaded successfully!")

    
# =========================================================
# FIX MULTIINDEX
# =========================================================
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# =========================================================
# CLEAN DATA
# =========================================================
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

df.dropna(inplace=True)

if len(df) < 60:
    st.error("❌ Not enough data for prediction.")
    st.stop()

# =========================================================
# MOVING AVERAGES
# =========================================================
df['MA20'] = df['Close'].rolling(20).mean()
df['MA50'] = df['Close'].rolling(50).mean()

# =========================================================
# KPI VALUES
# =========================================================
current_price = float(df['Close'].iloc[-1])
highest_price = float(df['High'].max())
lowest_price = float(df['Low'].min())
average_price = float(df['Close'].mean())

today = float(df['Close'].iloc[-1])
yesterday = float(df['Close'].iloc[-2])

change = today - yesterday
percent_change = (change / yesterday) * 100

# =========================================================
# KPI DASHBOARD
# =========================================================
st.markdown(
    '<div class="box"><h2>📊 Silver Market Dashboard</h2></div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">📍 Current Price</div>
        <div class="kpi-value">$ {current_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">📈 Highest Price</div>
        <div class="kpi-value">$ {highest_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">📉 Lowest Price</div>
        <div class="kpi-value">$ {lowest_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">📊 Average Price</div>
        <div class="kpi-value">$ {average_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:

    arrow = "🔺" if change > 0 else "🔻"

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">📈 Daily Change</div>
        <div class="kpi-value">{arrow} {percent_change:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# DATA PREVIEW
# =========================================================
st.markdown(
    '<div class="box"><h3>📋 Live Silver Dataset</h3></div>',
    unsafe_allow_html=True
)

st.dataframe(df.tail())

# =========================================================
# MOVING AVERAGE GRAPH
# =========================================================
st.markdown(
    '<div class="box"><h3>📉 Moving Average Analysis</h3></div>',
    unsafe_allow_html=True
)

fig, ax = plt.subplots(figsize=(14,6))

ax.plot(df['Date'], df['Close'], label='Close Price')
ax.plot(df['Date'], df['MA20'], label='20-Day MA')
ax.plot(df['Date'], df['MA50'], label='50-Day MA')

ax.set_title("Silver Price with Moving Averages")
ax.set_xlabel("Date")
ax.set_ylabel("Price")

ax.legend()

plt.xticks(rotation=45)

st.pyplot(fig)

# =========================================================
# CANDLESTICK CHART
# =========================================================
st.markdown(
    '<div class="box"><h3>📊 Candlestick Chart</h3></div>',
    unsafe_allow_html=True
)

candle_df = df.copy()

candle_df.set_index('Date', inplace=True)

fig2, axlist = mpf.plot(
    candle_df,
    type='candle',
    style='yahoo',
    mav=(20,50),
    volume=False,
    figsize=(14,7),
    returnfig=True
)

st.pyplot(fig2)

# =========================================================
# PREDICT BUTTON
# =========================================================
if st.button("🚀 Predict Future Silver Prices"):

    st.info("Training AI Model... Please wait ⏳")

    # =====================================================
    # DATA PREPARATION
    # =====================================================
    data = df[['Close']]

    scaler = MinMaxScaler(feature_range=(0,1))

    scaled_data = scaler.fit_transform(data)

    X = []
    y = []

    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])

    X = np.array(X)
    y = np.array(y)

    # =====================================================
    # LSTM MODEL
    # =====================================================
    model = Sequential()

    model.add(
        LSTM(
            64,
            return_sequences=True,
            input_shape=(60,1)
        )
    )

    model.add(LSTM(64))

    model.add(Dense(25))

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    # =====================================================
    # TRAIN MODEL
    # =====================================================
    model.fit(
        X,
        y,
        epochs=10,
        batch_size=32,
        verbose=0
    )

    st.success("✅ AI Model Trained Successfully!")

    # =====================================================
    # FUTURE PREDICTION
    # =====================================================
    last_60_days = scaled_data[-60:]

    current_batch = last_60_days.reshape(1,60,1)

    future_predictions = []

    for i in range(prediction_days):

        pred = model.predict(
            current_batch,
            verbose=0
        )[0][0]

        future_predictions.append(pred)

        current_batch = np.append(
            current_batch[:,1:,:],
            [[[pred]]],
            axis=1
        )

    # =====================================================
    # CONVERT BACK
    # =====================================================
    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1,1)
    )

    # =====================================================
    # FUTURE DATES
    # =====================================================
    future_dates = pd.date_range(
        start=df['Date'].iloc[-1],
        periods=prediction_days + 1
    )[1:]

    # =====================================================
    # PREDICTION DATAFRAME
    # =====================================================
    prediction_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": future_predictions.flatten()
    })

    # =====================================================
    # FUTURE GRAPH
    # =====================================================
    st.markdown(
        f'<div class="box"><h3>🔮 Future Prediction ({prediction_days} Days)</h3></div>',
        unsafe_allow_html=True
    )

    fig3, ax3 = plt.subplots(figsize=(14,6))

    ax3.plot(
        prediction_df['Date'],
        prediction_df['Predicted Price'],
        marker='o'
    )

    ax3.set_title("Future Silver Price Prediction")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Predicted Price")

    plt.xticks(rotation=45)

    st.pyplot(fig3)

    # =====================================================
# FUTURE KPI
# =====================================================
    final_price = float(
       prediction_df['Predicted Price'].iloc[-1]
    )

    st.markdown(
     f"""
      <div style="
        background: linear-gradient(135deg, #004d4d, #008080);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0px 5px 15px rgba(0,0,0,0.3);
      ">
        📈 Predicted Price After {prediction_days} Days
        <br><br>
        <span style="font-size:40px;">
            $ {final_price:.2f}
        </span>
      </div>
      """,
      unsafe_allow_html=True
    )
    # =====================================================
    # AI RECOMMENDATION
    # =====================================================
    st.markdown(
        '<div class="box"><h3>🤖 AI Recommendation</h3></div>',
        unsafe_allow_html=True
    )

    if final_price > current_price:
        st.success("✅ AI Recommendation: BUY")

    elif final_price < current_price:
        st.error("❌ AI Recommendation: SELL")

    else:
        st.warning("⚠️ AI Recommendation: HOLD")

    # =====================================================
    # PREDICTION TABLE
    # =====================================================
    st.markdown(
        '<div class="box"><h3>📋 Future Prediction Table</h3></div>',
        unsafe_allow_html=True
    )

    st.dataframe(prediction_df)

    # =====================================================
    # DOWNLOAD CSV
    # =====================================================
    csv = prediction_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction CSV",
        data=csv,
        file_name="silver_predictions.csv",
        mime="text/csv"
    )