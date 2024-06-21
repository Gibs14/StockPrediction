import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.stats import norm
import yfinance as yf
import matplotlib.pyplot as plt
import ta_py as ta
import streamlit as st

# COntoh simulasi GBM dengan parameter drift dan volatility konstan

# CONTOH GBM konstant
# Mengunduh data historis harga saham dari Yahoo Finance
data = yf.download("BBCA.JK", start="2022-01-01", end="2023-12-31")

# Membagi data menjadi dua bagian: data historis dan data untuk simulasi
n = int(len(data)/2)
histo = data.iloc[:n]

# Menghitung return harian
histo['return'] = [None]+[(i-j)/j for (i,j) in zip(histo['Close'].iloc[1:],histo['Close'].iloc[0:-1])]
histo = histo.dropna()

# Menghitung rata-rata dan standar deviasi dari return
mu = histo['return'].mean()
sig = histo['return'].std()

# Menyiapkan data untuk simulasi
M = len(data[n:])
price =[data['Close'].iloc[n-1]]
price2 = price.copy()
dt = 1

# Melakukan simulasi GBM
for i in range(M):
  eps = np.random.normal(scale=1)
  price.append(price[-1]*np.exp(mu*dt+sig*eps))
  price2.append(price[-1]*(1.+mu*dt+sig*np.sqrt(dt)*eps))

# Membuat plot perbandingan antara nilai aktual dan nilai yang diprediksi oleh model
plt.plot(data['Close'].iloc[n-1:], label='aktual')
plt.plot(data.index[n-1:],price2, label='model kontinyu')
plt.plot(data.index[n-1:],price, label='model diskrit')
plt.title('comparison between actual and predicted value of two models')
plt.legend()
plt.show()

# Membuat plot kesalahan absolut antara nilai aktual dan nilai yang diprediksi oleh model
actual = data['Close'].iloc[n-1:]
index = data.index[n-1:]
plt.title('Absolute error of continyu and discrete form GBM')
plt.plot(index, [abs(i-j) for (i,j) in zip(actual, price)], label ='model diskrit')
plt.hlines(y=np.mean([abs(i-j) for (i,j) in zip(actual, price)]), xmin=data.index[n-1:][0], xmax=data.index[n-1:][-1], label='MAE discrete')
plt.plot(index, [abs(i-j) for (i,j) in zip(actual, price2)], label ='model kontinyu')
plt.hlines(y=np.mean([abs(i-j) for (i,j) in zip(actual, price2)]), xmin=data.index[n-1:][0], xmax=data.index[n-1:][-1], label='MAE kontinyu', color='r')
plt.annotate("$\sigma$_discrit={:.2f}, $\sigma$_kontinyu={:.2f}".format(np.std(price), np.std(price2)), (index[int(len(index)/5)],1500))
plt.legend()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SImulation GBM for non constant drift and volatility rate ( σ )
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

scaler = MinMaxScaler()
period = 5 # length of TA indicator

# 1. Data Loading and Preprocessing
def load_data(start_year, end_year):
  # Mengunduh data historis harga saham dari Yahoo Finance
  data = yf.download("BBCA.JK", start=f"{start_year}-01-01", end=f"{end_year}-12-31")
  
  # Menghitung return harian
  data['return']=data['Adj Close'].pct_change(period).dropna()
  
  # Menghitung indikator teknikal
  data['sma']=[None for i in range(period-1)]+list(ta.sma(data['Adj Close'],period))
  data['ema']=[None for i in range(period-1)]+list(ta.ema(data['Adj Close'],period))
  data['rsi']=[None for i in range(period-1)]+list(ta.rsi(data['Adj Close'],period))
  
  # Menghitung volatilitas historis
  data['volatility'] = data['Adj Close'].rolling(window=period).std() * np.sqrt(252)
  
  data = data.dropna()
  
  # Menentukan fitur yang akan digunakan dalam model
  features = ['sma', 'ema', 'rsi', 'volatility']
  
  # Melakukan scaling pada fitur
  data[features] = scaler.fit_transform(data[features])
  
  return data, features

# 2. Machine Learning Model Training
def train_model(data, features, step, target):
  # Membuat data training
  X = list(data[features].iloc[:step].values)
  y = list(data[target].iloc[:step].values)
  
  # Melatih model RandomForestRegressor
  model = RandomForestRegressor()
  model.fit(X, y)
  
  return model

# 3. GBM Simulation with Machine Learning Predicted Drift and Volatility
def gbm_sim(spot_price, time_horizon, steps, drift_model, vol_model, features, data):
  dt=1
  actual = [spot_price]+list(data['Adj Close'].iloc[:].values)
  
  # Memprediksi drift dan volatilitas dengan model machine learning
  drift = drift_model.predict(scaler.fit_transform(data.loc[:][features]))
  volatility = vol_model.predict(scaler.fit_transform(data.loc[:][features]))
  
  paths = []
  paths.append(spot_price)
  
  # Melakukan simulasi GBM dengan drift dan volatilitas yang diprediksi
  for i in range(1, len(data)):
    paths.append(actual[i] * np.exp((drift[i-1] - 0.5 * (volatility[i-1]/252)**2) * dt + (volatility[i-1]/252) * np.random.normal(scale=np.sqrt(1/252))))
  
  return paths,drift

# 4. Main Function
if __name__ == "__main__":
    # Streamlit user interface
    st.title("Stock Price Prediction")
    start_year = st.slider("Enter start year", min_value=2004, max_value=2024, value=2023, step=1)
    end_year = st.slider("Enter end year", min_value=2004, max_value=2024, value=2024, step=1)
    st.markdown("Please enter the number of simulations. The minimum value is 1 and the maximum value is 10000.")
    simulations = st.number_input("", min_value=1, max_value=10000, value=1000, step=1)

    if start_year > end_year:
        st.error("End year should be greater than start year.")
    else:
        # Memuat data dan melatih model
        data, features = load_data(start_year, end_year)
        steps = int(len(data)/2)
        drift_model = train_model(data, features, steps, "return")
        vol_model = train_model(data, features, steps, "volatility")
        spot_price = data["Adj Close"].iloc[steps-1]
        time_horizon = len(data)-steps

        # Melakukan simulasi GBM
        simulated_paths,drifts = gbm_sim(spot_price, time_horizon, steps, drift_model, vol_model, features, data.iloc[steps:])
        
        # Monte Carlo simulation
        simulated_prices = np.zeros((simulations, len(data)-steps))
        for i in range(simulations):
          simulated_prices[i], _ = gbm_sim(spot_price, time_horizon, steps, drift_model, vol_model, features, data.iloc[steps:])
        confidence_interval = np.percentile(simulated_prices, [2.5, 97.5], axis=0)

        # Choose display format
        display_format = st.radio("Choose display format", ("Table", "Chart"))

        if display_format == "Table":
            st.dataframe(pd.DataFrame(simulated_paths, index=data.index[steps:], columns=["Simulated Price"]))
            st.dataframe(pd.DataFrame(confidence_interval.T, index=data.index[steps:], columns=["Lower Bound", "Upper Bound"]))
        else:
            st.line_chart(pd.DataFrame(simulated_paths, index=data.index[steps:], columns=["Simulated Price"]))
            st.line_chart(pd.DataFrame(confidence_interval.T, index=data.index[steps:], columns=["Lower Bound", "Upper Bound"]))
