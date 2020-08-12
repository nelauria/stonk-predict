import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timezone
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import tensorflow as tf
import streamlit as st


# function to get stonk data
def get_stonk(stonk_code,hist_date='01-01-2010'):
    today = dt.today()
    
    # converting dates to Unix timestamps
    unix_hist = dt.strptime(hist_date,'%m-%d-%Y').replace(tzinfo=timezone.utc).timestamp()
    unix_today = today.replace(tzinfo=timezone.utc).timestamp()
    
    # getting stonk history from Yahoo! Finance
    link = (f'https://query1.finance.yahoo.com/v7/finance/download/'
            f'{stonk_code}?period1={int(unix_hist)}&period2={int(unix_today)}&interval=1d&events=history')
    b = requests.get(link)
    s = str(b.content,'utf-8')
    s_io = StringIO(s)
    stonks = pd.read_csv(s_io,encoding='utf-8',parse_dates=['Date'])
    
    # converting Date column to numeric data
    try:
        stonks['Date'] = stonks['Date'].map(dt.toordinal)
    except:
        pass
    return stonks


# function for train/test splitting, scaling, and smoothing
def split_scale(stonk_df,smooth_window,EMA_interval=21):
    # splitting data
    train, test = train_test_split(stonk_df.dropna(),test_size=0.2,shuffle=False)
    try:
        train_date = train.pop('Date')
    except:
        pass
    try:
        test_date = test.pop('Date')
        test_date.reset_index(drop=True,inplace=True)
    except:
        pass
    
    # scaling data
    scaler = MinMaxScaler()
    train_scaled = pd.DataFrame()
    test_scaled = pd.DataFrame()
    for feat in train.columns:
        train_feat = []
        for di in range(0,len(train)-1,smooth_window):
            scaler.fit(train.loc[di:di+smooth_window-1,feat].values.reshape(-1,1))
            scaled_feat_train = scaler.transform(train.loc[di:di+smooth_window-1,feat].values.reshape(-1,1))
            train_feat = np.concatenate((train_feat,scaled_feat_train.reshape(-1)))
        train_scaled[feat] = train_feat
        scaled_feat_test = scaler.transform(test.loc[:,feat].values.reshape(-1,1))
        test_scaled[feat] = scaled_feat_test.reshape(-1)
    
    # smoothing training data using exponential moving average (EMA)
    mult = 2/(EMA_interval+1)
    EMA = 0.0
    for ind in range(len(train_scaled)):
        EMA = train_scaled.iloc[ind]*mult + EMA*(1-mult)
        train_scaled.iloc[ind] = EMA
    train_scaled['Date'] = train_date
    test_scaled['Date'] = test_date
    test_scaled.set_index(test_scaled.index+train_scaled.index[-1]+1,inplace=True)
    return train_scaled, test_scaled


st.title("Stonk Predictions, by Lana Elauria")
st.write("DISCLAIMER: This is a personal project for me, please don't blindly take these predictions at face value.")
st.write("Input the stock code you'd like to analyze and how many days into the future you'd like to look, "
         "and this app will give a general prediction of the stock movements!")
st.write("This app takes data from Yahoo! Finance.")


stonk_code = st.sidebar.text_input("What stock would you like to analyze? Default is AMC.","AMC")
stonks = get_stonk(stonk_code)
if st.sidebar.checkbox('Show stock data'):
    st.write(f'{stonk_code} stock data:\n',stonks)


train_scaled, test_scaled_full = split_scale(stonks,smooth_window=500)
# train_scaled
# test_scaled_full


# defining how far ahead user wants to predict
days_ahead = st.sidebar.text_input("How many days into the future would you like to predict? Default is 30.",30)
days_ahead = int(days_ahead)
dfs = [train_scaled,test_scaled_full]
for i in [0,1]:
    df = dfs[i]
    for ind in df.index:
        try:
            df.loc[ind,f'{days_ahead}_days'] = df.loc[ind+days_ahead,'Open']
        except:
            pass
train_scaled.dropna(inplace=True)
test_scaled = test_scaled_full.dropna()


# defining inputs & outputs
x_train = train_scaled.drop(f'{days_ahead}_days',axis=1)
y_train = train_scaled.loc[:,f'{days_ahead}_days']
x_test = test_scaled.drop(f'{days_ahead}_days',axis=1)
y_test = test_scaled.loc[:,f'{days_ahead}_days']

# reshaping inputs & outputs for LSTM model
x_train = np.reshape(x_train.values,(x_train.values.shape[0],x_train.values.shape[1],1))
x_test = np.reshape(x_test.values,(x_test.values.shape[0],x_test.values.shape[1],1))
y_train, y_test = y_train.values, y_test.values


# defining and building TensorFlow LSTM model
model_lstm = tf.keras.Sequential()
# model_lstm.add(tf.keras.layers.LSTM(units=64,return_sequences=True))
model_lstm.add(tf.keras.layers.LSTM(units=64,return_sequences=True))
model_lstm.add(tf.keras.layers.LSTM(units=64))
model_lstm.add(tf.keras.layers.Dense(units=1))
model_lstm.compile(loss='mse',optimizer='adam')


# training model
model_lstm.fit(x_train,y_train,epochs=5,batch_size=512,verbose=1)


# evaluating model
model_lstm.evaluate(x_test,y_test,verbose=1)


# checking the model's predictions; taking data from two months previous and predicting price movements for last month
# made to compare to real-world data in the last month
st.write("The graph below is made to check the predictions of the neural network made by this app. "
         "Compare general trends, and decide for yourself if the app is accurate enough for your use.")
check_data = test_scaled.iloc[-2*days_ahead:-days_ahead]
check_data.drop(f'{days_ahead}_days',axis=1,inplace=True)
x_check = np.reshape(check_data.values,(check_data.values.shape[0],check_data.values.shape[1],1))
check_pred = model_lstm.predict(x_check)
check_pred = check_pred - (check_pred[0]-check_data.iloc[-1].Open)


# plotting data fed into the model, the predictions from that data, and the real-world prices for the last month
plt.figure(figsize=(10,6))
plt.subplot(111)
plt.plot(test_scaled.iloc[-days_ahead:].Date.map(dt.fromordinal),check_pred,'r--',label='Predictions')
plt.plot(test_scaled.iloc[-days_ahead:].Date.map(dt.fromordinal),
         test_scaled.iloc[-days_ahead:].Open,'b-',label='Actual prices')
plt.plot(check_data.Date.map(dt.fromordinal),check_data.Open,'g-',label='Data fed into model')
plt.legend(loc=0,frameon=True)
plt.xlabel('Date')
plt.ylabel('Normalized Stock Price')
plt.title(f'{stonk_code} Stock Movement Prediction Check')
st.pyplot()


# actually predicting price movements in the future
predict_data = test_scaled_full.iloc[-days_ahead:]
predict_data.drop(f'{days_ahead}_days',axis=1,inplace=True)
x_predict = np.reshape(predict_data.values,(predict_data.values.shape[0],predict_data.values.shape[1],1))
predictions = model_lstm.predict(x_predict)
predictions = predictions - (predictions[0]-predict_data.iloc[-1].Open)


# plotting data fed into the model, the predictions from that data, and the real-world prices for the last month
pred_date = predict_data.iloc[-1].Date+days_ahead
pred_range = np.arange(predict_data.iloc[-1].Date,pred_date,dtype=int).astype(dt)
pred_dates = [dt.fromordinal(date) for date in pred_range]
plt.figure(figsize=(10,6))
plt.subplot(111)
plt.plot(pred_dates,predictions,'r--',label='Predictions')
plt.plot(predict_data.Date.map(dt.fromordinal),predict_data.Open,'g-',label='Data fed into model')
plt.legend(loc=0,frameon=True)
plt.xlabel('Date')
plt.ylabel('Normalized Stock Price')
plt.title(f'{stonk_code} Stock Movement Prediction')
st.pyplot()


st.write("NOTE: The y-axis for these graphs is the normalized stock price that the model works with. "
         "The stock market is extremely unpredictable, and this model won't be able to predict anything more than "
         "general trends in stock price movements. On the same token, be extremely wary of sharp spikes in the "
         "stock market predicted by this app. It extrapolates past behavior, and if there has been a spike recently, "
         "the model will believe there will be a spike later on as well, which is usually not the case.")
