import numpy as np
import model_fns
from datetime import datetime as dt
import get_stonk
from matplotlib import pyplot as plt
import tensorflow as tf
import streamlit as st


# function to get stonk data



# formatting the Streamlit webpage
st.title("Stonk Predictions, by Lana Elauria")
st.write("DISCLAIMER: This is a personal project for me, please don't blindly take these predictions at face value.")
st.write("Input the stock code you'd like to analyze and how many days into the future you'd like to look, "
         "and this app will give a general prediction of the stock movements!")
st.write("This app takes data from Yahoo! Finance.")

stonk_code = st.sidebar.text_input("What stock would you like to analyze? Default is AMC.", "AMC")
if st.sidebar.checkbox('Show stock data'):
    st.write(f'{stonk_code} stock data:\n', stonks)
days_ahead = st.sidebar.text_input("How many days into the future would you like to predict? Default is 30.", 30)
try:
    days_ahead = int(days_ahead)
except:
    raise Exception('You must input an integer.')

st.write("The graph below is made to check the predictions of the neural network made by this app. "
         "Compare general trends, and decide for yourself if the app is accurate enough for your use.")

stonks = get_stonk.get_stonk(stonk_code)
train_scaled, test_scaled_full = model_fns.split_scale(stonks, smooth_window=500)
# train_scaled
# test_scaled_full


# defining how far ahead user wants to predict
dfs = [train_scaled, test_scaled_full]
for i in [0, 1]:
    df = dfs[i]
    for ind in df.index:
        try:
            df.loc[ind, f'{days_ahead}_days'] = df.loc[ind + days_ahead, 'Open']
        except:
            pass
train_scaled.dropna(inplace=True)
test_scaled = test_scaled_full.dropna()

# defining inputs & outputs
x_train = train_scaled.drop(f'{days_ahead}_days', axis=1)
y_train = train_scaled.loc[:, f'{days_ahead}_days']
x_test = test_scaled.drop(f'{days_ahead}_days', axis=1)
y_test = test_scaled.loc[:, f'{days_ahead}_days']

# reshaping inputs & outputs for LSTM model
x_train = np.reshape(x_train.values, (x_train.values.shape[0], x_train.values.shape[1], 1))
x_test = np.reshape(x_test.values, (x_test.values.shape[0], x_test.values.shape[1], 1))
y_train, y_test = y_train.values, y_test.values

# defining and building TensorFlow LSTM model
model_lstm = tf.keras.Sequential()
# model_lstm.add(tf.keras.layers.LSTM(units=64,return_sequences=True))
model_lstm.add(tf.keras.layers.LSTM(units=64, return_sequences=True))
model_lstm.add(tf.keras.layers.LSTM(units=64))
model_lstm.add(tf.keras.layers.Dense(units=1))
model_lstm.compile(loss='mse', optimizer='adam')

# training model
model_lstm.fit(x_train, y_train, epochs=5, batch_size=512, verbose=1)

# evaluating model
model_lstm.evaluate(x_test, y_test, verbose=1)

# checking the model's predictions; taking data from two months previous and predicting price movements for last month
# made to compare to real-world data in the last month
check_data = test_scaled.iloc[-2 * days_ahead:-days_ahead]
check_data.drop(f'{days_ahead}_days', axis=1, inplace=True)
x_check = np.reshape(check_data.values, (check_data.values.shape[0], check_data.values.shape[1], 1))
check_pred = model_lstm.predict(x_check)
check_pred = check_pred - (check_pred[0] - check_data.iloc[-1].Open)

# plotting data fed into the model, the predictions from that data, and the real-world prices for the last month
plt.figure(figsize=(10, 6))
plt.subplot(111)
plt.plot(test_scaled.iloc[-days_ahead:].Date.map(dt.fromordinal), check_pred, 'r--', label='Predictions')
plt.plot(test_scaled.iloc[-days_ahead:].Date.map(dt.fromordinal),
         test_scaled.iloc[-days_ahead:].Open, 'b-', label='Actual prices')
plt.plot(check_data.Date.map(dt.fromordinal), check_data.Open, 'g-', label='Data fed into model')
plt.legend(loc=0, frameon=True)
plt.xlabel('Date')
plt.ylabel('Normalized Stock Price')
plt.title(f'{stonk_code} Stock Movement Prediction Check')
st.pyplot()

# actually predicting price movements in the future
predict_data = test_scaled_full.iloc[-days_ahead:]
predict_data.drop(f'{days_ahead}_days', axis=1, inplace=True)
x_predict = np.reshape(predict_data.values, (predict_data.values.shape[0], predict_data.values.shape[1], 1))
predictions = model_lstm.predict(x_predict)
predictions = predictions - (predictions[0] - predict_data.iloc[-1].Open)

# plotting data fed into the model, the predictions from that data, and the real-world prices for the last month
pred_date = predict_data.iloc[-1].Date + days_ahead
pred_range = np.arange(predict_data.iloc[-1].Date, pred_date, dtype=int).astype(dt)
pred_dates = [dt.fromordinal(date) for date in pred_range]
plt.figure(figsize=(10, 6))
plt.subplot(111)
plt.plot(pred_dates, predictions, 'r--', label='Predictions')
plt.plot(predict_data.Date.map(dt.fromordinal), predict_data.Open, 'g-', label='Data fed into model')
plt.legend(loc=0, frameon=True)
plt.xlabel('Date')
plt.ylabel('Normalized Stock Price')
plt.title(f'{stonk_code} Stock Movement Prediction')
st.pyplot()

st.write("NOTE: The y-axis for these graphs is the normalized stock price that the model works with. "
         "The stock market is extremely unpredictable, and this model won't be able to predict anything more than "
         "general trends in stock price movements. On the same token, be extremely wary of sharp spikes in the "
         "stock market predicted by this app. It extrapolates past behavior, and if there has been a spike recently, "
         "the model will believe there will be a spike later on as well, which is usually not the case.")
