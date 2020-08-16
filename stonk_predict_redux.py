import get_stonk
import model_fns
import numpy as np
from datetime import datetime as dt
from matplotlib import pyplot as plt
import tensorflow as tf
import streamlit as st


# formatting the Streamlit webpage
st.title("Stonk Predictions, by Lana Elauria")
st.write("DISCLAIMER: This is a personal project for me, please don't blindly take these predictions at face value.")
st.write("Input the stock code you'd like to analyze and how many days into the future you'd like to look, "
         "and this app will give a general prediction of the stock movements!")
st.write("This app takes data from Yahoo! Finance.")


class Stonk:
    def __init__(self, code, hist_date='01-01-2010'):
        # getting stock data from Yahoo! Finance
        self.code = code
        self.data = get_stonk.get_stonk(code, hist_date)

    def split_scale(self, smooth_window=500):
        # splitting stock data into train/test groups, scaling the data, and smoothing it
        self.train_scaled, self.test_scaled = model_fns.split_scale(self.data, smooth_window)

    def define_target(self, num_days):
        # defining inputs for LSTM model training & testing
        try:
            self.days_ahead = int(num_days)
        except:
            raise Exception('num_days must be an integer')
        self.x_train, self.y_train, self.x_test, self.y_test = model_fns.define_target(self.train_scaled,
                                                                                       self.test_scaled, num_days)

    def build_LSTM(self, num_layers, epochs, num_units=64, batch_size=512):
        # defining and building TensorFlow LSTM model. default 2 LSTM layers
        self.lstm = tf.keras.Sequential()
        for _ in range(num_layers - 1):
            self.lstm.add(tf.keras.layers.LSTM(units=num_units, return_sequences=True))
        self.lstm.add(tf.keras.layers.LSTM(units=num_units))
        self.lstm.add(tf.keras.layers.Dense(units=1))
        self.lstm.compile(loss='mse', optimizer='adam')

        # training LSTM model
        self.lstm.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def eval_LSTM(self):
        # evaluating LSTM model
        self.lstm.evaluate(self.x_test, self.y_test, verbose=1)

    def check_predictions(self):
        # checking the model's predictions; taking data from two months previous and predicting price movements for last month
        # made to compare to real-world data in the last month
        check_data = self.test_scaled.iloc[-2 * self.days_ahead:-self.days_ahead]
        check_data.drop(f'{self.days_ahead}_days', axis=1, inplace=True)
        x_check = np.reshape(check_data.values, (check_data.values.shape[0], check_data.values.shape[1], 1))
        check_pred = self.lstm.predict(x_check)
        check_pred = check_pred - (check_pred[0] - check_data.iloc[-1].Open)

        # plotting data fed into the model, the predictions from that data, and the real-world prices for the last month
        plt.figure(figsize=(10, 6))
        plt.subplot(111)
        plt.plot(self.test_scaled.iloc[-self.days_ahead:].Date.map(dt.fromordinal), check_pred, 'r--',
                 label='Predictions')
        plt.plot(self.test_scaled.iloc[-self.days_ahead:].Date.map(dt.fromordinal),
                 self.test_scaled.iloc[-self.days_ahead:].Open, 'b-', label='Actual prices')
        plt.plot(check_data.Date.map(dt.fromordinal), check_data.Open, 'g-', label='Data fed into model')
        plt.legend(loc=0, frameon=True)
        plt.xlabel('Date')
        plt.ylabel('Normalized Stock Price')
        plt.title(f'{self.code} Stock Movement Prediction Check')
        st.write("The graph below is made to check the predictions of the neural network made by this app. "
                 "Compare general trends, and decide for yourself if the app is accurate enough for your use.")
        st.pyplot()

    def predict(self):
        # predicting price movements in the future
        predict_data = self.test_scaled.iloc[-self.days_ahead:]
        predict_data.drop(f'{self.days_ahead}_days', axis=1, inplace=True)
        x_predict = np.reshape(predict_data.values, (predict_data.values.shape[0], predict_data.values.shape[1], 1))
        self.predictions = self.lstm.predict(x_predict)
        self.predictions = self.predictions - (self.predictions[0] - predict_data.iloc[-1].Open)

        # plotting data fed into the model, the predictions from that data, and the real-world prices for the last month
        pred_date = predict_data.iloc[-1].Date + self.days_ahead
        pred_range = np.arange(predict_data.iloc[-1].Date, pred_date, dtype=int).astype(dt)
        pred_dates = [dt.fromordinal(date) for date in pred_range]
        plt.figure(figsize=(10, 6))
        plt.subplot(111)
        plt.plot(pred_dates, self.predictions, 'r--', label='Predictions')
        plt.plot(predict_data.Date.map(dt.fromordinal), predict_data.Open, 'g-', label='Data fed into model')
        plt.legend(loc=0, frameon=True)
        plt.xlabel('Date')
        plt.ylabel('Normalized Stock Price')
        plt.title(f'{self.code} Stock Movement Prediction')
        st.pyplot()
        st.write("NOTE: The y-axis for these graphs is the normalized stock price that the model works with. "
         "The stock market is extremely unpredictable, and this model won't be able to predict anything more than "
         "general trends in stock price movements. On the same token, be extremely wary of sharp spikes in the "
         "stock market predicted by this app. It extrapolates past behavior, and if there has been a spike recently, "
         "the model will believe there will be a spike later on as well, which is usually not the case.")


stonk_code= st.sidebar.text_input("What stock would you like to analyze? Default is AMC.", "AMC")
stonks = Stonk(stonk_code)
if st.sidebar.checkbox('Show stock data'):
    st.write(f'{stonk_code} stock data:\n', stonks.data)
stonks.split_scale()
days_ahead = st.sidebar.text_input("How many days into the future would you like to predict? Default is 30.", 30)
try:
    days_ahead = int(days_ahead)
except:
    raise Exception('You must input an integer.')
stonks.define_target(num_days=days_ahead)
stonks.build_LSTM(num_layers=2,epochs=5)
stonks.eval_LSTM()
stonks.check_predictions()
stonks.predict()
