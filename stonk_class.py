import get_stonk
import model_fns
import numpy as np
from datetime import datetime as dt
from matplotlib import pyplot as plt
import tensorflow as tf

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

