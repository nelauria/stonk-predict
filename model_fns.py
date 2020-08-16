import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# function for train/test splitting, scaling, and smoothing
def split_scale(stonk_df, smooth_window, EMA_interval=21):
    # splitting data
    train, test = train_test_split(stonk_df.dropna(), test_size=0.2, shuffle=False)
    train_date = train.pop('Date')
    test_date = test.pop('Date')
    test_date.reset_index(drop=True, inplace=True)

    # scaling data
    scaler = MinMaxScaler()
    train_scaled = pd.DataFrame()
    test_scaled = pd.DataFrame()
    for feat in train.columns:
        train_feat = []
        for di in range(0, len(train) - 1, smooth_window):
            scaler.fit(train.loc[di:di + smooth_window - 1, feat].values.reshape(-1, 1))
            scaled_feat_train = scaler.transform(train.loc[di:di + smooth_window - 1, feat].values.reshape(-1, 1))
            train_feat = np.concatenate((train_feat, scaled_feat_train.reshape(-1)))
        train_scaled[feat] = train_feat
        scaled_feat_test = scaler.transform(test.loc[:, feat].values.reshape(-1, 1))
        test_scaled[feat] = scaled_feat_test.reshape(-1)

    # smoothing training data using exponential moving average (EMA)
    mult = 2 / (EMA_interval + 1)
    EMA = 0.0
    for ind in range(len(train_scaled)):
        EMA = train_scaled.iloc[ind] * mult + EMA * (1 - mult)
        train_scaled.iloc[ind] = EMA
    train_scaled['Date'] = train_date
    test_scaled['Date'] = test_date
    test_scaled.set_index(test_scaled.index + train_scaled.index[-1] + 1, inplace=True)
    return train_scaled, test_scaled

def define_target(train_df,test_df,num_days):
    # making a column in each DataFrame with the Open from {num_days} ahead
    dfs = [train_df,test_df]
    for i in range(len(dfs)):
        df = dfs[i]
        for ind in df.index:
            try:
                df.loc[ind, f'{num_days}_days'] = df.loc[ind + num_days, 'Open']
            except:
                pass


    # defining inputs & outputs
    x_train = train_df.dropna().drop(f'{num_days}_days', axis=1)
    y_train = train_df.dropna().loc[:, f'{num_days}_days']
    x_test = test_df.dropna().drop(f'{num_days}_days', axis=1)
    y_test = test_df.dropna().loc[:, f'{num_days}_days']


    # reshaping inputs & outputs for LSTM model
    x_train = np.reshape(x_train.values, (x_train.values.shape[0], x_train.values.shape[1], 1))
    x_test = np.reshape(x_test.values, (x_test.values.shape[0], x_test.values.shape[1], 1))
    y_train, y_test = y_train.values, y_test.values

    return x_train, y_train, x_test, y_test
