import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


COMPANY = 'META'
START = dt.datetime(2012, 1, 1)
END = dt.datetime(2020, 1, 1)

TEST_START = END
TEST_END = dt.datetime.now()


scaler = MinMaxScaler(feature_range=(0, 1))


def get_data(company: str, start: dt.datetime, end: dt.datetime) -> tuple:
    data = yf.download(company, start=start, end=end)
    date = data.index.values[1:]
    data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    x = data[:-1]
    y = data[1:]
    return (date, x, y)


def predict(x_train, y_train, x_test):
    model = Sequential(
        [LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        # Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        # Dropout(0.2),
        LSTM(units=50),
        # Dropout(0.2),
        Dense(units=1)]
    )
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train, epochs=100, batch_size=32)
    prediction = model.predict(x_test)
    return scaler.inverse_transform(prediction)


date_train, x_train, y_train = get_data(COMPANY, START, END)
date_test, x_test, y_test = get_data(COMPANY, TEST_START, TEST_END)
y_test = scaler.inverse_transform(y_test)

prediction = predict(x_train, y_train, x_test)


for d, y, y_pred in zip(date_test, y_test, prediction):
    print(d, y, y_pred)


plt.plot(date_test, y_test, 'b-', date_test, prediction, 'r-', alpha=0.75)
plt.legend(['Actual', 'Prediction'], loc='upper left')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{COMPANY} Stock Price Forecasting')
plt.show()
