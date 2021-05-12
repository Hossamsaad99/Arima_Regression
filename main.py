#!/usr/bin/env python
# coding: utf-8

# In[5]:


# for FastAPI
from fastapi import FastAPI
import uvicorn
import pydantic

from datetime import *
import pandas_datareader as pdr
import numpy as np
import pandas as pd

# for arima
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
# for LSTM
from tensorflow import keras
from keras.models import load_model
import pickle

# import holidays
# from prophet import Prophet
# In[ ]:

# def prophet (ticker):
#   """
#   Forcasting using prophet ! by Getting the desired data from yahoo, then doing some data manipulation, then the comes the prophet's turn
#   Args:
#       (str) ticket - the ticker of desired dataset (company)
#   Returns:
#       (float) prophet_output - the model out-put (the prediction of the next day)
#   """

#   # data_gathering
#   df = pdr.DataReader(ticker, data_source='yahoo', start='2015-01-01')

#   # data manipulation
#   holiday = pd.DataFrame([])
#   for date, name in sorted(holidays.UnitedStates(years=[2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]).items()):
#       holiday = holiday.append(pd.DataFrame({'ds': date, 'holiday': "US-Holidays"}, index=[0]), ignore_index=True)
#   holiday['ds'] = pd.to_datetime(holiday['ds'], format='%Y-%m-%d', errors='ignore')

#   # data frame modification to be accepted by prophet
#   data = df['Close'].reset_index()
#   data.columns = ['ds', 'y']

#   # model building
#   m = Prophet(holidays=holiday,seasonality_mode='additive', changepoint_prior_scale = 0.1, seasonality_prior_scale=0.01)
#   m.fit(data)

#   # model predictions
#   future = m.make_future_dataframe(periods=1)
#   model_prediction = m.predict(future) 
#   prophet_prediction = float(model_prediction[ 'yhat'][-1:])
#   return prophet_prediction


def lstm(data_set):
  """
  Getting the desired data from yahoo, then doing some data manipulation such as data
  reshaping
  Args:
      (str) data_set - the ticker of desired dataset (company)
  Returns:
      (float) diff_prediction - the model out-put (the prediction of the next day)
      (float) real_prediction - the model output + today's price (real price of tomorrow)
  """

  # data gathering
  df = pdr.DataReader(data_set, data_source='yahoo', start=date.today() - timedelta(100))

  # data manipulation

  # creating a new df with Xt - Xt-1 values of the close prices (most recent 60 days)
  close_df = df['2012-01-01':].reset_index()['Close'][-61:]
  close_diff = close_df.diff().dropna()
  data = np.array(close_diff).reshape(-1, 1)

  # reshaping the data to 3D to be accepted by our LSTM model
  model_input = np.reshape(data, (1, 60, 1))

  # loading the model and predicting
  loaded_model = load_model("lstm_f_60.hdf5")
  model_prediction = float(loaded_model.predict(model_input))
  real_prediction = model_prediction + df['Close'][-1]
  

  return model_prediction, real_prediction

def arima(ticker):
  """
  Forcasting using ARIMA ! by Getting the desired data from yahoo, 
  then finding the best order of arima params then the comes the ARIMA's turn
  Args:
      (str) ticket - the ticker of desired dataset (company)
  Returns:
      (float) arima_output - the model out-put (the prediction of the next day)
      (float) diff - the model output - today's price (the diff between tomorrow's prediction and today's real value)
  """
    
  # data gathering
  df = pdr.DataReader(ticker, data_source='yahoo', start='2016-01-01')
  df.index = pd.to_datetime(df.index, format="%Y/%m/%d")
  df = pd.Series(df['Close'])
  last_day=df[-1]

  # finding the best order
  auto_order = pm.auto_arima(df, start_p=0, start_q=0, test='adf', max_p=3, max_q=3, m=1,d=None,seasonal=False   
                    ,start_P=0,D=0, trace=True,error_action='ignore',suppress_warnings=True,stepwise=True)
  best_order = auto_order.order

  # model fitting
  model = ARIMA(df, order=best_order)
  model_fit = model.fit(disp=0)
  arima_prediction ,se, conf = model_fit.forecast(1)
  
  diff = arima_prediction - last_day
  
  return arima_prediction , diff

def Regression(ticker):
  """
  Forcasting using an ensambled model between SVR, Ridge and Linear regression! by Getting the desired data from yahoo, 
  then doing some data manipulation
  Args:
      (str) ticket - the ticker of desired dataset (company)
  Returns:
      (float) arima_output - the model out-put (the prediction of the next day)
      (float) diff - the model output - today's price (the diff between tomorrow's prediction and today's real value)
  """
  start_date = datetime.now() - timedelta(1)
  start_date = datetime.strftime(start_date, '%Y-%m-%d')

  df = pdr.DataReader(ticker, data_source='yahoo', start=start_date)  # read data
  df.drop('Volume', axis='columns', inplace=True)
  X = df[['High', 'Low', 'Open', 'Adj Close']]  # input columns
  y = df[['Close']]  # output column
  input = X
  loaded_model = pickle.load(open('regression_model.pkl', 'rb'))
  reg_prediction = loaded_model.predict(input)
  reg_diff=reg_prediction-df.Close[-1]

  return  reg_prediction,reg_diff

app = FastAPI()


@app.get('/')
def index():
    return {'message': 'This is your fav stock predictor!'}


@app.post('/predict')
async def predict_price(data: str):
    if data == 'F':
      
      model_prediction, lstm_prediction = lstm(data)
#       prophet_prediction = float(prophet(data))

      arima_prediction, diff = arima(data)

      reg_prediction,reg_diff = Regression(data)


      return {
#               'Prophet prediction': prophet_prediction,
              'LSTM prediction' : lstm_prediction,
              'Arima prediction' : arima_prediction[0],
              'regression prediction' : reg_prediction[0]

            }

    else:
      return {"the ticker not supported yet"}

    



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)

