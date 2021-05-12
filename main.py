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

import pickle



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
      arima_prediction, diff = arima(data)

      reg_prediction,reg_diff = Regression(data)


      return {'Arima prediction' : arima_prediction[0],'regression prediction' : reg_prediction[0]}

   
    else:
      return {"the ticker not supported yet"}

    



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)

