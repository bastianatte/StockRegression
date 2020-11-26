from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import os
import logging
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
from utils.misc import get_logger

model_log = get_logger(__name__)
model_log.setLevel(logging.INFO)

class models(object):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train	
        self.X_test = X_test
        self.y_test = y_test

    def linear_regression(self):
        from sklearn.metrics import mean_squared_error
        from sklearn import linear_model
        model = linear_model.LinearRegression().fit(self.X_train, self.y_train)
        train_pred = model.predict(self.X_test)
        print(self.X_train.shape, self.y_train.shape, self.X_test.shape, self.y_test.shape, train_pred.shape)
        rms=np.sqrt(np.mean(np.power((np.array(self.y_test)-np.array(train_pred)),2)))
        err = mean_squared_error(self.y_test, train_pred)
        model_log.info("RMS: {}, RMSE: {}".format(rms, err))
        return model, train_pred

    def random_forest(self):
        from sklearn.metrics import mean_squared_error
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
        model.fit(self.X_train, self.y_train)
        train_pred = model.predict(self.X_test)
        print(self.X_train.shape, self.y_train.shape, self.X_test.shape, self.y_test.shape, train_pred.shape)
        # Look at parameters used by our current forest
        # print('Parameters currently in use:\n', model.get_params())
        rms=np.sqrt(np.mean(np.power((np.array(self.y_test)-np.array(train_pred)),2)))
        err = mean_squared_error(self.y_test, train_pred)
        model_log.info("RMS: {}, RMSE: {}".format(rms, err))
        return model, train_pred
