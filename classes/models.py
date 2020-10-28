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
    def __init__(self, train, test, X_train, y_train, X_test, y_test, out_path):
        self.train = train
        self.test = test
        self.X_train = X_train
        self.y_train = y_train	
        self.X_test = X_test
        self.y_test = y_test
        self.out_path = out_path


    def arima_test(self):
        train = self.train['Clop_norm'].copy()
        test = self.test['Clop_norm'].copy()
        model = ARIMA(train, order=(2,1,0)).fit()
        pred = model.predict()
        print("Five prediction: ", pred[:5])
        print("ARIMA SHAPES: ", pred.shape, train.shape)
        plt.plot(train.values, color='r', linewidth=1, label='train')
        plt.plot(pred, color='b', linewidth=1, label='pred')
        plt.legend(loc='upper center', frameon=False)
        plt.ylabel('\u0394 Rel')
        figname = os.path.join(self.out_path, "ARIMA"+".png")
        plt.savefig(figname, dpi=200)
        plt.close()


    def linear_regression(self):
        from sklearn.metrics import mean_squared_error
        from sklearn import linear_model
        model = linear_model.LinearRegression().fit(self.X_train, self.y_train)
        train_pred = model.predict(self.X_test)
        rms=np.sqrt(np.mean(np.power((np.array(self.y_test)-np.array(train_pred)),2)))
        err = mean_squared_error(self.y_test, train_pred)
        model_log.info("RMS: {}, RMSE: {}".format(rms, err))
        return model, train_pred

    def random_forest(self):
        from sklearn.metrics import mean_squared_error
        from sklearn.ensemble import RandomForestRegressor
        # model = RandomForestRegressor(max_depth=2, random_state=0)
        model = RandomForestRegressor()
        model.fit(self.X_train, self.y_train)
        train_pred = model.predict(self.X_test)
        rms=np.sqrt(np.mean(np.power((np.array(self.y_test)-np.array(train_pred)),2)))
        err = mean_squared_error(self.y_test, train_pred)
        model_log.info("RMS: {}, RMSE: {}".format(rms, err))
        return model, train_pred
