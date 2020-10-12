from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

class models(object):
    def __init__(self, train, test, out_path):
        self.train = train
        self.test = test
        self.out_path = out_path

    def arima_test(self):
    	model = ARIMA(self.train, order=(3,4,1)).fit()
    	# model = ARIMA(self.train).fit()
    	train_pred = model.predict()
    	print("1: ", train_pred[:5])
    	print("1: ", train_pred.shape)
    	train_pred[0] += self.train.iloc[0,0]
    	train_pred = np.cumsum(train_pred)
    	print("2: ", train_pred.head())
    	print("2: ", train_pred.shape)
    	self.train["Pred"] = train_pred
    	self.train.dropna(inplace=True)
    	print("3: ", self.train.head())
    	print("3: ", self.train.shape)
    	self.train.plot(figsize=(18,6),
    		title='real vs forecaster price on training set',
    		color=['Teal','orangered'])
    	plt.ylabel('\u0394 Rel')
    	figname = os.path.join(self.out_path, "ARIMA"+".png")
    	plt.savefig(figname, dpi=200)
    	plt.close()

    def linear_regression(self):
    	from sklearn import linear_model
    	model = linear_model.LinearRegression().fit(self.train, self.train)
    	train_pred = model.predict(self.train)
    	train_pred[0] += self.train.iloc[0,0]
    	train_pred = np.cumsum(train_pred)
    	self.train["Pred"] = train_pred
    	self.train.plot(figsize=(18,6),
    		title='real vs forecaster price on training set',
    		color=['Teal','orangered'])
    	plt.ylabel('\u0394 Rel')
    	figname = os.path.join(self.out_path, "linearRegr"+".png")
    	plt.savefig(figname, dpi=200)
    	plt.close()