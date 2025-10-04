import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

class BiasVariance:
    def __init__(self,x,y, max_degree = 15, test_size = 0.2, random_state = 42, model = LinearRegression(fit_intercept=False)):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x.reshape(-1,1), y.reshape(-1,1), test_size=test_size,random_state = random_state) 
        self.max_degree = max_degree
        self.model = model
        self.y_mean = self.y_train.mean()

    def bootstrap(self, n_bootstraps = 100):
        mse = np.zeros(self.max_degree)
        bias = np.zeros(self.max_degree)
        variance = np.zeros(self.max_degree)
        for degree in range(1,self.max_degree+1):
            model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), StandardScaler(), self.model)
            y_pred = np.zeros((self.y_test.shape[0], n_bootstraps))

            for i in range(n_bootstraps):
                x_, y_ = resample(self.x_train, self.y_train - self.y_mean,random_state = i)
                y_pred[:, i] = model.fit(x_, y_).predict(self.x_test).ravel()

            mse[degree-1] = np.mean(np.mean(((self.y_test - self.y_mean) - y_pred)**2, axis=1, keepdims=True))
            bias[degree-1] = np.mean(((self.y_test - self.y_mean) - np.mean(y_pred, axis=1, keepdims=True))**2)
            variance[degree-1] = np.mean(np.var(y_pred, axis=1, keepdims=True))
        
        return mse, bias, variance
     
    def direct_mse(self,test_data = True): 
        mse = np.zeros(self.max_degree)
        if test_data == True: 
            y = self.y_test.copy()
            x = self.x_test.copy()
        else: 
            y = self.y_train.copy()
            x = self.x_train.copy()

        for degree in range(1,self.max_degree+1):
            model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False),StandardScaler(), self.model)
            y_pred = model.fit(self.x_train, self.y_train - self.y_mean).predict(x)
            mse[degree-1] = mean_squared_error(y - self.y_mean,y_pred)
        return mse
    

    def k_fold_cross_validation(self, k: int):
        mse = np.zeros(self.max_degree)
        kfold = KFold(n_splits=k)
        for degree in range(1, self.max_degree+1):
            model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), StandardScaler(), self.model)
            estimated_mse_folds = cross_val_score(model, self.x_train, self.y_train - self.y_mean, scoring='neg_mean_squared_error', cv=kfold)
            mse[degree-1] = np.mean(-estimated_mse_folds)
        return mse
        