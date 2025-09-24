import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

class BiasVariance:
     def __init__(self,x,y,max_degree = 15,test_size = 0.2,n_boostraps = 100,random_state = 42): 
          self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x.reshape(-1,1), y.reshape(-1,1), test_size=test_size,random_state = random_state)
          self.max_degree = max_degree
          self.n_boostraps = n_boostraps
          self.MSE = np.zeros(max_degree)
          self.bias = np.zeros(max_degree)
          self.variance = np.zeros(max_degree)

     def bootstrap(self):

        for degree in range(1,self.max_degree+1):
            model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
            y_pred = np.zeros((self.y_test.shape[0], self.n_boostraps))

            for i in range(self.n_boostraps):
                x_, y_ = resample(self.x_train, self.y_train,random_state = i)
                y_pred[:, i] = model.fit(x_, y_).predict(self.x_test).ravel()

            self.MSE[degree-1] = np.mean(np.mean((self.y_test - y_pred)**2, axis=1, keepdims=True))
            self.bias[degree-1] = np.mean((self.y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
            self.variance[degree-1] = np.mean(np.var(y_pred, axis=1, keepdims=True))
        
        return self.MSE,self.bias,self.variance
     
     def no_bootstrap(self,test_data = True): 
        if test_data == True: 
            y = self.y_test.copy()
            x = self.x_test.copy()
        else: 
            y = self.y_train.copy()
            x = self.x_train.copy()

        for degree in range(1,self.max_degree+1):
            model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
            y_pred = model.fit(self.x_train, self.y_train).predict(x)
            self.MSE[degree-1] = np.mean((y - y_pred)**2)
        return self.MSE

     def plot_bias_variance(self): 
        plt.plot(range(1,self.max_degree+1), self.MSE, label="MSE")
        plt.plot(range(1,self.max_degree+1), self.bias, label="Bias")
        plt.plot(range(1,self.max_degree+1), self.variance, label="Variance")
        plt.xlabel("Polynomial degree")
        plt.ylabel("Error")
        plt.legend()
        plt.show()