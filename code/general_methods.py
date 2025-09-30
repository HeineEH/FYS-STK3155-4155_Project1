import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from methods.training_methods import GradientDescent, StochasticGradientDescent
from methods.step_methods import ConstantGradientStep, MomentumGradientStep, ADAgradStep, RMSpropStep, AdamStep
from methods.regression_methods import OLS_Gradient, Ridge_Gradient, Lasso_Gradient
from sklearn.preprocessing import PolynomialFeatures

def OLS_parameters(X, y):
    X_transpose = np.transpose(X)
    return np.linalg.pinv(X_transpose @ X) @ X_transpose @ y

def Ridge_parameters(X, y, lambda_reg = 0.1):
    pred = X.shape[1]
    I = np.eye(pred) # size: (p,p)
    return np.linalg.pinv((X.T @ X)+I*lambda_reg) @ X.T @ y

def polynomial_features(x, p, intercept = False):
    n = len(x)
    if intercept:
        X = np.zeros((n,p+1))
        for i in range(p+1):
            X[:, i] = x**i
    else:
        X = np.zeros((n, p))
        for i in range(1, p + 1):
            X[:, i - 1] = x ** i
    return X

def MSE(y, y_pred):
    return 1/len(y) * np.sum((y-y_pred)**2)

def R2(y, y_pred):
    return 1 - np.sum((y-y_pred)**2)/np.sum((y-np.mean(y))**2)

def opt_theta(X,y,regression_type = "OLS",lamb = 0.0): 
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    y_offset = np.mean(y)
    if regression_type == "OLS": 
        return OLS_parameters(X_s, y - y_offset), scaler
    else: 
        return Ridge_parameters(X_s, y - y_offset,lambda_reg = lamb), scaler

def MSE_and_R2(x,y,polynomial_degrees,regression_type="OLS",lamb = 0.0): 
    mse_values = np.zeros(len(polynomial_degrees))
    r2_values = np.zeros(len(polynomial_degrees))
    for i in range(len(polynomial_degrees)):
        X = polynomial_features(x, polynomial_degrees[i])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
        theta, scaler = opt_theta(X_train,y_train,regression_type = regression_type, lamb = lamb)
        X_test_s = scaler.transform(X_test)
        y_pred = X_test_s @ theta + np.mean(y_train)
        ols_mse = MSE(y_test, y_pred)
        ols_r2 = R2(y_test, y_pred)
        mse_values[i] = ols_mse
        r2_values[i] = ols_r2
    return mse_values,r2_values