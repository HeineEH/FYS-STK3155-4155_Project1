from typing import Any
import numpy.typing as npt
import numpy as np


class _Gradient:
    def __call__(self, X: npt.NDArray[np.floating], y: npt.NDArray[np.floating], theta: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        ...

class OLS_Gradient(_Gradient):
    def __call__(self, X: npt.NDArray[np.floating], y: npt.NDArray[np.floating], theta: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        n = X.shape[0]
        return (2.0 / n) * X.T @ (X @ theta - y)

class Ridge_Gradient(_Gradient):
    def __init__(self, lambda_: float) -> None:
        self.lambda_ = lambda_

    def __call__(self, X: npt.NDArray[np.floating], y: npt.NDArray[np.floating], theta: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        n = X.shape[0]
        return (2.0 / n) * X.T @ (X @ theta - y) + 2 * self.lambda_ * theta
    
class Lasso_Gradient(_Gradient): 
    def __init__(self, lambda_: float) -> None:
        self.lambda_ = lambda_
    
    def __call__(self, X: npt.NDArray[np.floating], y: npt.NDArray[np.floating], theta: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        n = X.shape[0]
        return (2.0 / n) * X.T @ (X @ theta - y) + self.lambda_*(2*np.heaviside(theta,0.5*np.ones(len(theta))) - 1)