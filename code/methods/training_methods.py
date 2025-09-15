import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from .step_methods import _StepMethod

# Template for training methods, like gradient descent, and stochastic gradient descent
class _TrainingMethod:
    parameters: npt.NDArray[np.floating]
    def __init__(
        self,
        X: npt.NDArray[np.floating],
        y: npt.NDArray[np.floating],
        gradient_function: Callable[[npt.NDArray[np.floating],npt.NDArray[np.floating],npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        starting_parameters: npt.NDArray[np.floating],
        step_method: "_StepMethod",
    ) -> None:
        self.parameters = starting_parameters.copy()
        self.feature_amount = X.shape[1]
        self.X = X
        self.y = y
        self.gradient_function = gradient_function
        self.step_method = step_method
        self.step_method.setup(self.feature_amount)

        self.step_method.caller = self

        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        self.y_mean = self.y.mean()

        self.setup()

    def setup(self):
        ...
    
    def predict(self, X: npt.NDArray[np.floating], already_scaled: bool = False) -> npt.NDArray[np.floating]:
        if not already_scaled:
            X = self.scaler.transform(X)
        
        return X @ self.parameters + self.y_mean
        
    def mse(self):
        y_pred = self.predict(self.X, already_scaled=True)
        return mean_squared_error(self.y, y_pred)
    
    def train(self, iterations: int = 1000, store_mse: bool = False) -> tuple[npt.ArrayLike, npt.ArrayLike] | None:
        ...



# ========== Training methods ==========

class GradientDescent(_TrainingMethod):
    def train(self, iterations: int = 1000, store_mse: bool = False) -> tuple[npt.ArrayLike, npt.ArrayLike] | None:
        if store_mse:
            plot_steps = np.unique(np.logspace(0, np.log10(iterations-1), num=100, dtype=int))
            plot_step = 0
            mse_values = []
            for i in range(iterations):
                gradient = self.gradient_function(self.X, self.y-self.y_mean, self.parameters)
                self.step_method.training_step(gradient, iteration=i)
                
                if plot_step < len(plot_steps) and i == plot_steps[plot_step]:
                    mse_values.append(self.mse())
                    plot_step += 1
                    
            return plot_steps, mse_values

        else:
            for i in range(iterations):
                gradient = self.gradient_function(self.X, self.y, self.parameters)
                self.step_method.training_step(gradient, iteration=i)