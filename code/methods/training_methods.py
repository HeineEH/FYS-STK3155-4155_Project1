import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from .regression_methods import _Gradient
from typing import TYPE_CHECKING
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from .step_methods import _StepMethod

# Template for training methods, like gradient descent, and stochastic gradient descent
class _TrainingMethod:
    parameters: npt.NDArray[np.floating]
    def __init__(
        self,
        X: npt.NDArray[np.floating],
        y: npt.NDArray[np.floating],
        gradient: _Gradient,
        starting_parameters: npt.NDArray[np.floating],
        step_method: "_StepMethod",
    ) -> None:
        self.parameters = starting_parameters.copy()
        self.feature_amount = X.shape[1]
        #self.X = X
        #self.y = y
        self.X, self.X_test, self.y, self.y_test = train_test_split(X, y, test_size=0.3,random_state=42)     ##
        self.gradient = gradient
        self.step_method = step_method
        self.step_method.setup(self.feature_amount)

        self.step_method.caller = self

        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        self.X_test = self.scaler.transform(self.X_test)                            ##
        self.y_mean = self.y.mean()

        self.setup()

    def setup(self):
        ...
    
    def predict(self, X: npt.NDArray[np.floating], already_scaled: bool = False) -> npt.NDArray[np.floating]:
        if not already_scaled:
            X = self.scaler.transform(X)
        
        return X @ self.parameters + self.y_mean
        
    def mse(self):
        y_pred = self.predict(self.X_test, already_scaled=True)     ##
        return mean_squared_error(self.y_test, y_pred)     ##
    
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
                gradient = self.gradient(self.X, self.y-self.y_mean, self.parameters)
                self.step_method.training_step(gradient)
                
                if plot_step < len(plot_steps) and i == plot_steps[plot_step]:
                    mse_values.append(self.mse())
                    plot_step += 1
                    
            return plot_steps, mse_values

        else:
            for i in range(iterations):
                gradient = self.gradient(self.X, self.y - self.y_mean, self.parameters)
                self.step_method.training_step(gradient)

class StochasticGradientDescent(_TrainingMethod): 
    def train(self, epochs: int = 1000, n_batches: int = 5,store_mse: bool = True) -> tuple[npt.ArrayLike, npt.ArrayLike] | None:
        n_datapoints = self.X.shape[0]
        batch_size = int(n_datapoints/n_batches)
        if store_mse:
            plot_steps = np.unique(np.logspace(0, np.log10(epochs-1), num=100, dtype=int))
            plot_step = 0
            mse_values = []
            for i in range(epochs):
                shuffled_data = np.array(range(n_datapoints))
                np.random.shuffle(shuffled_data)
                for j in range(n_batches): 
                    gradient = self.gradient(self.X[shuffled_data][(batch_size*j):(batch_size*(j+1))], self.y[shuffled_data][(batch_size*j):(batch_size*(j+1))] - self.y_mean, self.parameters)
                    self.step_method.training_step(gradient)
                
                if plot_step < len(plot_steps) and i == plot_steps[plot_step]:
                    mse_values.append(self.mse())
                    plot_step += 1
                    
            return plot_steps, mse_values
        else: 
            for i in range(epochs):
                shuffled_data = np.array(range(n_datapoints))
                np.random.shuffle(shuffled_data)

                for j in range(n_batches): 
                    gradient = self.gradient(self.X[shuffled_data][(batch_size*j):(batch_size*(j+1))], self.y[shuffled_data][(batch_size*j):(batch_size*(j+1))] - self.y_mean, self.parameters)
                    self.step_method.training_step(gradient)