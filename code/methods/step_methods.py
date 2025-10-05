from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .training_methods import _TrainingMethod

# Template for step methods, like gd-momentum, RMSprop, ADAgrad
class _StepMethod:
    caller: "_TrainingMethod"
    learning_rate: float
    def setup(self, num_features: int) -> None:
        ...
    def training_step(self, gradient: npt.NDArray[np.floating]) -> None:
        ...



# ========== Step methods ==========

class ConstantLearningRateStep(_StepMethod):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
    
    def training_step(self, gradient: npt.NDArray[np.floating]) -> None:
        self.caller.parameters -= self.learning_rate * gradient


class MomentumStep(_StepMethod):
    def __init__(self, learning_rate: float, momentum: float, ) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def setup(self, num_features: int) -> None:
        self.velocity: npt.NDArray[np.floating] = np.zeros(num_features)
    
    def training_step(self, gradient: npt.NDArray[np.floating]) -> None:
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradient
        self.caller.parameters -= self.velocity


class ADAgradStep(_StepMethod):
    def __init__(self, learning_rate: float, error: float = 1e-8) -> None:
        self.learning_rate = learning_rate
        self.error = error
    
    def setup(self, num_features: int) -> None:
        self.accumulated_gradient = np.zeros(num_features)
        
    def training_step(self, gradient: npt.NDArray[np.floating]) -> None:
        self.accumulated_gradient += gradient**2  # Accumulate squared gradients
        adjusted_gradient = gradient / (np.sqrt(self.accumulated_gradient) + self.error)
        self.caller.parameters -= self.learning_rate * adjusted_gradient
        
class RMSpropStep(_StepMethod):
    def __init__(self, learning_rate: float, decay_rate: float, error: float = 1e-8) -> None:
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.error = error
        
    def setup(self, num_features: int) -> None:
        self.accumulated_gradient = np.zeros(num_features)
    
    def training_step(self, gradient: npt.NDArray[np.floating], iteration: int | None = None) -> None:
        self.accumulated_gradient = self.decay_rate * self.accumulated_gradient + (1 - self.decay_rate) * gradient**2
        adjusted_gradient = gradient / (np.sqrt(self.accumulated_gradient) + self.error)
        self.caller.parameters -= self.learning_rate * adjusted_gradient
        
    
class AdamStep(_StepMethod):
    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, error: float = 1e-8) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.error = error
    
    def setup(self, num_features: int) -> None:
        self.t = 0  # Time step
        self.s = np.zeros(num_features)  # First moment vector
        self.r = np.zeros(num_features)  # Second moment vector
    
    def training_step(self, gradient: npt.NDArray[np.floating], iteration: int | None = None) -> None:
        self.t += 1

        self.s = self.beta1 * self.s + (1 - self.beta1) * gradient
        self.r = self.beta2 * self.r + (1 - self.beta2) * (gradient ** 2)

        s_hat = self.s / (1 - self.beta1 ** self.t)
        r_hat = self.r / (1 - self.beta2 ** self.t)

        adjusted_gradient = s_hat / (np.sqrt(r_hat) + self.error)
        self.caller.parameters -= self.learning_rate * adjusted_gradient