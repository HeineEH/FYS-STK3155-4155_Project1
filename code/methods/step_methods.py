from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .training_methods import _TrainingMethod

# Template for step methods, like gd-momentum, RMSprop, ADAgrad
class _StepMethod:
    caller: "_TrainingMethod"
    def training_step(self, gradient: npt.NDArray[np.floating], iteration: int | None = None) -> None:
        ...



# ========== Step methods ==========

class ConstantGradientStep(_StepMethod):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
    
    def training_step(self, gradient: npt.NDArray[np.floating], iteration: int | None = None) -> None:
        self.caller.parameters -= self.learning_rate * gradient
        


# Momentum gradient descent
class MomentumGradientStep(_StepMethod):
    def __init__(self, learning_rate: float, momentum: float, num_features: int ) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity: npt.NDArray[np.floating] = np.zeros(num_features)
    
    def training_step(self, gradient: npt.NDArray[np.floating], iteration: int | None = None, ) -> None:
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradient
        self.caller.parameters -= self.velocity