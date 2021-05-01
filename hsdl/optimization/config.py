from typing import Optional

from hsdl.config.base import Config


class OptimizationConfig(Config):
    """Base class for optimization config."""

    def __init__(self, name: str, lr: float, weight_decay: float):
        super().__init__()
        self.name = name
        self.lr = lr
        self.weight_decay = weight_decay


class AdamConfig(OptimizationConfig):

    def __init__(self,
                 lr: float,
                 weight_decay: float = 0.,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-08):
        super().__init__(
            name='adam',
            lr=lr,
            weight_decay=weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps


class AdamWConfig(OptimizationConfig):

    def __init__(self,
                 lr: float,
                 weight_decay: float = 0.,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-08,
                 correct_bias: bool = True):
        super().__init__(
            name='adamw',
            lr=lr,
            weight_decay=weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.correct_bias = correct_bias


class SgdConfig(OptimizationConfig):

    def __init__(self,
                 lr: float,
                 weight_decay: Optional[float] = 0.,
                 momentum: Optional[float] = 0.,
                 damping: Optional[float] = 0.,
                 nesterov: Optional[bool] = False):
        super().__init__(
            name='sgd',
            lr=lr,
            weight_decay=weight_decay)
        self.momentum = momentum
        self.damping = damping
        self.nesterov = nesterov
