from hsdl.config import Config


class OptimizationConfig(Config):
    """Base class for optimization config."""

    def __init__(self, optimizer, lr, weight_decay):
        super().__init__()
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay


class AdamConfig(OptimizationConfig):
    """Config for Adam optimization."""

    def __init__(self, lr, weight_decay=0., beta1=0.9, beta2=0.999, eps=1e-08):
        super().__init__(
            optimizer='adam',
            lr=lr,
            weight_decay=weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps


class AdamWConfig(OptimizationConfig):
    """Config for AdamW."""

    def __init__(self, lr, weight_decay=0., beta1=0.9, beta2=0.999,
                 eps=1e-08, correct_bias=True):
        super().__init__(
            optimizer='adamw',
            lr=lr,
            weight_decay=weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.correct_bias = correct_bias
