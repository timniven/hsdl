from hsdl.config.base import Config


class StoppingConfig(Config):

    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy


class NoEarlyStoppingConfig(StoppingConfig):

    def __init__(self):
        super().__init__(strategy=None)


class NoValImprovementConfig(StoppingConfig):
    # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.early_stopping.html

    def __init__(self,
                 monitor: str,
                 min_delta: float,
                 patience: int,
                 mode: str = 'min',
                 strict: bool = True,
                 check_finite: bool = True,
                 verbose: bool = True):
        super().__init__(strategy='no_val_improvement')
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.strict = strict
        self.check_finite = check_finite
        self.verbose = verbose
