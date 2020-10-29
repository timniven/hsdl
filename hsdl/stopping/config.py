from hsdl.config.base import Config


class StoppingConfig(Config):

    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy


class NoEarlyStoppingConfig(StoppingConfig):

    def __init__(self):
        super().__init__(strategy=None)


class NoValImprovementConfig(StoppingConfig):

    def __init__(self,
                 patience: int,
                 k: int,
                 metric_config):
        if k > patience:
            raise ValueError(f'k ({k}) cannot be '
                             f'greater than patience ({patience}).')
        super().__init__(strategy='no_val_improvement')
        self.patience = patience
        self.k = k
        self.metric_config = metric_config
