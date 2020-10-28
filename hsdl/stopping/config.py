from hsdl.config.base import Config


class StoppingConfig(Config):

    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy


class NoEarlyStoppingConfig(StoppingConfig):

    def __init__(self):
        super().__init__(strategy=None)


class NoDevImprovementConfig(StoppingConfig):

    def __init__(self, patience, k, metric):
        super().__init__(strategy='no_dev_improvement')
        self.patience = patience
        self.k = k
        self.metric = metric
