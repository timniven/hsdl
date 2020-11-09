from hsdl.config.base import Config


class MetricConfig(Config):
    """Base class for metric config."""

    def __init__(self, name: str, criterion: str):
        super().__init__()
        self.name = name
        if criterion not in ['min', 'max']:
            raise ValueError(f'Unexpected criterion: {criterion}.')
        self.criterion = criterion


class MaxAccuracy(MetricConfig):

    def __init__(self):
        super().__init__('acc', 'max')
