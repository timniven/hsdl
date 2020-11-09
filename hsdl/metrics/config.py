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


class MaxFBeta(MetricConfig):

    def __init__(self, beta: float, num_classes: int, multi_class: bool):
        super().__init__('fbeta', 'max')
        self.beta = beta
        self.num_classes = num_classes
        self.multi_class = multi_class
