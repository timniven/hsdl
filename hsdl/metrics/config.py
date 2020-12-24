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


class FBeta(MetricConfig):

    def __init__(self, beta: float, num_classes: int, multi_label: bool,
                 mode: str = 'max'):
        super().__init__(name='fbeta', criterion=mode)
        self.beta = beta
        self.num_classes = num_classes
        self.multi_label = multi_label


class Loss(MetricConfig):

    def __init__(self, loss_fn: str = 'loss', criterion: str = 'min'):
        super().__init__(
            name='loss',
            criterion=criterion)
        self.loss_fn = loss_fn
