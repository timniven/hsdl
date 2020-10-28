from hsdl.config.base import Config


class MetricConfig(Config):
    """Base class for metric config."""

    def __init__(self, name: str, criterion: str):
        super().__init__()
        self.name = name
        self.criterion = criterion
