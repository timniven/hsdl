from hsdl.config import Config


class MetricConfig(Config):
    """Base class for metric config."""

    def __init__(self, name: str, criterion: str):
        self.name = name
        self.criterion = criterion
