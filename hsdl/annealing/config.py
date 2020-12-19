from hsdl.config.base import Config


class AnnealingConfig(Config):

    def __init__(self,
                 schedule: str,
                 epoch: bool,
                 iter: bool,
                 monitor: str,
                 mode: str):
        """Create a new AnnealingConfig.

        Args:
          schedule: String, defines the schedule.
          epoch: Bool, whether to execute each epoch.
          iter: Bool, whether to execute each iteration.
        """
        super().__init__()
        self.schedule = schedule
        self.epoch = epoch
        self.iter = iter
        self.monitor = monitor
        self.mode = mode


class NoAnnealingConfig(AnnealingConfig):

    def __init__(self):
        super().__init__(
            schedule='none', epoch=False, iter=False, monitor=None, mode=None)


class ReduceLROnPlateauConfig(AnnealingConfig):

    def __init__(self, factor: float, patience: int, monitor: str, mode: str):
        super().__init__(schedule='plateau', epoch=True, iter=False,
                         monitor=monitor, mode=mode)
        self.factor = factor
        self.patience = patience
