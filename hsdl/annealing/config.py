from hsdl.config import Config


class AnnealingConfig(Config):

    def __init__(self, schedule: str, epoch: bool, iter: bool):
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


class NoAnnealingConfig(AnnealingConfig):

    def __init__(self):
        super().__init__(schedule='none', epoch=False, iter=False)


class ReduceLROnPlateauConfig(AnnealingConfig):

    def __init__(self, factor, patience):
        super().__init__(schedule='plateau', epoch=True, iter=False)
        self.factor = factor
        self.patience = patience
