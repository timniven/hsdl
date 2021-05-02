from typing import Callable, Optional

from hsdl.config.base import Config


class AnnealingConfig(Config):

    def __init__(self,
                 schedule: str,
                 epoch: bool,
                 step: bool,
                 monitor: Optional[str] = None,
                 mode: Optional[str] = None):
        """Create a new AnnealingConfig.

        Args:
          schedule: String, defines the schedule.
          epoch: Bool, whether to execute each epoch.
          step: Bool, whether to execute each step.
        """
        super().__init__()
        self.schedule = schedule
        self.epoch = epoch
        self.iter = step
        self.monitor = monitor
        self.mode = mode


class NoAnnealingConfig(AnnealingConfig):

    def __init__(self):
        super().__init__(schedule='none', epoch=False, step=False)


class ReduceLROnPlateauConfig(AnnealingConfig):

    def __init__(self, factor: float, patience: int, monitor: str, mode: str):
        super().__init__(
            schedule='plateau',
            epoch=True,
            step=False,
            monitor=monitor,
            mode=mode)
        self.factor = factor
        self.patience = patience


class StepLrConfig(AnnealingConfig):

    def __init__(self,
                 step_size: int,
                 gamma: float,
                 last_epoch: int = -1,
                 verbose: bool = False):
        super().__init__(schedule='step', epoch=False, step=True)
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.verbose = verbose


class LambdaStep(AnnealingConfig):

    def __init__(self,
                 func: Callable[[float, int], float]):
        super().__init__(schedule='lambda_step', epoch=False, step=True)
        self.func = func
