from typing import Tuple, Union

from .base import *
from .config import *
from hsdl import metrics


def get(config: StoppingConfig, metric_config: metrics.MetricConfig):
    if not config.strategy:
        return NoEarlyStopping()
    elif config.strategy == 'no_dev_improvement':
        return NoDevImprovement(config.patience, config.k, metric_config)
    else:
        raise ValueError(f'Unexpected early stopping strategy: '
                         f'{config.strategy}')


class EarlyStopping:
    """Base early stopping class."""

    def __call__(self, train_state):
        return self.stop(train_state)

    def stop(self, train_state):
        raise NotImplementedError


class NoEarlyStopping(EarlyStopping):

    def __init__(self):
        super().__init__()

    def stop(self, train_state) -> Tuple[bool, Union[None, str]]:
        return False, None


class NoDevImprovement(EarlyStopping):
    """Early stopping with no improvement on the dev set."""

    def __init__(self, patience: int, k: int,
                 metric_config: metrics.MetricConfig):
        """Create a new NoDevImprovement.

        Args:
          patience: Int, how many epochs to let run before checking.
          k: Int, how many epochs without improvement before stopping.
          metric_config: MetricConfig, used for the conditions.
        """
        super().__init__()
        self.patience = patience
        self.k = k
        self.metric = metrics.get(metric_config)

    def stop(self, train_state) -> Tuple[bool, Union[None, str]]:
        if train_state.epoch >= self.patience:
            mets = [x[self.metric.abbr] for x in train_state.dev_metrics]
            k = mets[-self.k]  # baseline for checking improvement
            to_consider = mets[-self.k+1:]  # metrics to check
            # stop if all subsequent metrics are no better than the baseline
            stop = all(not self.metric.is_better(x, k) for x in to_consider)
            return stop, 'Early stopping condition met.'
        return False, ''
