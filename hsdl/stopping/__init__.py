import os
from typing import Tuple, Union

import pandas as pd
from pytorch_lightning.loggers import LightningLoggerBase

from hsdl import metrics
from hsdl.experiments.config import ExperimentConfig
from hsdl.stopping.config import *


def get(config: ExperimentConfig):
    if not config.stopping.strategy:
        return NoEarlyStopping()
    elif config.stopping.strategy == 'no_val_improvement':
        return NoValImprovement(
            patience=config.stopping.patience,
            k=config.stopping.k,
            metric_config=config.metric)
    else:
        raise ValueError(f'Unexpected early stopping strategy: '
                         f'{config.stopping.strategy}')


class EarlyStopping:
    """Base early stopping class."""

    def __init__(self):
        self.stop_called = False
        self.stop_epoch = None
        self.stop_reason = None

    def __call__(self, logger: LightningLoggerBase) \
            -> Tuple[bool, Union[None, str]]:
        if self.stop_called:
            return True, self.stop_reason
        stop, reason = self.stop(logger)
        self.stop_called = stop
        self.stop_reason = reason
        if stop:
            return stop, reason
        else:
            return False, None

    def stop(self, logger: LightningLoggerBase) \
            -> Tuple[bool, Union[None, str]]:
        raise NotImplementedError

    @staticmethod
    def load_val_metrics(logger: LightningLoggerBase):
        experiment_name = logger.name
        run_no = logger.version
        file_path = os.path.join(
            logger.experiment.get_data_path(experiment_name, run_no),
            'metrics.csv')
        if not os.path.exists(file_path):
            return []
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            return []
        # probably redundant, but safe
        df.sort_values(by='created_at', ascending=True, inplace=True)
        df = df[~pd.isnull(df.val_metric)]
        val_metrics = list(df.val_metric)
        return val_metrics


class NoEarlyStopping(EarlyStopping):

    def stop(self, logger: LightningLoggerBase) \
            -> Tuple[bool, Union[None, str]]:
        return False, None


class NoValImprovement(EarlyStopping):
    """Early stopping with no improvement on the validation set."""

    def __init__(self,
                 patience: int,
                 k: int,
                 metric_config: metrics.MetricConfig):
        super().__init__()
        self.patience = patience
        self.k = k
        self.metric = metrics.get(metric_config)

    def stop(self, logger: LightningLoggerBase) \
            -> Tuple[bool, Union[None, str]]:
        val_metrics = self.load_val_metrics(logger)
        n_epochs = len(val_metrics)
        if n_epochs >= self.patience:
            k = val_metrics[-self.k]  # baseline for checking improvement
            to_consider = val_metrics[-self.k+1:]  # metrics to check
            # stop if all subsequent metrics are no better than the baseline
            stop = all(not self.metric.is_better(x, k) for x in to_consider)
            return stop, 'Early stopping condition met.'
        return False, ''
