from typing import Any, Optional

from pytorch_lightning import metrics as pl_metrics
from pytorch_lightning.metrics.metric import Metric
import numpy as np
import torch
from torch import Tensor

from .config import MetricConfig


def get_lightning_metric(config, module):
    if config.metric is None or config.metric.name is None:
        return None
    if config.metric.name == 'acc':
        return pl_metrics.Accuracy()
    elif config.metric.name == 'fbeta':
        return pl_metrics.Fbeta(
            num_classes=config.metric.num_classes,
            beta=config.metric.beta,
            multilabel=config.metric.multi_label)
    elif config.metric.name == 'loss':
        loss_fn = getattr(module, config.metric.loss_fn)
        return Loss(loss_fn=loss_fn)
    # TODO: handle more metrics
    else:
        raise ValueError(f'Unexpected metric: {config.metric.name}.')


def get_criterion_fn(criterion: str):
    if criterion not in ['max', 'min']:
        raise ValueError(f'Unexpected criterion: {criterion}.')
    criterion = np.max if criterion == 'max' else np.min
    return criterion


def best(scores, criterion: str):
    criterion = get_criterion_fn(criterion)
    return criterion(scores)


def is_best(score, scores, criterion: str):
    return score == best(scores, criterion)


def is_better(score1, score0, criterion: str):
    """Determine if score1 improves on score0."""
    return score1 == best([score0, score1], criterion)


class Loss(Metric):

    def __init__(self,
                 loss_fn,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group)
        self.loss_fn = loss_fn
        self.add_state(
            'cum_loss', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state(
            'total', default=torch.tensor(0), dist_reduce_fx='sum')

    def compute(self):
        return self.cum_loss.float() / self.total

    def update(self, logits, y):
        loss = self.loss_fn(logits, y)
        self.cum_loss += loss
        self.total += 1  # assume loss is already meaned
