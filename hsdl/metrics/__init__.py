from pytorch_lightning import metrics as pl_metrics
import numpy as np

from .config import MetricConfig


def get_lightning_metric(config):
    if config.metric.name is None:
        return None
    if config.metric.name == 'acc':
        return pl_metrics.Accuracy()
    elif config.metric.name == 'fbeta':
        return pl_metrics.Fbeta(
            num_classes=config.metric.num_classes,
            beta=config.metric.beta,
            multilabel=config.metric.multi_label)
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
