from typing import Callable

from sklearn import metrics as sklearn_metrics
import numpy as np


def get(name):
    if name == 'acc':
        return Accuracy()
    else:
        raise ValueError(f'Unexpected metric: {name}')


class Metric:
    """Abstract base class, defining the Metric interface."""

    def __init__(self, name: str, criterion: Callable):
        self.name = name
        self.criterion = criterion

    def __call__(self, y_true, y_pred):
        raise NotImplementedError

    def best(self, scores):
        # handle the case where scores are an item in dicts
        if isinstance(scores[0], dict):
            scores = [x[self.name] for x in scores]
        return self.criterion(scores)

    def is_best(self, score, scores):
        return score == self.best(scores)

    def is_better(self, score1, score0):
        """Determine if score1 improves on score0."""
        return score1 == self.best([score0, score1])


class Accuracy(Metric):
    """Accuracy metric."""

    def __init__(self):
        super().__init__('accuracy', np.max)

    def __call__(self, y_true, y_pred):
        return sklearn_metrics.accuracy_score(y_true, y_pred)
