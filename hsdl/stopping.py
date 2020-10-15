"""Early stopping algorithms and config."""
from . import config, metrics


def get(cfg):
    if not cfg.strategy:
        return NoEarlyStopping()
    elif cfg.strategy == 'no_dev_improvement':
        return NoDevImprovement(cfg.patience, cfg.k, cfg.metric)
    else:
        raise ValueError(f'Unexpected early stopping strategy: {cfg.strategy}')


