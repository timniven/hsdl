from torch.optim import Optimizer, lr_scheduler

from .config import *


def get(config: AnnealingConfig, optimizer: Optimizer):
    """Get annealing algorithm from annealing config."""
    if config.schedule == 'none':
        return None
    elif config.schedule == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='max',
            factor=config.factor,
            patience=config.patience)
    else:
        raise ValueError(f'Unexpected lr_schedule: {config.schedule}')
