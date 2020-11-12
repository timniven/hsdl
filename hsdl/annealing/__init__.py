from torch.optim import Optimizer, lr_scheduler


def get(config, optimizer: Optimizer, verbose: bool = True):
    """Get annealing algorithm from annealing config."""
    if config.annealing is None or config.annealing.schedule == 'none':
        return None
    elif config.annealing.schedule == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='max',
            factor=config.annealing.factor,
            patience=config.annealing.patience,
            verbose=verbose)
    else:
        raise ValueError(f'Unexpected lr_schedule: {config.annealing.schedule}')
