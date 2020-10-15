from torch.optim import lr_scheduler

from hsdl import config


def get(cfg, optimizer):
    """Get annealing object from annealing config.

    Args:
      cfg: Config, just for annealing.
      optimizer: pytorch optimizer object.

    Returns:
      pytorch annealing object if annealing, or None if cfg.sched is None.
    """
    if not cfg.schedule:
        return None
    elif cfg.schedule == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='max',
            factor=cfg.factor,
            patience=cfg.patience)
    else:
        raise ValueError(f'Unexpected lr_schedule: {cfg.schedule}')
