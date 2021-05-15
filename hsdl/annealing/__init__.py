from torch.optim import Optimizer, lr_scheduler


def get(config, optimizer: Optimizer, verbose: bool = True):
    """Get annealing algorithm from annealing config."""
    if config.annealing is None \
            or config.annealing['schedule'] == 'none'\
            or config.annealing['schedule'] == 'model_defined':
        return None
    elif config.annealing['schedule'] == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=config.annealing.mode,
            factor=config.annealing.factor,
            patience=config.annealing.patience,
            verbose=verbose)
    elif config.annealing['schedule'] == 'step':
        return lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=config.annealing['step_size'],
            gamma=config.annealing['gamma'],
            last_epoch=config.annealing['last_epoch'],
            verbose=config.annealing['verbose'])
    else:
        raise ValueError(f'Unexpected lr_schedule: {config.annealing.schedule}')


class LambdaStepAnnealer(lr_scheduler._LRScheduler):
    # TODO: need to setup manual optimization in the Module itself
    #  so this is getting messy
    #  leaving it to upstream code for now

    pass
