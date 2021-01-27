from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def get(config):  # config: ExperimentConfig
    if not config.stopping.strategy:
        raise ValueError('No stopping configured.')
    elif config.stopping.strategy == 'no_val_improvement':
        return EarlyStopping(
            monitor=config.stopping.monitor,
            patience=config.stopping.patience,
            mode=config.stopping.criterion)
    else:
        raise ValueError(f'Unexpected early stopping strategy: '
                         f'{config.stopping.strategy}')
