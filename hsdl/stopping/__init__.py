from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def get(config):  # config: ExperimentConfig
    if not config.stopping.strategy:
        return None
    elif config.stopping.strategy == 'no_val_improvement':
        return EarlyStopping(
            monitor=config.stopping['monitor'],
            min_delta=config.stopping['min_delta'],
            patience=config.stopping['patience'],
            mode=config.stopping['mode'],
            strict=config.stopping['strict'],
            verbose=config.stopping['verbose'])
    else:
        raise ValueError(f'Unexpected early stopping strategy: '
                         f'{config.stopping.strategy}')
