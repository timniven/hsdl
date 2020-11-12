from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from hsdl.experiments.config import ExperimentConfig


def get(config: ExperimentConfig):
    if not config.stopping.strategy:
        raise ValueError('No stopping configured.')
    elif config.stopping.strategy == 'no_val_improvement':
        return EarlyStopping(
            monitor='val_metric',
            patience=config.stopping.patience,
            mode=config.metric.criterion)
    else:
        raise ValueError(f'Unexpected early stopping strategy: '
                         f'{config.stopping.strategy}')
