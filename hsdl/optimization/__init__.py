from typing import Iterable

from torch import optim
import transformers


def get(config, model_parameters: Iterable):
    if config.optimization.name == 'adam':
        return optim.Adam(
            params=model_parameters,
            lr=config.optimization.lr,
            betas=(config.optimization.beta1, config.optimization.beta2),
            eps=config.optimization.eps,
            weight_decay=config.optimization.weight_decay)
    elif config.optimization.name == 'adamw':
        return transformers.AdamW(
            params=model_parameters,
            lr=config.optimization.lr,
            betas=(config.optimization.beta1, config.optimization.beta2),
            eps=config.optimization.eps,
            weight_decay=config.optimization.weight_decay,
            correct_bias=config.optimization.correct_bias)
    elif config.optimization['name'] == 'sgd':
        return optim.SGD(
            params=model_parameters,
            lr=config.optimization['lr'],
            weight_decay=config.optimization['weight_decay'],
            momentum=config.optimization['momentum'],
            dampening=config.optimization['dampening'],
            nesterov=config.optimization['nesterov'])
    else:
        raise ValueError(f'Unexpected optimizer: {config.optimization.name}')
