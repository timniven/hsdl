from torch import optim
import transformers
from typing import List

from hsdl.config import ExperimentConfig


def get(config: ExperimentConfig, model_parameters: List):
    if config.optimization.name == 'adam':
        return optim.Adam(
            params=model_parameters,
            lr=config.optimization.lr,
            betas=(config.optimization.beta1, config.optimization.beta2),
            eps=config.optimization.eps,
            weight_decay=config.optimization.weight_decay)
    if config.optimization.name == 'adamw':
        return transformers.AdamW(
            params=model_parameters,
            lr=config.optimization.lr,
            betas=(config.optimization.beta1, config.optimization.beta2),
            eps=config.optimization.eps,
            weight_decay=config.optimization.weight_decay,
            correct_bias=config.optimization.correct_bias)
    else:
        raise ValueError(f'Unexpected optimizer: {config.optimization.name}')
