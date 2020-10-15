from torch import optim
import transformers
from typing import List

from .config import *


def get(config: OptimizationConfig, model_parameters: List):
    if config.name == 'adam':
        return optim.Adam(
            params=model_parameters,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay)
    if config.name == 'adamw':
        return transformers.AdamW(
            params=model_parameters,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            correct_bias=config.correct_bias)
    else:
        raise ValueError(f'Unexpected optimizer: {config.name}')
