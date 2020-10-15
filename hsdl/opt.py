"""Optimization config and helpers."""
from torch import optim
import transformers

from . import config


def get(cfg, model_parameters):
    if cfg.optimizer == 'adam':
        return optim.Adam(
            params=model_parameters,
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay)
    if cfg.optimizer == 'adamw':
        return transformers.AdamW(
            params=model_parameters,
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            correct_bias=cfg.correct_bias)
    else:
        raise ValueError(f'Unexpected optimizer: {cfg.optimizer}')


