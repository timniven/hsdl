# https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html#training

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from hsdl.training.trainers import HsdlTrainer


def get_trainer(config,  # ExperimentConfig
                logger: LightningLoggerBase,
                debug: bool) -> Trainer:  # TODO?
    # https://pytorch-lightning.readthedocs.io/en/stable/trainer.html
    return HsdlTrainer(config, logger)
