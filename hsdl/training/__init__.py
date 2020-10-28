# https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html#training

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase


def get_trainer(config,  # ExperimentConfig
                logger: LightningLoggerBase,
                debug: bool) -> Trainer:
    # make sure to set deterministic=True for reproducibility
    # https://pytorch-lightning.readthedocs.io/en/stable/trainer.html
    return Trainer(
        logger=logger,
        max_epochs=config.training.n_epochs,
        gpus=1,

    )
