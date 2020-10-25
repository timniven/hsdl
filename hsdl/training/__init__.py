# https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html#training

from pytorch_lightning import Trainer

from hsdl.experiments.config import ExperimentConfig


def get_trainer(config: ExperimentConfig,
                logger: None,
                debug: bool) -> Trainer:
    # make sure to set deterministic=True for reproducibility
    # https://pytorch-lightning.readthedocs.io/en/stable/trainer.html
    pass
