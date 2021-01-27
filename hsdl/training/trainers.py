from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase

from hsdl import stopping, util


tqdm = util.get_tqdm()


class HsdlTrainer(Trainer):

    # config: ExperimentConfig
    def __init__(self, config, logger: LightningLoggerBase):
        super().__init__(
            deterministic=True,
            logger=logger,
            max_epochs=config.training.max_epochs,
            min_epochs=config.training.min_epochs,
            gpus=config.training.n_gpus,
            gradient_clip_val=config.training.gradient_clip_val)
        self.config = config

        if config.stopping:
            early_stopping = stopping.get(config)
            self.callbacks.append(early_stopping)

        self.my_checkpoint_callback = ModelCheckpoint(
            monitor='val_metric',  # whatever is chosen, always goes in here
            save_top_k=2,
            mode=config.training.checkpoint_mode)
        self.callbacks.append(self.my_checkpoint_callback)

        if config.training.gradient_accumulation_steps:
            scheduler = GradientAccumulationScheduler(
                scheduling={config.training.gradient_accumulation_start:
                                config.training.gradient_accumulation_steps}
            )
            self.callbacks.append(scheduler)
