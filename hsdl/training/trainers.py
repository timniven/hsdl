from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase

from hsdl import util
from hsdl.experiments.config import ExperimentConfig


tqdm = util.get_tqdm()


class HsdlTrainer(Trainer):

    def __init__(self, config: ExperimentConfig, logger: LightningLoggerBase):
        super().__init__(
            deterministic=True,
            logger=logger,
            max_epochs=config.training.max_epochs,
            min_epochs=config.training.min_epochs,
            gpus=1,
            gradient_clip_val=config.training.gradient_clip_val)
        self.config = config

        if config.stopping and config.stopping.strategy == 'no_val_improvement':
            self.callbacks.append(
                EarlyStopping(
                    monitor='val_metric',
                    patience=config.stopping.patience,
                    mode=config.metric.criterion))

        self.my_checkpoint_callback = ModelCheckpoint(
            monitor='val_metric',
            save_top_k=2,
            mode=config.metric.criterion)
        self.callbacks.append(self.my_checkpoint_callback)

        if config.training.gradient_accumulation_steps:
            scheduler = GradientAccumulationScheduler(
                scheduling={config.training.gradient_accumulation_start:
                                config.training.gradient_accumulation_steps}
            )
            self.callbacks.append(scheduler)
