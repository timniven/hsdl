from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from hsdl import annealing, stopping, util
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
        # self.annealing = annealing.get(config, optimizer)
        self.stopping = stopping.get(config)
        self.stopped = False

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.stopped:
            return -1
        stop, reason = self.stopping(self.logger)
        if stop:
            tqdm.write(reason)
            self.stopped = True
            return -1
