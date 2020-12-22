from typing import Tuple, Union

from pytorch_lightning import LightningModule
from torch import Tensor

from hsdl import annealing, metrics, optimization
from hsdl.config import ExperimentConfig


class BaseModule(LightningModule):

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.train_metric = metrics.get_lightning_metric(config, self)
        self.val_metric = metrics.get_lightning_metric(config, self)
        self.test_metric = metrics.get_lightning_metric(config, self)
        # idea is overriding class defines the model in the constructor

    def configure_optimizers(self):
        optimizer = optimization.get(self.config, self.parameters())
        if self.config.annealing:
            annealer = annealing.get(self.config, optimizer, verbose=True)
            return {
                'optimizer': optimizer,
                'lr_scheduler': annealer,
                'monitor': self.config.annealing.monitor,
            }
        else:
            return optimizer

    def log_training_step(self,
                          logits: Tensor,
                          y: Tensor,
                          loss: Tensor):
        self.log('train_loss', loss)
        self.train_metric(logits, y)
        self.log('train_metric',
                 self.train_metric.compute(),
                 on_step=True,
                 on_epoch=True)

    def log_validation_step(self,
                            logits: Tensor,
                            y: Tensor,
                            loss: Tensor):
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        self.val_metric(logits, y)
        self.log('val_metric',
                 self.val_metric.compute(),
                 on_epoch=True,
                 on_step=False)

    def log_test_step(self,
                      logits: Tensor,
                      y: Tensor):
        self.test_metric(logits, y)
        self.log('test_metric',
                 self.val_metric.compute(),
                 on_epoch=True,
                 on_step=False)

    def training_step(self, batch: Tuple, batch_ix: int) -> Tensor:
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log_training_step(logits, y, loss)
        return loss

    def validation_step(self, batch: Tuple, batch_ix: int) -> Tensor:
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log_training_step(logits, y, loss)
        return loss

    def test_step(self, batch: Tuple, batch_ix: int) -> Tensor:
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log_training_step(logits, y, loss)
        return loss
