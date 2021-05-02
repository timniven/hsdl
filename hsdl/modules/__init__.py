from typing import Optional, Tuple, Union

from pytorch_lightning import LightningModule
from torch import Tensor

from hsdl import annealing, metrics, optimization
from hsdl.experiments import ExperimentConfig


class BaseModule(LightningModule):

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.metrics = {
            'train': metrics.get_lightning_metric(config, self),
            'val': metrics.get_lightning_metric(config, self),
            'test': metrics.get_lightning_metric(config, self),
        }
        self.train_metric = self.metrics['train']
        self.val_metric = self.metrics['val']
        self.test_metric = self.metrics['test']
        # idea is overriding class defines the model in the constructor

    def configure_optimizers(self):
        optimizer = optimization.get(self.config, self.parameters())
        if self.config.annealing:
            annealer = annealing.get(self.config, optimizer, verbose=True)
            return {
                'optimizer': optimizer,
                'lr_scheduler': annealer,
                'monitor': self.config.annealing['monitor'],
            }
        else:
            return optimizer

    def add_metric(self,
                   y_hat: Tensor,
                   y: Union[Tensor, Tuple],
                   subset: str):
        self.metrics[subset](y_hat, y)

    def log_step(self,
                 subset: str,
                 y_hat: Tensor,
                 y: Union[Tensor, Tuple],
                 loss: Optional[Tensor] = None):
        if loss is not None:
            self.log(f'{subset}_loss', loss)
        self.add_metric(y_hat, y, subset)
        self.log(f'{subset}_metric',
                 self.metrics[subset].compute(),
                 on_step=subset == 'train',
                 on_epoch=True)

    def training_step(self, batch: Tuple, batch_ix: int) -> Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log_step('train', y_hat, y, loss)
        return loss

    def validation_step(self, batch: Tuple, batch_ix: int) -> Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log_step('val', y_hat, y, loss)
        return loss

    def test_step(self, batch: Tuple, batch_ix: int) -> Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log_step('test', y_hat, y, loss)
        return loss
