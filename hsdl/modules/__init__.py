from pytorch_lightning import LightningModule

from hsdl import annealing, metrics, optimization
from hsdl.config import ExperimentConfig


class BaseModule(LightningModule):

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.train_metric = metrics.get_lightning_metric(config)
        self.val_metric = metrics.get_lightning_metric(config)
        self.test_metric = metrics.get_lightning_metric(config)
        # idea is overriding class defines the model in the constructor

    def log_training_step(self, logits, y, loss):
        self.log('train_loss', loss)
        self.train_metric(logits, y)
        self.log('train_metric',
                 self.train_metric.compute(),
                 on_step=True,
                 on_epoch=True)

    def log_validation_step(self, logits, y):
        self.val_metric(logits, y)
        self.log('val_metric',
                 self.val_metric.compute(),
                 on_epoch=True,
                 on_step=False)

    def log_test_step(self, logits, y):
        self.test_metric(logits, y)
        self.log('test_metric',
                 self.val_metric.compute(),
                 on_epoch=True,
                 on_step=False)

    def configure_optimizers(self):
        optimizer = optimization.get(self.config, self.parameters())
        if self.config.annealing:
            annealer = annealing.get(self.config, optimizer, True)
            return {
                'optimizer': optimizer,
                'lr_scheduler': annealer,
                'monitor': 'val_metric',
            }
        else:
            return optimizer
