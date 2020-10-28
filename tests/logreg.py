"""Linear regression experiment test case."""
from typing import List

import numpy as np
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.metrics import Accuracy
from sklearn.datasets import load_iris
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hsdl.config.base import Config
from hsdl.annealing.config import NoAnnealingConfig
from hsdl.experiments import Experiment, ExperimentConfig
from hsdl.metrics.config import MetricConfig
from hsdl.optimization.config import AdamConfig
from hsdl.parameter_search import SearchSpace, SearchSubSpace, GridDimension
from hsdl.stopping.config import NoEarlyStoppingConfig
from hsdl.training.config import TrainingConfig


class IrisDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, ix):
        return self.X[ix], self.y[ix]

    def __len__(self):
        return self.X.shape[0]


class IrisData(LightningDataModule):

    def __init__(self):
        super().__init__()
        data = load_iris()
        X = data['data']
        y = np.expand_dims(data['target'], axis=1)
        D = np.concatenate([X, y], axis=1)
        np.random.seed(42)
        D = np.random.permutation(D)
        X = D[:, 0:4]
        y = D[:, 4].squeeze().astype(int)
        self.train = IrisDataset(X[0:50, :], y[0:50])
        self.val = IrisDataset(X[50:100, :], y[50:100])
        self.test = IrisDataset(X[100:150, :], y[100:150])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=16, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=16, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=16, num_workers=4)


class LogisticRegressionConfig(Config):
    pass


class LogisticRegression(LightningModule):

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(4, 3)
        self.train_metric = Accuracy()
        self.val_metric = Accuracy()
        self.test_metric = Accuracy()

    def forward(self, X: Tensor) -> np.array:
        logits = self.linear(X.float())
        return logits

    def training_step(self, batch: List[Tensor], batch_ix: int):
        X, y = batch
        logits = self.linear(X.float())
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        self.train_metric(logits, y)
        self.log('train_metric', self.train_metric.compute(), on_step=True,
                 on_epoch=True)
        return loss

    def validation_step(self, batch: List[Tensor], batch_ix: int):
        X, y = batch
        logits = self.linear(X.float())
        self.val_metric(logits, y)
        self.log('val_metric', self.val_metric.compute(), on_epoch=True,
                 on_step=False)

    def test_step(self, batch: List[Tensor], batch_ix: int):
        X, y = batch
        logits = self.linear(X.float())
        self.test_metric(logits, y)
        self.log('test_metric', self.val_metric.compute(), on_epoch=True,
                 on_step=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.1)


config = ExperimentConfig(
    experiment_name='test_logreg',
    model=None,
    metric=MetricConfig(
        name='acc',
        criterion='max'),
    training=TrainingConfig(
        n_epochs=2,
        train_batch_size=16,
        tune_batch_size=16),
    annealing=NoAnnealingConfig(),
    optimization=AdamConfig(lr=0.1),
    stopping=NoEarlyStoppingConfig(),
    results_dir='temp',
    ckpt_dir='temp',
    n_runs=2)

search_space = SearchSpace([
    SearchSubSpace([GridDimension('optimization.lr', [0.3, 0.1, 0.09])])
])

experiment = Experiment(
    module_constructor=LogisticRegression,
    data=IrisData(),
    config=config,
    search_space=search_space)
