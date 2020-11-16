"""Linear regression experiment test case."""
import os
import shutil
from typing import List

import numpy as np
from pytorch_lightning import LightningDataModule
from sklearn.datasets import load_iris
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hsdl.config.base import Config
from hsdl.annealing.config import ReduceLROnPlateauConfig
from hsdl.experiments import Experiment, ExperimentConfig
from hsdl.metrics.config import MetricConfig
from hsdl.modules import BaseModule
from hsdl.optimization.config import AdamConfig
from hsdl.parameter_search import SearchSpace, SearchSubSpace, GridDimension
from hsdl.stopping.config import NoValImprovementConfig
from hsdl.training.config import TrainingConfig


exp_dir = 'temp/test_logreg'


def clear_dir():
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)


def create_dir():
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)


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


class LogisticRegression(BaseModule):

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.linear = nn.Linear(4, 3)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.linear(x.float())
        return logits

    def loss(self, logits: Tensor, y: Tensor) -> Tensor:
        return F.cross_entropy(logits, y)

    def training_step(self, batch: List[Tensor], batch_ix: int) -> Tensor:
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log_training_step(logits, y, loss)
        return loss

    def validation_step(self, batch: List[Tensor], batch_ix: int):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log_validation_step(logits, y, loss)

    def test_step(self, batch: List[Tensor], batch_ix: int):
        x, y = batch
        logits = self.forward(x)
        self.log_test_step(logits, y)


config = ExperimentConfig(
    experiment_name='test_logreg',
    model=None,
    metric=MetricConfig(
        name='acc',
        criterion='max'),
    training=TrainingConfig(
        max_epochs=2,
        train_batch_size=16,
        n_gpus=0,
        tune_batch_size=16),
    annealing=ReduceLROnPlateauConfig(
        factor=0.2,
        patience=2),
    optimization=AdamConfig(lr=0.1),
    stopping=NoValImprovementConfig(
        monitor='val_loss',
        criterion='min',
        patience=2,
        k=2),
    results_dir='temp',
    n_runs=2)
search_space = SearchSpace([
    SearchSubSpace([GridDimension('optimization.lr', [0.3, 0.1, 0.09])])
])
experiment = Experiment(
    module_constructor=LogisticRegression,
    data=IrisData(),
    config=config,
    search_space=search_space)
