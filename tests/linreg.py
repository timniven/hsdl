"""Linear regression experiment test case."""
import numpy as np
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.metrics import Accuracy
from sklearn.datasets import load_iris
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class IrisBatch:

    def __init__(self, X, y):
        self.X = X
        self.y = y


class IrisDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, ix):
        return IrisBatch(self.X[ix], self.y[ix])


class IrisData(LightningDataModule):

    def __init__(self):
        super().__init__()
        data = load_iris()
        self.train = IrisDataset(data['data'][0:50, :], data['target'][0:50])
        self.val = IrisDataset(data['data'][50:100, :], data['target'][50:100])
        self.test = IrisDataset(data['data'][100:150, :],
                                data['target'][100:150])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=16)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=16)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=16)


class LogisticRegression(LightningModule):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, batch: IrisBatch) -> np.array:
        X = batch.X
        logits = self.linear(X)
        preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
        return preds

    def training_step(self, batch: IrisBatch):
        X = batch.X
        y = batch.y
        logits = self.linear(X)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        self.train_acc(logits, y)
        self.log('train_acc', self.train_acc.compute(), on_step=True,
                 on_epoch=True)
        return loss

    def validation_step(self, batch: IrisBatch):
        X = batch.X
        y = batch.y
        logits = self.linear(X)
        self.val_acc(logits, y)
        self.log('val_acc', self.val_acc.compute(), on_epoch=True,
                 on_step=False)

    def test_step(self, batch: IrisBatch):
        X = batch.X
        y = batch.y
        logits = self.linear(X)
        self.test_acc(logits, y)
        self.log('test_acc', self.val_acc.compute(), on_epoch=True,
                 on_step=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.1)
