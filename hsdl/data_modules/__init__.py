import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class PandasDataset(Dataset):

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __getitem__(self, ix: int) -> pd.Series:
        return self.df.iloc[ix]

    def __len__(self):
        return len(self.df)


class HsdlDataModule(LightningDataModule):

    def __init__(self):
        super().__init__()

    def __getitem__(self, subset: str):
        if subset == 'train':
            return self.train_dataloader
        elif subset == 'val':
            return self.val_dataloader
        elif subset == 'test':
            return self.test_dataloader
        else:
            raise ValueError(f'Unexpected subset: "{subset}".')
