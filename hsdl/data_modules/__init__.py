from torch.utils.data import Dataset


class PandasDataset(Dataset):

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __getitem__(self, ix: int) -> pd.Series:
        return self.df.iloc[ix]

    def __len__(self):
        return len(self.df)
