from torch.utils.data import DataLoader


class ExperimentData:

    def train(self) -> DataLoader:
        raise NotImplementedError

    def dev(self) -> DataLoader:
        raise NotImplementedError

    def test(self) -> DataLoader:
        raise NotImplementedError
