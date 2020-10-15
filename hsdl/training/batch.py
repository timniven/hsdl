import torch


class Batch:
    """Base class for a batch."""

    def __init__(self, **kwargs):
        for attr, val in kwargs.items():
            setattr(self, attr, val)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __iter__(self):
        for key in self.__dict__.keys():
            yield key

    def __len__(self):
        return len(self.__dict__)

    def __contains__(self, item):
        return item in self.__dict__

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def get(self, item):
        return self.__dict__[item]

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return False

    def to(self, device):
        """Calls to(device) on all torch.Tensors on this batch object."""
        for attr, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))


class ClassificationBatch(Batch):
    """Base class for a batch for a classification task."""

    def __init__(self, labels, **kwargs):
        self.labels = labels
        super().__init__(**kwargs)
