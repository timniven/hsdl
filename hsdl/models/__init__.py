from torch import nn


class Model(nn.Module):
    """Base model."""

    def optim_params(self):
        raise NotImplementedError('Models must implement')
