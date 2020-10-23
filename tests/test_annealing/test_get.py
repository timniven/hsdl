import unittest

from torch import nn
from torch.optim import Adam, lr_scheduler

from hsdl import annealing
from hsdl.annealing import config


class TestGet(unittest.TestCase):

    def setUp(self):
        linear = nn.Linear(5, 5)
        self.optimizer = Adam(linear.parameters())

    def test_none(self):
        anneal = annealing.get(config.NoAnnealingConfig(), self.optimizer)
        self.assertIsNone(anneal)

    def test_plateau(self):
        anneal = annealing.get(config.ReduceLROnPlateauConfig(0.5, 1),
                               self.optimizer)
        self.assertIsInstance(anneal, lr_scheduler.ReduceLROnPlateau)
