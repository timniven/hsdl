import copy
import unittest

from torch import nn
from torch.optim import Adam, lr_scheduler

from hsdl.annealing import get
from tests import logreg


class TestGet(unittest.TestCase):

    def setUp(self):
        linear = nn.Linear(5, 5)
        self.optimizer = Adam(linear.parameters())

    def test_none(self):
        config = copy.deepcopy(logreg.config)
        config.annealing = None
        anneal = get(config, self.optimizer)
        self.assertIsNone(anneal)

    def test_plateau(self):
        anneal = get(logreg.config, self.optimizer)
        self.assertIsInstance(anneal, lr_scheduler.ReduceLROnPlateau)
