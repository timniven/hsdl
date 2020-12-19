import unittest

import torch

from hsdl.metrics import config, Loss
from tests import logreg


class TestLoss(unittest.TestCase):

    def test_cumulative_loss(self):
        loss_fn = lambda x, y: (x == y).sum()
        metric = Loss(loss_fn)

        # batch 1: loss = 0.75
        logits = torch.Tensor([0, 1, 0, 0]).long()
        y = torch.Tensor([1, 1, 0, 0]).long()
        metric(logits, y)

        # batch 2: loss = 0.25
        logits = torch.Tensor([0, 1, 1, 0]).long()
        y = torch.Tensor([0, 0, 0, 1]).long()
        metric(logits, y)

        result = metric.compute()
        self.assertEqual(0.5, result)

    def test_experiment(self):
        experiment = logreg.experiment
        experiment.config.metric = config.Loss('min')
        experiment.run()
        # basically just seeing if it runs normally.
