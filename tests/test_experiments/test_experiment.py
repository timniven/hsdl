import unittest

from tests import logreg


class TestExperiment(unittest.TestCase):

    def test_train(self):
        experiment = logreg.experiment
        experiment.train(42)
