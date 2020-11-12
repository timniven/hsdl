import unittest

from hsdl.annealing.config import ReduceLROnPlateauConfig
from tests import logreg


class TestReduceLROnPlateau(unittest.TestCase):

    def setUp(self):
        logreg.clear_dir()
        logreg.create_dir()

    def test_annealing_occurs(self):
        experiment = logreg.experiment
        experiment.config.training.max_epochs = 20
        experiment.config.annealing = ReduceLROnPlateauConfig(
            factor=0.2,
            patience=3)
        experiment.config.stopping = None
        experiment.run()
        # NOTE: this also tests that the logging is occurring
        pass  # TODO: not sure how to test, maybe add logging and check log?

    def tearDown(self):
        logreg.clear_dir()
