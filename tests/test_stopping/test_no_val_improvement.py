import unittest

from hsdl.stopping.config import NoValImprovementConfig
from tests import logreg


class TestNoValImprovement(unittest.TestCase):

    def setUp(self):
        logreg.clear_dir()
        logreg.create_dir()

    def test_stopping_occurs(self):
        experiment = logreg.experiment
        experiment.config.training.max_epochs = 20
        experiment.config.stopping = NoValImprovementConfig(
            patience=2,
            k=2,
            metric_config=experiment.config.metric)
        experiment.run()
        df = experiment.results.df_metrics()
        print(df)

    def tearDown(self):
        logreg.clear_dir()
