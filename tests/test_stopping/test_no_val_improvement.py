import unittest

from hsdl.stopping.config import NoValImprovementConfig
from tests import logreg


class TestNoValImprovement(unittest.TestCase):

    def setUp(self):
        logreg.clear_dir()
        logreg.create_dir()

    def test_stopping_occurs(self):
        experiment = logreg.experiment
        experiment.config.training.max_epochs = 8
        experiment.config.stopping = NoValImprovementConfig(
            patience=2,
            k=2,
            metric_config=experiment.config.metric)
        experiment.run()
        # NOTE: this also tests that the logging is occurring
        n_epochs = []
        for run_no in range(1, experiment.results.n_runs_completed + 1):
            df_run = experiment.results.df_run(run_no)
            n_epochs.append(df_run.epoch.max())
        self.assertNotEqual({8}, set(n_epochs))

    def tearDown(self):
        logreg.clear_dir()
