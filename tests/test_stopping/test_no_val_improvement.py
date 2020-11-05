import unittest

from tests import logreg


class TestNoValImprovement(unittest.TestCase):

    def setUp(self):
        logreg.clear_dir()
        logreg.create_dir()

    def test_stopping_occurs(self):
        max_epochs = 8
        experiment = logreg.experiment
        experiment.config.training.max_epochs = max_epochs
        experiment.run()
        # NOTE: this also tests that the logging is occurring
        n_epochs = []
        for run_no in range(1, experiment.results.n_runs_completed + 1):
            df_run = experiment.results.df_run(run_no)
            n_epochs.append(df_run.epoch.max())
        self.assertNotEqual({max_epochs}, set(n_epochs))

    def tearDown(self):
        logreg.clear_dir()
