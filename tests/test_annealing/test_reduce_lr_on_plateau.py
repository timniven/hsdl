import unittest

from hsdl.stopping.config import NoValImprovementConfig
from tests import logreg


class TestReduceLROnPlateau(unittest.TestCase):

    def setUp(self):
        logreg.clear_dir()
        logreg.create_dir()

    def test_annealing_occurs(self):
        # TODO: this is so far a copy of TestNoValImprovement
        experiment = logreg.experiment
        experiment.config.training.max_epochs = 20
        experiment.config.stopping = NoValImprovementConfig(
            patience=2,
            k=2,
            metric_config=experiment.config.metric)
        experiment.run()
        df = experiment.results.df_metrics()
        # NOTE: this also tests that the logging is occurring
        n_runs = []
        for run_no in range(1, experiment.results.n_runs_completed + 1):
            df_run = experiment.results.df_run(run_no)
            if 'epoch_stopped' in df_run.columns:
                n_runs.append(df_run.epoch_stopped.max())
        self.assertNotEqual({20}, set(n_runs))

    def tearDown(self):
        logreg.clear_dir()
