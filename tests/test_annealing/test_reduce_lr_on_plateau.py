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
        n_runs = []
        for run_no in range(1, experiment.results.n_runs_completed + 1):
            df_run = experiment.results.df_run(run_no)
            print(df_run)
            raise Exception
            # TODO: IT WORKS OK, NOT SURE HOW TO TEST HERE THO
            if 'epoch_stopped' in df_run.columns:
                n_runs.append(df_run.epoch_stopped.max())
        self.assertNotEqual({20}, set(n_runs))

    def tearDown(self):
        logreg.clear_dir()
