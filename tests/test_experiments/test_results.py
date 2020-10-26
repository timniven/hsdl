import os
import shutil
import unittest

from hsdl.experiments.config import ExperimentConfig
from hsdl.experiments.results import ExperimentResults
from hsdl.metrics import MetricConfig


class TestExperimentResults(unittest.TestCase):

    def setUp(self):
        self.config = ExperimentConfig(
            experiment_name='test1',
            model=None,
            metric=MetricConfig('Accuracy', 'max'),
            training=None,
            annealing=None,
            optimization=None,
            stopping=None,
            results_dir='temp',
            ckpt_dir='temp',
            n_runs=1)
        if os.path.exists('temp/test1'):
            shutil.rmtree('temp/test1')
        os.mkdir('temp/test1')

    def test_n_runs_completed(self):
        results = ExperimentResults(self.config)
        results.report_metric(1, 42, 'train', 0.42)
        results.report_metric(2, 69, 'train', 0.69)
        self.assertEqual(2, results.n_runs_completed)

    def test_report_metric(self):
        results = ExperimentResults(self.config)
        self.assertFalse(os.path.exists(results.metrics_path))
        results.report_metric(1, 42, 'train', 0.42)
        self.assertTrue(os.path.exists(results.metrics_path))
        # NOTE: test appending
        results.report_metric(2, 69, 'train', 0.69)
        expected = [
            {'run_no': 1, 'seed': 42, 'subset': 'train', 'Accuracy': 0.42},
            {'run_no': 2, 'seed': 69, 'subset': 'train', 'Accuracy': 0.69},
        ]
        # NOTE: tests df_metrics()
        df = results.df_metrics()
        self.assertEqual(expected, df.to_dict('records'))

    def test_save_best_params(self):
        # NOTE: also tests best_params()
        results = ExperimentResults(self.config)
        self.assertFalse(os.path.exists(results.best_params_path))
        params = {'a': 1}
        results.save_best_params(params)
        self.assertTrue(os.path.exists(results.best_params_path))
        best_params = results.best_params()
        self.assertEqual(params, best_params)

    def test_summarize(self):
        # NOTE: just makes sure it doesn't error
        error = False
        results = ExperimentResults(self.config)
        results.report_metric(1, 42, 'train', 0.42)
        results.report_metric(2, 69, 'train', 0.69)
        try:
            print(results.summarize())
        except Exception:
            error = True
        self.assertFalse(error)

    def tearDown(self):
        if os.path.exists('temp/test1'):
            shutil.rmtree('temp/test1')
