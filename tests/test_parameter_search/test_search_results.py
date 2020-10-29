import os
import shutil
import unittest

from hsdl.parameter_search import SearchResults
from tests import logreg


class TestSearchResults(unittest.TestCase):

    def setUp(self):
        self.results_dir = 'temp/test_logreg'
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.mkdir(self.results_dir)

    def test_contains(self):
        # nothing when new and fresh
        results = SearchResults(experiment=logreg.experiment)
        self.assertFalse(1 in results)
        results.report(1, {'lr': 0.1}, 0.6, 0.5)
        self.assertTrue(1 in results)

    def test_new_results(self):
        results = SearchResults(experiment=logreg.experiment)
        df = results.new_results()
        self.assertEqual(['ix', 'optimization.lr', 'train', 'val'],
                         list(df.columns))

    def test_report(self):
        results = SearchResults(experiment=logreg.experiment)
        results.report(1, {'optimization.lr': 0.1}, 0.6, 0.5)
        expected = [{'ix': 1, 'optimization.lr': 0.1, 'train': 0.6, 'val': 0.5}]
        self.assertEqual(expected, results.results.to_dict('records'))

    def test_previous_results_loaded(self):
        results = SearchResults(experiment=logreg.experiment)
        results.report(1, {'optimization.lr': 0.1}, 0.6, 0.5)
        results = SearchResults(experiment=logreg.experiment)
        expected = [{'ix': 1, 'optimization.lr': 0.1, 'train': 0.6, 'val': 0.5}]
        self.assertEqual(expected, results.results.to_dict('records'))

    def test_best(self):
        results = SearchResults(experiment=logreg.experiment)
        results.report(1, {'optimization.lr': 0.1}, 0.6, 0.5)
        results.report(2, {'optimization.lr': 0.3}, 0.7, 0.7)
        results.report(2, {'optimization.lr': 0.3}, 0.7, 0.5)
        expected = [{'ix': 2, 'optimization.lr': 0.3}]
        best_metric, best_params = results.best()
        self.assertEqual(expected, best_params)

    def tearDown(self):
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
