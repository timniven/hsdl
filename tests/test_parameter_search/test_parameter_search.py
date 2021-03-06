import os
import shutil
import unittest

from hsdl.parameter_search import ParameterSearch
from tests import logreg


class TestParameterSearch(unittest.TestCase):

    def setUp(self):
        self.results_dir = 'temp/test_logreg'
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.mkdir(self.results_dir)

    def test_call(self):
        search = ParameterSearch(logreg.experiment)
        error = False
        try:
            _ = search()
        except Exception as e:
            error = True
            raise e
        self.assertFalse(error)

    def test_evaluate(self):
        search = ParameterSearch(logreg.experiment)
        search.evaluate(1)
        self.assertEqual(1, len(search.results.results))
        result = search.results.results.iloc[0]
        self.assertEqual(1.0, result['ix'])
        self.assertEqual(0.3, result['optimization.lr'])
        self.assertEqual(0.41, round(result.val_loss, 2))

    def tearDown(self):
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
