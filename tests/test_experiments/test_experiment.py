import os
import shutil
import unittest

from tests import logreg


class TestExperiment(unittest.TestCase):

    def setUp(self):
        if os.path.exists('temp/test_logreg'):
            shutil.rmtree('temp/test_logreg')
        os.mkdir('temp/test_logreg')

    def test_train_does_not_raise_exception(self):
        experiment = logreg.experiment
        error = False
        try:
            experiment.train(42)
        except Exception as e:
            error = True
            # raise e  # if debugging
        self.assertFalse(error)

    def test_test_all(self):
        experiment = logreg.experiment
        trainer = experiment.train(42)
        train, val, test = experiment.test_all(trainer, 1, 42)
        self.assertEqual(0.70, round(train, 2))
        self.assertEqual(0.66, round(val, 2))
        self.assertEqual(0.64, round(test, 2))

    def test_run(self):
        pass

    def tearDown(self):
        if os.path.exists('temp/test_logreg'):
            shutil.rmtree('temp/test_logreg')
