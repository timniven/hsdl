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
            experiment.train(experiment.config, 42)
        except Exception as e:
            error = True
            # raise e  # if debugging
        self.assertFalse(error)

    def test_test_all(self):
        experiment = logreg.experiment
        trainer, module = experiment.train(experiment.config, 42)
        train, val, test = experiment.test_all(module, 1, 42)
        self.assertEqual(0.86, round(train, 2))
        self.assertEqual(0.86, round(val, 2))
        self.assertEqual(0.84, round(test, 2))

    def test_run(self):
        pass

    def test_best_model_returned(self):
        experiment = logreg.experiment
        experiment.config.training.max_epochs = 4
        _, module = experiment.train(experiment.config, 42, 1)
        _, val, __ = experiment.test_all(module, 1, 42)
        df1 = experiment.results.df_run(1)
        best_loss = df1.val_loss.min()
        best_acc = df1[df1.val_loss == best_loss].iloc[0].val_metric
        self.assertEqual(round(best_acc, 4), round(val, 4))

    def tearDown(self):
        if os.path.exists('temp/test_logreg'):
            shutil.rmtree('temp/test_logreg')
