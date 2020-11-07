import unittest

from hsdl.config.base import Config
from tests import logreg


class TestConfig(unittest.TestCase):

    def test_to_dict(self):
        config = Config(
            a=1,
            b=Config(c=2,
                     d=Config(e=3)))
        expected = {
            'a': 1,
            'b.c': 2,
            'b.d.e': 3
        }
        self.assertEqual(expected, config.to_dict())

    def test_merge(self):
        config = Config(
            a=1,
            b=Config(c=2,
                     d=Config(e=3)))
        config.merge({'a': 4, 'b.c': 5, 'b.d.e': 6})
        self.assertEqual(4, config['a'])
        self.assertEqual(5, config['b']['c'])
        self.assertEqual(6, config['b']['d']['e'])

    def test_contains_true_when_root_key_exists(self):
        config = logreg.config
        self.assertTrue('experiment_name' in config)

    def test_contains_false_when_root_key_does_not_exist(self):
        config = logreg.config
        self.assertFalse('some_garbage' in config)

    def test_contains_true_when_child_key_exists(self):
        config = logreg.config
        self.assertTrue('annealing.factor' in config)

    def test_contains_false_when_child_key_does_not_exist(self):
        config = logreg.config
        self.assertFalse('optimization.some_garbage' in config)
