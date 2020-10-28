import unittest

from hsdl.config.base import Config


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
