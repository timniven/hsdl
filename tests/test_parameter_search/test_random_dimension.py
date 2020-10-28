import unittest

from hsdl.parameter_search import RandomDimension


class TestRandomDimension(unittest.TestCase):

    def test_space_populates_deterministically(self):
        dim = RandomDimension(
            attr='test',
            low=0.1,
            high=0.3,
            k=5,
            seed=42,
            granularity=100)
        expected = [
            0.23106212424849698,
            0.12284569138276553,
            0.10480961923847695,
            0.25190380761523046,
            0.1561122244488978,
        ]
        self.assertEqual(expected, dim.values)
