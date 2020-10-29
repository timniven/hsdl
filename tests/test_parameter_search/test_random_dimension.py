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
            granularity=100,
            include_poles=False)
        expected = [
            0.10480961923847695,
            0.12284569138276553,
            0.1561122244488978,
            0.23106212424849698,
            0.25190380761523046,
        ]
        self.assertEqual(expected, dim.values)

    def test_space_includes_poles_when_requested(self):
        dim = RandomDimension(
            attr='test',
            low=0.1,
            high=0.3,
            k=5,
            seed=42,
            granularity=100)
        expected = [
            0.1,
            0.10521042084168337,
            0.12324649298597194,
            0.2314629258517034,
            0.3
        ]
        self.assertEqual(expected, dim.values)
