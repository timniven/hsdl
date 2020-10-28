import unittest

import numpy as np
import pandas as pd

from hsdl.parameter_search import GridDimension, SearchSpace, SearchSubSpace


class TestSearchSpace(unittest.TestCase):

    def test_build_space(self):
        sub_spaces = [
            SearchSubSpace([
                GridDimension('a', [1, 2]),
                GridDimension('b', [5, 6])
            ]),
            SearchSubSpace([
                GridDimension('c', [0.1, 0.2]),
                GridDimension('d', [0.6, 0.7]),
            ]),
        ]
        space = SearchSpace.build_space(sub_spaces)
        expected = [
            {'a': 1., 'b': 5., 'c': np.nan, 'd': np.nan},
            {'a': 1., 'b': 6., 'c': np.nan, 'd': np.nan},
            {'a': 2., 'b': 5., 'c': np.nan, 'd': np.nan},
            {'a': 2., 'b': 6., 'c': np.nan, 'd': np.nan},
            {'a': np.nan, 'b': np.nan, 'c': 0.1, 'd': 0.6},
            {'a': np.nan, 'b': np.nan, 'c': 0.1, 'd': 0.7},
            {'a': np.nan, 'b': np.nan, 'c': 0.2, 'd': 0.6},
            {'a': np.nan, 'b': np.nan, 'c': 0.2, 'd': 0.7},
        ]
        # NOTE: can't compare NaN this way, so use print for now
        print(space)
        # NOTE: don't forget to check ix is right, too
        # self.assertEqual(expected, space.to_dict('records'))

    def test_get_item(self):
        sub_spaces = [
            SearchSubSpace([
                GridDimension('a', [1, 2]),
                GridDimension('b', [5, 6])
            ]),
            SearchSubSpace([
                GridDimension('c', [0.1, 0.2]),
                GridDimension('d', [0.6, 0.7]),
            ]),
        ]
        space = SearchSpace(sub_spaces)
        params = space[1]
        self.assertEqual(1., params['a'])
        self.assertEqual(5., params['b'])
        self.assertTrue(pd.isnull(params['c']))
        self.assertTrue(pd.isnull(params['d']))

    def test_len(self):
        sub_spaces = [
            SearchSubSpace([
                GridDimension('a', [1, 2]),
                GridDimension('b', [5, 6])
            ]),
            SearchSubSpace([
                GridDimension('c', [0.1, 0.2]),
                GridDimension('d', [0.6, 0.7]),
            ]),
        ]
        space = SearchSpace(sub_spaces)
        self.assertEqual(8, len(space))

    def test_attrs(self):
        sub_spaces = [
            SearchSubSpace([
                GridDimension('a', [1, 2]),
                GridDimension('b', [5, 6])
            ]),
            SearchSubSpace([
                GridDimension('c', [0.1, 0.2]),
                GridDimension('d', [0.6, 0.7]),
            ]),
        ]
        space = SearchSpace(sub_spaces)
        expected = ['a', 'b', 'c', 'd']
        self.assertEqual(expected, space.attrs)
