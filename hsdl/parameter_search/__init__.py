import copy
import itertools
import json
import os
import random
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd

from hsdl import util


tqdm = util.get_tqdm()


class Placeholder:

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f'Placeholder for {self.name}'


class SearchDimension:

    def __init__(self, attr: str, values: List):
        self.attr = attr
        self.values = values


class GridDimension(SearchDimension):

    def __init__(self, attr: str, values: List):
        super().__init__(attr, values)


class RandomDimension(SearchDimension):

    def __init__(self, attr: str, low: float, high: float, k: int,
                 seed: int = 42, granularity: int = 100,
                 include_poles: bool = True):
        self.low = low
        self.high = high
        self.k = k
        self.seed = seed
        self.granularity = granularity

        random.seed(seed)

        candidates = list(
            np.linspace(self.low, self.high, self.k * granularity))
        if include_poles:
            candidates = [x for x in candidates if x not in [low, high]]
            values = [low, high]
            values += list(random.sample(candidates, k=k-2))
        else:
            values = random.sample(candidates, k=k)

        values = list(sorted(values))

        super().__init__(attr, values)


class SearchSubSpace:
    """A subspace is a collection of search dimensions."""

    def __init__(self, dimensions: List[SearchDimension]):
        self.dimensions = dimensions


class SearchSpace:
    """A search space is the union of cross products of 1+ sub spaces."""

    # TODO: pass config as defaults, track types in the loop, cast all columns,
    #  and fill NaN with the defaults - should avoid ints turning into floats.
    def __init__(self,
                 sub_spaces: List[SearchSubSpace],
                 config=None):
        self.space = self.build_space(sub_spaces)
        self.attrs = list(self.space.columns)  # NOTE: ix is index
        self.config = config

    def __len__(self):
        return len(self.space)

    def __getitem__(self, ix: int):
        return self.space.loc[ix].to_dict()

    @staticmethod
    def build_space(sub_spaces: List[SearchSubSpace]) -> pd.DataFrame:
        space = []
        ix = 0
        for sub_space in sub_spaces:
            attrs = [x.attr for x in sub_space.dimensions]
            values = [x.values for x in sub_space.dimensions]
            value_space = itertools.product(*values)
            for values in value_space:
                ix += 1
                row = {'ix': ix}
                for j in range(len(attrs)):
                    row[attrs[j]] = values[j]
                space.append(row)
        space = pd.DataFrame(space)
        space.set_index('ix', inplace=True)
        return space


class SearchResults:

    def __init__(self, experiment):
        self.experiment = experiment
        self.config = experiment.config
        self.space = experiment.search_space
        self.results = self.new_or_load_results()

    def __contains__(self, ix: int):
        return ix in self.results.ix.unique()

    def best(self):
        grouping = ['ix']
        grouping += self.space.attrs
        means = self.results.groupby(grouping).mean().reset_index()
        if self.config.metric['criterion'] == 'min':
            best_metric = means.val_metric.min()
        else:
            best_metric = means.val_metric.max()
        best_params = means[means.val_metric == best_metric]
        best_params = best_params[grouping]
        return best_metric, best_params.to_dict('records')

    @property
    def file_path(self):
        return os.path.join(self.experiment.config.results_dir,
                            self.experiment.config.experiment_name,
                            'param_search_results.csv')

    def load(self):
        return pd.read_csv(self.file_path)

    def new_or_load_results(self):
        if os.path.exists(self.file_path):
            return self.load()
        else:
            return self.new_results()

    def new_results(self):
        columns = ['ix']
        columns += self.space.attrs
        columns += ['val_metric']
        return pd.DataFrame(columns=columns, data=[])

    def report(self, ix: int, params: Dict, val_metric: float):
        self.results = self.results.append({
                'ix': ix,
                **params,
                'val_metric': val_metric,
            },
            ignore_index=True)
        self.save()

    def save(self):
        self.results.to_csv(self.file_path, index=False)


class ParameterSearch:

    def __init__(self,
                 experiment,
                 k_tie_break: int = 1,
                 max_tie_breaks: int = 1):
        self.experiment = experiment
        self.config = experiment.config
        self.search_space = experiment.search_space
        self.results = SearchResults(experiment)
        self.k_tie_break = k_tie_break
        self.max_tie_breaks = max_tie_breaks

    def __call__(self):
        # search over the space
        with tqdm(total=len(self.search_space), desc='Param Combination') \
                as pbar:
            for ix in range(1, len(self.search_space) + 1):
                if ix not in self.results:
                    self.evaluate(ix)
                pbar.update()

        # get best results after first pass
        best_metric, best_params = self.results.best()

        # tie break if needed
        tries = 0
        while len(best_params) > 1:
            tries += 1
            tqdm.write('Found %s combinations with best performance of %s.'
                       % (len(best_params), best_metric))
            tqdm.write('Performing tie break...')
            with tqdm(total=self.k_tie_break) as pbar:
                for _ in range(self.k_tie_break):
                    for params in best_params:
                        seed = random.choice(range(10000))
                        self.evaluate(params['ix'], seed)
                        pbar.update()
            best_metric, best_params = self.results.best()
            if tries >= self.max_tie_breaks and len(best_params) > 1:
                best_params = [random.choice(best_params)]

        best_params = best_params[0]
        best_params.pop('ix')
        tqdm.write('Grid search complete. Best: %s' % best_metric)
        util.aligned_print(best_params, indent=1)

        with open(self.best_params_path, 'w') as f:
            f.write(json.dumps(best_params))

        self.remove_grid_results_folders()

        return best_params

    @property
    def best_params_path(self):
        return os.path.join(self.experiment.config.results_dir,
                            self.experiment.config.experiment_name,
                            'best_params.json')

    def evaluate(self, ix: int, seed=42):
        config = copy.deepcopy(self.experiment.config)
        params = self.search_space[ix]
        config.merge(params)
        tqdm.write('Grid search for param combination %s/%s...'
                   % (ix, len(self.search_space)))
        util.aligned_print(params, indent=1)

        _, __ = self.experiment.train(
            config, seed, run_no=0, is_grid_search=True)
        df_run = self.experiment.results.df_run(0)  # always zero with deletes

        if self.config.metric['criterion'] == 'min':
            best_metric = df_run.val_metric.min()
        else:
            best_metric = df_run.val_metric.max()
        self.results.report(ix, params, best_metric)

        # remove the checkpoints after a grid search
        self.experiment.results.remove_run_checkpoints(
            self.experiment.results.n_runs_reported - 1)

    def remove_grid_results_folders(self):
        folders = [x for x in os.listdir(self.experiment.results.dir)
                   if x.startswith('version')]
        for folder in folders:
            path = os.path.join(self.experiment.results.dir, folder)
            shutil.rmtree(path)
            tqdm.write(f'Removed {path}.')
