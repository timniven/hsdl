import itertools
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd

from hsdl import util
from hsdl.experiments import Experiment


tqdm = util.get_tqdm()


class SearchDimension:

    def __init__(self, attr: str, values: List):
        self.attr = attr
        self.values = values


class GridDimension(SearchDimension):

    def __init__(self, attr: str, values: List):
        super().__init__(attr, values)


class RandomDimension(SearchDimension):

    def __init__(self, attr: str, low: float, high: float, k: int,
                 seed: int = 42, granularity: int = 100):
        self.low = low
        self.high = high
        self.k = k
        self.seed = seed
        self.granularity = granularity
        candidates = list(
            np.linspace(self.low, self.high, self.k * granularity))
        random.seed(seed)
        values = random.sample(candidates, k=k)
        super().__init__(attr, values)


class SearchSubSpace:

    def __init__(self, dimensions: List[SearchDimension]):
        self.dimensions = dimensions


class SearchSpace:

    def __init__(self, sub_spaces: List[SearchSubSpace]):
        self.space = self.build_space(sub_spaces)
        self.attrs = self.space.columns  # NOTE: ix is index

    def __len__(self):
        return len(self.space)

    def get_params(self, ix: int):
        return self.space.iloc[ix].to_dict()

    def build_space(self, sub_spaces: List[SearchSubSpace]):
        space = []
        ix = 0
        for sub_space in sub_spaces:
            attrs = [x.attr for x in sub_space.dimensions]
            values = [x.values for x in sub_space.dimensions]
            value_space = itertools.product(values)
            for values in value_space:
                ix += 1
                for attr, value in zip(attrs, values):
                    space.append({
                        'ix': ix,
                        'attr': attr,
                        'value': value,
                    })
        space = pd.DataFrame(space)
        space.set_index('ix', inplace=True)
        return space


class SearchResults:

    def __init__(self, experiment: Experiment, search_space: SearchSpace):
        self.experiment = experiment
        self.space = search_space
        self.results = self.new_or_load_results()

    def __contains__(self, item):
        return item in self.results

    def best(self):
        best_score = self.experiment.config.metric.best(self.results.dev.values)
        best_params = self.results[self.results.dev == best_score]
        best_params = best_params[self.space.attrs]
        return best_params.to_dict('records')

    @property
    def file_path(self):
        return os.path.join(self.experiment.results_dir,
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
        columns += ['train', 'dev']
        return pd.DataFrame(columns=columns, data=[])

    def report(self, ix: int, params: Dict, train: float, dev: float):
        self.results.append({
            'ix': ix,
            **params,
            'train': train,
            'dev': dev,
        })
        self.save()

    def save(self):
        # index True, since ix is index
        self.results.to_csv(self.file_path)


class ParameterSearch:

    def __init__(self, experiment: Experiment, search_space: SearchSpace):
        self.experiment = experiment
        self.search_space = search_space
        self.search_results = SearchResults(experiment, search_space)

    def __call__(self):
        # search over the space
        with tqdm(total=len(self.search_space), desc='Param Combination') \
                as pbar:
            for ix in range(len(self.search_space)):
                if ix not in self.search_results:
                    self.evaluate(ix)
                pbar.update()

        # get best results after first pass
        best_metric, best_params = self.search_results.best()

        # tie break if needed
        if len(best_params) > 1:
            tqdm.write('Found %s combinations with best performance of %s.'
                       % (len(best_params), best_metric))
            tqdm.write('Performing tie break...')
            for _ in range(3):
                seed = random.choice(range(10000))
                for combination in combinations:
                    self.evaluate(combination, seed, tie_break=True)
            best_acc, combinations = self.winning_combinations()

        best_params = combinations[0]
        tqdm.write('Grid search complete. Best acc: %s. Params:' % best_acc)
        util.aligned_print(
            keys=list(best_params.keys()),
            values=list(best_params.values()),
            indent=1)

        tqdm.write('Saving grid best params...')
        with open(self.params_path, 'w') as f:
            best_config = self.config.copy()
            for key, value in best_params.items():
                setattr(best_config, key, value)
            f.write(json.dumps(best_config.__dict__))

        return best_acc, best_params

    def evaluate(self, ix: int, seed=42):
        config = self.experiment.config.copy()
        params = self.search_space.get_params(ix)
        config = config.merge(params)
        tqdm.write('Grid search for param combination %s/%s...'
                   % (ix, len(self.search_space)))
        # TODO: pretty print?
        tqdm.write(params)

        train_metric, dev_metric = self.experiment. \
            train_and_validate(config, seed)

        self.search_results.report(ix, params, train_metric, dev_metric)











class GridSearch:

    def __init__(self, model, cfg, train_loader, dev_loader, search_space):
        self.experiment_name = experiment_name
        self.model_constructor = model_constructor
        self.data_loaders = data_loaders
        self.config = config
        self.search_space = search_space
        self.search_keys = search_space.keys()
        self.grid_path = grid_path(experiment_name)
        self.params_path = params_path(experiment_name)
        self.data, self.columns = self.get_or_load_data()

    def __call__(self):
        tqdm.write('Conducting grid search for %s...' % self.experiment_name)

        for combination in self.combinations:
            if not self.evaluated(combination):
                self.evaluate(combination)
            else:
                print('Already evaluated this combination:')
                for key, value in combination.items():
                    print('\t%s:\t%s' % (key, value))

        best_acc, combinations = self.winning_combinations()
        if best_acc == 0.5:  # i.e. random performance on dev
            tqdm.write('All dev accs are random - taking best train acc.')
            # take the run with the best training acc
            best_acc, combinations = self.winning_train_acc_combinations()
            while len(combinations) > 1:
                tqdm.write('Found %s combinations with best train acc of %s.'
                           % (len(combinations), best_acc))
                tqdm.write('Performing tie break...')
                for _ in range(5):
                    seed = random.choice(range(10000))
                    for combination in combinations:
                        self.evaluate(combination, seed, tie_break=True)
                best_acc, combinations = self.winning_combinations()
        else:
            while len(combinations) > 1:
                tqdm.write('Found %s combinations with best acc of %s.'
                           % (len(combinations), best_acc))
                tqdm.write('Performing tie break...')
                for _ in range(5):
                    seed = random.choice(range(10000))
                    for combination in combinations:
                        self.evaluate(combination, seed, tie_break=True)
                best_acc, combinations = self.winning_combinations()

        best_params = combinations[0]
        tqdm.write('Grid search complete. Best acc: %s. Params:' % best_acc)
        util.aligned_print(
            keys=list(best_params.keys()),
            values=list(best_params.values()),
            indent=1)

        tqdm.write('Saving grid best params...')
        with open(self.params_path, 'w') as f:
            best_config = self.config.copy()
            for key, value in best_params.items():
                setattr(best_config, key, value)
            f.write(json.dumps(best_config.__dict__))

        return best_acc, best_params

    @property
    def combinations(self):
        keys = self.search_space.keys()
        values = list(self.search_space.values())
        i = 0
        for _values in itertools.product(*values):
            combination = dict(zip(keys, _values))
            combination['id'] = i
            i += 1
            yield combination

    def evaluate(self, combination, seed=42, tie_break=False):
        tqdm.write('Evaluating param combination%s:'
                   % ' (tie break)' if tie_break else '')
        args = copy.deepcopy(self.config)
        for key, value in combination.items():
            setattr(args, key, value)
            self.data[key].append(value)
        args.seed = seed
        args.print()
        model = self.model_constructor(args)
        accs, _, __ = training.train(args, model, self.data_loaders)
        self.data['seed'].append(args.seed)
        self.data['train_acc'].append(accs['train'])
        self.data['dev_acc'].append(accs['dev'])
        self.data['test_acc'].append(accs['test'])
        df = pd.DataFrame(data=self.data, columns=self.columns)
        df.to_csv(grid_path(self.experiment_name), index=False)

    def evaluated(self, combination):
        if not os.path.exists(self.grid_path):
            return False
        df = pd.read_csv(self.grid_path)
        for key, value in combination.items():
            if isinstance(value, float):
                df = df[np.isclose(df[key], value)]
            else:
                df = df[df[key] == value]
        return len(df) > 0

    def get_or_load_data(self):
        # init the dict and columns
        data = {'id': []}
        columns = ['id']
        for key in self.search_keys:
            data[key] = []
            columns.append(key)
        data['seed'] = []
        data['train_acc'] = []
        data['dev_acc'] = []
        data['test_acc'] = []
        columns += ['seed', 'train_acc', 'dev_acc', 'test_acc']

        # load any old data
        if os.path.exists(self.grid_path):
            df = pd.read_csv(self.grid_path)
            data['id'] = list(df.id.values)
            for key in self.search_keys:
                data[key] = list(df[key].values)
            data['train_acc'] = list(df.train_acc.values)
            data['dev_acc'] = list(df.dev_acc.values)
            data['test_acc'] = list(df.test_acc.values)
            data['seed'] = list(df.seed.values)

        return data, columns

    @staticmethod
    def get_query(combination):
        query = ''
        for key, value in combination.items():
            if isinstance(value, str):
                value = "'%s'" % value
            else:
                value = str(value)
            query += ' & %s == %s' % (key, value)
        query = query[3:]
        return query

    @staticmethod
    def parse_dict(_dict):
        # wish I didn't need this hack for pandas
        # github issues reckons it should be solved in 24.0?
        keys = _dict.keys()
        values = []
        for value in _dict.values():
            if isinstance(value, np.bool_):
                value = bool(value)
            if isinstance(value, np.float64):
                value = float(value)
            if isinstance(value, np.int64):
                value = int(value)
            values.append(value)
        return dict(zip(keys, values))

    def winning_combinations(self):
        df = pd.read_csv(self.grid_path)
        best_acc = df.dev_acc.max()
        rows = df[df.dev_acc == best_acc]
        wanted_columns = list(self.search_keys) + ['id']
        column_selector = [c in wanted_columns for c in df.columns]
        if len(rows) > 1:  # have a tie break
            ids = rows.id.unique()
            ids_avgs = []
            for _id in ids:
                id_rows = df[df.id == _id]
                avg = id_rows.dev_acc.mean()
                ids_avgs.append((_id, avg))
            best_avg_acc = max(x[1] for x in ids_avgs)
            best_ids = [x[0] for x in ids_avgs if x[1] == best_avg_acc]
            combinations = []
            for _id in best_ids:
                rows = df[df.id == _id].loc[:, column_selector]
                combinations.append(rows.iloc[0].to_dict())
            best_acc = max(best_acc, best_avg_acc)
        else:
            rows = rows.loc[:, column_selector]
            combinations = [r[1].to_dict() for r in rows.iterrows()]
        combinations = [self.parse_dict(d) for d in combinations]
        return best_acc, combinations

    def winning_train_acc_combinations(self):
        df = pd.read_csv(self.grid_path)
        best_acc = df.train_acc.max()
        rows = df[df.train_acc == best_acc]
        wanted_columns = list(self.search_keys) + ['id']
        column_selector = [c in wanted_columns for c in df.columns]
        if len(rows) > 1:  # have a tie break
            ids = rows.id.unique()
            ids_avgs = []
            for _id in ids:
                id_rows = df[df.id == _id]
                avg = id_rows.train_acc.mean()
                ids_avgs.append((_id, avg))
            best_avg_acc = max(x[1] for x in ids_avgs)
            best_ids = [x[0] for x in ids_avgs if x[1] == best_avg_acc]
            combinations = []
            for _id in best_ids:
                rows = df[df.id == _id].loc[:, column_selector]
                combinations.append(rows.iloc[0].to_dict())
            best_acc = max(best_acc, best_avg_acc)
        else:
            rows = rows.loc[:, column_selector]
            combinations = [r[1].to_dict() for r in rows.iterrows()]
        combinations = [self.parse_dict(d) for d in combinations]
        return best_acc, combinations
