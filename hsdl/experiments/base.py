import json
import os
from typing import Callable, List, Optional, Tuple

import pandas as pd
from torch.utils.data import DataLoader

from hsdl.config import Config
from hsdl import training, util
from hsdl.experiments.config import ExperimentConfig
from hsdl.parameter_search import ParameterSearch, SearchSpace
from hsdl.training import TrainableModel


tqdm = util.get_tqdm()


class ExperimentData:

    def train(self) -> DataLoader:
        raise NotImplementedError

    def dev(self) -> DataLoader:
        raise NotImplementedError

    def test(self) -> DataLoader:
        raise NotImplementedError


class Experiment:
    """Default experiment class and algorithm."""

    def __init__(self,
                 model_constructor: Callable,
                 data: ExperimentData,
                 config: ExperimentConfig,
                 ckpt_root_dir: str,
                 results_root_dir: str,
                 search_space: Optional[SearchSpace] = None):
        self.model_constructor = model_constructor
        self.data = data
        self.config = config
        self.search_space = search_space
        self.ckpt_root_dir = ckpt_root_dir
        self.results_root_dir = results_root_dir
        self.results = Results(self)

    @property
    def ckpt_dir(self):
        return os.path.join(self.ckpt_root_dir, self.config.experiment_name)

    @property
    def results_dir(self):
        return os.path.join(self.results_root_dir, self.config.experiment_name)

    def run(self, memory_limit: Optional[int] = None):
        # if there is a memory limit, set it on the config
        if memory_limit:
            self.config.training.memory_limit = memory_limit

        # do parameter search if required
        if self.search_space:
            search = ParameterSearch(self, self.search_space)
            best_params = search()
            self.config.merge(best_params)

        for run_no in range(len(self.results) + 1, self.config.n_runs + 1):
            seed = util.new_random_seed()
            util.set_random_seed(seed)

            model = self.train(self.config, seed)

            # obtain evaluation results and predictions
            train_metric, train_preds = model.evaluate(self.data.train())
            dev_metric, dev_preds = model.evaluate(self.data.dev())
            test_metric, test_preds = model.evaluate(self.data.test())

            # save metrics
            self.results.report_metric(run_no, seed, 'train', train_metric)
            self.results.report_metric(run_no, seed, 'dev', dev_metric)
            self.results.report_metric(run_no, seed, 'test', test_metric)

            # save predictions
            self.results.report_preds(run_no, seed, 'train', train_preds)
            self.results.report_preds(run_no, seed, 'dev', dev_preds)
            self.results.report_preds(run_no, seed, 'test', test_preds)

        # report results
        tqdm.write(self.results.summarize())

    def train(self, config: ExperimentConfig, seed: int = 42) -> TrainableModel:
        util.set_random_seed(seed)

        model = self.model_constructor(**config.model)
        model = training.TrainableModel(model, config)
        model.train(self.data.train(), self.data.dev())

        return model

    def train_and_validate(self, config: ExperimentConfig, seed: int = 42) -> \
            Tuple[float, float]:
        # this is for parameter search

        util.set_random_seed(seed)

        # init model and train
        model = self.train(config, seed)

        # obtain evaluation results and predictions
        train_metric, _ = model.evaluate(self.data.train())
        dev_metric, _ = model.evaluate(self.data.dev())

        return train_metric, dev_metric


class Results:
    """Basic wrapper for experiment results."""

    def __init__(self, experiment: Experiment):
        self.experiment_name = experiment.config.experiment_name
        self.results_dir = experiment.results_dir
        self.metric_name = experiment.config.metric.name
        self.metrics, self.preds = self.new_or_load()

    def __len__(self):
        return len(set(x['run_no'] for x in self.metrics))

    def df_metrics(self):
        return pd.DataFrame(self.metrics)

    def df_preds(self):
        return pd.DataFrame(self.preds)

    @property
    def file_path(self):
        return os.path.join(self.results_dir, 'results.json')

    def load_data(self):
        with open(self.file_path, 'r') as f:
            data = json.loads(f.read())
        return data

    def new_or_load(self):
        if os.path.exists(self.file_path):
            data = self.load_data()
            return data['metrics'], data['preds']
        else:
            return [], []

    def report_metric(self, run_no: int, seed: int, subset: str, metric: float):
        self.metrics.append({
            'run_no': run_no,
            'seed': seed,
            'subset': subset,
            self.metric_name: metric,
        })

    def report_preds(self, run_no: int, seed: int, subset: str,
                     preds: List[float]):
        for pred in preds:
            self.preds.append({
                'run_no': run_no,
                'seed': seed,
                'subset': subset,
                **pred
            })

    def save(self):
        results = {
            'metrics': self.metrics,
            'preds': self.preds,
        }
        with open(self.file_path, 'w+') as f:
            f.write(json.dumps(results))

    def summarize(self):
        df = self.df_metrics()
        summary = f'{self.experiment_name} results:'
        for subset in df.subset.unique():
            m = df[df.subset == subset]
            summary += f'\t{subset} {self.metric_name}:'
            summary += '\t\tMax: %5.4f' % m[self.metric_name].max()
            summary += '\t\tMean: %5.4f' % m[self.metric_name].mean()
            summary += '\t\tStd: %5.4f' % m[self.metric_name].std()
        return summary
