import os
from typing import Callable, Optional, Tuple

from .data import ExperimentData
from .results import ExperimentResults
from hsdl import training, util
from hsdl.experiments.config import ExperimentConfig
from hsdl.parameter_search import ParameterSearch, SearchSpace
from hsdl.training import TrainableModel


tqdm = util.get_tqdm()


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
        self.results = ExperimentResults(self)

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
