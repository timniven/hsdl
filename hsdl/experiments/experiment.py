from typing import Callable, Optional, Tuple

from pytorch_lightning import LightningDataModule, LightningModule, \
    seed_everything

from hsdl import util
from hsdl.experiments.config import ExperimentConfig
from hsdl.experiments.results import ExperimentResults
from hsdl.parameter_search import ParameterSearch, SearchSpace
from hsdl.training import get_trainer


tqdm = util.get_tqdm()


class Experiment:
    """Default experiment class and algorithm."""

    def __init__(self,
                 model_constructor: Callable,
                 data: LightningDataModule,
                 config: ExperimentConfig,
                 search_space: Optional[SearchSpace] = None):
        self.model_constructor = model_constructor
        self.data = data
        self.config = config
        self.search_space = search_space
        self.results = ExperimentResults(self)

    def run(self, memory_limit: Optional[int] = None):
        # if there is a memory limit, set it on the config
        if memory_limit:
            self.config.training.memory_limit = memory_limit

        # do parameter search if required
        if self.search_space:
            search = ParameterSearch(self, self.search_space)
            best_params = search()
            self.config.merge(best_params)

        # train and get results
        for run_no in range(len(self.results) + 1, self.config.n_runs + 1):
            seed = util.new_random_seed()
            seed_everything(seed)

            model = self.train()

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

    def train(self) -> LightningModule:
        model = self.model_constructor(self.config.model)
        trainer = get_trainer(self.config)
        train_dataloader = self.data.train_dataloader()
        val_dataloader = self.data.val_dataloader()
        trainer.fit(model, train_dataloader, val_dataloader)
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
