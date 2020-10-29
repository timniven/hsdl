import os
from typing import Callable, Optional, Tuple

from pytorch_lightning import LightningDataModule, LightningModule, \
    seed_everything, Trainer
from pytorch_lightning.loggers import TestTubeLogger

from hsdl import util
from hsdl.experiments.config import ExperimentConfig as Config
from hsdl.experiments.results import ExperimentResults
from hsdl.parameter_search import ParameterSearch, SearchSpace
from hsdl.training import get_trainer


tqdm = util.get_tqdm()


class Experiment:
    """Default experiment class and algorithm."""

    def __init__(self,
                 module_constructor: Callable[[Config], LightningModule],
                 data: LightningDataModule,
                 config: Config,
                 search_space: SearchSpace):
        self.module_constructor = module_constructor
        self.data = data
        self.config = config
        self.search_space = search_space
        self.results = ExperimentResults(config)
        if not os.path.exists(config.results_dir):
            os.mkdir(config.results_dir)

    def run(self):
        # do parameter search if required
        if self.search_space and not self.results.best_params:
            search = ParameterSearch(self)
            best_params = search()
            self.results.save_best_params(best_params)
            self.config.merge(best_params)

        # train and get results
        for run_no in range(self.results.n_runs_completed + 1,
                            self.config.n_runs + 1):
            seed = util.new_random_seed()
            trainer, module = self.train(self.config, seed, run_no)
            self.test_all(module, run_no, seed)

        # report results
        tqdm.write(self.results.summarize())

    def train(self,
              config,
              seed: int,
              run_no: Optional[int] = None,
              debug: bool = False) -> Tuple[Trainer, LightningModule]:
        seed_everything(seed)
        tqdm.write('Running experiment with config:')
        config.print()
        module = self.module_constructor(config)
        logger = TestTubeLogger(
            name=config.experiment_name,
            save_dir=config.results_dir,
            version=run_no)
        trainer = get_trainer(config=config,
                              logger=logger,
                              debug=debug)
        train_dataloader = self.data.train_dataloader()
        val_dataloader = self.data.val_dataloader()
        trainer.fit(module, train_dataloader, val_dataloader)
        return trainer, module

    def test_all(self, module: LightningModule, run_no: int, seed: int):
        # NOTE: couldn't get trainer.test to work for me, so doing it manually
        # TODO: this is not flexible in any way to the specifics of data & model
        train_metric = self.test_train(module)
        val_metric = self.test_val(module)
        test_metric = self.test_test(module)

        self.results.report_metric(
            run_no=run_no, seed=seed, subset='train', metric=train_metric)
        self.results.report_metric(
            run_no=run_no, seed=seed, subset='val', metric=val_metric)
        self.results.report_metric(
            run_no=run_no, seed=seed, subset='test', metric=test_metric)

        # this return is for testing
        return train_metric, val_metric, test_metric

    def test_train(self, module: LightningModule) -> float:
        module.train_metric.reset()
        for batch in self.data.train_dataloader():
            X, y = batch
            preds = module(X)
            module.train_metric(preds, y)
        return float(module.train_metric.compute().detach().cpu().numpy())

    def test_val(self, module: LightningModule) -> float:
        module.val_metric.reset()
        for batch in self.data.val_dataloader():
            X, y = batch
            preds = module(X)
            module.val_metric(preds, y)
        return float(module.val_metric.compute().detach().cpu().numpy())

    def test_test(self, module: LightningModule) -> float:
        module.test_metric.reset()
        for batch in self.data.test_dataloader():
            X, y = batch
            preds = module(X)
            module.test_metric(preds, y)
        return float(module.test_metric.compute().detach().cpu().numpy())
