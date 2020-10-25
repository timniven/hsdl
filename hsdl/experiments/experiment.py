from typing import Callable, Optional, Tuple

from pytorch_lightning import LightningDataModule, LightningModule, \
    seed_everything, Trainer
from pytorch_lightning.loggers import TestTubeLogger
from torch.utils.data import DataLoader

from hsdl import util
from hsdl.config import Config
from hsdl.experiments.config import ExperimentConfig
from hsdl.experiments.results import ExperimentResults
from hsdl.parameter_search import ParameterSearch, SearchSpace
from hsdl.training import get_trainer


tqdm = util.get_tqdm()


class Experiment:
    """Default experiment class and algorithm."""

    def __init__(self,
                 model_constructor: Callable[[Config], LightningModule],
                 data: LightningDataModule,
                 config: ExperimentConfig,
                 search_space: Optional[SearchSpace] = None):
        self.model_constructor = model_constructor
        self.data = data
        self.config = config
        self.search_space = search_space
        self.logger = TestTubeLogger(
            name=config.experiment_name,
            save_dir=config.results_dir)
        self.results = ExperimentResults(self.logger.experiment)

    def run(self):
        # do parameter search if required
        if self.search_space:
            # TODO: is the second argument not redundant?
            search = ParameterSearch(self, self.search_space)
            best_params = search()
            self.config.merge(best_params)

        # train and get results
        for run_no in range(self.results.n_runs_completed + 1,
                            self.config.n_runs + 1):
            seed = util.new_random_seed()
            trainer = self.train(seed)
            self.test_all(trainer, run_no, seed)

        # report results
        tqdm.write(self.results.summarize())

    def train(self, seed: int, debug: bool = False) -> Trainer:
        seed_everything(seed)
        model = self.model_constructor(self.config.model)
        trainer = get_trainer(config=self.config,
                              logger=self.logger,
                              debug=debug)
        train_dataloader = self.data.train_dataloader()
        val_dataloader = self.data.val_dataloader()
        trainer.fit(model, train_dataloader, val_dataloader)
        return trainer

    def test_all(self, trainer: Trainer, run_no: int, seed: int):
        # TODO: mightn't always have train, val, test
        #  - so provide a way to control that
        train_metric, train_preds = trainer.test(
            self.data.train_dataloader())
        dev_metric, dev_preds = trainer.test(
            self.data.val_dataloader())
        test_metric, test_preds = trainer.test(
            self.data.test_dataloader())

        self.logger.experiment.log({
            'run_no': run_no,
            'seed': seed,
            'train_metric': train_metric,
            'dev_metric': dev_metric,
            'test_metric': test_metric,
        })

        # TODO: how to log preds?
