import os
import shutil
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.loggers import TestTubeLogger

from hsdl import util
from hsdl.data_modules import HsdlDataModule
from hsdl.experiments.config import ExperimentConfig as Config
from hsdl.experiments.results import ExperimentResults
from hsdl.parameter_search import ParameterSearch, SearchSpace
from hsdl.training import get_trainer


tqdm = util.get_tqdm()


class Experiment:
    """Default experiment class and algorithm."""

    def __init__(self,
                 module_constructor: Callable[[Config], LightningModule],
                 data_module: HsdlDataModule,
                 config: Config,
                 search_space: Optional[SearchSpace] = None,
                 grid_data_module: Optional[HsdlDataModule] = None):
        self.module_constructor = module_constructor
        self.data = data_module
        self.config = config
        self.validate_search_space(config, search_space)
        self.search_space = search_space
        self.grid_data = grid_data_module
        self.results = ExperimentResults(config)
        if not os.path.exists(config.results_dir):
            os.mkdir(config.results_dir)

    def best_epoch(self, run_no: int) -> int:
        df_run = self.results.df_run(run_no)
        best_run_metric = df_run.val_metric.min()
        best_epoch = int(
            df_run[df_run.val_metric == best_run_metric].iloc[0].epoch)
        return best_epoch

    def best_module(self, subset: str = 'test', run_no: Optional[int] = None):
        # get best run_no
        if not run_no:
            best_run = self.best_run(subset)
        else:
            best_run = run_no

        best_epoch = self.best_epoch(best_run)

        checkpoint_path = self.results.checkpoint_path(best_run, best_epoch)
        module = self.module_constructor.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=self.config)

        return module

    def best_metric(self, eval_metrics):
        fn = np.max if self.config.metric.criterion == 'max' else np.min
        return fn(eval_metrics)

    def best_run(self, subset: str = 'test') -> int:
        df_metrics = self.results.df_metrics()
        metric_name = self.config.metric.name
        eval_metrics = df_metrics[df_metrics.subset == subset]
        eval_metrics = eval_metrics[metric_name].values
        best_metric = self.best_metric(eval_metrics)
        best_run_no = df_metrics[
            df_metrics[metric_name] == best_metric].iloc[0].run_no
        return best_run_no

    def clean_run_folder(self, run_no: int) -> None:
        run_path = self.results.run_path(run_no)
        if os.path.exists(run_path):
            shutil.rmtree(run_path)

    def run(self):
        # do parameter search if required
        if self.search_space:
            tqdm.write('Running parameter search..')
            search = ParameterSearch(self)
            best_params = search()
            self.results.save_best_params(best_params)
            self.config.merge(best_params)

        # train and get results
        tqdm.write('Running final experiments...')
        remaining_runs = list(range(self.results.n_runs_completed + 1,
                                    self.config.n_runs + 1))
        with tqdm(total=len(remaining_runs)) as pbar:
            for run_no in remaining_runs:
                pbar.set_description(f'Run # {run_no} of {self.config.n_runs}')

                seed = util.new_random_seed()

                trainer, module = self.train(self.config, seed, run_no)

                # TODO: what's with the second condition?
                if self.config.metric is not None \
                        and self.config.metric['name'] is not None:
                    self.test_all(module, run_no, seed)

                # TODO: really need this?
                # try and prevent memory leak
                del module
                del trainer

                pbar.update()

        # report results
        tqdm.write(self.results.summarize())

    def train(self,
              config,
              seed: int,
              is_grid_search: bool = False,
              run_no: Optional[int] = None,
              debug: bool = False) -> Tuple[Trainer, LightningModule]:
        self.clean_run_folder(run_no)
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
        if is_grid_search and self.grid_data is not None:
            train_dataloader = self.grid_data.train_dataloader(config=config)
            val_dataloader = self.grid_data.val_dataloader(config=config)
        else:
            train_dataloader = self.data.train_dataloader(config=config)
            val_dataloader = self.data.val_dataloader(config=config)
        trainer.fit(
            model=module,
            train_dataloader=train_dataloader,
            val_dataloaders=[val_dataloader])
        if config.training['max_epochs'] > 1:
            ckpt_path = trainer.my_checkpoint_callback.best_model_path
            if ckpt_path is None:
                ckpt_path = trainer.my_checkpoint_callback.latest_model_path
            module = self.module_constructor.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                config=config)
        return trainer, module

    def test_all(self, module: LightningModule, run_no: int, seed: int) \
            -> List[float]:
        # NOTE: couldn't get trainer.test to work for me, so doing it manually
        metrics = []
        for subset in ['train', 'val', 'test']:
            metric = self.test(subset, module)
            if metric is not None:
                self.results.report_metric(
                    run_no=run_no,
                    seed=seed,
                    subset=subset,
                    metric=metric)
            metrics.append(metric)
        return metrics

    def test(self, subset: str, module: LightningModule) -> Union[float, None]:
        module.metrics[subset].reset()
        data = self.data[subset](self.config)
        if data is None:
            return None
        tqdm.write(f'Evaluating on {subset}...')
        with tqdm(total=len(data)) as pbar:
            for batch_ix, batch in enumerate(data):
                module.test_step(batch, batch_ix, subset)
                pbar.update()
        result = float(module.metrics[subset].compute().detach().cpu().numpy())
        tqdm.write(f'Result: {result}')
        return result

    @staticmethod
    def validate_search_space(config, search_space):
        if search_space is None:
            return
        error = False
        for attr in search_space.attrs:
            if attr not in config:
                tqdm.write(f'Search attribute "{attr}" not found in config.')
                error = True
        if error:
            raise ValueError('Search space attribute(s) not found in config.')
