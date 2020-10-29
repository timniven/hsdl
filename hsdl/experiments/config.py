from typing import Union

from hsdl.config.base import Config


class ExperimentConfig(Config):
    """Config for an experiment."""

    def __init__(self,
                 experiment_name: str,
                 model: Union[Config, None],
                 metric: Config,
                 training: Config,
                 annealing: Config,
                 optimization: Config,
                 stopping: Config,
                 results_dir: str,
                 n_runs: int = 20,
                 **kwargs):
        super().__init__(**kwargs)
        self.experiment_name = experiment_name
        self.model = model
        self.metric = metric
        self.training = training
        self.annealing = annealing
        self.optimization = optimization
        self.stopping = stopping
        self.n_runs = n_runs
        self.results_dir = results_dir
