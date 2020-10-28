from typing import Union

from hsdl.annealing.config import AnnealingConfig
from hsdl.config.base import Config
from hsdl.metrics.config import MetricConfig
from hsdl.optimization.config import OptimizationConfig
from hsdl.stopping.config import StoppingConfig
from hsdl.training.config import TrainingConfig


class ExperimentConfig(Config):
    """Config for an experiment."""

    def __init__(self,
                 experiment_name: str,
                 model: Union[Config, None],
                 metric: MetricConfig,
                 training: TrainingConfig,
                 annealing: AnnealingConfig,
                 optimization: OptimizationConfig,
                 stopping: StoppingConfig,
                 results_dir: str,
                 ckpt_dir: str,
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
        self.ckpt_dir = ckpt_dir
