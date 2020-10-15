from hsdl.annealing import AnnealingConfig
from hsdl.config import Config
from hsdl.metrics import Metric
from hsdl.optimization import OptimizationConfig
from hsdl.stopping import StoppingConfig
from hsdl.training import TrainingConfig


class ExperimentConfig(Config):
    """Config for an experiment."""

    def __init__(self,
                 experiment_name: str,
                 model: Config,
                 metric: Metric,
                 training: TrainingConfig,
                 annealing: AnnealingConfig,
                 optimization: OptimizationConfig,
                 stopping: StoppingConfig,
                 n_runs=20,
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
