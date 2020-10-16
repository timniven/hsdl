from hsdl.annealing import AnnealingConfig
from hsdl.config import Config
from hsdl.metrics import MetricConfig
from hsdl.optimization import OptimizationConfig
from hsdl.stopping import StoppingConfig
from hsdl.training import TrainingConfig


class ExperimentConfig(Config):
    """Config for an experiment."""

    def __init__(self,
                 experiment_name: str,
                 model: Config,
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
