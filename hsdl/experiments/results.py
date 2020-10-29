import json
import os
from typing import Any, Dict

import pandas as pd

from hsdl.experiments.config import ExperimentConfig


class ExperimentResults:
    """Wrapper for experiment results."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.dir = os.path.join(config.results_dir, config.experiment_name)

    def best_params(self):
        if not os.path.exists(self.best_params_path):
            return False
        with open(self.best_params_path) as f:
            params = json.loads(f.read())
            return params

    @property
    def best_params_path(self):
        return os.path.join(self.dir, 'best_params.json')

    def df_metrics(self):
        if os.path.exists(self.metrics_path):
            return pd.read_csv(self.metrics_path)
        else:
            return pd.DataFrame(
                columns=['run_no', 'seed', 'subset', self.config.metric.name],
                data=[])

    @property
    def metrics_path(self):
        return os.path.join(self.dir, 'metrics.csv')

    @property
    def n_runs_completed(self):
        df = self.df_metrics()
        if len(df) == 0:
            return 0
        return int(df.run_no.max())

    def report_metric(self, run_no: int, seed: int, subset: str, metric: float):
        df = self.df_metrics()
        df = df.append({
                'run_no': run_no,
                'seed': seed,
                'subset': subset,
                self.config.metric.name: metric,
            },
            ignore_index=True)
        df.to_csv(self.metrics_path, index=False)

    def save_best_params(self, params: Dict[str, Any]):
        with open(self.best_params_path, 'w+') as f:
            f.write(json.dumps(params))

    def summarize(self):
        df = self.df_metrics()
        summary = f'{self.config.experiment_name} results:\n'
        for subset in df.subset.unique():
            m = df[df.subset == subset]
            summary += f'\t{subset} {self.config.metric.name}:\n'
            summary += '\t\tMax: %5.4f\n' % m[self.config.metric.name].max()
            summary += '\t\tMean: %5.4f\n' % m[self.config.metric.name].mean()
            summary += '\t\tStd: %5.4f\n' % m[self.config.metric.name].std()
        return summary
