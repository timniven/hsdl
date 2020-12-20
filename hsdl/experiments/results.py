import json
import os
from typing import Any, Dict, Union

import pandas as pd

from hsdl import util
from hsdl.experiments.config import ExperimentConfig


tqdm = util.get_tqdm()


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

    def checkpoint_path(self, run_no: int, epoch: int):
        checkpoints_folder = os.path.join(
            self.config.results_dir,
            self.config.experiment_name,
            f'version_{run_no}',
            'checkpoints')
        checkpoints = os.listdir(checkpoints_folder)
        print(checkpoints)
        checkpoint_name = next(x for x in checkpoints
                               if f'epoch={epoch}' in x)
        return os.path.join(checkpoints_folder, checkpoint_name)

    def df_metrics(self) -> pd.DataFrame:
        if os.path.exists(self.metrics_path):
            return pd.read_csv(self.metrics_path)
        else:
            return pd.DataFrame(
                columns=['run_no', 'seed', 'subset', self.config.metric.name],
                data=[])

    def df_run(self, run_no: int) -> Union[pd.DataFrame, None]:
        if run_no > self.n_runs_completed:
            raise ValueError(f'Invalid run number: {run_no}. '
                             f'This experiment has {self.n_runs_completed} '
                             f'completed runs.')
        run_folder = os.path.join(self.dir, f'version_{run_no}')
        if not os.path.exists(run_folder):
            raise ValueError(f'Missing folder: {run_folder}')
        run_path = os.path.join(run_folder, 'metrics.csv')
        return pd.read_csv(run_path)

    @property
    def metrics_path(self):
        return os.path.join(self.dir, 'metrics.csv')

    @property
    def n_runs_completed(self):
        df = self.df_metrics()
        if len(df) == 0:
            return 0
        return int(df.run_no.max())

    @property
    def n_runs_reported(self):
        return sum(1 for x in os.listdir(self.dir) if x.startswith('version'))

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

    def remove_run_checkpoints(self, run_no: int):
        folder_path = os.path.join(self.dir, f'version_{run_no}', 'checkpoints')
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
            tqdm.write(f'Deleted {file_path}')

    def run_path(self, run_no: int):
        return os.path.join(self.dir, f'version_{run_no}')

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
