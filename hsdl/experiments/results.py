import json
import os
from typing import List

import pandas as pd


class ExperimentResults:
    """Basic wrapper for experiment results."""

    def __init__(self, experiment):
        self.experiment_name = experiment.config.experiment_name
        self.results_dir = experiment.results_dir
        self.metric_name = experiment.config.metric.name
        self.metrics, self.preds = self.new_or_load()

    def __len__(self):
        return len(set(x['run_no'] for x in self.metrics))

    def df_metrics(self):
        return pd.DataFrame(self.metrics)

    def df_preds(self):
        return pd.DataFrame(self.preds)

    @property
    def file_path(self):
        return os.path.join(self.results_dir, 'results.json')

    def load_data(self):
        with open(self.file_path, 'r') as f:
            data = json.loads(f.read())
        return data

    def new_or_load(self):
        if os.path.exists(self.file_path):
            data = self.load_data()
            return data['metrics'], data['preds']
        else:
            return [], []

    def report_metric(self, run_no: int, seed: int, subset: str, metric: float):
        self.metrics.append({
            'run_no': run_no,
            'seed': seed,
            'subset': subset,
            self.metric_name: metric,
        })

    def report_preds(self, run_no: int, seed: int, subset: str,
                     preds: List[float]):
        for pred in preds:
            self.preds.append({
                'run_no': run_no,
                'seed': seed,
                'subset': subset,
                **pred
            })

    def save(self):
        results = {
            'metrics': self.metrics,
            'preds': self.preds,
        }
        with open(self.file_path, 'w+') as f:
            f.write(json.dumps(results))

    def summarize(self):
        df = self.df_metrics()
        summary = f'{self.experiment_name} results:'
        for subset in df.subset.unique():
            m = df[df.subset == subset]
            summary += f'\t{subset} {self.metric_name}:'
            summary += '\t\tMax: %5.4f' % m[self.metric_name].max()
            summary += '\t\tMean: %5.4f' % m[self.metric_name].mean()
            summary += '\t\tStd: %5.4f' % m[self.metric_name].std()
        return summary
