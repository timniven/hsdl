from typing import List

import pandas as pd
from test_tube import Experiment as TestTubeExperiment


class ExperimentResults:
    """Wrapper for experiment results."""

    def __init__(self, experiment: TestTubeExperiment):
        self.experiment = experiment

    def __len__(self):
        return len(set(x['run_no'] for x in self.metrics))

    @property
    def df_metrics(self):
        return pd.DataFrame(self.metrics)

    @property
    def df_preds(self):
        return pd.DataFrame(self.preds)

    @property
    def n_runs_completed(self):
        return self.df_metrics.run_no.max()

    def report_preds(self, run_no: int, seed: int, subset: str,
                     preds: List[float]):
        for pred in preds:
            self.preds.append({
                'run_no': run_no,
                'seed': seed,
                'subset': subset,
                **pred
            })

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
