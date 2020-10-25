import json
import os
from typing import Optional


class TrainState:
    """Wraps info about state of training."""

    def __init__(self, ckpt_dir: str, run_no: Optional[int] = None):
        self.ckpt_dir = ckpt_dir
        self.run_no = run_no
        self.epoch = 0
        self.step = 0
        self.n_tr_x = 0
        self.cum_tr_loss = 0.
        self.cum_tr_metric = 0.
        self.dev_metrics = []
        if os.path.exists(self.file_path):
            self.load()

    def reset(self):  # TODO: delete?
        self.epoch = 0
        self.step = 0
        self.n_tr_x = 0
        self.cum_tr_loss = 0.
        self.cum_tr_metric = 0.
        self.dev_metrics = []

    @property
    def file_path(self):
        return os.path.join(self.ckpt_dir, 'train_state%s.json' % self.run_no)

    def load(self):
        with open(self.file_path) as f:
            data = json.loads(f.read())
        self.ckpt_dir = data['ckpt_dir']
        self.run_no = data['run_no']
        self.epoch = data['epoch']
        self.step = data['step']
        self.n_tr_x = data['n_tr_x']
        self.cum_tr_loss = data['cum_tr_loss']
        self.cum_tr_metric = data['cum_tr_metric']
        self.dev_metrics = data['dev_metrics']

    def save(self):
        data = {
            'ckpt_dir': self.ckpt_dir,
            'run_no': self.run_no,
            'epoch': self.epoch,
            'step': self.step,
            'n_tr_x': self.n_tr_x,
            'cum_tr_loss': self.cum_tr_loss,
            'cum_tr_metric': self.cum_tr_metric,
            'dev_metrics': self.dev_metrics,
        }
        with open(self.file_path, 'w+') as f:
            f.write(json.dumps(data))
