from typing import Optional

from hsdl.config.base import Config


class TrainingConfig(Config):
    """Config class for common training settings."""

    def __init__(self,
                 max_epochs: int,
                 train_batch_size: int,
                 tune_batch_size: int,
                 gradient_accumulation_steps: Optional[int] = None,
                 gradient_accumulation_start: Optional[int] = None,
                 dropout: float = 0,
                 min_epochs: int = 1,
                 gradient_clip_val: float = 0.,
                 auto_scale_batch_size: bool = False,
                 no_cuda=False,
                 num_collate_workers: int = 4,
                 **kwargs):
        super().__init__(**kwargs)
        self.train_batch_size = train_batch_size
        self.tune_batch_size = tune_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_accumulation_start = gradient_accumulation_start
        self.dropout = dropout
        self.no_cuda = no_cuda
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.gradient_clip_val = gradient_clip_val
        self.num_collate_workers = num_collate_workers
        self.auto_scale_batch_size = 'binsearch' \
            if auto_scale_batch_size else None
