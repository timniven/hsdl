from hsdl.config.base import Config


class TrainingConfig(Config):
    """Config class for common training settings."""

    def __init__(self,
                 max_epochs: int,
                 train_batch_size: int,
                 tune_batch_size: int,
                 min_epochs: int = 1,
                 gradient_clip_val: float = 0.,
                 auto_scale_batch_size: bool = False,
                 no_cuda=False,
                 num_collate_workers: int = 4,
                 **kwargs):
        super().__init__(**kwargs)
        # NOTE: this is a target, but account for memory limit, so use property
        self.train_batch_size = train_batch_size
        self.no_cuda = no_cuda
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.gradient_clip_val = gradient_clip_val
        self.num_collate_workers = num_collate_workers
        self.tune_batch_size = tune_batch_size
        self.auto_scale_batch_size = 'binsearch' \
            if auto_scale_batch_size else None
