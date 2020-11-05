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
        self.__train_batch_size = train_batch_size
        self.no_cuda = no_cuda
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.gradient_clip_val = gradient_clip_val
        self.num_collate_workers = num_collate_workers
        self.__tune_batch_size = tune_batch_size
        self.auto_scale_batch_size = 'binsearch' \
            if auto_scale_batch_size else None

    @property
    def grad_accum_steps(self):
        # this is a function of batch_size and memory limits for specific models
        # the memory limits are also computer (i.e. GPU) dependent.
        # the limits are controlled in this config by memory_limit.
        n_steps = max(int(self.__train_batch_size / self.memory_limit), 1)
        if n_steps == 0:
            raise ValueError(
                f'Erroring here: gradient_accumulation_steps should be '
                f'greater than zero.\n'
                f'\ttrain_batch_size: {self.__train_batch_size}\t'
                f'\tmemory_limit" {self.memory_limit}')
        return n_steps

    @property
    def memory_limit(self):
        if self.__memory_limit:
            return self.__memory_limit
        else:
            return max(self.__train_batch_size, self.__tune_batch_size)

    @memory_limit.setter
    def memory_limit(self, value):
        self.__memory_limit = value

    @property
    def train_batch_size(self):
        return int(self.__train_batch_size / self.grad_accum_steps)

    @property
    def tune_batch_size(self):
        return int(self.__tune_batch_size / self.grad_accum_steps)
