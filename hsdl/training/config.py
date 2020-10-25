from hsdl.config import Config


class TrainingConfig(Config):
    """Config class for common training settings."""

    def __init__(self, n_epochs, seed, train_batch_size, metric,
                 tune_batch_size, run_no=1, memory_limit=None, no_cuda=False,
                 **kwargs):
        super().__init__(**kwargs)
        # NOTE: this is a target, but account for memory limit, so use property
        self.__train_batch_size = train_batch_size
        self.no_cuda = no_cuda
        self.seed = seed
        self.run_no = run_no
        self.n_epochs = n_epochs
        self.__tune_batch_size = tune_batch_size
        self.metric = metric
        self.__memory_limit = memory_limit

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
