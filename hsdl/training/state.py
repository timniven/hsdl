


class TrainState:
    """Wraps info about state of training."""

    def __init__(self):
        self.epoch = 0
        self.step = 0
        self.n_tr_x = 0
        self.cum_tr_loss = 0.
        self.cum_tr_metric = 0.
        self.train_losses = []
        self.train_metrics = []
        self.dev_metrics = []

    def reset(self):
        self.epoch = 0
        self.step = 0
        self.n_tr_x = 0
        self.cum_tr_loss = 0.
        self.cum_tr_metric = 0.
        self.train_losses = []
        self.train_metrics = []
        self.dev_metrics = []

    # TODO: load and save
