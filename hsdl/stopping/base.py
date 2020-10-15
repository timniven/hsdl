


class EarlyStopping:
    """Base early stopping class."""

    def __init__(self):
        pass

    def __call__(self, train_state):
        return self.stop(train_state)

    def stop(self, train_state):
        """Determine whether to stop.

        Args:
          train_state: TrainState object.

        Returns:
          stop: Bool, whether or not to stop.
          message: String, details of the decision.
        """
        raise NotImplementedError


class NoEarlyStopping(EarlyStopping):

    def __init__(self):
        super().__init__()

    def stop(self, train_state):
        return False, None


class NoDevImprovement(EarlyStopping):
    """Early stopping with no improvement on the dev set."""

    def __init__(self, patience, k, metric):
        """Create a new NoDevImprovement.

        Args:
          patience: Int, how many epochs to let run before checking.
          k: Int, how many epochs without improvement before stopping.
          metric: Metric object, used for the conditions.
        """
        super().__init__()
        self.patience = patience
        self.k = k
        self.metric = metrics.get(name=metric)

    def stop(self, train_state):
        if train_state.epoch >= self.patience:
            mets = [x[self.metric.abbr] for x in train_state.dev_metrics]
            k = mets[-self.k]  # baseline for checking improvement
            to_consider = mets[-self.k+1:]  # metrics to check
            # stop if all subsequent metrics are no better than the baseline
            stop = all(not self.metric.is_better(x, k) for x in to_consider)
            return stop, 'Early stopping condition met.'
        return False, ''

