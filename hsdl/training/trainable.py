from typing import Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from hsdl import annealing, metrics, optimization, stopping, util
from hsdl.experiments.config import ExperimentConfig
from hsdl.models import Model
from hsdl.training.saver import Saver
from hsdl.training.state import TrainState
from hsdl.util import TqdmWrapper


tqdm = util.get_tqdm()


class Trainable:
    """Default training wrapper for a nn.Module."""

    def __init__(self, model: Model, config: ExperimentConfig,
                 run_no: Optional[int] = None):
        self.model = model
        self.config = config
        self.run_no = run_no
        self.metric = metrics.get(config.metric)
        self.optimizer = optimization.get(
            config.optimization, model.optim_params())
        self.stop = stopping.get(config.stopping, config.metric)
        self.anneal = annealing.get(config.annealing, self.optimizer)
        self.train_state = TrainState(config.ckpt_dir, run_no)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.saver = Saver(experiment_name=config.experiment_name,
                           ckpt_dir=config.ckpt_dir)
        # set these when running
        self.epoch_pbar = None
        self.iter_pbar = None
        self.eval_pbar = None

    def train(self, train_loader, dev_loader):
        # use cuda if available
        self.model.to(self.device)

        # variables for training loop
        self.epoch_pbar = TqdmWrapper(self.config.training.n_epochs, 'epoch')
        self.iter_pbar = TqdmWrapper(len(train_loader), 'iter')
        self.eval_pbar = TqdmWrapper(len(dev_loader), 'tune')

        # put model in training state
        self.model.train()

        # start an epoch
        for epoch in range(self.train_state.epoch + 1,
                           self.config.training.n_epochs + 1):
            self.train_state.epoch += 1
            self.iter_pbar.restart()

            # start an iter
            for batch in train_loader:
                self.train_state.step += 1

                # forward pass
                batch.to(self.device)
                loss, logits = self.model(batch)
                preds = logits.max(dim=1).indices.detach().cpu().numpy()
                labels = batch.labels.detach().cpu().numpy()

                # backward pass
                if self.config.training.grad_accum_steps > 1:
                    loss = loss / self.config.training.grad_accum_steps
                loss.backward()

                # record loss and metrics
                tr_metric = self.metric(labels, preds)
                self.train_state.n_tr_x += batch.labels.size(0)
                self.train_state.cum_tr_metric += tr_metric
                self.train_state.cum_tr_loss += loss.detach().cpu().numpy()

                # for gradient accumulation
                if (self.train_state.step + 1) \
                        % self.config.training.grad_accum_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                    metric = self.train_state.cum_tr_metric \
                             / self.train_state.step
                    loss = self.train_state.cum_tr_loss \
                           / self.train_state.step

                    # report progress to the pbar
                    self.iter_pbar.set_description(
                        '(J: %4.3f, M: %3.2f)' % (loss, metric))

                # thus ends an iteration
                self.iter_pbar.update()

            # tuning
            dev_metric, _, = self.evaluate(dev_loader)
            self.train_state.dev_metrics.append({
                'step': self.train_state.step,
                self.metric.abbr: dev_metric
            })
            is_best = self.metric.is_best(
                score=dev_metric, scores=self.train_state.dev_metrics)

            # learning rate annealing
            if self.config.annealing.epoch:
                self.anneal.step(dev_metric)

            # save params
            self.saver.save(self.model, self.config.experiment_name, is_best)

            # thus ends an epoch
            self.epoch_pbar.update()
            self.epoch_pbar.set_description(
                '(best: %5.2f)'
                % self.metric.best(self.train_state.dev_metrics))

            # early stopping
            stop, message = self.stop(self.train_state)
            if stop:
                tqdm.write(message)
                break

        # report end of training and load best model
        tqdm.write('Training completed.')
        self.load_best()

    def load_best(self) -> None:
        # TODO: will raise error if no ckpt, should have readable error message
        self.saver.load(
            model=self.model,
            name=self.config.experiment_name,
            is_best=True,
            load_optimizer=False)

    def evaluate(self, data_loader: DataLoader):
        self.model.eval()
        cum_metric = 0.
        n_steps, n_x = 0, 0
        predictions = []
        self.eval_pbar.restart()

        for i, batch in enumerate(data_loader):
            i += 1
            batch.to(self.device)

            with torch.no_grad():
                _, logits = self.model(batch)

            n_x += batch.labels.size(0)
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            preds = logits.max(dim=1).indices.detach().cpu().numpy()
            labels = batch.labels.detach().cpu().numpy()
            tmp_metric = self.metric(labels, preds)
            cum_metric += tmp_metric
            n_steps += 1
            correct = preds == labels

            # TODO: make more general
            for i in range(len(batch)):
                predictions.append({
                    'prob': probs[i],
                    'pred': preds[i],
                    'correct': correct[i]})

            self.eval_pbar.update()
            self.eval_pbar.set_description('(%5.2f)' % (cum_metric / i))

        metric = cum_metric / len(data_loader)

        return metric, predictions
