import os
from typing import Callable

import torch
from torch import nn

from hsdl import annealing, metrics, optimization, stopping
from hsdl.experiments.config import ExperimentConfig
from hsdl.models import Model
from hsdl.training.batch import Batch
from hsdl.training.saver import Saver
from hsdl.training.state import TrainState
from hsdl.util import TqdmWrapper


class TrainableModel:
    """Training wrapper for a nn.Module."""

    def __init__(self, model: Model, config: ExperimentConfig):
        self.model = model
        self.config = config
        self.metric = metrics.get(config.metric)
        self.optimizer = optimization.get(
            config.optimization, model.optim_params())
        self.stop = stopping.get(config.stopping, config.metric)
        self.anneal = annealing.get(config.annealing, self.optimizer)

        self.train_state = TrainState()  # TODO: get or load
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.saver = Saver(ckpt_dir=self.config.ckpt_dir)  # TODO

    # TODO: feels like there are different algorithms for this
    #  some are epochs
    #  others maybe just iterations, no epoch concept
    def train(self, train_loader, dev_loader):
        # use cuda if available
        self.model.to(self.device)

        # variables for training loop
        self.train_state.reset()  # TODO: resume?
        epoch_pbar = TqdmWrapper(self.config.training.n_epochs, 'epoch')
        iter_pbar = TqdmWrapper(len(train_loader), 'iter')
        eval_pbar = TqdmWrapper(len(dev_loader), 'tune')

        # put model in training state
        self.model.train()

        # start an epoch
        # TODO: resume
        for epoch in range(1, self.config.training.n_epochs + 1):
            self.train_state.epoch += 1
            iter_pbar.restart()

            # start an iter
            for batch in train_loader:
                self.train_state.step += 1
                batch.to(self.device)
                loss, logits = self.model(**batch)
                preds = logits.max(dim=1).indices.detach().cpu().numpy()
                labels = batch.labels.detach().cpu().numpy()
                tr_metric = self.metric(labels, preds)
                if self.cfg.train.grad_accum_steps > 1:
                    loss = loss / self.cfg.train.grad_accum_steps
                loss.backward()
                self.train_state.n_tr_x += batch.labels.size(0)
                self.train_state.cum_tr_metric += tr_metric
                self.train_state.cum_tr_loss += loss.detach().cpu().numpy()

                if (self.train_state.step + 1) \
                        % self.cfg.train.grad_accum_steps == 0:
                    optimizer.step()
                    self.model.zero_grad()
                    metric = self.train_state.cum_tr_metric \
                             / self.train_state.step
                    loss = self.train_state.cum_tr_loss \
                           / self.train_state.step
                    self.train_state.train_metrics.append({
                        'step': self.train_state.step,
                        self.metric.abbr: self.train_state.cum_tr_metric
                                          / self.train_state.step,
                    })
                    self.train_state.train_losses.append({
                        'step': self.train_state.step,
                        'loss': loss,
                    })
                    iter_pbar.set_description(
                        '(J: %4.3f, M: %3.2f)' % (loss, metric))

                # thus ends an iteration
                iter_pbar.update()

            # tuning
            dev_metric, _, = self.evaluate(dev_loader, eval_pbar)
            self.train_state.dev_metrics.append({
                'step': self.train_state.step,
                self.metric.abbr: dev_metric
            })
            is_best = self.metric.is_best(
                score=dev_metric, scores=self.train_state.dev_metrics)

            # learning rate annealing
            if self.cfg.anneal.epoch:
                annealing.step(dev_metric)

            # plot diagnostics
            self.plot_diagnostics()

            # save params
            self.saver.save(self.model, self.cfg.experiment_name, is_best)

            # thus ends an epoch
            epoch_pbar.update()
            epoch_pbar.set_description(
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













    def load_best(self):
        # TODO: will raise error if no ckpt, should have readable error message
        self.saver.load(
            model=self.model,
            name=self.cfg.experiment_name,
            is_best=True,
            load_optimizer=False)

    def evaluate(self, data_loader, pbar=None):
        self.model.eval()
        cum_metric = 0.
        n_steps, n_x = 0, 0
        predictions = []
        if not pbar:
            pbar = TqdmWrapper(len(data_loader), 'eval')
        else:
            pbar.restart()

        for i, batch in enumerate(data_loader):
            i += 1
            batch.to(self.device)

            with torch.no_grad():
                _, logits = self.model(**batch)

            n_x += batch.labels.size(0)
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            preds = logits.max(dim=1).indices.detach().cpu().numpy()
            labels = batch.labels.detach().cpu().numpy()
            tmp_metric = self.metric(labels, preds)
            cum_metric += tmp_metric
            n_steps += 1
            correct = preds == labels

            # TODO: make more general
            # for i in range(len(batch)):
            #     predictions.append({
            #         'prob0': probs[i][0],
            #         'prob1': probs[i][1],
            #         'pred': preds[i],
            #         'correct': correct[i]})

            pbar.update()
            pbar.set_description('(%5.2f)' % (cum_metric / i))

        metric = cum_metric / len(data_loader)

        return metric, predictions
