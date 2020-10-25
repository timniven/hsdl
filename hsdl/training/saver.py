import os

import torch


class Saver:

    def __init__(self, experiment_name, ckpt_dir):
        self.experiment_name = experiment_name
        self.ckpt_dir = ckpt_dir

    def ckpt_path(self, name, module, is_best):
        return os.path.join(
            self.ckpt_dir,
            '%s_%s_%s' % (name, module, 'best' if is_best else 'latest'))

    def load(self, model, name, is_best=False, load_to_cpu=False,
             load_optimizer=True, replace_model=None, ignore_missing=False):
        model_path = self.ckpt_path(name, 'model', is_best)
        model_state_dict = self.get_state_dict(model_path, load_to_cpu)
        model_state_dict = self.replace_model_state(
            model_state_dict, replace_model)
        if ignore_missing:
            model_state_dict = self.drop_missing(model, model_state_dict)
        model.load_state_dict(model_state_dict)
        if load_optimizer:
            optim_path = self.ckpt_path(name, 'optim', is_best)
            optim_state_dict = self.get_state_dict(optim_path, load_to_cpu)
            model.optimizer.load_state_dict(optim_state_dict)

    @staticmethod
    def drop_missing(model, saved_state_dict):
        return {k: v for k, v in saved_state_dict.items()
                if k in model.state_dict().keys()}

    @staticmethod
    def replace_model_state(state_dict, replace):
        if replace is not None:
            for name, tensor in replace.items():
                state_dict[name] = tensor
        return state_dict

    @staticmethod
    def filter_optim_state_dict(state_dict, exclude):
        if exclude is not None:
            raise NotImplementedError  # TODO
        else:
            return state_dict

    @staticmethod
    def get_state_dict(path, load_to_cpu):
        if not torch.cuda.is_available() or load_to_cpu:
            return torch.load(path, map_location=lambda storage, loc: storage)
        else:
            return torch.load(path)

    def save(self, model, name, is_best, save_optim=False):
        model_path = self.ckpt_path(name, 'model', False)
        torch.save(model.state_dict(), model_path)
        if is_best:
            model_path = self.ckpt_path(name, 'model', True)
            torch.save(model.state_dict(), model_path)
        if save_optim:
            optim_path = self.ckpt_path(name, 'optim', is_best)
            torch.save(model.optimizer.state_dict(), optim_path)
