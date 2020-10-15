"""Base config class and utilities."""
from collections import Mapping
import inspect
import json
from typing import Dict

import numpy as np


class Config(Mapping):
    """Base config class."""

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if isinstance(val, dict):
                setattr(self, key, Config(**val))
            else:
                setattr(self, key, val)

    def __getitem__(self, key: str):
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self):
        return len(self.to_dict())

    def __repr__(self):
        cfg = self.to_dict()
        lens = []  # attribute name lengths
        parent_attrs = {k: v for k, v in cfg.items()
                        if not isinstance(v, Config)}
        child_attrs = {k: v for k, v in cfg.items()
                       if isinstance(v, Config)}
        lens += [len(k) for k in parent_attrs]
        lens += [len(k) for k in child_attrs]
        for children in child_attrs.values():
            lens += [len(k) for k in children]
        max_len = np.max(lens)
        cfg = 'Config:\n'

        def pad(attr):
            while len(attr) < max_len:
                attr += ' '
            return attr

        for attr, val in sorted(parent_attrs.items()):
            attr = pad(attr)
            cfg += f'{attr}:\t\t{val}\n'
        for child in sorted(child_attrs.items()):
            cfg += f'{child}\n'
            for attr, val in sorted(child.items()):
                attr = pad(attr)
                cfg += f'{attr}:\t\t{val}'

        return cfg

    def copy(self):
        return Config(**self.to_dict())

    @classmethod
    def load(cls, file_path: str):
        with open(file_path) as f:
            params = json.loads(f.read())
        return cls(**params)

    def merge(self, params: Dict):
        for param, value in params.items():
            setattr(self, param, value)

    def properties(self):
        def is_property(v):
            return isinstance(v, property)
        return inspect.getmembers(self, is_property)

    def save(self, file_path: str):
        with open(file_path, 'w+') as f:
            f.write(json.dumps(self.to_dict()))

    def to_dict(self):
        cfg = {}
        for attr, val in self.__dict__.items():
            # NOTE: at most one level of depth in these config trees.
            if isinstance(val, Config):
                cfg[attr] = {}
                for child_attr, child_val in val.__dict__.items():
                    cfg[attr][child_attr] = child_val
                for child_attr, child_val in val.properties():
                    cfg[attr][child_attr] = child_val
            else:
                cfg[attr] = val
        return cfg
