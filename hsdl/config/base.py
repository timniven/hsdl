"""Base config class and utilities."""
from collections import Mapping
from typing import Dict

from hsdl import util


class Config(Mapping):
    """Base config class."""

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if isinstance(val, dict):
                setattr(self, key, Config(**val))
            else:
                setattr(self, key, val)

    def __getitem__(self, key: str):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self):
        return len(self.to_dict())

    def merge(self, params: Dict):
        for param, value in params.items():
            if param == 'ix':
                continue
            if '.' in param:
                attrs = param.split('.')
                o = self
                for attr in attrs[0:-1]:
                    o = o[attr]
                setattr(o, attrs[-1], value)
            else:
                setattr(self, param, value)

    def print(self):
        cfg = self.to_dict()
        util.aligned_print(cfg)

    def to_dict(self):
        cfg = {}
        for attr, val in self.__dict__.items():
            if isinstance(val, Config):
                child_dict = val.to_dict()
                for attr2, val2 in child_dict.items():
                    cfg[f'{attr}.{attr2}'] = val2
            else:
                cfg[attr] = val
        return cfg
