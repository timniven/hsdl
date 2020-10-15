"""General utilities."""
import json
import random

import numpy as np
import torch
try:
    get_ipython()
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm


class IxDict:

    def __init__(self, entities):
        self.entities = list(sorted(entities))
        self.ent_to_ix = dict(zip(self.entities, range(len(self.entities))))
        self.ix_to_ent = {v: k for k, v in self.ent_to_ix.items()}

    def __contains__(self, item):
        return item in self.ent_to_ix

    def __getitem__(self, item):
        try:
            if isinstance(item, str):
                return self.ent_to_ix[item]
            elif isinstance(item, int):
                return self.ix_to_ent[item]
            else:
                raise ValueError(type(item))
        except Exception as e:
            print(item)
            print(type(item))
            raise e

    def __len__(self):
        return len(self.entities)

    def keys(self, ent_to_ix=True):
        if ent_to_ix:
            return self.ent_to_ix.keys()
        else:
            return self.ix_to_ent.keys()

    def items(self, ix_to_ent=True):
        if ix_to_ent:
            return self.ix_to_ent.items()
        else:
            return self.ent_to_ix.items()

    @classmethod
    def load(cls, file_name):
        with open(file_name, 'r') as f:
            entities = json.loads(f.read())
            return cls(entities)

    def save(self, file_name):
        with open(file_name, 'w+') as f:
            f.write(json.dumps(self.entities))


def entropy(p, axis=0):
    """Calculate information entropy.

    Args:
      p: Vector representing a probability distribution.
      axis: Int, the axis along which to perform the calculation.

    Returns:
      Float.
    """
    l2 = np.log2(p)
    prod = p * l2
    h = -prod.sum(axis=axis)
    return h


def new_random_seed():
    return random.choice(range(10000))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tqdm():
    return tqdm
