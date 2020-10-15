"""General utilities."""
import json
import math
import random
from typing import Dict

import numpy as np
import torch
try:
    get_ipython()
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm


# define a global pad for all models so I can refer to it when saving attentions
pad = '<PAD>'


# progress bars
epoch_pbar = None
iter_pbar = None
eval_train_pbar = None
eval_dev_pbar = None
eval_test_pbar = None


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


class TqdmWrapper:
    """Wraps tqdm progress bars with a restart method and base description."""

    def __init__(self, total, desc):
        self.pbar = tqdm(total=total, desc=desc)
        self.base_desc = desc

    def reset(self, total=None):
        self.pbar.reset(total)
        self.restart()

    def restart(self):
        self.pbar.n = 0
        self.pbar.last_print_n = 0
        self.pbar.refresh()
        self.pbar.set_description(self.base_desc)

    def set_description(self, desc):
        desc = f'{self.base_desc}: {desc}'
        self.pbar.set_description(desc)

    def update(self):
        self.pbar.update()


def aligned_print(params: Dict, indent: int = 0) -> None:
    keys = list(params.keys())
    values = list(params.values())
    # assume keys and values already sorted
    key_lengths = [len(x) for x in keys]
    # reason for +1: if on the border, i.e. 16 on 2, need to count as 3.
    # noting that a tab in python 3 is 8 spaces.
    key_tab_depth = [math.ceil((x + 1) / 8.) for x in key_lengths]
    max_tab = max(key_tab_depth) + 1
    num_tabs = [max_tab - x for x in key_tab_depth]
    for ix in range(len(keys)):
        prefix = ''
        if indent:
            prefix = '\t' * indent
        tqdm.write('%s%s%s%s'
                   % (prefix, keys[ix], '\t' * num_tabs[ix], values[ix]))


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


def get_pbar_name(pbar) -> str:
    pbar_name = pbar.desc.replace(':', '')
    if '(' in pbar_name:
        pbar_name = pbar_name.split('(')[0].strip()
    return pbar_name


def get_tqdm():
    return tqdm


def new_random_seed():
    return random.choice(range(10000))


def reset_pbar(pbar: tqdm) -> None:
    pbar.n = 0
    pbar.last_print_n = 0
    pbar.refresh()
    pbar.set_description(get_pbar_name(pbar))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
