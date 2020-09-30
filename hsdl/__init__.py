"""General util functions."""
import math
import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm


epoch_pbar = None
iter_pbar = None
eval_train_pbar = None
eval_dev_pbar = None
eval_test_pbar = None


# define a global pad for all models so I can refer to it when saving attentions
pad = '<PAD>'


def aligned_print(keys: List, values: List, indent: int = 0) -> None:
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


def check_period(sent: str, period_char: str = '.') -> str:
    """Make sure a sentence has a period on the end."""
    if sent[-1] != period_char:
        sent += period_char
    return sent


def get_folder_path(*folders) -> str:
    """Gets a relative path, creating intermediate folders if necessary."""
    folders = list(folders)
    if len(folders) == 1:
        raise ValueError(folders)
    current = folders[0]
    if not os.path.exists(current):
        os.mkdir(current)
    for subfolder in folders[1:]:
        current = os.path.join(current, subfolder)
        if not os.path.exists(current):
            os.mkdir(current)
    return current


def get_pbar_name(pbar: tqdm) -> str:
    pbar_name = pbar.desc.replace(':', '')
    if '(' in pbar_name:
        pbar_name = pbar_name.split('(')[0].strip()
    return pbar_name


def in_ipynb() -> bool:
    try:
        get_ipython()
        return True
    except NameError:
        return False


def report(metrics: Dict) -> None:
    tqdm.write('Experiment results (%s):' % len(metrics['train']))
    for dataset in ['train', 'dev', 'test']:
        tqdm.write('\t%s' % dataset)
        tqdm.write('\t\tMean:   %s' % np.mean(metrics[dataset]))
        tqdm.write('\t\tMedian: %s' % np.median(metrics[dataset]))
        tqdm.write('\t\tMin:    %s' % min(metrics[dataset]))
        tqdm.write('\t\tMax:    %s' % max(metrics[dataset]))
        tqdm.write('\t\tStd:    %s' % np.std(metrics[dataset]))


def reset_pbar(pbar: tqdm) -> None:
    pbar.n = 0
    pbar.last_print_n = 0
    pbar.refresh()
    pbar.set_description(get_pbar_name(pbar))
