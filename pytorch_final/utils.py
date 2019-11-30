"""
Utility functions for loading data, models etc.
"""
import json
import pprint
import sys
import os

import numpy as np
import torch
import h5py

def load_config(args):
    config_file = os.path.join('configs', args['config_folder'], 'config_{}.json'.format(args['config_num']))

    config = json.load(open(config_file, 'r'))

    if 'batch_size' in args:
        # command-line arg overrides config
        batch_size = args['batch_size']
        args.update(config)
        args['batch_size'] = batch_size
    else:
        args.update(config)

def print_config(args):
    """
    Pretty-print config
    """
    pprint.pprint(args, stream=sys.stderr, indent=2)

"""
Loading data
"""

def load_data(args):
    fname_start = os.path.join('data', args['dataset'], args['dataset'])
    print(fname_start, 'fname_start is here !!!')
    load_fns = {
            'drum': load_drum,
    }
    load_fn = load_drum
    train, test= load_fn(args, fname_start)

    def torchify(arr):
        return torch.tensor(arr, dtype=torch.float, device=args['device'])

    train = torchify(train)
    test = torchify(test)

    return train, test


def load_drum(args, fname_start):
    # fname = 'drum'
    fname = fname_start + '.h5'
    #fname = fname_start + '_2parts.h5'
    with h5py.File(fname, 'r') as f:
        train = f['train'][:]
        test = f['test'][:]

    print('shapes lol', train.shape, test.shape)
    return train, test

