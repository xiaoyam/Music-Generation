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
    train_spect, train_pcm, test_spect, test_pcm = load_fn(args, fname_start)

    def torchify(arr):
        return torch.tensor(arr, dtype=torch.float, device=args['device'])

    train_spect = torchify(train_spect)
    train_pcm = torchify(train_pcm)

    test_spect = torchify(test_spect)
    test_pcm = torchify(test_pcm)

    return (train_spect, train_pcm), (test_spect, test_pcm)


def load_drum(args, fname_start):
    # fname = 'drum'
    fname = fname_start + '.h5'
    with h5py.File(fname, 'r') as f:
        train_spect = f['train'][:]
        train_pcm = f['trainPCM'][:]

        test_spect = f['test'][:]
        test_pcm = f['testPCM'][:]

    print('shapes lol', train_spect.shape, test_spect.shape)
    return train_spect, train_pcm, test_spect, test_pcm

