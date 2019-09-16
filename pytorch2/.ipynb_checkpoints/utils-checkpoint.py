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
    config_file = os.path.join('../configs', args['config_folder'], 'config_{}.json'.format(args['config_num']))

    config = json.load(open(config_file, 'r'))

    if 'batch_size' in config:
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
    if args['dataset_path']:
        fname_start = args['dataset_path']
    elif args['dataset'] == 'artificial':
        raise ValueError('--dataset_path option must be set for artificial dataset')
    else:
        fname_start = os.path.join('data', args['dataset'], args['dataset'])
    # fname_start = args['data_file']
    print(fname_start, 'fname_start is here !!!')
    load_fns = {
            'sfm': load_sfm,
            'artificial': load_artificial,
            'mosi': load_mosi,
            'mosei': load_mosi,
            'mnist': load_mnist,
            'fashion_mnist': load_mnist,
            'drum': load_drum,
    }

    if args['dataset'] not in load_fns.keys():
        raise NotImplementedError
    else:
        load_fn = load_fns[args['dataset']]
        train_data, train_mask, test_data, test_mask, test_true = load_fn(args, fname_start)

    def torchify(arr):
        return torch.tensor(arr, dtype=torch.float, device=args['device'])

    train_data = torchify(train_data)
    train_mask = torchify(train_mask)

    test_data = torchify(test_data)
    test_mask = torchify(test_mask)
    test_true = torchify(test_true)

    return (train_data, train_mask), (test_data, test_mask), test_true

def load_artificial(args, fname_start):
    """
    Load synthetic dataset
    """
    # fname_start = args['data_file']

    if args['masked'] is not None:
        fname = fname_start + '_' + str(args['masked']) + '.h5'
        print(fname)

        with h5py.File(fname, 'r') as f:
            train_data = f['train_y'][:]
            train_mask = f['train_mask'][:]

            test_data = f['test_y'][:]
            test_mask = f['test_mask'][:]
            test_true = f['test_y_true'][:]

    else:
        fname = fname_start + '.h5'
        print(fname)

        with h5py.File(fname, 'r') as f:
            train_data = f['train_data'][:]
            train_mask = np.ones_like(train_data)

            test_data = f['test_data'][:]
            test_mask = np.ones_like(test_data)
            test_true = test_data

    return train_data, train_mask, test_data, test_mask, test_true

def load_sfm(args, fname_start):
    """
    Load facial landmark data
    """
    # fname_start = args['data_file']

    if args['masked'] is not None:
        fname = fname_start + '_' + str(args['masked']) + '.h5'
        # fname = os.path.join(fname_start, 'sfm_{}.h5'.format(args['masked']))
        print(fname)

        with h5py.File(fname, 'r') as f:
            train_data = f['train_2d'][:]
            train_mask = f['train_mask'][:]
            train_mask = np.repeat(np.expand_dims(train_mask, -1), 2, axis=-1)

            valid_data = f['valid_2d'][:]
            valid_mask = f['valid_mask'][:]
            valid_mask = np.repeat(np.expand_dims(valid_mask, -1), 2, axis=-1)

            test_data = f['test_2d'][:]
            test_mask = f['test_mask'][:]
            test_mask = np.repeat(np.expand_dims(test_mask, -1), 2, axis=-1)
            test_true = test_data

    else:
        fname = fname_start + '.h5'
        # fname = os.path.join(fname_start, 'sfm.h5')
        print(fname)

        with h5py.File(fname, 'r') as f:
            train_data = f['train_2d'][:]
            train_mask = np.ones_like(train_data)

            valid_data = f['valid_2d'][:]
            valid_mask = np.ones_like(valid_data)

            test_data = f['test_2d'][:]
            test_mask = np.ones_like(test_data)
            test_true = test_data

    # flatten landmarks
    print(train_data.shape)
    train_data = train_data.reshape(train_data.shape[0], -1)
    valid_data = valid_data.reshape(valid_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)

    train_mask = train_mask.reshape(train_mask.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)
    test_mask = test_mask.reshape(test_mask.shape[0], -1)

    test_true = test_true.reshape(test_true.shape[0], -1)
    
    print(train_data.shape)

    return train_data, train_mask, test_data, test_mask, test_true

def load_drum(args, fname_start):
    fname = fname_start + '.h5'
    with h5py.File(fname, 'r') as f:
        train_data = f['dataset_2'][:2000]
        train_mask = np.ones_like(train_data)

        test_data = f['dataset_2'][2000:]
        test_mask = np.ones_like(test_data)
        test_true = test_data

    train_data = train_data.reshape([train_data.shape[0], -1])
    train_mask = train_mask.reshape([train_mask.shape[0], -1])
    test_data = test_data.reshape([test_data.shape[0], -1])
    test_mask = test_mask.reshape([test_mask.shape[0], -1])
    test_true = test_true.reshape([test_true.shape[0], -1])
    print('shapes lol', train_data.shape, test_data.shape)
    return train_data, train_mask, test_data, test_mask, test_true

def load_mnist(args, fname_start):
    """
    Load MNIST-format data (MNIST, Fashion-MNIST)
    """
    # fname_start = args['data_file']
    print('args masked is here !!', args['masked'])

    if args['masked'] is not None:
        fname = f"{fname_start}_{args['masked']}.h5"
        print(fname, 'hahaha')

        with h5py.File(fname, 'r') as f:
            train_data = f['train_y'][:]
            train_mask = f['train_mask'][:]

            test_data = f['test_y'][:]
            test_mask = f['test_mask'][:]
            test_true = f['test_y_true'][:]

    else:
        fname = fname_start + '.h5'
        print(fname)

        with h5py.File(fname, 'r') as f:
            train_data = f['train_x'][:]
            train_mask = np.ones_like(train_data)

            test_data = f['test_x'][:]
            test_mask = np.ones_like(test_data)
            test_true = test_data

    # reshape data to flat array
    train_data = train_data.reshape([train_data.shape[0], -1])
    train_mask = train_mask.reshape([train_mask.shape[0], -1])
    test_data = test_data.reshape([test_data.shape[0], -1])
    test_mask = test_mask.reshape([test_mask.shape[0], -1])
    test_true = test_true.reshape([test_true.shape[0], -1])

    # normalize data to [0, 1]
    train_data = train_data.astype(np.float) / 255
    test_data = test_data.astype(np.float) / 255
    test_true = test_true.astype(np.float) / 255

    return train_data, train_mask, test_data, test_mask, test_true

def load_mosi(args, fname_start):
    """
    Load MOSI/MOSEI data
    """
    # fname_start = args['data_file']

    if args['masked'] is not None:
        fname = fname_start + '_' + str(args['masked']) + '.h5'
        print(fname)

        with h5py.File(fname, 'r') as f:
            train_data = f['train_y'][:]
            train_mask = f['train_mask'][:]

            test_data = f['test_y'][:]
            test_mask = f['test_mask'][:]
            test_true = f['test_y_true'][:]

    else:
        fname = fname_start + '.h5'
        print(fname)

        with h5py.File(fname, 'r') as f:
            train_data = f['train_data'][:]
            train_mask = np.ones_like(train_data)

            test_data = f['test_data'][:]
            test_mask = np.ones_like(test_data)
            test_true = test_data

    return train_data, train_mask, test_data, test_mask, test_true

