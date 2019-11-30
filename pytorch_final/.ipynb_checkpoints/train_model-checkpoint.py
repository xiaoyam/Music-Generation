import sys
import math
import os
import argparse
import time
import itertools
import json
import warnings

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.decomposition import PCA

from vad_model import MaskedDecoder, MaskedVAD, MaskedVAD2, MaskedVAD_free
from utils import load_config, print_config, load_data

def torch_mse_mask(y_true, y_pred, mask):
    """
    Returns the mean MSE
    """
    sq_errors = ((y_pred - y_true) * mask) ** 2
    mean_error = sq_errors.sum() / mask.sum()
    return mean_error

def optimize_network(args, model, y, mask, mode, **kwargs):
    assert mode in ['train', 'test']
    print(y.shape)

    # load appropriate hyper-parameters
    if mode == 'train':
        n_epochs = args['n_train_epochs']
        n_epochs = 4000
        batch_size = args['batch_size']
        param_init = args['latent_param_init']

    elif mode == 'test':
        n_epochs = args['n_test_epochs']
        if args.get('test_batch_size') is not None:
            batch_size = args['test_batch_size']
        else:
            batch_size = args['batch_size']
        param_init = args['test_latent_param_init']
        #n_epochs = 1
        #batch_size = 50

    print(f"Mode: {mode}")
    print(f"Batch size: {batch_size}")

    n_points = y.size()[0]

    # initialize latent variables
    if param_init == 'pca':
        pca = PCA(model.latent_size)
        pca.fit(y.cpu())

        latents = torch.tensor(pca.explained_variance_ratio_, dtype=torch.float, device=args['device'])
        latents = latents.repeat(n_points, 1)
        print(latents.size())

    elif param_init == 'train':
        assert mode != 'train'

        print("Initializing latents using training latents as mean", file=sys.stderr)
        train_latents = kwargs['train_latents']
        train_means = torch.mean(train_latents, 0)
        train_std = torch.std(train_latents, 0)
        latents = torch.tensor(np.random.normal(train_means, train_std, size=(n_points, model.latent_size)), device=args['device'])

    else:
        latents = model.init_latents(n_points, args['device'], param_init)

    # latent parameters to update
    latents.requires_grad = True
    latent_params = [latents]
    if args['model'] == 'vae_free':
        # randomly init log_var
        latent_log_var = torch.randn_like(latents, device=args['device'])
        latent_log_var.requires_grad = True
        latent_params.append(latent_log_var)

    epoch = 0
    if mode == 'test':
        # freeze the network weights
        model.freeze_hiddens()

    if mode == 'train':
        lr = args['net_lr']
        latent_lr = args['latent_param_lr']
        if args['use_adam']:
            net_optimizer = optim.Adam(model.parameters(), lr=lr)
            latent_optimizer = optim.Adam(latent_params, lr=latent_lr)
        else:
            net_optimizer = optim.SGD(model.parameters(), lr=lr)
            latent_optimizer = optim.SGD(latent_params, lr=latent_lr)
        # for reduce lr on plateau
        net_scheduler = optim.lr_scheduler.ReduceLROnPlateau(net_optimizer, mode='min',
                factor=0.5, patience=10, verbose=True)
        latent_scheduler = optim.lr_scheduler.ReduceLROnPlateau(latent_optimizer, mode='min',
                factor=0.5, patience=10, verbose=True)

        optimizers = [net_optimizer, latent_optimizer]
        schedulers = [net_scheduler, latent_scheduler]
        print(f"Optimizer: {net_optimizer}, {latent_optimizer}", file=sys.stderr)

    elif mode == 'test':
        latent_lr = args['test_latent_param_lr']

        if args['use_adam']:
            optimizer = optim.Adam(latent_params, lr=latent_lr)
        else:
            optimizer = optim.SGD(latent_params, lr=latent_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                factor=0.5, patience=10, verbose=True)

        optimizers = [optimizer]
        schedulers = [scheduler]
        print(f"Test optimizer: {optimizer}", file=sys.stderr)

    # start optimization loop
    start_time = time.time()
    losses = []

    while True:
        epoch += 1

        order = np.random.permutation(n_points)
        cumu_loss = 0
        cumu_total_loss = 0
        cumu_kl_loss = 0
        #
        n_batches = n_points // batch_size
        # model.set_verbose(False)
        for i in range(n_batches):
            # model.zero_grad()
            for op in optimizers:
                op.zero_grad()
            # net_optimizer.zero_grad()
            # latent_optimizer.zero_grad()

            idxes = order[i * batch_size: (i + 1) * batch_size]

            if args['model'] == 'vae_free':
                pred_y = model(latents[idxes], latent_log_var[idxes])
            elif args['sfm_transform']:
                pred_y, transform_mat = model(latents[idxes])
            else:
                pred_y = model(latents[idxes])
            # model.set_verbose(False)

            masked_train = y[idxes] * mask[idxes]

            # loss with masking
            loss = torch_mse_mask(y[idxes], pred_y, mask[idxes])
            
            if args['kl']:
                if args['model'] == 'vae':
                    z_var = torch.full_like(latents[idxes], args['log_var'])
                    kl_loss = 0.5 * torch.sum(torch.exp(z_var) + latents[idxes]**2 - 1. - z_var) / batch_size
                    total_loss = loss + args['ratio_kl'] * kl_loss

                elif args['model'] == 'vae_free':
                    kl_loss = 0.5 * torch.sum(torch.exp(latent_log_var[idxes]) + latents[idxes]**2 - 1. - latent_log_var[idxes])
                    kl_loss /= batch_size
                    total_loss = loss + args['ratio_kl'] * kl_loss

                else:
                    raise NotImplementedError
            else:
                kl_loss = 0.
                total_loss = loss

            # loss = loss_fn(pred_y, train_y[idxes]
            # loss *= train_mask[idxes]
            cumu_total_loss += float(total_loss)
            cumu_loss += float(loss)
            cumu_kl_loss += float(kl_loss)

            total_loss.backward()
            for op in optimizers:
                op.step()
            # net_optimizer.step()
            # latent_optimizer.step()

        curr_time = time.time() - start_time
        avg_loss = cumu_loss / n_batches

        avg_kl_loss = cumu_kl_loss / n_batches
        avg_total_loss = cumu_total_loss / n_batches

        print("Epoch {} - Average loss: {:.6f}, Cumulative loss: {:.6f}, KL loss: {:.6f}, Average total loss: {:.6f} ({:.2f} s)".format(epoch, avg_loss, cumu_loss, avg_kl_loss, avg_total_loss, curr_time),
                file=sys.stderr)
        losses.append([float(avg_loss), float(avg_kl_loss), float(avg_total_loss)])

        # early stopping etc.
        if epoch >= n_epochs:
            print("Max number of epochs reached!", file=sys.stderr)
            break

        if args.get('reduce', False):
            for sch in schedulers:
                sch.step(cumu_loss)
            # net_scheduler.step(cumu_loss)
            # latent_scheduler.step(cumu_loss)

        sys.stderr.flush()
        sys.stdout.flush()

    if mode == 'train':
        # return final latent variables, to possibly initialize during testing
        if args['model'] == 'vae_free':
            train_latents = latents, latent_log_var
        else:
            train_latents = latents
        return train_latents, losses

    elif mode == 'test':
        print("Final test loss: {}".format(losses[-1]), file=sys.stderr)

        # get final predictions to get loss wrt unmasked test data
        all_pred = []
        with torch.no_grad():
            idxes = np.arange(n_points)
            n_batches = math.ceil(n_points / batch_size)

            for i in range(n_batches):
                idx = idxes[i*batch_size : (i+1)*batch_size]
                if args['model'] == 'vae_free':
                    pred_y = model(latents[idx], latent_log_var[idx])
                elif args['sfm_transform']:
                    pred_y, transform_mat = model(latents[idx])
                else:
                    pred_y = model(latents[idx])
                all_pred.append(pred_y)

        all_pred = torch.cat(all_pred, dim=0)

        if kwargs['clean_y'] is not None:
            clean_y = kwargs['clean_y']
            #final_test_loss = float(loss_fn(all_pred * test_mask, clean_y * test_mask))
            #final_clean_loss = float(loss_fn(all_pred, clean_y))
            final_test_loss = float(torch_mse_mask(clean_y, all_pred, mask))
            final_clean_loss = float(torch_mse_mask(clean_y, all_pred, torch.ones_like(all_pred)))
            print("Masked test loss: {}".format(final_test_loss), file=sys.stderr)
            print("Clean test loss: {}".format(final_clean_loss), file=sys.stderr)

            mse = torch.mean(torch.mean((all_pred - clean_y) ** 2, -1), -1)
            print("Manual calculation: {}".format(mse), file=sys.stderr)

        if args['model'] == 'vae_free':
            test_latents = latents, latent_log_var
        else:
            test_latents = latents

        return losses, (final_test_loss, final_clean_loss), all_pred, test_latents

def create_model(args):
    latent_size = args['n_latent_params']
    n_hidden = args['n_hidden']
    n_hidden = 400
    if type(n_hidden) == int:
        hidden_sizes = [n_hidden] * args['n_hidden_layers']

    output_size = args['output_size']
    layer_norm = args['layer_norm']
    layer_norm = True
    dropout = args['dropout']
    activation = args['activation_fn']

    if args['model'] == 'ae':
        if args['kl']:
            warnings.warn('Adding KL term for non-variational autoencoder...', UserWarning)

        print("making MaskedDecoder", file=sys.stderr)
        return MaskedDecoder(latent_size, hidden_sizes, output_size, layer_norm,
                    dropout, activation)
    elif args['model'] == 'vae':
        if args['log_var'] is None:
            error_msg = "log_var cannot be None for vae. "
            error_msg += "Use --model vae_free if you want to optimize the log-variance."
            raise ValueError(error_msg)

        if args.get('sfm_transform', False):
            print("making MaskedVAD2", file=sys.stderr)
            output_size = 84*3
            return MaskedVAD2(latent_size, hidden_sizes, output_size, layer_norm,
                        dropout, activation, log_var=args['log_var'])
        
        else:
            print("making MaskedVAD", file=sys.stderr)
            return MaskedVAD(latent_size, hidden_sizes, output_size, layer_norm,
                        dropout, activation, log_var=args['log_var'])
    elif args['model'] == 'vae_free':
        if not args['kl']:
            warnings.warn("Training vae_free without KL term...", UserWarning)

        return MaskedVAD_free(latent_size, hidden_sizes, output_size, layer_norm,
                        dropout, activation)

    raise NotImplementedError

def test_clean(args):
    """
    Load model and test on clean (ground-truth) data
    """

    print("Loading pretrained model and testing on clean data...", file=sys.stderr)
    args['masked'] = None
    # load clean data
    data = load_data(args)
    (train_y, train_mask), (test_y, test_mask), test_y_true = data

    args['output_size'] = train_y.size()[-1]
    model = create_model(args)

    # load pretrained model
    basedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_folder = os.path.join(basedir, 'model_saves', 'pytorch', args['config_folder'], 'config_' + str(args['config_num']))
    model_file = os.path.join(model_folder, 'model.h5')
    if not os.path.isfile(model_file):
        print("Model not found at {}, cannot test!".format(model_file))
        sys.exit()

    model.load_state_dict(torch.load(model_file))
    model = model.to(args['device'])

    test_loss, final_losses, final_pred, test_latents = optimize_network(args, model, test_y, test_mask, 'test', clean_y=test_y_true)
    final_test_loss, final_clean_loss = final_losses

    # save results
    test_clean_results_file = os.path.join(model_folder, 'test_clean_results.json')
    print(f"writing to {test_clean_results_file}", file=sys.stderr)
    with open(test_clean_results_file, 'w') as f:
        json.dump({
            'test_loss': test_loss,
            'final_clean_loss': final_clean_loss,
            'final_test_loss': final_test_loss,
        }, f, indent=4)

    print("---------------DONE---------------", file=sys.stderr)

    sys.exit()

def make_missing(true_y, noise_ratio):
    """
    Randomly mask features with probability=noise ratio
    """
    p = np.random.random(true_y.shape)

    mask = np.zeros_like(p)
    mask[p > noise_ratio] = 1
    missing_y = np.copy(true_y)

    missing_y[mask == 0] = -100

    # print(true_y[:5, :5])
    # print(missing_y[:5, :5])
    # print(mask[:5, :5])

    return missing_y, mask

def make_missing_sfm(true_y, noise_ratio):
    """
    Remove full landmarks instead of individual coordinates
    """
    true_y = true_y.view(true_y.size()[0], -1, 2)

    # p = np.random.random(true_y.shape[:-1])
    # p = np.repeat(np.expand_dims(p, -1), 2, axis=-1)
    p = np.random.random(true_y.size()[:-1])
    p = np.repeat(np.expand_dims(p, -1), 2, axis=-1)

    mask = np.zeros_like(p)
    mask[p > noise_ratio] = 1
    missing_y = np.copy(true_y)

    missing_y[mask == 0] = -10.

    # print(true_y[:5, :5])
    # print(missing_y[:5, :5])
    # print(mask[:5, :5])

    # flatten
    missing_y = missing_y.reshape(missing_y.shape[0], -1)
    mask = mask.reshape(mask.shape[0], -1)

    return missing_y, mask

def test_missing(args):
    """
    Load model and test on partial data (either random or pre-generated)
    """

    if args['missing_mode'] is None:
        raise ValueError("--missing_mode should be set for test_missing")

    print("Loading pretrained model and testing on {} missing data...".format(args['missing_mode']),
            file=sys.stderr)
    args['masked'] = None

    # load clean data
    print("Loading clean data...")
    _, _, test_y_true = load_data(args)

    args['output_size'] = test_y_true.size()[-1]
    model = create_model(args)

    # load pretrained model
    basedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_folder = os.path.join(basedir, 'model_saves', 'pytorch', args['config_folder'], 'config_' + str(args['config_num']))
    model_file = os.path.join(model_folder, 'model.h5')
    if not os.path.isfile(model_file):
        print("Model not found at {}, cannot test!".format(model_file))
        sys.exit()

    print(f"Loading model from {model_file}")
    model.load_state_dict(torch.load(model_file))
    model = model.to(args['device'])

    results = []

    f = h5py.File(os.path.join(model_folder, 'missing_pred_{}.h5'.format(args['missing_mode'])), 'w')

    for noise in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if args['missing_mode'] == 'random':
            if args['dataset'] == 'sfm':
                missing_data, missing_mask = make_missing_sfm(test_y_true.cpu(), noise)
            else:
                missing_data, missing_mask = make_missing(test_y_true.cpu(), noise)
            missing_data = torch.tensor(missing_data, dtype=torch.float, device=args['device'])
            missing_mask = torch.tensor(missing_mask, dtype=torch.float, device=args['device'])
        elif args['missing_mode'] == 'fixed':
            args['masked'] = noise
            _, (missing_data, missing_mask), _ = load_data(args)

        test_loss, final_losses, final_pred, test_latents = optimize_network(args, model, missing_data, missing_mask, 'test', clean_y=test_y_true)
        final_test_loss, final_clean_loss = final_losses

        g = f.create_group('{}'.format(noise))
        g.create_dataset("pred", data=final_pred.cpu())
        g.create_dataset("input", data=missing_data.cpu())

        print(f"{noise}:\ttest: {final_test_loss}\tclean: {final_clean_loss}")
        results.append((noise, final_test_loss, final_clean_loss))

    f.close()

    print("###############", file=sys.stderr)
    print("Summary", file=sys.stderr)
    print("###############", file=sys.stderr)

    for noise, final_test_loss, final_clean_loss in results: 
        print(f"{noise}:\ttest: {final_test_loss}\tclean: {final_clean_loss}", file=sys.stderr)

    # check that output file isn't already written to
    test_missing_results_file = os.path.join(model_folder, 'test_missing_results_{}.json'.format(args['missing_mode']))
    if os.path.isfile(test_missing_results_file):
        print(f"{test_missing_results_file} already exists!", file=sys.stderr)
        sys.exit()
    # save results
    print(f"writing to {test_missing_results_file}", file=sys.stderr)
    results_dict = {noise: {'final_clean_loss': clean, 'final_test_loss': test} for (noise, test, clean) in results}
    with open(test_missing_results_file, 'w') as f:
        json.dump(results_dict, f, indent=4)

    print("---------------DONE---------------", file=sys.stderr)

    sys.exit()

def save_results(args, model_folder, results):
    train_latents = results['train_latents']
    test_latents = results['test_latents']
    test_loss = results['test_loss']
    final_clean_loss = results['final_clean_loss']
    final_test_loss = results['final_test_loss']
    train_loss = results['train_loss']
    final_pred = results['final_pred']
    model = results['model']

    # write the latents to a file
    train_latents_file = os.path.join(model_folder, 'final_latents.h5')
    print("saving final train and test latents to {}".format(train_latents_file), file=sys.stderr)
    with h5py.File(train_latents_file, 'w') as f:
        if args['model'] == 'vae_free':
            # save both mean and log variance
            g = f.create_group('train_latents')
            g.create_dataset('mu', data=train_latents[0].detach().cpu().numpy())
            g.create_dataset('log_var', data=train_latents[1].detach().cpu().numpy())

            g = f.create_group('test_latents')
            g.create_dataset('mu', data=test_latents[0].detach().cpu().numpy())
            g.create_dataset('log_var', data=test_latents[1].detach().cpu().numpy())
        else:
            f.create_dataset('train_latents', data=train_latents.detach().cpu().numpy())
            f.create_dataset('test_latents', data=test_latents.detach().cpu().numpy())

    results_file = os.path.join(model_folder, 'results.json')
    print("writing to {}".format(results_file), file=sys.stderr)

    # convert to 2d list
    if type(test_loss[0]) is not list:
        test_loss = [[x] for x in test_loss]
    if type(train_loss[0]) is not list:
        train_loss = [[x] for x in train_loss]

    with open(results_file, 'w') as f:
        json.dump({
            'test_loss': test_loss,
            'final_clean_loss': final_clean_loss,
            'final_test_loss': final_test_loss,
            'train_loss': train_loss
        }, f, indent=4)

    # write final predictions to h5py file
    pred_file = os.path.join(model_folder, 'pred.h5')
    print("writing predictions to {}".format(pred_file), file=sys.stderr)
    with h5py.File(pred_file, 'w') as f:
        f.create_dataset('y_pred', data=final_pred.cpu())

    # write arguments + config
    arg_file = os.path.join(model_folder, 'args.json')
    with open(arg_file, 'w') as f:
        json.dump(args, f, indent=4, default=lambda x: None)

    # save model dict
    model_file = os.path.join(model_folder, 'model.h5')
    print("saving model state_dict to {}".format(model_file), file=sys.stderr)
    torch.save(model.state_dict(), model_file)

def parse_arguments():
    datasets = [
        'sfm',
        'artificial',
        'mosi',
        'mosei',
        'mnist',
        'fashion_mnist',
        'drum',
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'test', 'test_clean', 'test_missing'], type=str)
    parser.add_argument('dataset', choices=datasets)
    parser.add_argument('config_folder')
    parser.add_argument('config_num', type=int)

    parser.add_argument('--batch_size', type=int, default=32,
            help='Batch size')
    parser.add_argument('--test_batch_size', type=int,
            help='Batch size during testing (overrides batch_size during test)')
    parser.add_argument('--n_train_epochs', type=int, default=50)
    parser.add_argument('--n_test_epochs', type=int, default=50)
    parser.add_argument('--cuda_device', type=int)
    parser.add_argument('--cuda', action='store_true', help='enable cuda')
    parser.add_argument('--model', choices=['ae', 'vae', 'vae_free'], default='ae',
            help='choose the model variant to train')
    parser.add_argument('--log_var', type=float,
            help='log of variance (for fixed-variance vae)')
    parser.add_argument('--reduce', action='store_true',
            help='reduce learning rate when loss plateaus')
    parser.add_argument('--sfm_transform', action='store_true')
    parser.add_argument('--kl', action='store_true',
            help='add KL-divergence term to VAE loss')
    parser.add_argument('--ratio_kl', type=float, default=1,
            help='ratio of KL term to reconstruction loss in VAE loss (if --kl is set). default is 1')

    parser.add_argument('--dataset_path', help='override default dataset location')

    parser.add_argument('--missing_mode', choices=['random', 'fixed'],
            help='randomly generate missing data, or load from data files')

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    return vars(args)

def main():
    # read arguments
    args = parse_arguments()

    if args['cuda_device']:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args['cuda_device'])

    if args['cuda']:
        args['device'] = torch.device('cuda')
    else:
        args['device'] = torch.device('cpu')

    load_config(args)
    print_config(args)

    if args['action'] == 'test_clean':
        test_clean(args)
        sys.exit()
    elif args['action'] == 'test_missing':
        test_missing(args)
        sys.exit()

    # load data
    data = load_data(args)
    (train_y, train_mask), (test_y, test_mask), test_y_true = data

    args['output_size'] = train_y.size()[-1]

    # create model
    model = create_model(args)

    model = model.to(args['device'])

    if args['action'] == 'train':
        # initialize training latents and train model
        train_latents, train_loss = optimize_network(args, model, train_y, train_mask, 'train', debug=args['debug'])

        test_loss, final_losses, final_pred, test_latents = optimize_network(args, model, test_y, test_mask, 'test', clean_y=test_y_true, train_latents=train_latents)

    elif args['action'] == 'test':
        # initialize testing latents and test model
        test_loss, final_losses, final_pred, test_latents = optimize_network(args, model, test_y, test_mask, 'test', clean_y=test_y_true)

    final_test_loss, final_clean_loss = final_losses

    # save statistics
    basedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_folder = os.path.join(basedir, 'model_saves', 'pytorch',
            args['config_folder'], 'config_' + str(args['config_num']))
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)

    save_results(args, model_folder, {
        'train_latents': train_latents,
        'test_latents': test_latents,
        'test_loss': test_loss,
        'final_clean_loss': final_clean_loss,
        'final_test_loss': final_test_loss,
        'train_loss': train_loss,
        'final_pred': final_pred,
        'model': model,
    })

if __name__ == '__main__':
    print(os.getcwd())
#     os.chdir('../')
#     print(os.getcwd())
    sys.argv = ['dummy','train', 'drum', '1', '4']
    main()
    sys.stdout.flush()




