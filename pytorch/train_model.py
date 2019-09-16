import sys
import math
import os
sys.path.append(os.getcwd())
import argparse
import time
import itertools
import json
import warnings
import csv

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dill as pickle

from vad_model import Decoder, LSTM
from transformer_model import Transformer
from utils import load_config, print_config, load_data
from collections import OrderedDict
def compute_corr(y_true, y_pred):
    row_mean_true = torch.mean(y_true, 2)
    col_mean_true = torch.mean(y_true, 1)
    row_mean_pred = torch.mean(y_pred, 2)
    col_mean_pred = torch.mean(y_pred, 1)
    col_diff_true = y_true - col_mean_true[:,None,:]
    col_diff_pred = y_pred - col_mean_pred[:,None,:]
    row_diff_true = y_true - row_mean_true[:,:,None]
    row_diff_pred = y_pred - row_mean_pred[:,:,None]
    row_stdev_true = torch.std(y_true, 2)
    row_stdev_pred = torch.std(y_pred, 2)
    col_stdev_true = torch.std(y_true, 1)
    col_stdev_pred = torch.std(y_pred, 1)
    row_stdevs = row_stdev_true*row_stdev_pred
    col_stdevs = col_stdev_true*col_stdev_pred
    row_stdevs[row_stdevs == 0] = 1
    col_stdevs[col_stdevs == 0] = 1
    row_corr = torch.sum(row_diff_true*row_diff_pred, 2)/row_stdevs
    col_corr = torch.sum(col_diff_true*col_diff_pred, 1)/col_stdevs
    #row_corr = torch.mean(row, 1)
    #col_corr = torch.mean(col, 1)
    return -row_corr, -col_corr
    

def compute_loss(spect_true, spect_pred, pcm_true, pcm_pred, corr, mode, epoch, total_epochs):
    """
    Returns the mean MSE
    """
    sq_errors= []
    w = 5
    eps = 1
    C = w - w * math.log(1 + w / eps)
    x =  torch.abs(spect_true - spect_pred)
    inner = x[x < w]
    outer = x[x >= w]
    inner = w * torch.log(1 + inner / eps)
    outer = outer - C
    spect_errors = torch.cat([inner, outer])
    #spect_errors = (spect_pred - spect_true) ** 2
    sq_errors.append(spect_errors)
    if pcm_pred.size() != (1,1,1):
        pcm_errors = (pcm_true - pcm_pred) ** 2
        sq_errors.append(pcm_errors)
    if corr:
        row_corr, col_corr = compute_corr(spect_true, spect_pred)
        row_unnorm = torch.mean(row_corr)
        col_unnorm = torch.mean(col_corr)
        if mode == 'torch_norm' and epoch > total_epochs - 2000:
            sq_final = torch.zeros(1,1,1)
            for sq in sq_errors:
                sq_det = sq.detach()
                sq_det[sq_det == 0] = 1
                cur_error = sq * 1/sq_det
                if sq_mat.size() == (1,1,1):
                    sq_final = torch.mean(cur_error)
                else:
                    sq_final += torch.mean(cur_error)
            row_det = row_corr.detach()
            row_det[row_det == 0] = 1
            col_det = col_corr.detach()
            col_det[col_det == 0] = 1
            row_mat = row_corr*1/row_det
            col_mat = col_corr*1/col_det
            row_final = torch.mean(row_mat)
            col_final = torch.mean(col_mat)
            #sq_final = torch.mean(sq_mat)
        else:
            row_final = torch.mean(row_corr)
            col_final = torch.mean(col_corr)
            sq_final = torch.zeros(1,1,1)
            for sq in sq_errors:
                if sq_final.size() == (1,1,1):
                    sq_final = torch.mean(sq)
                else:
                    sq_final += torch.mean(sq)
            #sq_final = torch.mean(sq_mat)
    else:
        row_final = 0
        col_final = 0
        row_unnorm = 0
        col_unnorm = 0
        if mode == 'torch_norm' and epoch > total_epochs - 2000:
            sq_final = torch.zeros(1,1,1)
            for sq in sq_errors:
                sq_det = sq.detach()
                sq_det[sq_det == 0] = 1
                cur_error = sq * 1/sq_det
                if sq_final.size() == (1,1,1):
                    sq_final = torch.mean(cur_error)
                else:
                    sq_final += torch.mean(cur_error)
        else:
            sq_final = torch.zeros(1,1,1)
            for sq in sq_errors:
                if sq_final.size() == (1,1,1):
                    sq_final = torch.mean(sq)
                else:
                    sq_final += torch.mean(sq)
    loss_prep = row_final + col_final + sq_final
    if mode == 'scalar_norm':
        loss_det = loss_prep.detach()
        loss_det[loss_det == 0] = 1
        loss = loss_prep/loss_det
    else:
        loss = loss_prep
    sq_unnorm = torch.zeros(1,1,1)
    for sq in sq_errors:
        if sq_unnorm.size() == (1,1,1):
            sq_unnorm = torch.mean(sq)
        else:
            sq_unnorm += torch.mean(sq)
    loss_unnorm = sq_unnorm + row_unnorm + col_unnorm
    return loss, loss_unnorm

def optimize_network(args, model, spect, pcm, mode, **kwargs):
    assert mode in ['train', 'test']
    print(spect.shape)

    

    # load appropriate hyper-parameters
    if mode == 'train':
        n_epochs = args['n_train_epochs']
        batch_size = args['train_batch_size']
        param_init = args['train_latents_init']

    elif mode == 'test':
        n_epochs = args['n_test_epochs']
        batch_size = args['test_batch_size']
        param_init = args['test_latents_init']

    n_points = spect.size()[0]
    input_dim = spect.size()[1]
    
    # initialize latent variables
    #latents = model.init_latents(n_points, input_dim, args['device'], param_init)
    latents = []
    adams = []
    if mode == 'train':
        lr = args['train_latents_Adam_lr']
    else:
        lr = args['test_latents_Adam_lr']
    latent_size = args['n_latents']
    stdev = 0.001
    device = args['device']
    for i in range(n_points):
        cur_latent = torch.tensor(np.random.normal(0, stdev, size = (1, input_dim, latent_size)),
            dtype = torch.float, requires_grad = True,  device = device)
        latents.append(cur_latent)
        cur_Adam = optim.Adam([cur_latent], lr = lr)
        adams.append(cur_Adam)  

    epoch = 0
    if mode == 'test':
        # freeze the network weights
        model.freeze_hiddens()
    
    optimizers = []
    schedulers = []
    if mode == 'train':
        model_lr = args['model_lr']
        net_optimizer = optim.Adam(model.parameters(), lr=model_lr)
        # for reduce lr on plateau
        net_scheduler = optim.lr_scheduler.ReduceLROnPlateau(net_optimizer, mode='min',
                factor=0.5, patience=10, verbose=True)

        optimizers = [net_optimizer]
        schedulers = [net_scheduler]


    # start optimization loop
    start_time = time.time()
    losses = []

    best_loss = None
    best_model = None
    best_epoch = None

    while True:
        epoch += 1
                

        order = np.random.permutation(n_points)
        cumu_loss = 0
        cumu_loss_unnorm = 0
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
            for j in idxes:
                adams[j].zero_grad()
            latents_use = torch.tensor(np.random.normal(0, stdev, size = (1,1,1)), dtype = torch.float, requires_grad = True, device = device)
            for j in idxes:
                cur_latent = latents[j]
                if latents_use.size() == (1,1,1):
                    latents_use = cur_latent
                else:
                    latents_use = torch.cat((latents_use, cur_latent), 0)

            preds = model(latents_use)
            if args['backprop_pcm']:
                [spect_pred_unmask, pcm_pred], mask = preds
                spect_pred = spect_pred_unmask*mask
            else:
                pcm_pred = torch.tensor(np.random.normal(0,stdev,size=(1,1,1)), dtype=torch.float, requires_grad=False, device=device)
                res, mask = preds
                spect_pred = res[0]*mask
            # loss with masking
            spect_true = spect[idxes]
            pcm_true = pcm[idxes]
            norm_mode = args['loss_normalization']
            corr = args['correlation_loss']
            if mode == 'train':
                total_epochs = args['n_train_epochs']
            else:
                total_epochs = args['n_test_epochs']
            loss, loss_unnorm = compute_loss(spect_true, spect_pred, pcm_true, pcm_pred, corr, norm_mode, epoch, total_epochs)
            
            

            cumu_loss += float(loss)
            cumu_loss_unnorm += float(loss_unnorm)
            loss.backward()
            for op in optimizers:
                op.step()
            for j in idxes:
                adams[j].step()
            # net_optimizer.step()
            # latent_optimizer.step()
             

        curr_time = time.time() - start_time
        avg_loss = cumu_loss / n_batches
        avg_unnormed_loss = cumu_loss_unnorm / n_batches
        if mode == 'train' and (best_loss == None or avg_unnormed_loss < best_loss):
            best_loss = avg_unnormed_loss
            best_model = pickle.loads(pickle.dumps(model))        
            best_model = best_model.to(torch.device('cpu'))
            best_epoch = epoch 

        if mode == 'train':
            print("Epoch {} - Average loss: {:.6f}, Cumulative loss: {:.6f}, Average Unnormed loss: {:.6f}, Best Loss: {:.6f}, Best Loss at Epoch {}({:.2f} s)".format(epoch, avg_loss, cumu_loss, avg_unnormed_loss,best_loss, best_epoch, curr_time),
                file=sys.stderr)
        else:
            print("Epoch {} - Average loss: {:.6f}, Cumulative loss: {:.6f}, Average Unnormed loss: {:.6f}({:.2f} s)".format(epoch, avg_loss, cumu_loss, avg_unnormed_loss,curr_time),
                file=sys.stderr)
        losses.append([float(avg_loss), float(avg_unnormed_loss)])

        # early stopping etc.
        if epoch >= n_epochs:
            print("Max number of epochs reached!", file=sys.stderr)
            break

        #for sch in schedulers:
        #    sch.step(cumu_loss)
            # net_scheduler.step(cumu_loss)
            # latent_scheduler.step(cumu_loss)

        sys.stderr.flush()
        sys.stdout.flush()

    if mode == 'train':
        # return final latent variables, to possibly initialize during testing
        train_latents = torch.cat(latents, dim=0)
        print('Best train epoch reached at {}'.format(best_epoch), file = sys.stderr)
        return train_latents, losses, float(best_loss), best_epoch, best_model

    elif mode == 'test':
        print("Final test loss: {}".format(losses[-1]), file=sys.stderr)

        # get final predictions to get loss wrt unmasked test data
        all_pred = []
        with torch.no_grad():
            idxes = np.arange(n_points)
            n_batches = math.ceil(n_points / batch_size)

            for i in range(n_batches):
                idx = idxes[i*batch_size : (i+1)*batch_size]
                latents_use = torch.tensor(np.random.normal(0, stdev, size = (1,1,1)), dtype = torch.float, requires_grad = True, device = device)
                for j in idx:
                    cur_latent = latents[j]
                    if latents_use.size() == (1,1,1):
                        latents_use = cur_latent
                    else:
                        latents_use = torch.cat((latents_use, cur_latent), 0)
                if args['backprop_pcm']:
                    [spect_pred, pcm_pred], mask = model(latents_use)
                else:
                    res, mask = model(latents_use)
                    spect_pred = res[0]*mask
                pcm_pred =  torch.tensor(np.random.normal(0, stdev, size = (1,1,1)), dtype = torch.float, requires_grad = True, device = device)
                all_pred.append(spect_pred)

        all_pred = torch.cat(all_pred, dim=0)
        
        _, final_test_loss = compute_loss(spect, all_pred, pcm, pcm_pred, False, None, epoch, total_epochs)
        final_test_loss = float(final_test_loss)


        test_latents = torch.cat(latents, dim=0)

        return losses, final_test_loss, all_pred, test_latents

def create_model(args):
    latent_size = args['n_latents']
    n_hidden_units = args['n_hidden_units']
    n_layers = args['n_decoder_layers']
    if type(n_hidden_units) == int and args['decoder'] == 'fc':
        hidden_sizes = [n_hidden_units] * n_layers
    mask_activ_fn = args['mask_activation']
    spect_dim = args['spect_dim']
    pcm_dim = args['pcm_dim']
    #activation = args['activation_fn']
    n_heads = args['n_heads']
    if not args['backprop_pcm']:
        pcm_dim = None
    if args['decoder'] == 'fc':
        print("making FC Decoder", file=sys.stderr)
        return Decoder(latent_size, hidden_sizes, spect_dim, pcm_dim, 'relu')
    elif args['decoder'] == 'transformer':
        print("making Transformer Decoder", file = sys.stderr)
        return Transformer(spect_dim, pcm_dim, latent_size, n_layers,n_heads, mask_activ_fn) 
    raise NotImplementedError


def write_to_csv(path, args, results):
    with open(path, 'a') as f:
        f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow([args['config_num'], args['decoder'], args['n_train_epochs'], args['n_test_epochs'], args['model_lr'], args['train_latents_Adam_lr'], args['test_latents_Adam_lr'], args['n_decoder_layers'], args['n_hidden_units'], args['train_batch_size'], args['test_batch_size'], args['backprop_pcm'], args['n_latents'], args['loss_normalization'], args['n_heads'], results['final_test_loss'], results['best_train_loss'], results['best_train_epoch'], results['model_file']])
    f.close()

def save_results(args, model_folder, results):
    train_latents = results['train_latents']
    test_latents = results['test_latents']
    test_loss = results['test_loss']
    final_test_loss = results['final_test_loss']
    train_loss = results['train_loss']
    final_pred = results['final_pred']
    model = results['best_model']
    best_loss = results['best_train_loss']
    best_epoch = results['best_train_epoch']
    spect_gt = results['spect_true']

    # write the latents to a file
    train_latents_file = os.path.join(model_folder, 'final_latents.h5')
    print("saving final train and test latents to {}".format(train_latents_file), file=sys.stderr)
    with h5py.File(train_latents_file, 'w') as f:
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
            'test_losses': test_loss,
            'final_test_loss': final_test_loss,
            'train_losses': train_loss,
            'best_train_loss': best_loss
        }, f, indent=4)
    # write final predictions to h5py file
    pred_file = os.path.join(model_folder, 'pred.h5')
    print("writing predictions to {}".format(pred_file), file=sys.stderr)
    with h5py.File(pred_file, 'w') as f:
        f.create_dataset('spect_pred', data=final_pred.cpu())
        f.create_dataset('spect_gt', data=spect_gt.cpu())
        f.create_dataset('spect_diff', data =(spect_gt-final_pred).cpu())

    # write arguments + config
    arg_file = os.path.join(model_folder, 'args.json')
    with open(arg_file, 'w') as f:
        json.dump(args, f, indent=4, default=lambda x: None)

    # save model dict
    model_file = os.path.join(model_folder, 'model.h5')
    print("saving model state_dict to {}".format(model_file), file=sys.stderr)
    torch.save(model.state_dict(), model_file)
    path = os.path.join(model_folder, '..', 'grid_search.csv')
    results['model_file'] = model_file
    print("writing to {}".format(path), file = sys.stderr)
    write_to_csv(path, args, results)
    print("All Done!!!")

def parse_arguments():
    datasets = [
        'drum',
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'test'], type=str)
    parser.add_argument('dataset', choices=datasets)
    parser.add_argument('config_folder')
    parser.add_argument('config_num', type=int)

    #parser.add_argument('--batch_size', type=int, default=32,
    #        help='Batch size')
    #parser.add_argument('--test_batch_size', type=int,
    #        help='Batch size during testing (overrides batch_size during test)')
    parser.add_argument('--n_train_epochs', type=int, default=5000)
    parser.add_argument('--n_test_epochs', type=int, default=5000)

    args = parser.parse_args()
    return vars(args)

def main():
    # read arguments
    args = parse_arguments()


    args['device'] = torch.device('cuda')
    #args['device'] = torch.device('cpu')

    load_config(args)
    print_config(args)


    # load data
    data = load_data(args)
    (train_spect, train_pcm), (test_spect, test_pcm) = data

    args['spect_dim'] = train_spect.size()[-1]
    args['pcm_dim'] = train_pcm.size()[-1]

    # create model
    model = create_model(args)

    model = model.to(args['device'])

    if args['action'] == 'train':
        # initialize training latents and train model
        train_latents, train_loss, best_loss, best_epoch, best_model = optimize_network(args, model, train_spect, train_pcm, 'train')
        model = best_model.to(args['device'])
        test_loss, final_loss, final_pred, test_latents = optimize_network(args, model, test_spect, test_pcm, 'test')

    elif args['action'] == 'test':
        # initialize testing latents and test model
        test_loss, final_loss, final_pred, test_latents = optimize_network(args, model, test_spect, test_pcm, 'test')


    # save statistics
    basedir = os.getcwd()
    #basedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_folder = os.path.join(basedir, 'model_saves', 'pytorch',
            args['config_folder'], 'config_' + str(args['config_num']))
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    save_results(args, model_folder, {
        'train_latents': train_latents,
        'test_latents': test_latents,
        'test_loss': test_loss,
        'final_test_loss': final_loss,
        'train_loss': train_loss,
        'final_pred': final_pred,
        'best_train_loss': best_loss,
        'best_train_epoch': best_epoch,
        'best_model': model,
        'spect_true': test_spect
    })

if __name__ == '__main__':
    main()
    sys.stdout.flush()




