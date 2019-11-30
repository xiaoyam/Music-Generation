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

from vad_model import Decoder, LSTM, FFLayer
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
    

def compute_loss(spect_true, spect_pred, corr, mode, epoch, total_epochs):
    """
    Returns the mean MSE
    """
    w = 5
    eps = 1
    C = w - w * math.log(1 + w / eps)
    x =  torch.abs(spect_true - spect_pred)
    inner = x[x < w]
    outer = x[x >= w]
    inner = w * torch.log(1 + inner / eps)
    outer = outer - C
    spect_errors = torch.cat([inner, outer])
    sq_errors = (spect_true - spect_pred)**2
    if corr:
        row_corr, col_corr = compute_corr(spect_true, spect_pred)
        row_unnorm = torch.mean(row_corr)
        col_unnorm = torch.mean(col_corr)
        row_final = torch.mean(row_corr)
        col_final = torch.mean(col_corr)
        wing_final = torch.mean(spect_errors)
    else:
        row_final = 0
        col_final = 0
        row_unnorm = 0
        col_unnorm = 0
        wing_final = torch.mean(spect_errors)
    loss_prep = row_final + col_final + wing_final
    loss = loss_prep
    #wing_unnorm = torch.mean(spect_errors)
    sq_unnorm = torch.mean(sq_errors)
    loss_unnorm = sq_unnorm + row_unnorm + col_unnorm
    return loss, loss_unnorm

def optimize_network(args, model_low, model_high, joint_model_low, joint_model_high, spect_low, spect_high, mode, **kwargs):
    assert mode in ['train', 'test']
    print(spect_low.shape, spect_high.shape)

    

    # load appropriate hyper-parameters
    if mode == 'train':
        n_epochs = args['n_train_epochs']
        batch_size = args['train_batch_size']
        param_init = args['train_latents_init']

    elif mode == 'test':
        n_epochs = args['n_test_epochs']
        batch_size = args['test_batch_size']
        param_init = args['test_latents_init']

    n_points = spect_low.size()[0]
    input_dim = spect_low.size()[1]
    
    
    # initialize latent variables
    #latents = model.init_latents(n_points, input_dim, args['device'], param_init)
    #latents_low = []
    #latents_high = []
    #adams_low = []
    #adams_high = []
    low_level_adams = []
    if mode == 'train':
        lr = args['train_latents_Adam_lr']
    else:
        lr = args['test_latents_Adam_lr']
    joint_latent_size = args['n_joint_dim']
    low_level_latents = []
    stdev = 0.001
    device = args['device']
    latent_size = args['n_latents']
    #joint_latent = torch.tensor(np.random.normal(0, stdev, size = (1, input_dim, joint_latent_size)),
    #	dtype = torch.float, requires_grad = True,  device = device)
    #adam_joint = optim.Adam([joint_latent], lr = lr)
    for i in range(n_points):
        cur_latent = torch.tensor(np.random.normal(0, stdev, size = (1, joint_latent_size)),
		dtype = torch.float, requires_grad = True,  device = device) 
        low_level_latents.append(cur_latent)
	#cur_latent_low = joint_model(joint_latent)
	#cur_latent_high = joint_model(joint_latent)
        #cur_latent_low = torch.tensor(np.random.normal(0, stdev, size = (1, input_dim, latent_size)),
        #    dtype = torch.float, requires_grad = True,  device = device)
        #cur_latent_high = torch.tensor(np.random.normal(0, stdev, size = (1, input_dim, latent_size)),
        #    dtype = torch.float, requires_grad = True,  device = device)
        cur_Adam = optim.Adam([cur_latent], lr = lr)
        low_level_adams.append(cur_Adam)

	#latents_low.append(cur_latent_low)
        #latents_high.append(cur_latent_high)
        #cur_Adam_low = optim.Adam([cur_latent_low], lr = lr)
        #cur_Adam_high = optim.Adam([cur_latent_high], lr = lr)
        #adams_low.append(cur_Adam_low)
        #adams_high.append(cur_Adam_high)  

    epoch = 0
    if mode == 'test':
        # freeze the network weights
        model_low.freeze_hiddens()
        model_high.freeze_hiddens()
        joint_model_low.freeze_weights()
        joint_model_high.freeze_weights()
    optimizers = []
    schedulers = []
    if mode == 'train':
        model_lr = args['model_lr']
        net_optimizer_low = optim.Adam(model_low.parameters(), lr=model_lr)
        net_optimizer_high = optim.Adam(model_high.parameters(), lr=model_lr)
        net_optimizer_joint_low = optim.Adam(joint_model_low.parameters(), lr=model_lr)
        net_optimizer_joint_high = optim.Adam(joint_model_high.parameters(), lr = model_lr)
        # for reduce lr on plateau
        net_scheduler_low = optim.lr_scheduler.ReduceLROnPlateau(net_optimizer_low, mode='min',
                factor=0.5, patience=10, verbose=True)
        net_scheduler_high = optim.lr_scheduler.ReduceLROnPlateau(net_optimizer_high, mode='min',
                factor=0.5, patience=10, verbose=True)
        net_scheduler_joint_low = optim.lr_scheduler.ReduceLROnPlateau(net_optimizer_joint_low, mode='min',
                factor=0.5, patience=10, verbose=True)
        net_scheduler_joint_high = optim.lr_scheduler.ReduceLROnPlateau(net_optimizer_joint_high, mode='min',
		factor=0.5, patience=10, verbose=True)

        optimizers = [net_optimizer_low, net_optimizer_high, net_optimizer_joint_low, net_optimizer_joint_high]
        schedulers = [net_scheduler_low, net_scheduler_high, net_scheduler_joint_low, net_scheduler_joint_high]


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
                low_level_adams[j].zero_grad()
                #adams_low[j].zero_grad()
                #adams_high[j].zero_grad()
            latents_use = torch.tensor(np.random.normal(0, stdev, size = (1,1,1)), dtype = torch.float, requires_grad = True, device = device)
            for j in idxes:
                cur_latent_low = torch.reshape(joint_model_low(low_level_latents[j]), (1, input_dim, latent_size))
                cur_latent_high = torch.reshape(joint_model_high(low_level_latents[j]), (1, input_dim, latent_size))
                #cur_latent_low = latents_low[j]
                #cur_latent_high = latents_high[j]
                if latents_use.size() == (1,1,1):
                    latents_use = cur_latent_low
                    latents_use_high = cur_latent_high
                else:
                    latents_use = torch.cat((latents_use, cur_latent_low), 0)
                    latents_use_high = torch.cat((latents_use_high, cur_latent_high), 0)
            
            preds_low = model_low(latents_use)
            preds_high = model_high(latents_use_high)
            res_low, mask_low = preds_low
            res_high, mask_high = preds_high
            spect_pred_low = res_low*mask_low
            spect_pred_high = res_high*mask_high
            # loss with masking
            spect_true_low = spect_low[idxes]
            spect_true_high = spect_high[idxes]
            norm_mode = args['loss_normalization']
            corr = args['correlation_loss']
            if mode == 'train':
                total_epochs = args['n_train_epochs']
            else:
                total_epochs = args['n_test_epochs']
            loss_low, loss_unnorm_low = compute_loss(spect_true_low, spect_pred_low, corr, norm_mode, epoch, total_epochs)
            loss_high, loss_unnorm_high = compute_loss(spect_true_high, spect_pred_high, corr, norm_mode, epoch, total_epochs)
            
            loss = (float(loss_low)+float(loss_high))/2
            loss_unnorm = (float(loss_unnorm_low)+float(loss_unnorm_high))/2

            cumu_loss += float(loss)
            cumu_loss_unnorm += float(loss_unnorm)
            loss_low.backward()
            loss_high.backward()
            for op in optimizers:
                op.step()
            for j in idxes:
                low_level_adams[j].step()
                #adams_low[j].step()
                #adams_high[j].step()
	    #adam_joint.step()
            # net_optimizer.step()
            # latent_optimizer.step()
             

        curr_time = time.time() - start_time
        avg_loss = cumu_loss / n_batches
        avg_unnormed_loss = cumu_loss_unnorm / n_batches
        if mode == 'train' and (best_loss == None or avg_unnormed_loss < best_loss):
            best_loss = avg_unnormed_loss
            best_model_low = pickle.loads(pickle.dumps(model_low))        
            best_model_low = best_model_low.to(torch.device('cpu'))
            best_model_high = pickle.loads(pickle.dumps(model_high))
            best_model_high = best_model_high.to(torch.device('cpu'))
            best_joint_model_low = pickle.loads(pickle.dumps(joint_model_low))
            best_joint_model_low = best_joint_model_low.to(torch.device('cpu'))
            best_joint_model_high = pickle.loads(pickle.dumps(joint_model_high))
            best_joint_model_high = best_joint_model_high.to(torch.device('cpu'))
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
        train_low_level_latents = torch.cat(low_level_latents, dim=0)
        train_latents_low = torch.reshape(joint_model_low(train_low_level_latents), (n_points, input_dim, latent_size))
        train_latents_high = torch.reshape(joint_model_high(train_low_level_latents), (n_points, input_dim, latent_size))
        #train_latents_low = torch.cat(latents_low, dim=0)
        #train_latents_high = torch.cat(latents_high, dim=0)
        
        print('Best train epoch reached at {}'.format(best_epoch), file = sys.stderr)
        return train_low_level_latents, train_latents_low, train_latents_high, losses, float(best_loss), best_epoch, best_model_low, best_model_high, best_joint_model_low, best_joint_model_high

    elif mode == 'test':
        print("Final test loss: {}".format(losses[-1]), file=sys.stderr)

        # get final predictions to get loss wrt unmasked test data
        all_pred_low = []
        all_pred_high = []
        with torch.no_grad():
            idxes = np.arange(n_points)
            n_batches = math.ceil(n_points / batch_size)

            for i in range(n_batches):
                idx = idxes[i*batch_size : (i+1)*batch_size]
                latents_use = torch.tensor(np.random.normal(0, stdev, size = (1,1,1)), dtype = torch.float, requires_grad = True, device = device)
                
                for j in idx:
                    cur_latent_low = torch.reshape(joint_model_low(low_level_latents[j]), (1, input_dim, latent_size))
                    cur_latent_high = torch.reshape(joint_model_high(low_level_latents[j]), (1, input_dim, latent_size))
                    #cur_latent_low = latents_low[j]
                    #cur_latent_high = latents_high[j]
                    if latents_use.size() == (1,1,1):
                        latents_use = cur_latent_low
                        latents_use_high = cur_latent_high
                    else:
                        latents_use = torch.cat((latents_use, cur_latent_low), 0)
                        latents_use_high = torch.cat((latents_use_high, cur_latent_high), 0)
                res_low, mask_low = model_low(latents_use)
                res_high, mask_high = model_high(latents_use_high)
                spect_pred_low = res_low*mask_low
                spect_pred_high = res_high*mask_high
                all_pred_low.append(spect_pred_low)
                all_pred_high.append(spect_pred_high)
       
        all_pred_low = torch.cat(all_pred_low, dim=0)
        all_pred_high = torch.cat(all_pred_high, dim=0)
        _, final_test_loss_low = compute_loss(spect_low, all_pred_low, False, None, epoch, total_epochs)
        _, final_test_loss_high = compute_loss(spect_high, all_pred_high, False, None, epoch, total_epochs)
        final_test_loss = (float(final_test_loss_low)+float(final_test_loss_high))/2

        test_low_level_latents = torch.cat(low_level_latents, dim = 0)
        test_latents_low = joint_model_low(test_low_level_latents).reshape((n_points, input_dim, latent_size))
        test_latents_high = joint_model_high(test_low_level_latents).reshape((n_points, input_dim, latent_size))
        #test_latents_low = torch.cat(latents_low, dim=0)
        #test_latents_high = torch.cat(latents_high, dim=0)

        return losses, final_test_loss, all_pred_low, all_pred_high, test_latents_low, test_latents_high, test_low_level_latents

def sample(train_joint_latents, joint_model_low, joint_model_high, model_low, model_high, args):
    #train_latents_low = train_latents_low.cpu().detach()
    #train_latents_high = train_latents_high.cpu().detach()
    train_joint_latents = train_joint_latents.cpu().detach()
    joint_model_low = joint_model_low.to('cpu')
    joint_model_high = joint_model_high.to('cpu')
    model_low = model_low.to('cpu')
    model_high = model_high.to('cpu')
    low_level_generated_low = []
    low_level_generated_high = []
    generated_low = []
    generated_high = []
    n_points = train_joint_latents.shape[0]
    latent_size = args['n_latents']
    time_step = args['time_step']
    for i in range(5):
        idxes = np.random.permutation(n_points)
        
        low_level = train_joint_latents[idxes[:20]]
        low_level_mean = torch.mean(low_level, 0)
        print('mean latent', low_level_mean.size())
        latent_low = torch.reshape(joint_model_low(low_level_mean), (1, time_step, latent_size))
        res_low,mask_low = model_low(latent_low)
        spect_low = res_low*mask_low
        low_level_generated_low.append(spect_low)
        latent_high = torch.reshape(joint_model_high(low_level_mean), (1, time_step, latent_size))
        res_high, mask_high = model_high(latent_high)
        spect_high = res_high*mask_high
        low_level_generated_high.append(spect_high)
        
        #normal_latents_low = train_latents_low[idxes[:100]]
        #normal_latents_high = train_latents_high[idxes[:100]]
        #normal_low_mean = torch.mean(normal_latents_low, 0)
        #normal_high_mean = torch.mean(normal_latents_high, 0)
        #normal_res_low, normal_m_low = model_low(normal_low_mean)
        #normal_res_high, normal_m_high = model_high(normal_high_mean)
        #normal_spect_low = normal_res_low*normal_m_low
        #normal_spect_high = normal_res_high*normal_m_high
        #generated_low.append(normal_spect_low)
        #generated_high.append(normal_spect_high)
    low_level_sample_low = torch.cat(low_level_generated_low, dim=0)
    low_level_sample_high = torch.cat(low_level_generated_high, dim=0)
    #sample_low = torch.cat(generated_low, dim=0)
    #sample_high = torch.cat(generated_high, dim=0)
    return low_level_sample_low, low_level_sample_high#, sample_low, sample_high

def create_model(args):
    latent_size = args['n_latents']
    n_layers = args['n_decoder_layers']
    mask_activ_fn = args['mask_activation']
    spect_dim_low = args['low_dim']
    spect_dim_high = args['high_dim']
    time_step = args['time_step']
    #activation = args['activation_fn']
    joint_latent_size = args['n_joint_dim']
    n_heads = args['n_heads']
    print("making Transformer Decoder", file = sys.stderr)
    
    return Transformer(spect_dim_low, latent_size, n_layers,n_heads, mask_activ_fn) ,\
           Transformer(spect_dim_high, latent_size, n_layers,n_heads, mask_activ_fn), \
           FFLayer(joint_latent_size, latent_size * time_step, 'relu'), \
	   FFLayer(joint_latent_size, latent_size * time_step, 'relu')
	   


def write_to_csv(path, args, results):
    with open(path, 'a') as f:
        f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow([args['config_num'], args['decoder'], args['n_train_epochs'], args['n_test_epochs'], args['model_lr'], args['train_latents_Adam_lr'], args['test_latents_Adam_lr'], args['n_decoder_layers'], args['train_batch_size'], args['test_batch_size'], args['n_latents'], args['n_joint_dim'], args['correlation_loss'], 'Wing Loss', args['loss_normalization'], args['mask_activation'], args['n_heads'], results['final_test_loss'], results['best_train_loss'], results['best_train_epoch'], results['model_file']])
    f.close()

def save_results(args, model_folder, results):
    train_latents_low = results['train_latents_low']
    train_latents_high = results['train_latents_high']
    train_low_level_latents = results['train_low_level_latents']
    test_low_level_latents = results['test_low_level_latents']
    test_latents_low = results['test_latents_low']
    test_latents_high = results['test_latents_high']
    test_loss = results['test_loss']
    final_test_loss = results['final_test_loss']
    train_loss = results['train_loss']
    final_pred_low = results['final_pred_low']
    final_pred_high = results['final_pred_high']
    model_low = results['best_model_low']
    model_high = results['best_model_high']
    joint_model_low = results['best_joint_model_low']
    joint_model_high = results['best_joint_model_high']
    best_loss = results['best_train_loss']
    best_epoch = results['best_train_epoch']
    spect_gt_low = results['spect_true_low']
    spect_gt_high = results['spect_true_high']

    gen_low1D, gen_high1D = sample(train_low_level_latents
	, joint_model_low, joint_model_high, model_low, model_high, args)

    # write the latents to a file
    train_latents_file = os.path.join(model_folder, 'final_latents.h5')
    print("saving final train and test latents to {}".format(train_latents_file), file=sys.stderr)
    with h5py.File(train_latents_file, 'w') as f:
        f.create_dataset('train_latents_low', data=train_latents_low.detach().cpu().numpy())
        f.create_dataset('train_latents_high', data=train_latents_low.detach().cpu().numpy())
        f.create_dataset('test_latents_low', data=test_latents_high.detach().cpu().numpy())
        f.create_dataset('test_latents_high', data=test_latents_high.detach().cpu().numpy())
        f.create_dataset('train_low_level_latents', data=train_low_level_latents.detach().cpu().numpy())
        f.create_dataset('test_low_level_latents', data=test_low_level_latents.detach().cpu().numpy())

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
        f.create_dataset('spect_pred_low', data=final_pred_low.cpu())
        f.create_dataset('spect_gt_low', data=spect_gt_low.cpu())
        f.create_dataset('spect_diff_low', data =(spect_gt_low-final_pred_low).cpu())
        f.create_dataset('spect_pred_high', data=final_pred_high.cpu())
        f.create_dataset('spect_gt_high', data=spect_gt_high.cpu())
        f.create_dataset('spect_diff_high', data =(spect_gt_high-final_pred_high).cpu())
        f.create_dataset('spect_gen1D_low', data=gen_low1D.detach().cpu())
        f.create_dataset('spect_gen1D_high', data=gen_high1D.detach().cpu())
        #f.create_dataset('spect_gen_low', data=gen_low.detach().cpu())
        #f.create_dataset('spect_gen_high', data=gen_high.detach().cpu())

    # write arguments + config
    arg_file = os.path.join(model_folder, 'args.json')
    with open(arg_file, 'w') as f:
        json.dump(args, f, indent=4, default=lambda x: None)

    # save model dict
    model_file = os.path.join(model_folder, 'model.h5')
    print("saving model state_dict to {}".format(model_file), file=sys.stderr)
    torch.save(model_low.state_dict(), model_file)
    torch.save(model_high.state_dict(), model_file)
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
    parser.add_argument('--n_train_epochs', type=int, default=20000)
    parser.add_argument('--n_test_epochs', type=int, default=20000)

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
    (train_low, train_high), (test_low, test_high) = data

    args['low_dim'] = train_low.size()[-1]
    args['high_dim'] = train_high.size()[-1]
    args['time_step'] = train_low.size()[-2]
    args['n_latents'] = 100

    # create model
    model_low, model_high, joint_model_low, joint_model_high = create_model(args)
    model_low = model_low.to(args['device'])
    model_high = model_high.to(args['device'])
    joint_model_low = joint_model_low.to(args['device'])
    joint_model_high = joint_model_high.to(args['device'])

    if args['action'] == 'train':
        # initialize training latents and train model
        train_low_level_latents, train_latents_low, train_latents_high, train_loss, best_loss, best_epoch, best_model_low, best_model_high, best_joint_model_low, best_joint_model_high = optimize_network(args, model_low, model_high, joint_model_low, joint_model_high, train_low, train_high, 'train')
        model_low = best_model_low.to(args['device'])
        model_high = best_model_high.to(args['device'])
        joint_model_low = best_joint_model_low.to(args['device'])
        joint_model_high = best_joint_model_high.to(args['device'])
        test_loss, final_loss, final_pred_low, final_pred_high, test_latents_low, test_latents_high, test_low_level_latents = optimize_network(args, model_low, model_high, joint_model_low, joint_model_high, test_low, test_high, 'test')

    elif args['action'] == 'test':
        # initialize testing latents and test model
        test_loss, final_loss, final_pred_low, final_pred_high, test_latents_low, test_latents_high, test_low_level_latents = optimize_network(args, model_low, model_high, joint_model_low, joint_model_high, test_low, test_high, 'test')


    # save statistics
    basedir = os.getcwd()
    #basedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_folder = os.path.join(basedir, 'model_saves', 'pytorch',
            args['config_folder'], 'config_' + str(args['config_num']))
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    save_results(args, model_folder, {
        'train_latents_low': train_latents_low,
        'train_latents_high': train_latents_high,
	'train_low_level_latents': train_low_level_latents,
        'test_latents_low': test_latents_low,
        'test_latents_high': test_latents_high,
	'test_low_level_latents': test_low_level_latents,
        'test_loss': test_loss,
        'final_test_loss': final_loss,
        'train_loss': train_loss,
        'final_pred_low': final_pred_low,
        'final_pred_high': final_pred_high,
        'best_train_loss': best_loss,
        'best_train_epoch': best_epoch,
        'best_model_low': model_low,
        'best_model_high': model_high,
	'best_joint_model_low': joint_model_low,
	'best_joint_model_high': joint_model_high,
        'spect_true_low': test_low,
        'spect_true_high': test_high
    })

if __name__ == '__main__':
    main()
    sys.stdout.flush()




