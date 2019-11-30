import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_model import *

class FFLayer(nn.Module):
    """
    Feed-forward layer with layer-norm, dropout, activation
    """
    def __init__(self, input_size, output_size, activation):
        super(FFLayer, self).__init__()

        self.layer = nn.Linear(input_size, output_size)
        self.layer_norm = nn.LayerNorm(input_size)
        
        if activation == 'relu':
            self.activation = lambda x: F.relu(x)
        elif activation == 'linear':
            self.activation = lambda x: x
        else:
            raise NotImplementedError

    def forward(self, inputs):
        normed = self.layer_norm(inputs)
        output = self.layer(normed)
        return self.activation(output)

    def freeze_weights(self):
        for param in self.layer.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.layer.parameters():
            param.requires_grad = True

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_sizes, spect_size, pcm_size, activation):
        super(Decoder, self).__init__()

        self.latent_size = latent_size
        print('latentsize:', self.latent_size)
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        # nn layers
        sizes = [self.latent_size] + self.hidden_sizes
        self.hiddens = []
        for i, s in enumerate(sizes[:-2]):
            self.hiddens.append(FFLayer(s, sizes[i+1], self.activation))
        # last layer has no activation
        self.hiddens.append(FFLayer(sizes[-1], spect_size, 'linear'))
        self.backprop_pcm = False
        if pcm_size != None:
            self.backprop_pcm = True
            self.hiddens.append(FFLayer(sizes[-1], pcm_size, 'linear')) 
        self.hiddens = nn.ModuleList(self.hiddens)

    def init_latents(self, n_points, input_d, device, mode):
        if mode == 'zeros':
            return torch.zeros(n_points, self.latent_size, dtype=torch.float,
                        requires_grad=True, device=device)
        elif mode == 'normal':
            stdev = 0.001
            print("Initializing from normal distribution with stdev {}".format(stdev), file=sys.stderr)

            return torch.tensor(np.random.normal(0, stdev, size=(n_points, input_d, self.latent_size)), dtype=torch.float,
                requires_grad=True, device=device)

        else:
            raise NotImplementedError

    def forward(self, latents):
        x = latents
        results = []
        if self.backprop_pcm:
            for layer in self.hiddens[:-2]:
                x = layer(x)
            spect = self.hiddens[-2](x)
            pcm = self.hiddens[-1](x)
            results.append(spect)
            results.append(pcm)
        else:
            for layer in self.hiddens:
                x = layer(x)
            results.append(x)
        mask = torch.ones_like(results[0])
        return results, mask

    def freeze_hiddens(self):
        for h in self.hiddens:
            h.freeze_weights()

    def unfreeze_hiddens(self):
        for h in self.hiddens:
            h.unfreeze_weights()


class LSTM(nn.Module):

    def __init__(self, latent_size, hidden_layers, output_size, layer_norm, dropout, activation):
        super(LSTM, self).__init__()

        self.latent_size = latent_size
        print('latentsize:', self.latent_size)
        
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        
        # lstm layer
        self.lstm = nn.LSTM(self.latent_size, self.latent_size, self.hidden_layers, batch_first = True)
        self.linear = nn.Linear(self.latent_size, self.output_size)


    def init_latents(self, n_points, input_d, device, mode):
        if mode == 'zeros':
            return torch.zeros(n_points, self.latent_size, dtype=torch.float,
                        requires_grad=True, device=device)
        elif mode == 'normal':
            stdev = 0.001
            print("Initializing from normal distribution with stdev {}".format(stdev), file=sys.stderr)

            return torch.tensor(np.random.normal(0, stdev, size=(n_points, input_d, self.latent_size)), dtype=torch.float,
                requires_grad=True, device=device)

            #return torch.randn(n_points, self.latent_size, dtype=torch.float,
            #            requires_grad=True, device=device) * stdev
        else:
            raise NotImplementedError

    def forward(self, latents):
        # latents = latents.transpose(0,1)
        # print('latent size is now', latents.shape)

        lstm_out, self.hidden = self.lstm(latents)
        y_pred = self.linear(lstm_out)
        # print('y_pred size is now', y_pred.shape)
        return y_pred
    def freeze_hiddens(self):
        for param in self.lstm.parameters():
            param.requires_grad = False
        for param in self.linear.parameters():
            param.requires_grad = False

    def unfreeze_hiddens(self):
        for param in self.lstm.parameters():
            param.requires_grad = True
        for param in self.linear.parameters():
            param.requires_grad = True

class TransformerUnit(nn.Module):
    def __init__(self, d_model, channels, heads, k_size):
        super().__init__()
        self.attn = MultiHeadAttention(heads, d_model) 
        self.activa = lambda x: F.relu(x)
        self.conv1D = nn.Conv1d(channels,channels,k_size, padding = k_size // 2)
        self.ff = FFLayer(d_model, d_model, 'relu')
        #self.out_spect = FFLayer(d_model, spect_size, 'relu')
        #self.out_mask = nn.Linear(d_model, spect_size)
        #self.mask_activa = lambda x: F.sigmoid(x)
    def forward(self, latents):
        e_output = self.attn(latents, latents, latents)
        e_acti = self.activa(e_output)
        conv_output = self.conv1D(latents)
        conv_acti = self.activa(conv_output)
        output = self.ff(conv_acti)
        return output
        #output = self.out_spect(conv_acti)
        #mask = self.out_mask(conv_acti)
        #mask = self.mask_activa(mask)
        #return output, mask
    def freeze_hiddens(self):
        self.attn.freeze_weights()
        for p in self.conv1D.parameters():
            p.requires_grad = False
        for p2 in self.ff.parameters():
            p2.requires_grad = False

class MultiTransformerLayer(nn.Module):
    def __init__(self, joint_size, spect_size, time_step, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.time_step = time_step
        self.pitch_unit1 = TransformerUnit(d_model, time_step, heads, 1)
        self.pitch_unit2 = TransformerUnit(d_model, time_step, heads, 3)
        self.pitch_unit3 = TransformerUnit(d_model, time_step, heads, 5)
        self.time_unit1 = TransformerUnit(time_step, d_model, heads, 1)
        self.time_unit2 = TransformerUnit(time_step, d_model, heads, 3)
        self.time_unit3 = TransformerUnit(time_step, d_model, heads, 5)
        self.unitList = [self.pitch_unit1, self.pitch_unit2, self.pitch_unit3,\
            self.time_unit1, self.time_unit2, self.time_unit3]
        self.norm = Norm(d_model)
    def forward(self, latents):
        pitch_out1 = self.pitch_unit1(latents)
        pitch_out2 = self.pitch_unit2(latents)
        pitch_out3 = self.pitch_unit3(latents)
        latents_t = torch.transpose(latents, 1,2)
        time_out1 = self.time_unit1(latents_t)
        time_out2 = self.time_unit2(latents_t)
        time_out3 = self.time_unit3(latents_t)
        pitch_out = pitch_out1 + pitch_out2 + pitch_out3
        time_out = time_out1 + time_out2 + time_out3
        time_out = torch.transpose(time_out, 1, 2)
        output = pitch_out + time_out
        out = self.norm(output)
        return out
    def freeze_hiddens(self):
        for u in self.unitList:
            u.freeze_hiddens()
        self.norm.freeze_weights()
class MultiTransformer(nn.Module):
    def __init__(self, joint_size, spect_size, time_step, d_model, N, heads):
        super().__init__()
        self.d_model = d_model
        self.time_step = time_step
        self.layers = get_clones(MultiTransformerLayer(joint_size, spect_size, time_step
                                , d_model, heads), N)
        self.project = nn.Linear(joint_size, time_step * d_model)
        self.out_spect = nn.Linear(d_model, spect_size)
        self.out_mask = nn.Linear(d_model, spect_size)
        self.activation = lambda x: F.sigmoid(x)
    def forward(self, joint_latents):
        latents = self.project(joint_latents)
        latents = torch.reshape(latents, (-1, self.time_step, self.d_model))
        for l in self.layers:
            latents = l(latents)
        spect_out = self.out_spect(latents)
        mask_out = self.out_mask(latents)
        mask = self.activation(mask_out)
        return spect_out, mask
    def freeze_hiddens(self):
        for l in self.layers:
            l.freeze_hiddens()
        for p in self.out_spect.parameters():
            p.requires_grad = False
        for p2 in self.out_mask.parameters():
            p2.requires_grad = False
        for p3 in self.project.parameters():
            p3.requires_grad = False
